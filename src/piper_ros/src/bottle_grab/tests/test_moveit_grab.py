"""Offline regression tests: no ROS master, camera, CAN or motion commands."""

import copy
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import rospy
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    PlanningScene,
    RobotState,
    RobotTrajectory,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

BOTTLE_GRAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTTLE_GRAB))

import moveit_grab as m
import detector as detector_module
import hand_eye as hand_eye_module
import joint_feedback as joint_feedback_module
import scene_manager as scene_manager_module
from arm_controller import ArmController
from detector import BottleDetector
from grasp_task import GraspTask
from gripper_controller import GripperController
from hand_eye import HandEyeCalibration
from joint_feedback import JointFeedback
from place_task import PlaceTask
from scene_manager import SceneManager


def trajectory(positions=((0.,) * 6, (0.02,) * 6)):
    result = RobotTrajectory()
    result.joint_trajectory.joint_names = list(m.ARM_JOINTS)
    for index, values in enumerate(positions):
        result.joint_trajectory.points.append(JointTrajectoryPoint(
            positions=list(values), time_from_start=rospy.Duration(index + 1)))
    return result


def state():
    result = RobotState()
    result.joint_state.name = m.ARM_JOINTS + m.FINGER_JOINTS
    result.joint_state.position = [0.] * 6 + [0.035, -0.035]
    return result


def make_config(**overrides):
    config = SimpleNamespace(
        execute_motion=True,
        model_path="bottle.pt",
        table_model_path="table.json",
        target_label=None,
        show_detection_window=False,
        support_edge_margin=0.01,
        lift_clearance=0.01,
        bottle_height=0.2,
        bottle_radius=0.025,
        grasp_height=0.1,
        pregrasp_distance=0.08,
        lift_distance=0.15,
        lift_table_margin=0.005,
        approach_yaws=[0.0, 30.0],
        opening_open=0.07,
        opening_close_command=0.0,
        grasp_opening_min=0.03,
        grasp_opening_max=0.05,
        grasp_effort_min=0.5,
        grasp_check_service="",
        gripper_effort=1.0,
        gripper_tolerance=0.002,
        gripper_feedback_timeout=0.5,
        gripper_settle_time=0.0,
        move_to_photo_pose=False,
        photo_joint_target=[0.0] * 6,
        post_grasp_joint_target=None,
        enable_place=True,
        place_mode="card",
        place_approach_height=0.05,
        place_tool_pose=[0.4, 0.0, 0.1, 0.0, 0.0, 0.0],
        place_offset_from_card=0.1,
        place_card_offset_xy=[0.0, 0.0],
        place_card_conf_thres=0.35,
        place_release_clearance=0.005,
        place_card_label=None,
        place_front_mode="toward_robot",
        place_front_yaw=0.0,
        place_photo_joint_target=None,
        bottle_center_offset_xy=[0.0, 0.0],
        estimate_bottle_center_from_table=True,
        T_link6_tool=np.eye(4),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_state(**overrides):
    state = SimpleNamespace(
        current_ee_pose_mat=np.eye(4),
        grasp_confirmed=False,
        T_link6_bottle=None,
        bottle_center_at_grasp=np.array([0.4, 0.0, 0.1]),
        created_bottle=False,
        gripper_service_sent=False,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


class MoveItGrabTests(unittest.TestCase):
    def test_no_legacy_command_paths(self):
        source = (Path(__file__).resolve().parents[1]
                  / 'moveit_grab.py').read_text()
        self.assertNotIn('PosCmd', source)
        self.assertNotIn('/pin_pos_cmd', source)
        self.assertNotIn('joint_pub', source)
        self.assertNotIn('move_to_target_smooth', source)
        self.assertNotIn('get_max_velocity_scaling_factor', source)

    # ------------------------------------------------------------------
    # Hand-eye / pose helpers.
    # ------------------------------------------------------------------
    def hand_eye(self):
        he = HandEyeCalibration.__new__(HandEyeCalibration)
        he.T_link6_color_optical = np.eye(4)
        return he

    def test_transform_has_explicit_direction(self):
        he = self.hand_eye()
        he.T_link6_color_optical[:3, 3] = [0, 0, 0.1]
        ee = np.eye(4)
        ee[:3, 3] = [1, 2, 3]
        pose = he.make_bottle_pose_in_base([0.1, 0.2, 0.3], ee)
        np.testing.assert_allclose(pose[:3, 3], [1.1, 2.2, 3.4])
        np.testing.assert_allclose(pose[:3, 0], [0, 0, -1])

    def test_flat_card_center_uses_table_plane(self):
        he = self.hand_eye()
        table_model = {
            'plane_normal': [0., 0., 1.], 'plane_offset': 0.0}
        ee = np.eye(4)
        ee[:3, 3] = [0., 0., 1.]
        he.T_link6_color_optical = np.array([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ])
        intrinsics = SimpleNamespace(
            fx=100., fy=100., ppx=320., ppy=240., width=640, height=480)
        with patch.object(
                hand_eye_module, 'deproject',
                side_effect=lambda intr, px, py, d: [
                    (px - intr.ppx) / intr.fx * d,
                    (py - intr.ppy) / intr.fy * d, d]):
            pose = he.make_bottle_center_pose_in_base(
                (320, 240), intrinsics, ee, table_model,
                center_height=0.0, bottle_height=0.2)
            np.testing.assert_allclose(pose[:3, 3], [0., 0., 0.], atol=1e-9)
            pose = he.make_bottle_center_pose_in_base(
                (320, 240), intrinsics, ee, table_model,
                center_height=0.1, bottle_height=0.2)
            np.testing.assert_allclose(pose[:3, 3], [0., 0., 0.1], atol=1e-9)

    def test_held_bottle_filter_uses_cylinder_not_height(self):
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.0, 0.10]
        ee = np.eye(4)
        ee[:3, 3] = [0.4, 0.0, 0.2]
        # 瓶子中心：属于夹爪中的瓶子。
        self.assertTrue(HandEyeCalibration.held_bottle_contains(
            [0.4, 0.0, 0.30], ee, T, 0.17, 0.03))
        # 卡片在瓶身外 10cm：即使高度相同也不能被过滤。
        self.assertFalse(HandEyeCalibration.held_bottle_contains(
            [0.4, 0.10, 0.30], ee, T, 0.17, 0.03))

    # ------------------------------------------------------------------
    # Arm controller.
    # ------------------------------------------------------------------
    def arm_controller(self, execute_motion=False):
        scene = MagicMock()
        scene.require_unchanged_scan = MagicMock()
        scene.get_scene = MagicMock(
            return_value=SimpleNamespace(scene=PlanningScene(robot_state=state())))
        scene.validate_robot_state = MagicMock()
        scene.tf_buffer = MagicMock()
        jf = MagicMock()
        jf._fresh_joint_positions.return_value = {name: 0.0 for name in m.ARM_JOINTS}
        task_state = SimpleNamespace(grasp_confirmed=False)
        ctl = ArmController(
            MagicMock(), MagicMock(), scene, jf, MagicMock(), execute_motion,
            task_state, MagicMock(), 0.05, 0.05, 0.10)
        return ctl, task_state, scene, jf

    def test_partial_and_nonfinite_cartesian_never_execute(self):
        for fraction in (0.5, float('nan')):
            ctl, _, _, _ = self.arm_controller()
            ctl.arm.compute_cartesian_path.return_value = (trajectory(), fraction)
            with self.assertRaises(RuntimeError):
                ctl.move_cartesian(np.eye(4))
            ctl.arm.execute.assert_not_called()
            ctl.arm.retime_trajectory.assert_not_called()

    def test_preview_displays_without_execute(self):
        ctl, _, scene, jf = self.arm_controller()
        ctl.display_pub = MagicMock()
        ctl.validate_trajectory = MagicMock()
        ctl.robot.get_current_state.return_value = state()
        scene.require_unchanged_scan = MagicMock()
        with patch.object(m.rospy, 'sleep'):
            with self.assertRaises(m.PlanningOnlyStop):
                ctl.finish_motion(trajectory())
        ctl.display_pub.publish.assert_called_once()
        ctl.arm.execute.assert_not_called()

    def test_changed_scene_blocks_execute(self):
        ctl, _, scene, jf = self.arm_controller(execute_motion=True)
        ctl.display_pub = MagicMock()
        ctl.validate_trajectory = MagicMock()
        ctl.robot.get_current_state.return_value = state()
        scene.require_unchanged_scan.side_effect = [None, RuntimeError('changed map')]
        with self.assertRaises(RuntimeError):
            ctl.finish_motion(trajectory())
        ctl.arm.execute.assert_not_called()

    def test_invalid_trajectory_and_between_sample_collision(self):
        ctl, _, scene, jf = self.arm_controller()
        ctl.validate_trajectory = ArmController.validate_trajectory.__get__(ctl)
        scene.get_scene = MagicMock(
            return_value=SimpleNamespace(scene=PlanningScene(robot_state=state())))
        checked = []

        def check(robot_state):
            x = robot_state.joint_state.position[0]
            checked.append(x)
            if 0.04 < x < 0.08:
                raise RuntimeError('between-point collision')
        scene.validate_robot_state = check
        with self.assertRaisesRegex(RuntimeError, 'between-point'):
            ctl.validate_trajectory(trajectory(((0.,) * 6, (0.12,) * 6)))
        self.assertGreater(len(checked), 2)
        with self.assertRaisesRegex(RuntimeError, '非有限'):
            ctl.validate_trajectory(trajectory(((float('nan'),) * 6,)))

    def test_stale_trajectory_start_is_rejected(self):
        ctl, _, _, jf = self.arm_controller()
        ctl.check_trajectory_start(trajectory())
        jf._fresh_joint_positions = MagicMock(return_value={
            **{name: 0.0 for name in m.ARM_JOINTS}, 'joint5': -0.013})
        with self.assertRaisesRegex(RuntimeError, '起点与实机状态'):
            ctl.check_trajectory_start(trajectory())

    # ------------------------------------------------------------------
    # Gripper controller.
    # ------------------------------------------------------------------
    def gripper_controller(self):
        jf = MagicMock()
        jf._fresh_joint_positions.return_value = {'joint7': 0.035, 'joint8': -0.035}
        jf.effort.return_value = 0.0
        scene = MagicMock()
        scene.require_unchanged_scan = MagicMock()
        scene.get_scene = MagicMock(
            return_value=SimpleNamespace(scene=PlanningScene(robot_state=state())))
        scene.validate_robot_state = MagicMock()
        task_state = SimpleNamespace(gripper_service_sent=False)
        gripper = GripperController(
            jf, scene, MagicMock(), True, task_state, 1.0, 0.002, 0.5, 0.0,
            0.03, 0.05, 0.5, "")
        return gripper, jf, scene, task_state

    def test_gripper_sweep_checks_both_fingers_and_table(self):
        gripper, jf, scene, task_state = self.gripper_controller()
        checked = []

        def valid(robot_state):
            joint = dict(zip(
                robot_state.joint_state.name, robot_state.joint_state.position))
            self.assertAlmostEqual(joint['joint8'], -joint['joint7'])
            checked.append(joint['joint7'])
            if joint['joint7'] < 0.025:
                raise RuntimeError('table/link8 collision')
        scene.validate_robot_state = valid
        with self.assertRaisesRegex(RuntimeError, 'table/link8'):
            gripper.command(0.01)
        self.assertGreater(len(checked), 2)
        gripper.gripper_service.assert_not_called()

    def test_grasp_stability_requires_load_not_only_opening(self):
        gripper, jf, scene, task_state = self.gripper_controller()
        jf._fresh_joint_positions.return_value = {'joint7': 0.02, 'joint8': -0.02}
        jf.effort.return_value = 0.0
        gripper.gripper_feedback_timeout = 0.05
        gripper.gripper_settle_time = 0
        with patch.object(m.rospy, 'is_shutdown', return_value=False), \
                patch.object(m.rospy, 'sleep'):
            with self.assertRaises(RuntimeError):
                gripper.confirm()
        jf.effort.return_value = 0.6
        with patch.object(m.rospy, 'is_shutdown', return_value=False), \
                patch.object(m.rospy, 'loginfo'):
            gripper.confirm()

    # ------------------------------------------------------------------
    # Bottle detector loop.
    # ------------------------------------------------------------------
    def detection_detector(self):
        det = BottleDetector.__new__(BottleDetector)
        det.model_path = "bottle.pt"
        det.imgsz = 640
        det.conf_thres = 0.5
        det.model = MagicMock()
        det.model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(
                numpy=lambda: np.array([[300, 200, 340, 260]]))),
            conf=SimpleNamespace(cpu=lambda: SimpleNamespace(
                numpy=lambda: np.array([0.9])))))]
        return det

    def detection_components(self):
        camera = MagicMock()
        camera.frames.return_value = None
        camera.depth_scale = 0.001
        camera.filters = []
        raw_depth = MagicMock()
        raw_depth.get_width.return_value = 640
        raw_depth.get_height.return_value = 480
        color_frame = MagicMock()
        color_frame.get_data.return_value = np.zeros((480, 640, 3), np.uint8)
        camera.align.process.return_value = SimpleNamespace(
            get_depth_frame=lambda: raw_depth,
            get_color_frame=lambda: color_frame)
        intrinsics = SimpleNamespace(
            fx=455., fy=455., ppx=320., ppy=240., width=640, height=480)
        filtered_depth = SimpleNamespace(
            get_width=lambda: 640, get_height=lambda: 480,
            profile=SimpleNamespace(
                as_video_stream_profile=lambda: SimpleNamespace(
                    intrinsics=intrinsics)))

        arm = MagicMock()
        arm.current_ee_matrix.return_value = np.eye(4)
        jf = MagicMock()
        jf._fresh_joint_positions.return_value = {name: 0.0 for name in m.ARM_JOINTS}
        task_state = make_state(T_link6_bottle=None)
        he = MagicMock()
        surface = np.eye(4)
        surface[:3, 3] = [0.5, 0.0, 0.35]
        table = np.eye(4)
        table[:3, 3] = [0.7, 0.0, 0.1]
        he.make_bottle_pose_in_base.return_value = surface
        he.make_bottle_center_pose_in_base.return_value = table
        he.held_bottle_contains.return_value = False
        return camera, arm, jf, task_state, he, surface, table, filtered_depth

    def run_detection(self, det, **kwargs):
        camera, arm, jf, task_state, he, surface, table, filtered_depth = \
            self.detection_components()
        with patch.object(detector_module, 'classify_bottle_color',
                          return_value='bottle-g'), \
                patch.object(detector_module, 'get_depth_percentile',
                             return_value=0.35), \
                patch.object(detector_module, 'deproject',
                             return_value=[0., 0., 0.35]), \
                patch.object(detector_module, 'apply_filters',
                             return_value=MagicMock(
                                 as_depth_frame=lambda: filtered_depth)), \
                patch.object(detector_module.rospy, 'is_shutdown',
                             return_value=False):
            return det.get_pose(
                camera, arm, jf, task_state, he,
                table_model={'plane_normal': [0., 0., 1.],
                             'plane_offset': 0.0},
                bottle_height=0.2, bottle_radius=0.03,
                estimate_bottle_center_from_table=True,
                show_detection_window=False, max_frames=12, **kwargs)

    def test_card_detection_can_use_depth_surface_point(self):
        det = self.detection_detector()
        pose = self.run_detection(det, use_surface_point=True)
        # 立着的卡片必须返回深度表面点，而不是桌面平面交点。
        np.testing.assert_allclose(pose[:3, 3], [0.5, 0.0, 0.35])

    def test_bottle_detection_keeps_table_ray_center_by_default(self):
        det = self.detection_detector()
        pose = self.run_detection(det)
        np.testing.assert_allclose(pose[:3, 3], [0.7, 0.0, 0.1])

    # ------------------------------------------------------------------
    # Scene manager.
    # ------------------------------------------------------------------
    def scene_manager(self):
        return SceneManager(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            "table.json", MagicMock(), support_edge_margin=0.01)

    def test_acm_preserves_existing_pairs_and_defaults(self):
        sm = self.scene_manager()
        acm = AllowedCollisionMatrix(
            entry_names=['link6', 'table'],
            entry_values=[AllowedCollisionEntry(enabled=[True, False]),
                          AllowedCollisionEntry(enabled=[False, False])],
            default_entry_names=['fixture'], default_entry_values=[False])
        sm.get_acm = MagicMock(return_value=copy.deepcopy(acm))
        sm.apply_acm = MagicMock()
        sm.set_contact_pairs([('target_bottle', 'link7')], True)
        updated = sm.apply_acm.call_args[0][0]
        self.assertTrue(updated.entry_values[0].enabled[0])
        self.assertFalse(updated.entry_values[0].enabled[1])
        self.assertEqual(list(updated.default_entry_names), ['fixture'])
        i = updated.entry_names.index('target_bottle')
        j = updated.entry_names.index('link7')
        self.assertTrue(updated.entry_values[i].enabled[j])
        self.assertTrue(updated.entry_values[j].enabled[i])
        self.assertTrue(all(len(row.enabled) == len(updated.entry_names)
                            for row in updated.entry_values))

    def test_existing_attached_bottle_never_removed(self):
        sm = self.scene_manager()
        sm.require_support_footprint = MagicMock()
        sm.table_model = {'bounds_max': [1, 1, 0.02]}
        sm.scene.get_attached_objects.return_value = {'target_bottle': object()}
        sm.scene.get_known_object_names.return_value = []
        task_state = make_state()
        with self.assertRaisesRegex(RuntimeError, '已有target_bottle'):
            sm.add_bottle([0.4, 0, 0.1], 0.2, 0.025, 0.15, 0.005, task_state)
        sm.scene.remove_world_object.assert_not_called()
        sm.scene.add_cylinder.assert_not_called()

    def test_wait_object_allows_ignoring_known_after_detach(self):
        sm = self.scene_manager()
        sm.scene.get_known_object_names.return_value = ['target_bottle']
        sm.scene.get_attached_objects.return_value = {}
        with patch.object(m.rospy, 'sleep'):
            sm.wait_object('target_bottle', known=None, attached=False)
        sm.scene.get_attached_objects.return_value = {'target_bottle': object()}
        with patch.object(m.rospy, 'sleep'), \
                patch.object(scene_manager_module.time, 'monotonic',
                             side_effect=[0., 0., 10.]):
            with self.assertRaisesRegex(RuntimeError, 'attached=True'):
                sm.wait_object('target_bottle', known=None, attached=False)

    # ------------------------------------------------------------------
    # Joint feedback settling.
    # ------------------------------------------------------------------
    def test_settle_waits_for_release_rebound(self):
        jf = JointFeedback()

        def pose(**values):
            result = {name: 0.0 for name in m.ARM_JOINTS}
            result.update(values)
            return result

        frames = [pose(), pose(joint5=-0.013)]
        jf._fresh_joint_positions = MagicMock(
            side_effect=lambda: frames.pop(0) if frames else pose(joint5=-0.013))
        clock = itertools.count(0.0, 0.1)
        with patch.object(m.rospy, 'sleep'), \
                patch.object(joint_feedback_module.time, 'monotonic',
                             side_effect=lambda: next(clock)):
            jf.wait_for_arm_settled(timeout=2.0, hold=0.2)

    def test_settle_times_out_while_arm_keeps_moving(self):
        jf = JointFeedback()

        def pose(**values):
            result = {name: 0.0 for name in m.ARM_JOINTS}
            result.update(values)
            return result

        frames = itertools.cycle([pose(), pose(joint5=-0.013)])
        jf._fresh_joint_positions = MagicMock(side_effect=lambda: next(frames))
        clock = itertools.count(0.0, 0.1)
        with patch.object(m.rospy, 'sleep'), \
                patch.object(joint_feedback_module.time, 'monotonic',
                             side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(RuntimeError, '未静止'):
                jf.wait_for_arm_settled(timeout=1.0, hold=0.2)

    # ------------------------------------------------------------------
    # Grasp task workflow.
    # ------------------------------------------------------------------
    def grasp_task(self, **kwargs):
        config = make_config(**kwargs.get('config', {}))
        task_state = make_state()
        arm = MagicMock()
        arm.move_joints = MagicMock()
        arm.finish_motion = MagicMock()
        arm.move_cartesian = MagicMock()
        lifted = np.eye(4)
        lifted[2, 3] = 0.15
        arm.current_ee_matrix = MagicMock(
            side_effect=[np.eye(4), np.eye(4), lifted])
        gripper = MagicMock()
        gripper.command = MagicMock()
        gripper.confirm = MagicMock()
        gripper.current_opening = MagicMock(return_value=0.07)
        scene = MagicMock()
        scene.require_scan_scene = MagicMock()
        scene.table_model = {
            'bounds_min': [0.1, -0.3, -0.05],
            'bounds_max': [0.8, 0.3, 0.02],
            'plane_normal': [0., 0., 1.],
            'plane_offset': 0.0,
        }
        scene.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        scene.apply_acm = MagicMock()
        scene.set_contact_pairs = MagicMock()
        scene.wait_object = MagicMock()
        scene.validate_robot_state = MagicMock()
        scene.get_scene = MagicMock(
            return_value=SimpleNamespace(scene=PlanningScene()))
        scene.scene.get_attached_objects.return_value = {}
        jf = MagicMock()
        jf.wait_for_joint_feedback = MagicMock()
        detector = MagicMock()
        detector.load = MagicMock()
        detector.get_pose = MagicMock(return_value=np.eye(4))
        camera = MagicMock()
        camera.start = MagicMock()
        hand_eye = MagicMock()
        place_task = MagicMock()
        place_task.execute = MagicMock()
        task = GraspTask(
            config, task_state, arm, gripper, scene, jf, detector, camera,
            hand_eye, place_task)
        return task, task_state, arm, gripper, scene, jf, detector, place_task

    def test_unknown_grasp_stops_before_attach_or_lift(self):
        task, task_state, arm, gripper, scene, jf, detector, place = \
            self.grasp_task()
        task.plan_grasp_candidate = MagicMock(
            return_value=(trajectory(), np.eye(4)))
        task.get_bottle_pose = MagicMock(return_value=np.eye(4))
        gripper.confirm.side_effect = RuntimeError('no contact')
        with self.assertRaisesRegex(RuntimeError, 'no contact'):
            task.execute_vision_grasp()
        scene.scene.attach_object.assert_not_called()
        self.assertEqual(arm.move_cartesian.call_count, 1)  # only approach

    def test_successful_grasp_only_enables_bottle_table_for_lift(self):
        task, task_state, arm, gripper, scene, jf, detector, place = \
            self.grasp_task()
        task.plan_grasp_candidate = MagicMock(
            return_value=(trajectory(), np.eye(4)))
        task.get_bottle_pose = MagicMock(return_value=np.eye(4))
        task.execute_vision_grasp()
        self.assertTrue(task_state.grasp_confirmed)
        scene.scene.attach_object.assert_called_once_with(
            'target_bottle', link='link6', touch_links=['link7', 'link8'])
        pairs = [call.args for call in scene.set_contact_pairs.call_args_list]
        self.assertIn(([('target_bottle', 'table')], True), pairs)
        self.assertFalse(any(('link7', 'table') in args[0] for args in pairs))
        self.assertEqual(arm.move_cartesian.call_count, 2)
        scene.scene.remove_attached_object.assert_not_called()

    # ------------------------------------------------------------------
    # Place task workflow.
    # ------------------------------------------------------------------
    def place_task(self, **kwargs):
        config = make_config(**kwargs.get('config', {}))
        task_state = make_state(T_link6_bottle=np.eye(4), grasp_confirmed=True)
        arm = MagicMock()
        gripper = MagicMock()
        scene = MagicMock()
        scene.table_model = {
            'bounds_min': [0.1, -0.3, -0.05],
            'bounds_max': [0.8, 0.3, 0.02],
            'plane_normal': [0., 0., 1.],
            'plane_offset': 0.0,
        }
        scene.require_support_footprint = MagicMock()
        scene.table_surface_z = MagicMock(return_value=0.0)
        scene.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        scene.apply_acm = MagicMock()
        scene.set_contact_pairs = MagicMock()
        scene.wait_object = MagicMock()
        scene.scene.get_attached_objects.return_value = {'target_bottle': object()}
        scene.robot.get_link_names.return_value = ['link6', 'link7', 'link8']
        jf = MagicMock()
        get_pose = MagicMock()
        task = PlaceTask(
            config, task_state, arm, gripper, scene, jf, get_pose)
        return task, task_state, arm, gripper, scene, jf, get_pose

    def test_card_place_target_is_in_front_and_upright(self):
        task, task_state, arm, gripper, scene, jf, get_pose = \
            self.place_task()
        card = np.eye(4)
        card[:3, 3] = [0.5, 0.0, 0.0]
        pre, place, center = task.build_place_targets(card)
        # 卡片检测点在桌面高度(0)时，瓶底被抬到桌面+间隙，避免压进桌面。
        np.testing.assert_allclose(center, [0.4, 0.0, 0.105])
        np.testing.assert_allclose(place[:3, 3], [0.4, 0.0, 0.105])
        np.testing.assert_allclose(place[:3, 2], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(pre[:3, 3], [0.4, 0.0, 0.155])
        scene.scene.add_cylinder.assert_not_called()

    def test_card_place_uses_detected_card_height_for_release(self):
        task, task_state, arm, gripper, scene, jf, get_pose = \
            self.place_task()
        card = np.eye(4)
        # 立在桌面上的卡片，检测点比桌面高 0.2m。
        card[:3, 3] = [0.5, 0.0, 0.2]
        pre, place, center = task.build_place_targets(card)
        np.testing.assert_allclose(center, [0.4, 0.0, 0.2])
        np.testing.assert_allclose(place[:3, 3], [0.4, 0.0, 0.2])
        np.testing.assert_allclose(pre[:3, 3], [0.4, 0.0, 0.25])

    def test_place_rejects_insufficient_preplace_clearance(self):
        task, task_state, arm, gripper, scene, jf, get_pose = \
            self.place_task(config={'place_approach_height': 0.02})
        card = np.eye(4)
        card[:3, 3] = [0.5, 0.0, 0.0]
        with self.assertRaisesRegex(RuntimeError, '预放置高度不足'):
            task.build_place_targets(card)

    def test_place_outside_support_is_rejected_before_motion(self):
        task, task_state, arm, gripper, scene, jf, get_pose = \
            self.place_task()
        card = np.eye(4)
        card[:3, 3] = [5.0, 5.0, 0.0]
        task.get_pose = MagicMock(return_value=card)
        scene.require_support_footprint.side_effect = RuntimeError('table边界')
        arm.move_pose = MagicMock()
        gripper.command = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'table边界'):
            task.execute_place()
        task.get_pose.assert_called_once()
        arm.move_pose.assert_not_called()
        gripper.command.assert_not_called()

    def test_place_release_detaches_then_rebuilds_world_bottle(self):
        task, task_state, arm, gripper, scene, jf, get_pose = \
            self.place_task()
        card = np.eye(4)
        card[:3, 3] = [0.5, 0.0, 0.0]
        pre = np.eye(4)
        pre[:3, 3] = [0.4, 0.0, 0.15]
        place = np.eye(4)
        place[:3, 3] = [0.4, 0.0, 0.1]
        task.get_pose = MagicMock(return_value=card)
        task.build_place_targets = MagicMock(
            return_value=(pre, place, place[:3, 3]))
        arm.move_pose = MagicMock()
        arm.move_cartesian = MagicMock()
        arm.current_ee_matrix = MagicMock(return_value=place)
        gripper.command = MagicMock()
        scene.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        scene.apply_acm = MagicMock()
        scene.set_contact_pairs = MagicMock()
        scene.wait_object = MagicMock()
        scene.scene.remove_attached_object = MagicMock()
        scene.scene.add_cylinder = MagicMock()
        scene.table_surface_z = MagicMock(return_value=0.0)
        task.execute_place()
        scene.scene.remove_attached_object.assert_called_once_with(
            'link6', name='target_bottle')
        scene.scene.add_cylinder.assert_called_once()
        pose = scene.scene.add_cylinder.call_args[0][1]
        np.testing.assert_allclose(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            [0.4, 0.0, 0.1])
        self.assertFalse(task_state.grasp_confirmed)
        self.assertEqual(arm.move_cartesian.call_count, 2)
        gripper.command.assert_called_once_with(
            task.config.opening_open, wait_for_target=True)


if __name__ == '__main__':
    unittest.main()

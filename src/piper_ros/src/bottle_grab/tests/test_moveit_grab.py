"""Offline regression tests: no ROS master, camera, CAN or motion commands."""

import copy
import itertools
import math
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
from pose_utils import level_grasp_rotation_for_azimuth
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
        model_path="bottle.pt",
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
        home_joint_target=[0.0] * 6,
        dry_run=False,
        enable_place=True,
        place_mode="card",
        place_approach_height=0.05,
        place_tool_pose=[0.4, 0.0, 0.1, 0.0, 0.0, 0.0],
        place_offset_from_card=0.1,
        place_card_offset_xy=[0.0, 0.0],
        place_card_clearance=0.015,
        place_release_clearance=0.005,
        place_front_mode="card_normal",
        place_front_yaw=0.0,
        place_yaws=[0.0, 15.0],
        bottle_center_offset_xy=[0.0, 0.0],
        T_link6_tool=np.eye(4),
        scan_mode="semantic",
        scan_start_joint_target=[0.0, 0.3, -0.65, 0.0, 0.88, 0.0],
        scan_joint1_offsets=[0.0, 0.357, -0.357],
        scan_corridor_radius=0.05,
        scan_corridor_samples=24,
        scan_obstacle_padding=0.01,
        scan_card_planarity_m=0.012,
        scan_card_min_extent=0.10,
        scan_card_max_thickness=0.03,
        scan_cylinder_radius_tol=0.6,
        scan_cylinder_height_tol=0.5,
        scan_non_target_cylinder=True,
        scan_table_clearance=0.01,
        scan_footprint_margin=0.02,
        scan_cluster_voxel_m=0.012,
        scan_min_cluster_points=30,
        scan_voxel_m=0.006,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_state(**overrides):
    task_state = SimpleNamespace(
        current_ee_pose_mat=np.eye(4),
        grasp_confirmed=False,
        T_link6_bottle=None,
        T_base_bottle=None,
        bottle_center_at_grasp=np.array([0.4, 0.0, 0.1]),
        created_bottle=False,
        gripper_service_sent=False,
        scan_result=None,
        card_model=None,
    )
    for key, value in overrides.items():
        setattr(task_state, key, value)
    return task_state


def card_model(center=(0.5, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
               width=0.20, length=0.15, thickness=0.005):
    return {
        "center": np.asarray(center, dtype=float),
        "normal": np.asarray(normal, dtype=float),
        "u": np.array([0.0, 1.0, 0.0]),
        "v": np.array([0.0, 0.0, 1.0]),
        "width": width,
        "length": length,
        "thickness": thickness,
    }


class MoveItGrabTests(unittest.TestCase):
    def test_no_legacy_command_paths(self):
        source = (Path(__file__).resolve().parents[1]
                  / 'moveit_grab.py').read_text()
        self.assertNotIn('PosCmd', source)
        self.assertNotIn('/pin_pos_cmd', source)
        self.assertNotIn('joint_pub', source)
        self.assertNotIn('move_to_target_smooth', source)
        self.assertNotIn('get_max_velocity_scaling_factor', source)
        # 预览模式已取消。
        self.assertNotIn('PlanningOnlyStop', source)
        self.assertNotIn('execute_motion', source)

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

    def test_static_transforms_compose_back_to_ee_camera(self):
        he = HandEyeCalibration.__new__(HandEyeCalibration)
        link6_T_optical = np.eye(4)
        link6_T_optical[:3, :3] = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        link6_T_optical[:3, 3] = [0.08, 0.02, 0.04]
        camera_link_T_optical = np.eye(4)
        camera_link_T_optical[:3, 3] = [0.0, 0.015, 0.0]
        he.T_link6_color_optical = link6_T_optical
        he.T_camera_link_color_optical = camera_link_T_optical
        he.frames = {"ee": "link6", "camera_link": "camera_link",
                     "camera_optical": "camera_color_optical_frame"}
        transforms = he.static_transforms()
        self.assertEqual(
            [(parent, child) for parent, child, _ in transforms],
            [("link6", "camera_link"),
             ("camera_link", "camera_color_optical_frame")])
        composed = transforms[0][2] @ transforms[1][2]
        np.testing.assert_allclose(composed, link6_T_optical, atol=1e-12)
        he.T_camera_link_color_optical = None
        with self.assertRaisesRegex(RuntimeError, "camera_link_T_optical"):
            he.static_transforms()

    # ------------------------------------------------------------------
    # Arm controller.
    # ------------------------------------------------------------------
    def arm_controller(self):
        scene = MagicMock()
        scene.require_unchanged = MagicMock()
        scene.get_scene = MagicMock(
            return_value=SimpleNamespace(scene=PlanningScene(robot_state=state())))
        scene.validate_robot_state = MagicMock()
        scene.tf_buffer = MagicMock()
        jf = MagicMock()
        jf._fresh_joint_positions.return_value = {name: 0.0 for name in m.ARM_JOINTS}
        task_state = SimpleNamespace(grasp_confirmed=False)
        ctl = ArmController(
            MagicMock(), MagicMock(), scene, jf, MagicMock(),
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

    def test_blocked_cartesian_can_fall_back_to_planning(self):
        ctl, _, _, _ = self.arm_controller()
        ctl.arm.compute_cartesian_path.return_value = (trajectory(), 0.735)
        ctl.move_pose = MagicMock()
        ctl.move_cartesian(np.eye(4), allow_plan=True)
        ctl.move_pose.assert_called_once()
        ctl.arm.execute.assert_not_called()

    def test_invalid_fraction_never_falls_back(self):
        ctl, _, _, _ = self.arm_controller()
        ctl.arm.compute_cartesian_path.return_value = (
            trajectory(), float('nan'))
        ctl.move_pose = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "完成比例无效"):
            ctl.move_cartesian(np.eye(4), allow_plan=True)
        ctl.move_pose.assert_not_called()

    def test_scan_motion_skips_guard_and_executes(self):
        ctl, _, scene, _ = self.arm_controller()
        ctl.display_pub = MagicMock()
        ctl.validate_trajectory = MagicMock()
        ctl.robot.get_current_state.return_value = state()
        ctl.arm.execute.return_value = True
        ctl.arm.get_active_joints.return_value = list(m.ARM_JOINTS)
        ctl.arm.get_current_joint_values.return_value = [0.02] * 6
        ctl.check_trajectory_start = MagicMock()
        ctl.finish_motion(trajectory(), guarded=False)
        scene.require_unchanged.assert_not_called()
        ctl.display_pub.publish.assert_called_once()
        ctl.arm.execute.assert_called_once()

    def test_changed_scene_blocks_execute(self):
        ctl, _, scene, _ = self.arm_controller()
        ctl.display_pub = MagicMock()
        ctl.validate_trajectory = MagicMock()
        ctl.robot.get_current_state.return_value = state()
        scene.require_unchanged.side_effect = [None, RuntimeError('changed map')]
        with self.assertRaises(RuntimeError):
            ctl.finish_motion(trajectory())
        ctl.arm.execute.assert_not_called()

    def test_diagnose_pose_tries_multiple_seeds(self):
        ctl, _, scene, _ = self.arm_controller()
        proxy = MagicMock()
        ctl._ik_proxy = proxy
        solution = SimpleNamespace(
            error_code=SimpleNamespace(val=1),
            solution=SimpleNamespace(joint_state=state().joint_state))
        failure = SimpleNamespace(
            error_code=SimpleNamespace(val=-31),
            solution=SimpleNamespace(joint_state=state().joint_state))

        # 单种子失败不算失败：后面的初值解出来就算可达。
        proxy.side_effect = [failure, failure, solution]
        self.assertIsNone(ctl.diagnose_pose(np.eye(4), "预抓取位"))

        proxy.side_effect = None
        proxy.return_value = failure
        reason = ctl.diagnose_pose(np.eye(4), "预抓取位")
        self.assertIn("KDL无IK解", reason)
        self.assertIn("组初值", reason)

        # 有解但全部碰撞时，要报出碰撞对象。
        proxy.return_value = solution
        scene.validate_robot_state.side_effect = RuntimeError(
            "机器人状态无效/碰撞: [('link7', 'scan_obstacle_0')]")
        reason = ctl.diagnose_pose(np.eye(4), "预放置位")
        self.assertIn("scan_obstacle_0", reason)

    def test_plan_pose_nearest_prefers_joint_target(self):
        ctl, _, _, _ = self.arm_controller()
        ctl._limits = {}
        ctl.arm.get_joint_value_target.return_value = [0.0] * 6
        ctl.arm.plan.return_value = (True, trajectory())
        ctl.validate_trajectory = MagicMock()
        result = ctl.plan_pose_nearest(np.eye(4))
        ctl.arm.set_joint_value_target.assert_called_once()
        ctl.arm.set_pose_target.assert_not_called()
        self.assertTrue(result.joint_trajectory.points)

    def test_plan_pose_nearest_falls_back_to_pose_target(self):
        from moveit_commander.exception import MoveItCommanderException

        ctl, _, _, _ = self.arm_controller()
        ctl._limits = {}
        ctl.arm.set_joint_value_target.side_effect = \
            MoveItCommanderException("no ik")
        ctl.arm.plan.return_value = (True, trajectory())
        ctl.validate_trajectory = MagicMock()
        ctl.plan_pose_nearest(np.eye(4))
        ctl.arm.set_pose_target.assert_called_once()

    def test_diagnose_cartesian_distinguishes_ik_from_collision(self):
        ctl, _, scene, _ = self.arm_controller()
        ctl._limits = {}
        ctl.arm.compute_cartesian_path.return_value = (trajectory(), 0.4)
        reason = ctl.diagnose_cartesian(np.eye(4), 0.4)
        self.assertIn("IK/关节限位", reason)

        ctl.arm.compute_cartesian_path.return_value = (
            trajectory(((0.,) * 6, (0.02,) * 6, (0.04,) * 6)), 1.0)
        scene.inspect_robot_state = MagicMock(
            side_effect=[(True, []), (False, [("link6", "table")]), (True, [])])
        reason = ctl.diagnose_cartesian(np.eye(4), 0.4)
        self.assertIn("第2/3步", reason)
        self.assertIn("table", reason)

    def test_invalid_trajectory_and_between_sample_collision(self):
        ctl, _, scene, _ = self.arm_controller()
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
        scene.require_unchanged = MagicMock()
        scene.get_scene = MagicMock(
            return_value=SimpleNamespace(scene=PlanningScene(robot_state=state())))
        scene.validate_robot_state = MagicMock()
        task_state = SimpleNamespace(gripper_service_sent=False)
        gripper = GripperController(
            jf, scene, MagicMock(), task_state, 1.0, 0.002, 0.5, 0.0,
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

    def test_scan_gripper_command_skips_scene_guard(self):
        gripper, jf, scene, task_state = self.gripper_controller()
        gripper.command(0.07, wait_for_target=False, guarded=False)
        scene.require_unchanged.assert_not_called()
        gripper.gripper_service.assert_called_once()

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
            MagicMock(), support_edge_margin=0.01, voxel_size=0.006)

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

    def test_reset_for_scan_rejects_unknown_objects(self):
        sm = self.scene_manager()
        sm.scene.get_attached_objects.return_value = {}
        sm.scene.get_known_object_names.return_value = ['table', 'someone_else']
        with self.assertRaisesRegex(RuntimeError, '未知对象'):
            sm.reset_for_scan()

    def test_reset_for_scan_removes_own_objects_and_clears_map(self):
        sm = self.scene_manager()
        sm.scene.get_attached_objects.return_value = {}
        sm.scene.get_known_object_names.return_value = [
            'table', 'scan_obstacle_1', 'target_bottle']
        sm.apply_world = MagicMock()
        sm.clear_octomap = MagicMock()
        sm.reset_contact_pairs = MagicMock()
        sm.table_model = {'stale': True}
        sm.reset_for_scan()
        sm.apply_world.assert_called_once_with(
            [], ['table', 'scan_obstacle_1', 'target_bottle'])
        sm.clear_octomap.assert_called_once()
        sm.reset_contact_pairs.assert_called_once()
        self.assertIsNone(sm.table_model)
        self.assertIsNone(sm.signature)

    def test_freeze_and_require_unchanged(self):
        sm = self.scene_manager()
        sm.current_signature = MagicMock(return_value=('a', ()))
        sm.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        sm.freeze({'table': {'x': 1}, 'obstacles': []})
        sm.require_unchanged()
        sm.current_signature.return_value = ('b', ())
        with self.assertRaisesRegex(RuntimeError, '改变'):
            sm.require_unchanged()

    def test_support_footprint_uses_explicit_voxel(self):
        sm = self.scene_manager()
        sm.table_model = {
            'bounds_min': [0.0, 0.0, 0.0], 'bounds_max': [0.5, 0.5, 0.1]}
        sm.require_support_footprint([0.25, 0.25, 0.05], 0.03)
        with self.assertRaisesRegex(RuntimeError, 'table边界'):
            sm.require_support_footprint([0.03, 0.25, 0.05], 0.03)

    def test_existing_attached_bottle_never_removed(self):
        sm = self.scene_manager()
        sm.require_support_footprint = MagicMock()
        sm.table_model = {'bounds_max': [1, 1, 0.02]}
        sm.scene.get_attached_objects.return_value = {'target_bottle': object()}
        sm.scene.get_known_object_names.return_value = []
        task_state = make_state()
        with self.assertRaisesRegex(RuntimeError, '已有target_bottle'):
            sm.add_bottle([0.4, 0, 0.1], 0.2, 0.025, 0.15, 0.005, task_state)
        sm.apply_world = getattr(sm, "apply_world")
        self.assertFalse(task_state.created_bottle)

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
        task_state = make_state(
            T_base_bottle=np.eye(4), created_bottle=True,
            scan_result={'obstacles': []})
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
        scene.table_model = {
            'bounds_min': [0.1, -0.3, -0.05],
            'bounds_max': [0.8, 0.3, 0.02],
            'plane_normal': [0., 0., 1.],
            'plane_offset': 0.0,
        }
        scene.table_surface_z = MagicMock(return_value=0.0)
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
        place_task = MagicMock()
        place_task.execute = MagicMock()
        task = GraspTask(
            config, task_state, arm, gripper, scene, jf, place_task)
        return task, task_state, arm, gripper, scene, jf, place_task

    def test_unknown_grasp_stops_before_attach_or_lift(self):
        task, task_state, arm, gripper, scene, jf, place = \
            self.grasp_task()
        task.plan_grasp_candidate = MagicMock(
            return_value=(trajectory(), np.eye(4)))
        gripper.confirm.side_effect = RuntimeError('no contact')
        with self.assertRaisesRegex(RuntimeError, 'no contact'):
            task.execute_vision_grasp()
        scene.scene.attach_object.assert_not_called()
        self.assertEqual(arm.move_cartesian.call_count, 1)  # only approach

    def test_successful_grasp_only_enables_bottle_table_for_lift(self):
        task, task_state, arm, gripper, scene, jf, place = \
            self.grasp_task()
        task.plan_grasp_candidate = MagicMock(
            return_value=(trajectory(), np.eye(4)))
        task.execute_vision_grasp()
        self.assertTrue(task_state.grasp_confirmed)
        scene.scene.attach_object.assert_called_once_with(
            'target_bottle', link='link6', touch_links=['link7', 'link8'])
        pairs = [call.args for call in scene.set_contact_pairs.call_args_list]
        self.assertIn(([('target_bottle', 'table')], True), pairs)
        self.assertFalse(any(('link7', 'table') in args[0] for args in pairs))
        self.assertEqual(arm.move_cartesian.call_count, 2)
        scene.scene.remove_attached_object.assert_not_called()
        place.execute_place.assert_called_once()
        # 抓取后先回零位再放置，放置完成后再回零位。
        self.assertEqual(arm.move_joints.call_count, 2)
        for call in arm.move_joints.call_args_list:
            self.assertEqual(call.args[0], [0.0] * 6)

    def test_dry_run_stops_before_any_grasp_motion(self):
        task, task_state, arm, gripper, scene, jf, place = self.grasp_task(
            config={'dry_run': True})
        task.plan_grasp_candidate = MagicMock(
            return_value=(trajectory(), np.eye(4)))
        task.execute_vision_grasp()
        arm.finish_motion.assert_not_called()
        arm.move_cartesian.assert_not_called()
        scene.scene.attach_object.assert_not_called()
        self.assertFalse(task_state.grasp_confirmed)

    def test_candidates_use_radial_azimuth(self):
        task, task_state, arm, gripper, scene, jf, place = self.grasp_task()
        observed = np.eye(4)
        observed[:3, 3] = [0.62, -0.08, 0.18]
        candidates = task._candidates(observed)
        self.assertEqual([c["label"] for c in candidates], [0.0, 30.0])
        radial = math.atan2(-0.08, 0.62)
        for candidate in candidates:
            rotation = candidate["grasp"][:3, :3]
            np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, -1.0],
                                       atol=1e-9)
            azimuth = math.atan2(rotation[1, 2], rotation[0, 2])
            expected = radial + math.radians(candidate["label"])
            self.assertAlmostEqual(
                math.atan2(math.sin(azimuth - expected),
                           math.cos(azimuth - expected)), 0.0, places=9)

    def test_candidate_cost_penalizes_wrist_twist(self):
        task, task_state, arm, gripper, scene, jf, place = self.grasp_task()
        arm.joint_limit_margins.return_value = [1.0] * 6
        centered = trajectory(((0.,) * 6, (0., 0., 0., 0.1, 0., 0.1)))
        twisted = trajectory(((0.,) * 6, (0., 0., 0., 0.8, 0., 0.8)))
        centered_approach = trajectory(
            ((0., 0., 0., 0.1, 0., 0.1), (0.01, 0., 0., 0.11, 0., 0.11)))
        twisted_approach = trajectory(
            ((0., 0., 0., 0.8, 0., 0.8), (0.01, 0., 0., 0.81, 0., 0.81)))
        centered_cost, _ = task._candidate_cost(
            centered, centered_approach)
        twisted_cost, _ = task._candidate_cost(
            twisted, twisted_approach)
        self.assertLess(centered_cost, twisted_cost)

    def test_plan_grasp_candidate_chooses_lowest_cost(self):
        task, task_state, arm, gripper, scene, jf, place = self.grasp_task()
        observed = np.eye(4)
        observed[:3, 3] = [0.62, -0.08, 0.18]
        candidates = task._candidates(observed)
        self.assertEqual(len(candidates), 2)

        arm.robot.get_current_state.return_value = state()
        arm.cartesian_joint_step_limit = 0.5
        arm.joint_limit_margins.return_value = [1.0] * 6
        low_pre = trajectory(((0.,) * 6, (0., 0., 0., 0.1, 0., 0.1)))
        high_pre = trajectory(((0.,) * 6, (0., 0., 0., 0.8, 0., 0.8)))
        low_approach = trajectory(
            ((0., 0., 0., 0.1, 0., 0.1), (0.01, 0., 0., 0.11, 0., 0.11)))
        high_approach = trajectory(
            ((0., 0., 0., 0.8, 0., 0.8), (0.01, 0., 0., 0.81, 0., 0.81)))
        arm.plan_pose_nearest.side_effect = [low_pre, high_pre]
        arm.arm.compute_cartesian_path.side_effect = [
            (low_approach, 1.0), (high_approach, 1.0)]

        chosen_pre, chosen_grasp = task.plan_grasp_candidate(
            observed, candidates)
        self.assertIs(chosen_pre, low_pre)
        np.testing.assert_allclose(chosen_grasp, candidates[0]["grasp"])
        self.assertEqual(arm.plan_pose_nearest.call_count, 2)
        self.assertEqual(arm.arm.compute_cartesian_path.call_count, 2)

    def test_corridor_rejects_blocked_approach_before_planning(self):
        task, task_state, arm, gripper, scene, jf, place = self.grasp_task()
        pose = np.eye(4)
        pose[:3, 3] = [0.3, 0.0, 0.0]
        box = {
            "id": "scan_obstacle_0", "kind": "box",
            "dims": (0.2, 0.2, 0.2), "pose": np.eye(4), "label": "card",
            "points": 100,
        }
        # 预抓取直线位于 x=0.3, y=0, z∈[0.02,0.10]；把这个盒子放在走廊里。
        box["pose"][:3, 3] = [0.3, 0.0, 0.06]
        task_state.scan_result = {"obstacles": [box]}
        with self.assertRaisesRegex(RuntimeError, '没有可行的接近走廊'):
            task._candidates(pose)

    # ------------------------------------------------------------------
    # Place task workflow.
    # ------------------------------------------------------------------
    def place_task(self, **kwargs):
        config = make_config(**kwargs.get('config', {}))
        task_state = make_state(T_link6_bottle=np.eye(4), grasp_confirmed=True,
                                card_model=card_model())
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
        task = PlaceTask(config, task_state, arm, gripper, scene, jf)
        return task, task_state, arm, gripper, scene, jf

    def test_card_place_target_is_in_front_and_upright(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        pre, place, center = task.build_place_targets(card_model())
        # 卡片法向为-x、朝机器人；松手点在卡面前0.1m，高度只由桌面决定。
        np.testing.assert_allclose(center, [0.4, 0.0, 0.105])
        np.testing.assert_allclose(place[:3, 3], [0.4, 0.0, 0.105])
        np.testing.assert_allclose(place[:3, 2], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(pre[:3, 3], [0.4, 0.0, 0.155])
        scene.scene.add_cylinder.assert_not_called()

    def test_place_candidates_follow_place_radial(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        # 抓取时 link6 的 z 指向 +20°；T_link6_bottle 的旋转是其逆。
        grasp_R = level_grasp_rotation_for_azimuth(np.deg2rad(20.0))
        grasp_T = np.eye(4)
        grasp_T[:3, :3] = np.linalg.inv(grasp_R)
        task_state.T_link6_bottle = grasp_T
        candidates = task.build_place_candidates(card_model())
        self.assertEqual([c["label"] for c in candidates], [0.0, 15.0])
        place_azimuth = math.atan2(0.0, 0.4)
        for candidate in candidates:
            rotation = candidate["place"][:3, :3]
            azimuth = math.atan2(rotation[1, 2], rotation[0, 2])
            expected = place_azimuth + math.radians(candidate["label"])
            self.assertAlmostEqual(
                math.atan2(math.sin(azimuth - expected),
                           math.cos(azimuth - expected)), 0.0, places=9)
            bottle = candidate["place"] @ grasp_T
            np.testing.assert_allclose(
                bottle[:3, 2], [0.0, 0.0, 1.0], atol=1e-9)
            np.testing.assert_allclose(
                candidate["bottle_center"], [0.4, 0.0, 0.105])

    def test_place_cost_penalizes_wrist_twist(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        arm.joint_limit_margins.return_value = [1.0] * 6
        centered_pre = trajectory(
            ((0.,) * 6, (0., 0., 0., 0.1, 0., 0.1)))
        centered = trajectory(
            ((0., 0., 0., 0.1, 0., 0.1), (0.01, 0., 0., 0.11, 0., 0.11)))
        twisted_pre = trajectory(
            ((0.,) * 6, (0., 0., 0., 0.8, 0., 0.8)))
        twisted = trajectory(
            ((0., 0., 0., 0.8, 0., 0.8), (0.01, 0., 0., 0.81, 0., 0.81)))
        centered_cost, _ = task._candidate_cost(
            centered_pre, centered, True)
        twisted_cost, _ = task._candidate_cost(
            twisted_pre, twisted, True)
        self.assertLess(centered_cost, twisted_cost)

    def test_place_prefers_complete_descent(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        pre = np.eye(4)
        pre[:3, 3] = [0.4, 0.0, 0.15]
        place = np.eye(4)
        place[:3, 3] = [0.4, 0.0, 0.1]
        candidates = [
            {"label": 0.0, "pre": pre.copy(), "place": place.copy(),
             "bottle_center": place[:3, 3].copy()},
            {"label": 15.0, "pre": pre.copy(), "place": place.copy(),
             "bottle_center": place[:3, 3].copy()},
        ]
        arm.robot.get_current_state.return_value = state()
        arm.cartesian_joint_step_limit = 0.5
        arm.joint_limit_margins.return_value = [1.0] * 6
        first_pre = trajectory(((0.,) * 6, (0., 0., 0., 0.05, 0., 0.05)))
        second_pre = trajectory(((0.,) * 6, (0., 0., 0., 0.05, 0., 0.05)))
        descent = trajectory(
            ((0., 0., 0., 0.05, 0., 0.05), (0.01, 0., 0., 0.06, 0., 0.06)))
        arm.plan_pose_nearest.side_effect = [first_pre, second_pre]
        arm.arm.compute_cartesian_path.side_effect = [
            (descent, 0.8), (descent, 1.0)]
        chosen, chosen_pre = task.plan_place_candidates(candidates)
        self.assertEqual(chosen["label"], 15.0)
        self.assertIs(chosen_pre, second_pre)

    def test_release_height_uses_table_not_card_center(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        # 20cm高卡片的重心在桌面上方10cm，松手高度不能被它抬起来。
        card = card_model(center=(0.5, 0.0, 0.2))
        pre, place, center = task.build_place_targets(card)
        np.testing.assert_allclose(center, [0.4, 0.0, 0.105])
        np.testing.assert_allclose(pre[:3, 3], [0.4, 0.0, 0.155])

    def test_toward_robot_mode_still_supported(self):
        task, task_state, arm, gripper, scene, jf = self.place_task(
            config={'place_front_mode': 'toward_robot'})
        pre, place, center = task.build_place_targets(card_model())
        np.testing.assert_allclose(center, [0.4, 0.0, 0.105])

    def test_place_rejects_insufficient_card_clearance(self):
        task, task_state, arm, gripper, scene, jf = self.place_task(
            config={'place_offset_from_card': 0.03})
        with self.assertRaisesRegex(RuntimeError, '不足以避开卡片'):
            task.build_place_targets(card_model())

    def test_place_rejects_insufficient_preplace_clearance(self):
        task, task_state, arm, gripper, scene, jf = self.place_task(
            config={'place_approach_height': 0.02})
        with self.assertRaisesRegex(RuntimeError, '预放置高度不足'):
            task.build_place_targets(card_model())

    def test_place_outside_support_is_rejected_before_motion(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        scene.require_support_footprint.side_effect = RuntimeError('table边界')
        arm.move_pose = MagicMock()
        gripper.command = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'table边界'):
            task.execute_place()
        arm.move_pose.assert_not_called()
        gripper.command.assert_not_called()

    def test_place_without_scanned_card_is_rejected(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        task_state.card_model = None
        with self.assertRaisesRegex(RuntimeError, '没有识别出卡片'):
            task.execute_place()

    def test_place_release_detaches_then_rebuilds_world_bottle(self):
        task, task_state, arm, gripper, scene, jf = self.place_task()
        pre = np.eye(4)
        pre[:3, 3] = [0.4, 0.0, 0.15]
        place = np.eye(4)
        place[:3, 3] = [0.4, 0.0, 0.1]
        candidate = {
            "label": 0.0, "pre": pre, "place": place,
            "bottle_center": place[:3, 3]}
        task.build_place_candidates = MagicMock(return_value=[candidate])
        task.plan_place_candidates = MagicMock(
            return_value=(candidate, trajectory()))
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

    def test_place_keeps_bottle_table_contact_allowed_after_release(self):
        # 回归：table碰撞盒是膨胀包络，放置后的瓶子必然落在盒内；
        # 张爪和放置后的回零都必须保留瓶子-桌面允许关系。
        task, task_state, arm, gripper, scene, jf = self.place_task()
        pre = np.eye(4)
        pre[:3, 3] = [0.4, 0.0, 0.15]
        place = np.eye(4)
        place[:3, 3] = [0.4, 0.0, 0.1]
        candidate = {
            "label": 0.0, "pre": pre, "place": place,
            "bottle_center": place[:3, 3]}
        task.build_place_candidates = MagicMock(return_value=[candidate])
        task.plan_place_candidates = MagicMock(
            return_value=(candidate, trajectory()))
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
        pairs = [call.args for call in scene.set_contact_pairs.call_args_list]
        self.assertIn(([("target_bottle", "table")], True), pairs)
        # 最后一个动作是把table对瓶子的关系设回True，而不是随机器人link
        # 一起设成False。
        self.assertEqual(
            scene.set_contact_pairs.call_args_list[-1].args,
            ([("target_bottle", "table")], True))


if __name__ == '__main__':
    unittest.main()

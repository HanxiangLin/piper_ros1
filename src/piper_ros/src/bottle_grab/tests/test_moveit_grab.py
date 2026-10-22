"""Offline regression tests: no ROS master, camera, CAN or motion commands."""
import copy
import importlib.util
import itertools
from pathlib import Path
import threading
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
from moveit_msgs.msg import (
    AllowedCollisionEntry, AllowedCollisionMatrix, PlanningScene, RobotState,
    RobotTrajectory,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import rospy

SCRIPT = Path(__file__).resolve().parents[1] / 'moveit_grab.py'
spec = importlib.util.spec_from_file_location('moveit_grab', str(SCRIPT))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
Controller = m.PiperVisionController


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


class MoveItGrabTests(unittest.TestCase):
    def controller(self):
        controller = Controller.__new__(Controller)
        controller.arm = MagicMock()
        controller.robot = MagicMock()
        controller.scene = MagicMock()
        controller.execute_motion = False
        controller.grasp_confirmed = False
        controller.lift_table_margin = 0.005
        controller.require_unchanged_scan = MagicMock()
        controller._fresh_joint_positions = MagicMock(return_value={
            name: 0. for name in m.ARM_JOINTS})
        controller.validate_trajectory = MagicMock()
        return controller

    def test_no_legacy_command_paths(self):
        source = SCRIPT.read_text()
        self.assertNotIn('PosCmd', source)
        self.assertNotIn('/pin_pos_cmd', source)
        self.assertNotIn('joint_pub', source)
        self.assertNotIn('move_to_target_smooth', source)
        self.assertNotIn('get_max_velocity_scaling_factor', source)

    def test_transform_has_explicit_direction(self):
        c = self.controller()
        c.current_ee_pose_mat = np.eye(4)
        c.current_ee_pose_mat[:3, 3] = [1, 2, 3]
        c.T_link6_color_optical = np.eye(4)
        c.T_link6_color_optical[:3, 3] = [0, 0, 0.1]
        pose = c.make_bottle_pose_in_base([0.1, 0.2, 0.3])
        np.testing.assert_allclose(pose[:3, 3], [1.1, 2.2, 3.4])
        np.testing.assert_allclose(pose[:3, 0], [0, 0, -1])

    def test_partial_and_nonfinite_cartesian_never_execute(self):
        for fraction in (0.5, float('nan')):
            c = self.controller()
            c.arm.compute_cartesian_path.return_value = (trajectory(), fraction)
            with self.assertRaises(RuntimeError):
                c.move_cartesian(np.eye(4))
            c.arm.execute.assert_not_called()
            c.arm.retime_trajectory.assert_not_called()

    def test_preview_displays_without_execute(self):
        c = self.controller()
        c.display_pub = MagicMock()
        c.robot.get_current_state.return_value = state()
        with patch.object(m.rospy, 'sleep'):
            with self.assertRaises(m.PlanningOnlyStop):
                c.finish_motion(trajectory())
        c.display_pub.publish.assert_called_once()
        c.arm.execute.assert_not_called()

    def test_changed_scene_blocks_execute(self):
        c = self.controller()
        c.execute_motion = True
        c.display_pub = MagicMock()
        c.robot.get_current_state.return_value = state()
        c.require_unchanged_scan.side_effect = [None, RuntimeError('changed map')]
        with self.assertRaises(RuntimeError):
            c.finish_motion(trajectory())
        c.arm.execute.assert_not_called()

    def test_gripper_sweep_checks_both_fingers_and_table(self):
        c = self.controller()
        scene = PlanningScene(robot_state=state())
        c.get_scene = MagicMock(return_value=SimpleNamespace(scene=scene))
        c.current_gripper_opening = MagicMock(return_value=0.07)
        checked = []
        def valid(robot_state):
            joint = dict(zip(robot_state.joint_state.name, robot_state.joint_state.position))
            self.assertAlmostEqual(joint['joint8'], -joint['joint7'])
            checked.append(joint['joint7'])
            if joint['joint7'] < 0.025:
                raise RuntimeError('table/link8 collision')
        c.validate_robot_state = valid
        c.gripper_service = MagicMock()
        c.execute_motion = True
        with self.assertRaisesRegex(RuntimeError, 'table/link8'):
            c.command_gripper(0.01)
        self.assertGreater(len(checked), 2)
        c.gripper_service.assert_not_called()

    def test_acm_preserves_existing_pairs_and_defaults(self):
        c = self.controller()
        acm = AllowedCollisionMatrix(
            entry_names=['link6', 'table'],
            entry_values=[AllowedCollisionEntry(enabled=[True, False]),
                          AllowedCollisionEntry(enabled=[False, False])],
            default_entry_names=['fixture'], default_entry_values=[False])
        c.get_acm = MagicMock(return_value=copy.deepcopy(acm))
        c.apply_acm = MagicMock()
        c.set_contact_pairs([('target_bottle', 'link7')], True)
        updated = c.apply_acm.call_args[0][0]
        self.assertTrue(updated.entry_values[0].enabled[0])
        self.assertFalse(updated.entry_values[0].enabled[1])
        self.assertEqual(list(updated.default_entry_names), ['fixture'])
        i, j = updated.entry_names.index('target_bottle'), updated.entry_names.index('link7')
        self.assertTrue(updated.entry_values[i].enabled[j])
        self.assertTrue(updated.entry_values[j].enabled[i])
        self.assertTrue(all(len(row.enabled) == len(updated.entry_names)
                            for row in updated.entry_values))

    def test_existing_attached_bottle_never_removed(self):
        c = self.controller()
        c.require_support_footprint = MagicMock()
        c.bottle_radius, c.bottle_height = 0.025, 0.2
        c.lift_distance, c.lift_clearance = 0.15, 0.01
        c.table_model = {'bounds_max': [1, 1, 0.02]}
        c.scene.get_attached_objects.return_value = {'target_bottle': object()}
        with self.assertRaisesRegex(RuntimeError, '已有target_bottle'):
            c.add_bottle([0.4, 0, 0.1])
        c.scene.remove_world_object.assert_not_called()
        c.scene.add_cylinder.assert_not_called()

    def test_unknown_grasp_stops_before_attach_or_lift(self):
        c = self.workflow()
        c.confirm_grasp.side_effect = RuntimeError('no contact')
        with self.assertRaisesRegex(RuntimeError, 'no contact'):
            c.execute_vision_grasp()
        c.scene.attach_object.assert_not_called()
        self.assertEqual(c.move_cartesian.call_count, 1)  # only approach

    def test_successful_grasp_only_enables_bottle_table_for_lift(self):
        c = self.workflow()
        c.execute_vision_grasp()
        self.assertTrue(c.grasp_confirmed)
        c.scene.attach_object.assert_called_once_with(
            'target_bottle', link='link6', touch_links=['link7', 'link8'])
        pairs = [call.args for call in c.set_contact_pairs.call_args_list]
        self.assertIn(([('target_bottle', 'table')], True), pairs)
        self.assertFalse(any(('link7', 'table') in args[0] for args in pairs))
        self.assertEqual(c.move_cartesian.call_count, 2)
        c.scene.remove_attached_object.assert_not_called()

    def workflow(self):
        c = self.controller()
        c.execute_motion = True
        c.wait_for_joint_feedback = MagicMock()
        c.require_scan_scene = MagicMock()
        c.init_camera_and_detector = MagicMock()
        c.scene.get_attached_objects.return_value = {}
        c.move_to_photo_pose = False
        c.target_label = None
        c.get_bottle_pose = MagicMock(return_value=np.eye(4))
        c.plan_grasp_candidate = MagicMock(return_value=(trajectory(), np.eye(4)))
        c.finish_motion = MagicMock()
        c.command_gripper = MagicMock()
        c.opening_open, c.opening_close_command = 0.07, 0.
        c.confirm_grasp = MagicMock()
        c.wait_object = MagicMock()
        c.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        c.apply_acm = MagicMock()
        c.set_contact_pairs = MagicMock()
        initial, lifted = np.eye(4), np.eye(4)
        lifted[2, 3] = 0.15
        c.current_ee_matrix = MagicMock(side_effect=[initial, initial, lifted])
        c.move_cartesian = MagicMock()
        c.bottle_center_at_grasp = np.array([0.4, 0, 0.1])
        c.bottle_height, c.lift_distance, c.lift_clearance = 0.2, 0.15, 0.01
        c.table_model = {'bounds_max': [1, 1, 0.02]}
        c.get_scene = MagicMock(return_value=SimpleNamespace(scene=PlanningScene()))
        c.validate_robot_state = MagicMock()
        c.post_grasp_joint_target = None
        c.execute_place = MagicMock()
        return c

    def test_grasp_stability_requires_load_not_only_opening(self):
        c = self.controller()
        c.current_gripper_opening = MagicMock(return_value=0.04)
        c.grasp_opening_min, c.grasp_opening_max = 0.03, 0.05
        c.grasp_effort_min = 0.5
        c.gripper_feedback_timeout = 0.05
        c.gripper_settle_time = 0
        c.grasp_check_service = ''
        c._joint_lock = threading.Lock()
        c._joint_efforts = {'joint7': 0.0}
        with patch.object(m.rospy, 'is_shutdown', return_value=False), patch.object(m.rospy, 'sleep'):
            with self.assertRaises(RuntimeError):
                c.confirm_grasp()
        c._joint_efforts['joint7'] = 0.6
        with patch.object(m.rospy, 'is_shutdown', return_value=False), patch.object(m.rospy, 'loginfo'):
            c.confirm_grasp()

    def test_invalid_trajectory_and_between_sample_collision(self):
        c = self.controller()
        c.validate_trajectory = Controller.validate_trajectory.__get__(c)
        scene = PlanningScene(robot_state=state())
        c.get_scene = MagicMock(return_value=SimpleNamespace(scene=scene))
        checked=[]
        def check(robot_state):
            x = robot_state.joint_state.position[0]
            checked.append(x)
            if 0.04 < x < 0.08:
                raise RuntimeError('between-point collision')
        c.validate_robot_state = check
        with self.assertRaisesRegex(RuntimeError, 'between-point'):
            c.validate_trajectory(trajectory(((0.,)*6, (0.12,)*6)))
        self.assertGreater(len(checked), 2)
        with self.assertRaisesRegex(RuntimeError, '非有限'):
            c.validate_trajectory(trajectory(((float('nan'),)*6,)))

    def test_candidate_partial_approach_restores_permissions_without_moving(self):
        c = self.controller()
        c.build_grasp_targets = MagicMock(return_value=(np.array([0.4, 0, 0.1]), None, None))
        c.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        c.apply_acm = MagicMock()
        c.set_contact_pairs = MagicMock()
        c.approach_yaws = [0, 30]
        c.table_surface_z = MagicMock(return_value=0.)
        c.grasp_height, c.pregrasp_distance = 0.1, 0.08
        c.T_link6_tool = np.eye(4)
        c.arm.plan.return_value = (True, trajectory(), 0., None)
        c.arm.compute_cartesian_path.return_value = (trajectory(), 0.5)
        c.robot.get_current_state.side_effect = lambda: state()
        with self.assertRaisesRegex(RuntimeError, '没有有效'):
            c.plan_grasp_candidate(np.eye(4))
        self.assertEqual(c.apply_acm.call_count, 2)
        c.arm.execute.assert_not_called()

    def place_controller(self):
        c = self.controller()
        c.enable_place, c.grasp_confirmed = True, True
        c.place_mode = 'card'
        c.place_approach_height = 0.05
        c.place_offset_from_card = 0.1
        c.place_card_offset_xy = [0., 0.]
        c.place_card_conf_thres = 0.35
        c.place_release_clearance = 0.005
        c.place_front_mode = 'toward_robot'
        c.place_front_yaw = 0.0
        c.place_photo_joint_target = None
        c.opening_open = 0.07
        c.target_label = None
        c.place_card_label = None
        c.T_link6_bottle = np.eye(4)
        c.bottle_radius, c.bottle_height = 0.025, 0.2
        c.lift_clearance = 0.01
        c.bottle_center_offset_xy = [0., 0.]
        c.table_model = {
            'bounds_min': [0.1, -0.3, -0.05],
            'bounds_max': [0.8, 0.3, 0.02],
            'plane_normal': [0., 0., 1.],
            'plane_offset': 0.0,
        }
        c.support_edge_margin = 0.01
        c.scan_signature = ('dummy_link', 'OcTree', True, 0.02)
        c.scene.get_attached_objects.return_value = {'target_bottle': object()}
        c.acm_before_task = AllowedCollisionMatrix()
        c.robot.get_link_names.return_value = ['link6', 'link7', 'link8']
        return c

    def detection_controller(self):
        c = self.controller()
        c.current_ee_pose_mat = np.eye(4)
        c.current_ee_matrix = MagicMock(return_value=np.eye(4))
        c.T_link6_color_optical = np.eye(4)
        c.T_link6_bottle = None
        c.bottle_radius, c.bottle_height = 0.03, 0.17
        c.estimate_bottle_center_from_table = True
        c.show_detection_window = False
        c.table_model = {}
        c.depth_scale = 0.001
        c.filters = []
        color_frame = MagicMock()
        color_frame.get_data.return_value = np.zeros((480, 640, 3), np.uint8)
        raw_depth = MagicMock()
        raw_depth.get_width.return_value = 640
        raw_depth.get_height.return_value = 480
        c.pipeline = MagicMock()
        c.pipeline.wait_for_frames.return_value = None
        c.align = MagicMock()
        c.align.process.return_value = SimpleNamespace(
            get_depth_frame=lambda: raw_depth,
            get_color_frame=lambda: color_frame)
        intrinsics = SimpleNamespace(
            fx=455., fy=455., ppx=320., ppy=240., width=640, height=480)
        c.filtered_depth = SimpleNamespace(
            get_width=lambda: 640, get_height=lambda: 480,
            profile=SimpleNamespace(
                as_video_stream_profile=lambda: SimpleNamespace(
                    intrinsics=intrinsics)))
        c.model = MagicMock()
        c.model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(
                numpy=lambda: np.array([[300, 200, 340, 260]]))),
            conf=SimpleNamespace(cpu=lambda: SimpleNamespace(
                numpy=lambda: np.array([0.9])))))]
        return c

    def run_detection(self, c, **kwargs):
        surface = np.eye(4)
        surface[:3, 3] = [0.5, 0.0, 0.35]
        table = np.eye(4)
        table[:3, 3] = [0.7, 0.0, 0.1]
        with patch.object(m, 'classify_bottle_color', return_value='bottle-g'), \
                patch.object(m, 'get_depth_percentile', return_value=0.35), \
                patch.object(m, 'deproject', return_value=[0., 0., 0.35]), \
                patch.object(m, 'apply_filters', return_value=MagicMock(
                    as_depth_frame=lambda: c.filtered_depth)), \
                patch.object(c, 'make_bottle_pose_in_base', return_value=surface), \
                patch.object(c, 'make_bottle_center_pose_in_base',
                             return_value=table), \
                patch.object(m.rospy, 'is_shutdown', return_value=False):
            return c.get_bottle_pose(max_frames=12, **kwargs)

    def test_card_detection_can_use_depth_surface_point(self):
        c = self.detection_controller()
        pose = self.run_detection(c, use_surface_point=True)
        # 立着的卡片必须返回深度表面点，而不是桌面平面交点。
        np.testing.assert_allclose(pose[:3, 3], [0.5, 0.0, 0.35])

    def test_bottle_detection_keeps_table_ray_center_by_default(self):
        c = self.detection_controller()
        pose = self.run_detection(c)
        np.testing.assert_allclose(pose[:3, 3], [0.7, 0.0, 0.1])

    def test_held_bottle_filter_uses_cylinder_not_height(self):
        c = self.controller()
        c.bottle_radius, c.bottle_height = 0.03, 0.17
        c.T_link6_bottle = np.eye(4)
        c.T_link6_bottle[:3, 3] = [0.0, 0.0, 0.10]
        c.current_ee_pose_mat = np.eye(4)
        c.current_ee_pose_mat[:3, 3] = [0.4, 0.0, 0.2]
        # 瓶子中心：属于夹爪中的瓶子。
        self.assertTrue(c.held_bottle_contains([0.4, 0.0, 0.30]))
        # 卡片在瓶身外 10cm：即使高度相同也不能被过滤。
        self.assertFalse(c.held_bottle_contains([0.4, 0.10, 0.30]))

    def test_wait_object_allows_ignoring_known_after_detach(self):
        c = self.controller()
        # detach后对象会回到world：known=True、attached=False 必须能通过。
        c.scene.get_known_object_names.return_value = ['target_bottle']
        c.scene.get_attached_objects.return_value = {}
        with patch.object(m.rospy, 'sleep'):
            c.wait_object('target_bottle', known=None, attached=False)
        # 仍然附着时必须超时，并在错误里带上实际状态。
        c.scene.get_attached_objects.return_value = {'target_bottle': object()}
        with patch.object(m.rospy, 'sleep'), \
                patch.object(m.time, 'monotonic', side_effect=[0., 0., 10.]):
            with self.assertRaisesRegex(RuntimeError, 'attached=True'):
                c.wait_object('target_bottle', known=None, attached=False)

    def test_settle_waits_for_release_rebound(self):
        c = self.controller()
        def pose(**values):
            result = {name: 0.0 for name in m.ARM_JOINTS}
            result.update(values)
            return result
        frames = [pose(), pose(joint5=-0.013)]
        c._fresh_joint_positions = MagicMock(
            side_effect=lambda: frames.pop(0) if frames else pose(joint5=-0.013))
        clock = itertools.count(0.0, 0.1)
        with patch.object(m.rospy, 'sleep'), \
                patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)):
            c.wait_for_arm_settled(timeout=2.0, hold=0.2)

    def test_settle_times_out_while_arm_keeps_moving(self):
        c = self.controller()
        def pose(**values):
            result = {name: 0.0 for name in m.ARM_JOINTS}
            result.update(values)
            return result
        frames = itertools.cycle([pose(), pose(joint5=-0.013)])
        c._fresh_joint_positions = MagicMock(side_effect=lambda: next(frames))
        clock = itertools.count(0.0, 0.1)
        with patch.object(m.rospy, 'sleep'), \
                patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(RuntimeError, '未静止'):
                c.wait_for_arm_settled(timeout=1.0, hold=0.2)

    def test_stale_trajectory_start_is_rejected(self):
        c = self.controller()
        c.check_trajectory_start(trajectory())
        c._fresh_joint_positions = MagicMock(return_value={
            **{name: 0.0 for name in m.ARM_JOINTS}, 'joint5': -0.013})
        with self.assertRaisesRegex(RuntimeError, '起点与实机状态'):
            c.check_trajectory_start(trajectory())

    def test_flat_card_center_uses_table_plane(self):
        c = self.controller()
        c.table_model = {
            'plane_normal': [0., 0., 1.], 'plane_offset': 0.0}
        c.current_ee_pose_mat = np.eye(4)
        c.current_ee_pose_mat[:3, 3] = [0., 0., 1.]
        c.T_link6_color_optical = np.array([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ])
        c.bottle_height = 0.2
        intrinsics = SimpleNamespace(
            fx=100., fy=100., ppx=320., ppy=240., width=640, height=480)
        with patch.object(m, 'deproject', side_effect=lambda intr, px, py, d: [
                (px - intr.ppx) / intr.fx * d,
                (py - intr.ppy) / intr.fy * d, d]):
            pose = c.make_bottle_center_pose_in_base(
                (320, 240), intrinsics, center_height=0.0)
            np.testing.assert_allclose(pose[:3, 3], [0., 0., 0.], atol=1e-9)
            pose = c.make_bottle_center_pose_in_base(
                (320, 240), intrinsics, center_height=0.1)
            np.testing.assert_allclose(pose[:3, 3], [0., 0., 0.1], atol=1e-9)

    def test_card_place_target_is_in_front_and_upright(self):
        c = self.place_controller()
        card = np.eye(4)
        card[:3, 3] = [0.5, 0.0, 0.0]
        pre, place, center = c.build_place_targets(card)
        # 卡片检测点在桌面高度(0)时，瓶底被抬到桌面+间隙，避免压进桌面。
        np.testing.assert_allclose(center, [0.4, 0.0, 0.105])
        np.testing.assert_allclose(place[:3, 3], [0.4, 0.0, 0.105])
        np.testing.assert_allclose(place[:3, 2], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(pre[:3, 3], [0.4, 0.0, 0.155])
        c.scene.add_cylinder.assert_not_called()

    def test_card_place_uses_detected_card_height_for_release(self):
        c = self.place_controller()
        card = np.eye(4)
        # 立在桌面上的卡片，检测点比桌面高 0.2m。
        card[:3, 3] = [0.5, 0.0, 0.2]
        pre, place, center = c.build_place_targets(card)
        np.testing.assert_allclose(center, [0.4, 0.0, 0.2])
        np.testing.assert_allclose(place[:3, 3], [0.4, 0.0, 0.2])
        np.testing.assert_allclose(pre[:3, 3], [0.4, 0.0, 0.25])

    def test_place_rejects_insufficient_preplace_clearance(self):
        c = self.place_controller()
        c.place_approach_height = 0.02
        card = np.eye(4)
        card[:3, 3] = [0.5, 0.0, 0.0]
        with self.assertRaisesRegex(RuntimeError, '预放置高度不足'):
            c.build_place_targets(card)

    def test_place_outside_support_is_rejected_before_motion(self):
        c = self.place_controller()
        card = np.eye(4)
        card[:3, 3] = [5.0, 5.0, 0.0]
        c.get_bottle_pose = MagicMock(return_value=card)
        c.move_pose = MagicMock()
        c.command_gripper = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'table边界'):
            c.execute_place()
        c.get_bottle_pose.assert_called_once()
        c.move_pose.assert_not_called()
        c.command_gripper.assert_not_called()

    def test_place_release_detaches_then_rebuilds_world_bottle(self):
        c = self.place_controller()
        card = np.eye(4)
        card[:3, 3] = [0.5, 0.0, 0.0]
        pre = np.eye(4)
        pre[:3, 3] = [0.4, 0.0, 0.15]
        place = np.eye(4)
        place[:3, 3] = [0.4, 0.0, 0.1]
        c.get_bottle_pose = MagicMock(return_value=card)
        c.build_place_targets = MagicMock(return_value=(pre, place, place[:3, 3]))
        c.move_pose = MagicMock()
        c.move_cartesian = MagicMock()
        c.command_gripper = MagicMock()
        c.get_acm = MagicMock(return_value=AllowedCollisionMatrix())
        c.apply_acm = MagicMock()
        c.set_contact_pairs = MagicMock()
        c.wait_object = MagicMock()
        c.current_ee_matrix = MagicMock(return_value=place)
        c.pose_from_matrix = Controller.pose_from_matrix
        c.execute_place()
        c.scene.remove_attached_object.assert_called_once_with(
            'link6', name='target_bottle')
        c.scene.add_cylinder.assert_called_once()
        pose = c.scene.add_cylinder.call_args[0][1]
        np.testing.assert_allclose(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            [0.4, 0.0, 0.1])
        self.assertFalse(c.grasp_confirmed)
        self.assertEqual(c.move_cartesian.call_count, 2)
        c.command_gripper.assert_called_once_with(c.opening_open,
                                                  wait_for_target=True)


if __name__ == '__main__':
    unittest.main()

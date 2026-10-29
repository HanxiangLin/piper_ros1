"""Offline orchestration test for ScanTask (camera and arm fully mocked)."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

BOTTLE_GRAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTTLE_GRAB))

import scan_task as scan_task_module  # noqa: E402
from scan_task import ScanTask  # noqa: E402


def scan_config(root):
    return SimpleNamespace(
        scan_mode="semantic",
        scan_cloud_topic="/piper/static_scan/points",
        scan_camera_frame="camera_color_optical_frame",
        scan_start_joint_target=[0.0, 0.297, -0.648, 0.0, 0.878, 0.0],
        scan_joint1_offsets=[0.0, 0.357, -0.357],
        scan_joint2_offsets=[0.0],
        scan_settle_time=0.0,
        scan_frames_per_view=2,
        scan_frame_interval=0.0,
        scan_min_frame_votes=1,
        scan_min_views=1,
        scan_voxel_m=0.006,
        scan_cluster_voxel_m=0.012,
        scan_min_points=100,
        scan_min_cluster_points=30,
        scan_stride=2,
        scan_min_depth=0.25,
        scan_max_range=2.0,
        scan_roi_min=[0.15, -0.45, -0.10],
        scan_roi_max=[0.80, 0.45, 0.75],
        scan_detect_frames=2,
        scan_table_clearance=0.010,
        scan_footprint_margin=0.10,
        scan_plane_threshold=0.006,
        scan_table_voxel=0.008,
        scan_table_max_tilt=3.0,
        scan_table_min_inliers=300,
        scan_table_min_ratio=0.25,
        scan_table_iterations=200,
        scan_table_edge_padding=0.03,
        scan_table_support_depth=0.02,
        scan_table_mask_padding=0.02,
        scan_table_focus_radius=0.30,
        scan_table_focus_down=0.30,
        scan_table_focus_up=0.02,
        scan_table_footprint_radius=0.60,
        scan_table_surface_epsilon=0.002,
        scan_table_max_jump=0.03,
        scan_map_cloud_voxel=0.015,
        scan_map_base_keepout_radius=0.16,
        scan_map_target_carve=0.02,
        scan_map_target_height_margin=0.04,
        scan_map_support_carve=0.015,
        scan_map_support_margin=0.05,
        scan_map_settle_quiet=0.0,
        scan_map_settle_timeout=1.0,
        scan_map_min_bytes=1000,
        scan_map_require_current_valid=False,
        scan_card_planarity_m=0.012,
        scan_card_min_extent=0.10,
        scan_card_max_thickness=0.03,
        scan_card_max_top_height=0.30,
        scan_cylinder_radius_tol=0.6,
        scan_cylinder_height_tol=0.5,
        scan_non_target_cylinder=True,
        scan_obstacle_padding=0.01,
        scan_artifact_root=root,
        bottle_height=0.17,
        bottle_radius=0.03,
        lift_distance=0.04,
        lift_table_margin=0.005,
        opening_open=0.07,
        enable_place=True,
        place_mode="card",
        place_offset_from_card=0.07,
        place_card_clearance=0.015,
        show_detection_window=False,
        target_label=None,
    )


def scene_cloud():
    rng = np.random.default_rng(7)
    xs = np.arange(0.30, 0.80, 0.006)
    ys = np.arange(-0.30, 0.30, 0.006)
    grid_x, grid_y = np.meshgrid(xs, ys)
    table = np.stack([grid_x.ravel(), grid_y.ravel(),
                      rng.normal(0.0, 0.0007, grid_x.size)], axis=1)

    angles = np.deg2rad(np.arange(90.0, 271.0, 3.0))
    heights = np.arange(0.0, 0.17, 0.004)
    grid_a, grid_z = np.meshgrid(angles, heights)
    bottle = np.stack([
        0.50 + 0.03 * np.cos(grid_a).ravel(),
        -0.15 + 0.03 * np.sin(grid_a).ravel(),
        grid_z.ravel() + 0.001], axis=1)

    card_y = np.arange(0.075, 0.225, 0.006)
    card_z = np.arange(0.0, 0.20, 0.006)
    grid_cy, grid_cz = np.meshgrid(card_y, card_z)
    card = np.stack([
        np.full(grid_cy.size, 0.50),
        grid_cy.ravel(),
        grid_cz.ravel()], axis=1)
    return np.vstack([table, bottle, card])


class ScanTaskTests(unittest.TestCase):
    def build(self):
        root = tempfile.mkdtemp(dir="/tmp/opencode")
        config = scan_config(root)
        state = SimpleNamespace(
            T_link6_bottle=None, T_base_bottle=None, card_model=None,
            scan_result=None, bottle_center_at_grasp=None,
            created_bottle=False, current_ee_pose_mat=np.eye(4),
            grasp_confirmed=False, gripper_service_sent=False)
        arm = MagicMock()
        arm.current_ee_matrix.return_value = np.eye(4)
        arm.move_joints = MagicMock()
        gripper = MagicMock()
        scene = MagicMock()
        scene.obstacles = []
        scene.table_surface_z.return_value = 0.0

        def set_obstacles(specs):
            scene.obstacles = list(specs)

        scene.set_obstacles.side_effect = set_obstacles
        joint_feedback = MagicMock()
        joint_feedback._fresh_joint_positions.return_value = {
            name: 0.0 for name in ("joint1", "joint2")}
        camera = MagicMock()
        hand_eye = MagicMock()
        hand_eye.T_link6_color_optical = np.eye(4)
        detector = MagicMock()
        detector.detect_scene.return_value = [{
            "label": "bottle-g", "conf": 0.9,
            "surface_base": np.array([0.47, -0.15, 0.08]),
            "center_base": np.array([0.50, -0.15, 0.085]),
        }]
        task = ScanTask(config, state, arm, gripper, scene, joint_feedback,
                        camera, hand_eye, detector)
        return task, state, arm, scene, detector, root

    def test_scan_builds_target_card_and_obstacles(self):
        task, state, arm, scene, detector, root = self.build()
        frames = [scene_cloud()[:12000], scene_cloud()[:12000]]
        task._capture = MagicMock(
            return_value=(frames, None, np.eye(4), None))
        result = task.execute()

        # 三个视点：起始位 + 两个 joint1 偏移。
        self.assertEqual(arm.move_joints.call_count, 3)
        self.assertAlmostEqual(arm.move_joints.call_args_list[0][0][0][0], 0.0)
        self.assertAlmostEqual(arm.move_joints.call_args_list[1][0][0][0], 0.357)
        self.assertAlmostEqual(arm.move_joints.call_args_list[2][0][0][0], -0.357)
        for call in arm.move_joints.call_args_list:
            self.assertFalse(call[1].get("guarded", True))

        detector.detect_scene.assert_called_once()
        scene.reset_for_scan.assert_called_once()
        scene.add_bottle.assert_called_once()
        scene.set_table.assert_called()
        scene.freeze.assert_called_once()
        self.assertIsNotNone(state.T_base_bottle)
        self.assertIsNotNone(state.card_model)
        np.testing.assert_allclose(
            result["target_center"], [0.50, -0.15, 0.085], atol=0.01)
        self.assertEqual(result["views"], 3)
        self.assertTrue(any(spec["label"] == "card"
                            for spec in result["obstacles"]))
        # 快照落盘。
        folders = list(Path(root).glob("scan_*"))
        self.assertEqual(len(folders), 1)
        self.assertTrue((folders[0] / "cloud.npz").is_file())
        self.assertTrue((folders[0] / "table_model.json").is_file())

    def test_map_mode_publishes_one_cloud_per_view_and_no_semantic_boxes(self):
        task, state, arm, scene, detector, root = self.build()
        task.config.scan_mode = "map"
        task.map_mode = True
        task.semantic_obstacles = False
        task._cloud_pub = MagicMock()

        signatures = {"count": 0}

        def octomap_signature():
            signatures["count"] += 1
            return ("sig", signatures["count"])

        scene.octomap_signature.side_effect = octomap_signature
        scene.wait_octomap_quiet.side_effect = (
            lambda *args, **kwargs: ("sig", signatures["count"] + 1))
        scene.octomap_snapshot.return_value = {
            "frame": "dummy_link", "id": "OcTree", "binary": True,
            "resolution": 0.02, "data": np.zeros(4000, dtype=np.uint8),
            "bytes": 4000, "sha256": "deadbeef"}
        frames = [scene_cloud()[:12000], scene_cloud()[:12000]]
        stamp = SimpleNamespace(to_sec=lambda: 123.0)
        task._capture = MagicMock(
            return_value=(frames, None, np.eye(4), stamp))
        result = task.execute()
        self.assertEqual(task._cloud_pub.publish.call_count, 3)
        for call in task._cloud_pub.publish.call_args_list:
            self.assertEqual(
                call[0][0].header.frame_id, "camera_color_optical_frame")
        # 默认map模式不建语义障碍盒。
        scene.set_obstacles.assert_not_called()
        self.assertEqual(result["obstacles"], [])
        self.assertEqual(len(task._map_views), 3)
        # 目标瓶的体素必须被挖掉，否则指尖会撞到瓶子自己的点。
        published = task._map_views[0]["points_base"]
        target = np.array([0.50, -0.15])
        radial = np.linalg.norm(published[:, :2] - target, axis=1)
        self.assertTrue(np.all(radial > task.config.bottle_radius + 0.01))
        # 地图快照落盘。
        folders = list(Path(root).glob("scan_*"))
        self.assertTrue((folders[0] / "map" / "octomap.npz").is_file())
        self.assertTrue((folders[0] / "map" / "meta.json").is_file())

    def test_table_tilt_limit_is_strict_only_on_last_view(self):
        task, *_ = self.build()
        self.assertAlmostEqual(task._table_tilt_limit(0), 5.0)
        self.assertAlmostEqual(task._table_tilt_limit(1), 5.0)
        # 最后一个视点的融合桌面才是冻结给抓取用的，必须严格。
        self.assertAlmostEqual(task._table_tilt_limit(2), 3.0)

    def test_frozen_table_must_pass_the_strict_tilt_limit(self):
        import math
        task, *_ = self.build()
        task._require_final_table_tilt({"plane_normal": [0.0, 0.0, 1.0]})
        angle = math.radians(6.0)
        with self.assertRaisesRegex(RuntimeError, "最终桌面倾角"):
            task._require_final_table_tilt(
                {"plane_normal": [math.sin(angle), 0.0, math.cos(angle)]})

    def test_missing_card_aborts_before_freeze(self):
        task, state, arm, scene, detector, root = self.build()
        table_only = scene_cloud()[:12000]
        task._capture = MagicMock(
            return_value=([table_only], None, np.eye(4), None))
        with patch.object(scan_task_module.obstacle_model, "build",
                          return_value=({"target": {
                              "center": np.array([0.5, -0.15, 0.085]),
                              "radius": 0.03, "height": 0.17,
                              "measured_radius": 0.03,
                              "measured_height": 0.17,
                              "points": 100, "label": None},
                              "card": None, "obstacles": [],
                              "clusters": []}, ["stub"])):
            with self.assertRaisesRegex(RuntimeError, "未识别出卡片"):
                task.execute()
        scene.freeze.assert_not_called()

    @staticmethod
    def built(center_y=-0.15):
        return ({
            "target": {"center": np.array([0.50, center_y, 0.085]),
                       "radius": 0.03, "height": 0.17,
                       "measured_radius": 0.03, "measured_height": 0.17,
                       "points": 100, "label": None},
            "card": {"center": np.array([0.50, 0.15, 0.10]),
                     "normal": np.array([-1.0, 0.0, 0.0]),
                     "u": np.array([0.0, 1.0, 0.0]),
                     "v": np.array([0.0, 0.0, 1.0]),
                     "width": 0.20, "length": 0.15, "thickness": 0.005,
                     "points": 100},
            "obstacles": [], "clusters": []}, ["stub"])

    def test_target_jump_between_views_is_rejected(self):
        task, state, arm, scene, detector, root = self.build()
        cloud = scene_cloud()
        task._capture = MagicMock(
            return_value=([cloud], None, np.eye(4), None))
        with patch.object(scan_task_module.obstacle_model, "build") as build:
            build.side_effect = [
                self.built(center_y=-0.15),
                self.built(center_y=0.05),
                self.built(center_y=0.05),
            ]
            with self.assertRaisesRegex(RuntimeError, "目标瓶位置跳变"):
                task.execute()
        scene.freeze.assert_not_called()


if __name__ == "__main__":
    unittest.main()

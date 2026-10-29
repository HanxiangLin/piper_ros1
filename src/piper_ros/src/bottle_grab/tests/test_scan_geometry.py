"""Offline tests for the pure scan-geometry modules (no ROS needed)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BOTTLE_GRAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTTLE_GRAB))

import obstacle_model  # noqa: E402
import table_fit  # noqa: E402
from cloud_fusion import (  # noqa: E402
    FusedCloud,
    crop_points,
    depth_points,
    transform_points,
)
from grasp_corridor import (  # noqa: E402
    distance_to_box,
    distance_to_cylinder,
    filter_by_corridor,
)


def scan_config(**overrides):
    values = dict(
        bottle_height=0.17,
        bottle_radius=0.03,
        target_label=None,
        scan_card_planarity_m=0.012,
        scan_card_min_extent=0.10,
        scan_card_max_thickness=0.03,
        scan_cylinder_radius_tol=0.6,
        scan_cylinder_height_tol=0.5,
        scan_table_clearance=0.010,
        scan_footprint_margin=0.02,
        scan_cluster_voxel_m=0.012,
        scan_min_cluster_points=30,
        scan_obstacle_padding=0.01,
        scan_card_padding=0.005,
        scan_stand_max_distance=0.25,
        scan_stand_max_thickness=0.03,
        scan_non_target_cylinder=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def table_model():
    return {
        "version": 1, "frame_id": "base_link", "object_id": "table",
        "bounds_min": [0.30, -0.30, -0.03],
        "bounds_max": [0.80, 0.30, 0.02],
        "plane_normal": [0.0, 0.0, 1.0],
        "plane_offset": 0.0,
    }


def table_points(step=0.005, noise=0.0007):
    rng = np.random.default_rng(1)
    xs = np.arange(0.30, 0.80 + step, step)
    ys = np.arange(-0.30, 0.30 + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel(),
                       np.zeros(grid_x.size)], axis=1)
    points[:, 2] += rng.normal(0.0, noise, len(points))
    return points


def tilted_table_points(tilt_deg, noise=0.0015, step=0.006):
    """Plane tilted around the y axis, with depth noise like the real sensor."""
    rng = np.random.default_rng(3)
    xs = np.arange(0.30, 0.80 + step, step)
    ys = np.arange(-0.30, 0.30 + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    slope = np.tan(np.deg2rad(tilt_deg))
    points = np.stack([grid_x.ravel(), grid_y.ravel(),
                       slope * grid_x.ravel()], axis=1)
    points[:, 2] += rng.normal(0.0, noise, len(points))
    return points


def bottle_points(center=(0.50, -0.15), radius=0.03, height=0.17,
                  noise=0.0004):
    rng = np.random.default_rng(2)
    angles = np.deg2rad(np.arange(90.0, 271.0, 2.0))
    heights = np.arange(0.0, height + 0.002, 0.003)
    grid_a, grid_z = np.meshgrid(angles, heights)
    # 只保留朝向机器人（-x）的半圆柱，和真实相机看到的一致。
    x = center[0] + radius * np.cos(grid_a)
    y = center[1] + radius * np.sin(grid_a)
    z = grid_z + 0.001
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    points[:, :2] += rng.normal(0.0, noise, (len(points), 2))
    return points


def card_points(center=(0.50, 0.15), width=0.15, height=0.20,
                thickness=0.002, noise=0.0004):
    rng = np.random.default_rng(3)
    ys = np.arange(center[1] - width / 2.0,
                   center[1] + width / 2.0 + 0.005, 0.005)
    zs = np.arange(0.0, height + 0.005, 0.005)
    grid_y, grid_z = np.meshgrid(ys, zs)
    points = np.stack([
        np.full(grid_y.size, center[0]),
        grid_y.ravel(),
        grid_z.ravel()], axis=1)
    points[:, 0] += rng.normal(0.0, thickness / 4.0, len(points))
    points[:, 1:] += rng.normal(0.0, noise, (len(points), 2))
    return points


def scene_points():
    return np.vstack([
        table_points(), bottle_points(), card_points()])


class CloudFusionTests(unittest.TestCase):
    def test_depth_points_uses_intrinsics(self):
        depth = np.full((4, 4), 1000, dtype=np.uint16)
        intrinsics = SimpleNamespace(fx=100.0, fy=100.0, ppx=1.5, ppy=1.5)
        points = depth_points(depth, 0.001, intrinsics, stride=1,
                              depth_min=0.1, depth_max=2.0)
        self.assertEqual(points.shape, (16, 3))
        np.testing.assert_allclose(points[0], [-0.015, -0.015, 1.0])

    def test_depth_points_drops_invalid(self):
        depth = np.zeros((2, 2), dtype=np.uint16)
        depth[0, 0] = 500
        intrinsics = SimpleNamespace(fx=100.0, fy=100.0, ppx=0.0, ppy=0.0)
        points = depth_points(depth, 0.001, intrinsics, stride=1,
                              depth_min=0.1, depth_max=2.0)
        self.assertEqual(len(points), 1)

    def test_transform_and_crop(self):
        points = np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        transform = np.eye(4)
        transform[0, 3] = 0.5
        moved = transform_points(points, transform)
        np.testing.assert_allclose(moved[:, 0], [1.5, 5.5])
        kept = crop_points(moved, [0.0, -1.0, -1.0], [2.0, 1.0, 1.0])
        self.assertEqual(len(kept), 1)

    def test_frame_voting_drops_single_frame_points(self):
        fused = FusedCloud(voxel_m=0.006, min_frame_votes=2)
        first = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        second = np.array([[0.0, 0.0, 0.001], [1.0, 1.0, 1.0]])
        fused.add_view([first, second])
        points, observations = fused.points()
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0005])
        self.assertEqual(observations.tolist(), [1])

    def test_view_support_is_counted(self):
        fused = FusedCloud(voxel_m=0.006, min_frame_votes=1)
        fused.add_view([np.array([[0.0, 0.0, 0.0]])])
        fused.add_view([np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])])
        both, both_obs = fused.points(min_views=2)
        self.assertEqual(len(both), 1)
        np.testing.assert_allclose(both[0], [0.0, 0.0, 0.0])
        self.assertEqual(both_obs.tolist(), [2])

    def test_rejects_non_camera_input(self):
        with self.assertRaises(ValueError):
            depth_points(np.zeros((2, 2)), 0.001,
                         SimpleNamespace(fx=0.0, fy=1.0, ppx=0.0, ppy=0.0))


class TableFitTests(unittest.TestCase):
    def fit(self, points):
        return table_fit.fit_table(
            points, roi_min=[0.25, -0.35, -0.10], roi_max=[0.85, 0.35, 0.45],
            plane_threshold_m=0.006, voxel_m=0.008, max_tilt_deg=3.0,
            min_inliers=300, min_inlier_ratio=0.25, ransac_iterations=200)

    def test_recovers_plane_and_bounds(self):
        model = self.fit(scene_points())
        np.testing.assert_allclose(model["plane_normal"], [0.0, 0.0, 1.0],
                                   atol=0.01)
        self.assertAlmostEqual(table_fit.surface_z(model, 0.5, 0.0), 0.0,
                               delta=0.003)
        self.assertLessEqual(model["bounds_min"][0], 0.30)
        self.assertGreaterEqual(model["bounds_max"][0], 0.80)
        # 顶面现在精确等于支撑平面+epsilon，而不是 max(inlier z)+2cm padding。
        self.assertAlmostEqual(model["bounds_max"][2], 0.002, delta=0.003)

    def test_noisy_one_degree_plane_passes(self):
        # 回归：真实桌面约 1°，三点 RANSAC 预筛不应把它误判成倾斜。
        model = self.fit(tilted_table_points(1.0, noise=0.0025))
        normal = np.asarray(model["plane_normal"], dtype=float)
        tilt = np.degrees(np.arccos(min(1.0, float(normal[2]))))
        self.assertLess(tilt, 3.0)

    def test_genuinely_tilted_plane_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "倾角"):
            self.fit(tilted_table_points(6.0, noise=0.0015))

    def test_rejects_empty_roi(self):
        with self.assertRaises(RuntimeError):
            self.fit(np.zeros((10, 3)))

    def test_model_validation(self):
        model = self.fit(scene_points())
        table_fit.validate_model(model)
        broken = dict(model)
        broken["plane_normal"] = [0.0, 1.0, 0.0]
        with self.assertRaises(ValueError):
            table_fit.validate_model(broken)

    def test_focus_window_ignores_a_larger_floor_plane(self):
        # 大ROI里地板比桌子大得多；没有焦点时全局拟合会选中地板。
        rng = np.random.default_rng(11)
        xs = np.arange(-0.20, 2.00, 0.02)
        ys = np.arange(-1.00, 1.00, 0.02)
        grid_x, grid_y = np.meshgrid(xs, ys)
        floor = np.stack([grid_x.ravel(), grid_y.ravel(),
                          np.full(grid_x.size, -0.60)], axis=1)
        floor[:, 2] += rng.normal(0.0, 0.0007, len(floor))
        points = np.vstack([floor, table_points()])
        lower = [-0.30, -1.20, -0.90]
        upper = [2.10, 1.20, 0.50]
        common = dict(
            roi_min=lower, roi_max=upper, plane_threshold_m=0.006,
            voxel_m=0.008, max_tilt_deg=3.0, min_inliers=300,
            min_inlier_ratio=0.25, ransac_iterations=200)
        global_model = table_fit.fit_table(points, **common)
        self.assertLess(
            table_fit.surface_z(global_model, 0.5, 0.0), -0.30,
            "大ROI的全局拟合应当被地板主导（本测试的前提）")
        focused = table_fit.fit_table(
            points, focus_xyz=[0.50, -0.15, 0.08], focus_radius_m=0.30,
            focus_down_m=0.30, focus_up_m=0.02, **common)
        self.assertAlmostEqual(
            table_fit.surface_z(focused, 0.5, 0.0), 0.0, delta=0.005)
        self.assertEqual(focused["fit_scope"], "focus")

    def test_footprint_radius_caps_stray_coplanar_points(self):
        count = 40
        stray = np.stack([
            np.linspace(1.35, 1.55, count),
            np.zeros(count),
            np.zeros(count)], axis=1)
        points = np.vstack([table_points(), stray])
        common = dict(
            roi_min=[0.20, -0.60, -0.20], roi_max=[1.70, 0.60, 0.50],
            plane_threshold_m=0.006, voxel_m=0.008, max_tilt_deg=3.0,
            min_inliers=300, min_inlier_ratio=0.25, ransac_iterations=200)
        uncapped = table_fit.fit_table(points, **common)
        self.assertGreater(uncapped["bounds_max"][0], 1.4)
        capped = table_fit.fit_table(
            points, focus_xyz=[0.55, 0.0, 0.05], footprint_radius_m=0.60,
            **common)
        self.assertLess(capped["bounds_max"][0], 1.30)

    def test_footprint_top_is_the_plane_not_the_highest_inlier(self):
        model = self.fit(scene_points())
        # 瓶子和卡片远高于桌面，但桌面盒顶面只能高出平面 epsilon。
        self.assertLessEqual(model["bounds_max"][2], 0.01)
        self.assertGreaterEqual(model["bounds_max"][2], -0.002)


class ObstacleModelTests(unittest.TestCase):
    def test_wall_like_panel_is_not_a_card(self):
        config = scan_config()
        plane = {"rms": 0.001, "ratio": 0.99, "thickness": 0.004,
                 "extent_u": 1.20, "extent_v": 1.00}
        wall = {"plane": plane, "top_height": 1.20, "circle": None,
                "height": 1.20}
        self.assertEqual(obstacle_model.classify(wall, config), "unknown")
        card = {"plane": {"rms": 0.001, "ratio": 0.99, "thickness": 0.004,
                          "extent_u": 0.15, "extent_v": 0.20},
                "top_height": 0.20, "circle": None, "height": 0.20}
        self.assertEqual(obstacle_model.classify(card, config), "card")

    @staticmethod
    def cylinder(centroid):
        return {
            "kind": "cylinder", "centroid": np.asarray(centroid, dtype=float),
            "circle": {"radius": 0.03, "center": None, "rms": 0.001},
            "height": 0.17, "plane": {"ratio": 0.4},
        }

    def test_two_bottles_use_the_confident_detection_as_target(self):
        config = scan_config()
        descriptions = [
            self.cylinder([0.50, 0.28, 0.085]),
            self.cylinder([0.50, -0.30, 0.085]),
        ]
        detections = [
            {"conf": 0.60, "surface_base": np.array([0.50, -0.31, 0.09])},
            {"conf": 0.92, "surface_base": np.array([0.50, 0.27, 0.09])},
        ]
        self.assertEqual(
            obstacle_model.choose_target(descriptions, config, detections), 0)

    def test_ambiguous_bottles_are_refused(self):
        config = scan_config()
        descriptions = [
            self.cylinder([0.50, 0.30, 0.085]),
            self.cylinder([0.50, -0.30, 0.085]),
        ]
        detections = [{"conf": 0.90,
                       "surface_base": np.array([0.50, 0.00, 0.09])}]
        with self.assertRaisesRegex(RuntimeError, "无法用检测唯一确定"):
            obstacle_model.choose_target(descriptions, config, detections)

    def build(self):
        config = scan_config()
        table = table_model()
        return obstacle_model.build(
            scene_points(), table, config,
            detections=[{"label": "bottle-g",
                         "surface_base": np.array([0.47, -0.15, 0.08])}])

    def test_splits_bottle_from_card(self):
        result, summary = self.build()
        np.testing.assert_allclose(
            result["target"]["center"], [0.50, -0.15, 0.085], atol=0.008)
        self.assertIsNotNone(result["card"])
        self.assertAlmostEqual(result["card"]["center"][0], 0.50, delta=0.01)
        self.assertAlmostEqual(result["card"]["center"][1], 0.15, delta=0.01)
        sizes = sorted([result["card"]["width"], result["card"]["length"]])
        np.testing.assert_allclose(sizes, [0.15, 0.20], atol=0.02)
        self.assertEqual(len(result["obstacles"]), 1)
        self.assertEqual(result["obstacles"][0]["label"], "card")
        self.assertTrue(any("目标瓶" in line for line in summary))

    def test_card_box_is_a_proper_rotation(self):
        result, _ = self.build()
        pose = result["obstacles"][0]["pose"]
        np.testing.assert_allclose(
            pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0, places=6)

    def test_ambiguous_target_is_rejected(self):
        points = np.vstack([scene_points(), bottle_points(center=(0.60, 0.15))])
        with self.assertRaisesRegex(RuntimeError, "多个圆柱候选"):
            obstacle_model.build(points, table_model(), scan_config())

    def test_colour_label_picks_between_two_bottles(self):
        points = np.vstack([scene_points(), bottle_points(center=(0.60, 0.15))])
        detections = [
            {"label": "bottle-y", "surface_base": np.array([0.57, 0.15, 0.08])},
            {"label": "bottle-g", "surface_base": np.array([0.47, -0.15, 0.08])},
        ]
        result, _ = obstacle_model.build(
            points, table_model(), scan_config(target_label="bottle-y"),
            detections=detections)
        np.testing.assert_allclose(
            result["target"]["center"][:2], [0.60, 0.15], atol=0.01)

    def test_unknown_cluster_never_uses_a_measured_circle(self):
        # 回归：卡片+桌面噪声曾被拟合成 r=0.15 的"圆柱"，把放置位整个包住。
        description = {
            "kind": "unknown",
            "points": 2256,
            "aabb_min": np.array([0.44, -0.09, 0.11]),
            "aabb_max": np.array([0.80, 0.25, 0.30]),
            "height": 0.19,
            "plane": {"normal": np.array([0.0, 0.0, 1.0]),
                      "u": np.array([1.0, 0.0, 0.0]),
                      "v": np.array([0.0, 1.0, 0.0]),
                      "center": np.array([0.62, 0.08, 0.20]),
                      "extent_u": 0.36, "extent_v": 0.34,
                      "thickness": 0.19, "rms": 0.03, "ratio": 0.2},
            "circle": {"center": np.array([0.646, 0.157]),
                       "radius": 0.152, "rms": 0.033},
            "centroid": np.array([0.55, 0.13, 0.19]),
        }
        spec = obstacle_model.spec_from_description(description, 0, 0.01, True)
        self.assertEqual(spec["kind"], "box")
        self.assertEqual(spec["label"], "unknown")
        np.testing.assert_allclose(spec["dims"], [0.38, 0.36, 0.21], atol=1e-6)

    def test_bottle_like_cluster_keeps_measured_cylinder(self):
        description = {
            "kind": "cylinder",
            "points": 600,
            "aabb_min": np.array([0.43, -0.15, 0.11]),
            "aabb_max": np.array([0.51, -0.07, 0.28]),
            "height": 0.17,
            "plane": {"normal": np.array([1.0, 0.0, 0.0]),
                      "u": np.array([0.0, 1.0, 0.0]),
                      "v": np.array([0.0, 0.0, 1.0]),
                      "center": np.array([0.47, -0.11, 0.19]),
                      "extent_u": 0.17, "extent_v": 0.06,
                      "thickness": 0.05, "rms": 0.01, "ratio": 0.5},
            "circle": {"center": np.array([0.465, -0.108]),
                       "radius": 0.028, "rms": 0.003},
            "centroid": np.array([0.46, -0.11, 0.19]),
        }
        spec = obstacle_model.spec_from_description(description, 1, 0.01, True)
        self.assertEqual(spec["kind"], "cylinder")
        np.testing.assert_allclose(spec["dims"], [0.038, 0.19], atol=1e-6)

    def test_table_surface_noise_is_dropped_by_min_height(self):
        config = scan_config(scan_min_obstacle_height=0.02)
        noise = np.concatenate([
            bottle_points(center=(0.30, 0.00), height=0.006),
            bottle_points(center=(0.60, 0.00), height=0.006)])
        groups = [noise]
        descriptions = obstacle_model._describe_groups(
            groups, table_model(), config)
        self.assertEqual(descriptions, [])

    def test_locked_card_is_not_rebuilt_from_a_side_view(self):
        config = scan_config()
        first, _ = self.build()
        card = first["card"]
        self.assertIsNotNone(card)
        # 侧视角：卡片自己的点又出现一次（残影/重复），外加一团真实障碍。
        # 抠除外扩 2cm 的卡片包围盒后，卡片必须只出现一次（稳定 id），
        # 真实障碍仍然被保留。
        grid_y, grid_z = np.meshgrid(np.arange(-0.14, -0.08, 0.004),
                                     np.arange(0.12, 0.18, 0.004))
        clutter = np.column_stack([
            np.full(grid_y.size, 0.72), grid_y.ravel(), grid_z.ravel()])
        points = np.vstack([scene_points(), card_points(), clutter])
        second, summary = obstacle_model.build(
            points, table_model(), config,
            target_center=first["target"]["center"], locked_card=card)
        self.assertIs(second["card"], card)
        labels = [spec["label"] for spec in second["obstacles"]]
        self.assertEqual(labels.count("card"), 1)
        self.assertIn("scan_obstacle_card",
                      [spec["id"] for spec in second["obstacles"]])
        self.assertIn("unknown", labels)
        np.testing.assert_allclose(
            second["target"]["center"][:2], [0.50, -0.15], atol=0.01)

    def test_card_stand_is_excluded_from_obstacles(self):
        config = scan_config()
        first, _ = self.build()
        card = first["card"]
        self.assertIsNotNone(card)
        # 卡片旁边 15cm 处的一块薄板 = 支架：平面度好、很薄、离卡片近。
        grid_y, grid_z = np.meshgrid(np.arange(0.255, 0.315, 0.004),
                                     np.arange(0.12, 0.18, 0.004))
        stand = np.column_stack([
            np.full(grid_y.size, 0.58), grid_y.ravel(), grid_z.ravel()])
        second, summary = obstacle_model.build(
            np.vstack([scene_points(), stand]), table_model(), config,
            target_center=first["target"]["center"], locked_card=card)
        self.assertEqual(
            [spec["label"] for spec in second["obstacles"] if spec["label"] == "unknown"],
            [])
        self.assertTrue(any("已排除卡片支架 1 个" in line for line in summary))

    def test_planar_unknown_uses_thin_oriented_box(self):
        config = scan_config()
        first, _ = self.build()
        card = first["card"]
        grid_y, grid_z = np.meshgrid(np.arange(-0.14, -0.08, 0.004),
                                     np.arange(0.12, 0.18, 0.004))
        plate = np.column_stack([
            np.full(grid_y.size, 0.72), grid_y.ravel(), grid_z.ravel()])
        second, _ = obstacle_model.build(
            np.vstack([scene_points(), plate]), table_model(), config,
            target_center=first["target"]["center"], locked_card=card)
        unknown = [spec for spec in second["obstacles"]
                   if spec["label"] == "unknown"]
        self.assertEqual(len(unknown), 1)
        dims = unknown[0]["dims"]
        self.assertLess(min(dims), 0.03)          # 薄
        self.assertLess(max(dims), 0.20)          # 不是把整片空间都包进去的 AABB
        self.assertFalse(np.allclose(unknown[0]["pose"][:3, :3], np.eye(3)))

    def test_card_box_uses_the_small_card_padding(self):
        config = scan_config()
        first, _ = self.build()
        card = first["card"]
        spec = [item for item in first["obstacles"]
                if item["id"] == "scan_obstacle_card"][0]
        np.testing.assert_allclose(
            spec["dims"][0], card["width"] + 2 * config.scan_card_padding, atol=1e-6)
        np.testing.assert_allclose(
            spec["dims"][1], card["length"] + 2 * config.scan_card_padding, atol=1e-6)

    def test_cluster_separates_two_blobs(self):
        first = np.random.default_rng(4).normal(0.0, 0.002, (200, 3))
        second = np.random.default_rng(5).normal(0.0, 0.002, (200, 3))
        second[:, 0] += 0.5
        groups = obstacle_model.cluster(
            np.vstack([first, second]), voxel_m=0.01, min_points=10)
        self.assertEqual(len(groups), 2)


class CorridorTests(unittest.TestCase):
    def box_spec(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.5, 0.0, 0.20]
        return {"id": "scan_obstacle_0", "kind": "box",
                "dims": (0.05, 0.05, 0.05), "pose": pose}

    def cylinder_spec(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.5, 0.10, 0.10]
        return {"id": "scan_obstacle_1", "kind": "cylinder",
                "dims": (0.03, 0.17), "pose": pose}

    def test_box_distance(self):
        spec = self.box_spec()
        self.assertAlmostEqual(
            distance_to_box([0.5, 0.0, 0.10], spec["pose"], spec["dims"]),
            0.075, places=6)
        self.assertLess(
            distance_to_box([0.5, 0.0, 0.20], spec["pose"], spec["dims"]), 0.0)

    def test_cylinder_distance(self):
        spec = self.cylinder_spec()
        self.assertAlmostEqual(
            distance_to_cylinder([0.5, 0.10, 0.10], spec["pose"],
                                 spec["dims"]), -0.03, places=6)
        self.assertAlmostEqual(
            distance_to_cylinder([0.5, 0.20, 0.10], spec["pose"],
                                 spec["dims"]), 0.07, places=6)

    def test_corridor_filter_keeps_clear_direction(self):
        specs = [self.box_spec()]
        candidates = [
            {"label": 0.0, "start": np.array([0.5, 0.0, 0.35]),
             "end": np.array([0.5, 0.0, 0.20])},
            {"label": 30.0, "start": np.array([0.6, 0.0, 0.35]),
             "end": np.array([0.6, 0.0, 0.20])},
        ]
        accepted, rejected = filter_by_corridor(
            candidates, specs, radius=0.05, samples=20)
        self.assertEqual([item["label"] for item in rejected], [0.0])
        self.assertEqual([item["label"] for item in accepted], [30.0])
        self.assertEqual(rejected[0]["blocking"], "scan_obstacle_0")


if __name__ == "__main__":
    unittest.main()

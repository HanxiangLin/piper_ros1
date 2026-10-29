"""Offline tests for the map carving filters (no ROS needed)."""

import sys
import unittest
from pathlib import Path

import numpy as np

BOTTLE_GRAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTTLE_GRAB))

import map_filter  # noqa: E402


def table_model(z=0.0):
    return {
        "bounds_min": [0.30, -0.30, z - 0.03],
        "bounds_max": [0.90, 0.30, z + 0.002],
        "plane_normal": [0.0, 0.0, 1.0],
        "plane_offset": -float(z),
    }


class MapFilterTests(unittest.TestCase):
    def test_downsample_averages_cells(self):
        points = np.array([
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.100, 0.100, 0.100],
        ])
        result = map_filter.downsample(points, 0.01)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0], [0.0015, 0.0015, 0.0015])
        with self.assertRaises(ValueError):
            map_filter.downsample(points, 0.0)

    def test_base_keepout_is_a_full_height_cylinder(self):
        points = np.array([
            [0.10, 0.00, -0.20],
            [0.15, 0.00, 0.40],
            [0.17, 0.00, 0.40],
            [0.00, 0.20, 0.05],
        ])
        kept, removed = map_filter.carve_base_keepout(points, 0.16)
        self.assertEqual(removed, 2)
        self.assertEqual(len(kept), 2)

    def test_target_cylinder_respects_height(self):
        points = np.array([
            [0.50, 0.00, 0.10],   # inside body
            [0.53, 0.00, 0.10],   # inside radius+margin
            [0.50, 0.00, 0.30],   # above the bottle
            [0.60, 0.00, 0.10],   # outside
        ])
        kept, removed = map_filter.carve_cylinder(
            points, [0.50, 0.0, 0.085], 0.05, 0.21)
        self.assertEqual(removed, 2)
        self.assertEqual(len(kept), 2)

    def test_support_band_only_inside_footprint(self):
        points = np.array([
            [0.50, 0.00, 0.001],   # support surface inside footprint
            [1.50, 0.00, 0.001],   # same height, outside footprint
            [0.50, 0.00, 0.05],    # above the support
        ])
        kept, removed = map_filter.carve_support_band(
            points, table_model(), 0.015, 0.05)
        self.assertEqual(removed, 1)
        np.testing.assert_allclose(kept[:, 2], [0.001, 0.05])

    def test_support_band_tolerance_keeps_thin_objects_above(self):
        points = np.array([[0.50, 0.00, 0.016]])
        kept, removed = map_filter.carve_support_band(
            points, table_model(), 0.015, 0.05)
        self.assertEqual(removed, 0)
        self.assertEqual(len(kept), 1)

    def test_to_optical_round_trip(self):
        rng = np.random.default_rng(5)
        angles = rng.normal(size=3)
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_rotvec(angles).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.18, 0.03, 0.48]
        points = rng.normal(size=(40, 3)) * 0.3 + [0.5, 0.0, 0.1]
        optical = map_filter.to_optical(points, transform)
        restored = optical @ rotation.T + transform[:3, 3]
        np.testing.assert_allclose(restored, points, atol=1e-12)

    def test_build_map_view_removes_robot_target_and_support(self):
        table = table_model()
        support = np.stack([
            np.arange(0.35, 0.85, 0.01),
            np.zeros(int((0.85 - 0.35) / 0.01)),
            np.zeros(int((0.85 - 0.35) / 0.01))], axis=1)
        base = np.array([[0.05, 0.02, 0.30], [0.12, -0.03, 0.10]])
        bottle = np.array([
            [0.55, 0.10, 0.05], [0.55, 0.10, 0.12], [0.52, 0.10, 0.08]])
        wall = np.array([[1.20, 0.00, 0.10], [1.20, 0.20, 0.60]])
        points = np.vstack([support, base, bottle, wall])
        kept, stats = map_filter.build_map_view(
            points, table, target_center=[0.55, 0.10, 0.085],
            target_radius=0.03, target_height=0.17,
            base_keepout_m=0.16, target_carve_m=0.02,
            support_carve_m=0.015, support_xy_margin_m=0.05,
            voxel_m=0.005)
        self.assertEqual(stats["base_keepout_removed"], 2)
        self.assertEqual(stats["target_removed"], 3)
        self.assertEqual(stats["support_removed"], len(support))
        # 只有远处墙面保留下来。
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.all(kept[:, 0] > 1.0))
        self.assertEqual(stats["published"], 2)

    def test_build_map_view_never_returns_nonfinite(self):
        points = np.array([[np.nan, 0.0, 0.0], [0.5, 0.0, 0.1]])
        kept, _ = map_filter.build_map_view(points, None)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()

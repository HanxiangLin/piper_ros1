#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-view point-cloud fusion for the live scene scan.

Pure numpy: no ROS and no camera SDK, so every function here is unit-testable
offline.

Two noise filters are applied:

1. per-frame voxel voting -- a voxel must be observed in at least
   ``min_frame_votes`` frames of the same stationary view.  Single-frame
   speckle, mirror reflections and someone walking through the view are
   dropped;
2. per-view merging -- the fused cloud keeps how many *views* observed each
   voxel.  Callers can require multi-view support where it matters, but a
   real obstacle that is visible from one side only is still kept.
"""

import numpy as np

DEFAULT_STRIDE = 2
# Collision-free linearisation of a 3D integer cell index into one int64.
_KEY_BASE = np.int64(200003)
_KEY_OFFSET = 100000


def depth_points(depth_image, depth_scale, intrinsics, stride=DEFAULT_STRIDE,
                 depth_min=0.1, depth_max=8.0):
    """Convert an aligned depth image into (N, 3) points in the optical frame.

    ``intrinsics`` is any object exposing ``fx``, ``fy``, ``ppx`` and ``ppy``
    (a pyrealsense2 ``intrinsics`` works directly).
    """
    depth = np.asarray(depth_image, dtype=np.float32) * float(depth_scale)
    if depth.ndim != 2:
        raise ValueError("深度图必须是二维数组")
    if stride < 1:
        raise ValueError("stride必须为正整数")
    for name in ("fx", "fy"):
        value = getattr(intrinsics, name, None)
        if value is None or not np.isfinite(value) or value <= 0.0:
            raise ValueError("相机内参无效: " + name)
    for name in ("ppx", "ppy"):
        value = getattr(intrinsics, name, None)
        if value is None or not np.isfinite(value):
            raise ValueError("相机内参无效: " + name)

    depth = depth[::stride, ::stride]
    height, width = depth.shape
    xs = np.arange(width, dtype=np.float64) * stride - float(intrinsics.ppx)
    ys = np.arange(height, dtype=np.float64) * stride - float(intrinsics.ppy)
    grid_x, grid_y = np.meshgrid(xs, ys)
    valid = np.isfinite(depth) & (depth > float(depth_min)) & (depth < float(depth_max))
    if not valid.any():
        return np.zeros((0, 3))
    z = depth[valid].astype(np.float64)
    x = grid_x[valid] * z / float(intrinsics.fx)
    y = grid_y[valid] * z / float(intrinsics.fy)
    return np.stack((x, y, z), axis=1)


def transform_points(points, transform):
    """Apply a 4x4 rigid transform to an (N, 3) array."""
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("点云变换矩阵无效")
    if len(points) == 0:
        return np.zeros((0, 3))
    return points @ transform[:3, :3].T + transform[:3, 3]


def crop_points(points, lower, upper):
    """Return the points inside an axis-aligned base_link box."""
    points = np.asarray(points, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3
            or lower.shape != (3,) or upper.shape != (3,)
            or np.any(upper <= lower)):
        raise ValueError("ROI或点云形状无效")
    if len(points) == 0:
        return points
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    return points[mask]


def cell_keys(points, voxel_m):
    """Linearise ``floor(points / voxel_m)`` into one int64 key per point."""
    voxel_m = float(voxel_m)
    if not np.isfinite(voxel_m) or voxel_m <= 0.0:
        raise ValueError("体素尺寸必须为正有限值")
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    index = np.floor(points / voxel_m).astype(np.int64)
    if np.any(np.abs(index) >= _KEY_OFFSET // 2):
        raise ValueError("点云超出体素索引范围，检查相机外参")
    index = index + _KEY_OFFSET
    return (index[:, 0] * _KEY_BASE + index[:, 1]) * _KEY_BASE + index[:, 2]


class FusedCloud:
    """Voxel accumulator that remembers per-frame votes and per-view support."""

    def __init__(self, voxel_m=0.006, min_frame_votes=2):
        if int(min_frame_votes) < 1:
            raise ValueError("min_frame_votes至少为1")
        self.voxel_m = float(voxel_m)
        self.min_frame_votes = int(min_frame_votes)
        self._cells = {}
        self.view_count = 0
        self.frame_count = 0
        # points of the most recent view (base_link), i.e. what a single
        # camera pose actually saw.  The static obstacle map is published per
        # view so its ray-casting origin stays the real camera position.
        self._view_points = np.zeros((0, 3), dtype=float)

    def add_view(self, frames):
        """Add one stationary view made of several frames.

        ``frames`` is a sequence of (N, 3) arrays in base_link.  A cell is kept
        only when at least ``min_frame_votes`` frames of this view saw it.
        """
        if not frames:
            raise ValueError("一个视点必须包含至少一帧")
        qualified = {}
        used_frames = 0
        for points in frames:
            points = np.asarray(points, dtype=float)
            if len(points) == 0:
                continue
            used_frames += 1
            keys = cell_keys(points, self.voxel_m)
            unique, inverse = np.unique(keys, return_inverse=True)
            sums = np.stack([
                np.bincount(inverse, weights=points[:, axis],
                            minlength=unique.size)
                for axis in range(3)], axis=1)
            counts = np.bincount(inverse, minlength=unique.size)
            for key, center_sum, count in zip(
                    unique.tolist(), sums, counts.tolist()):
                entry = qualified.get(key)
                if entry is None:
                    qualified[key] = [center_sum, count, 1]
                else:
                    entry[0] = entry[0] + center_sum
                    entry[1] += count
                    entry[2] += 1
        if used_frames == 0:
            raise RuntimeError("当前视点没有有效深度帧")

        self.view_count += 1
        self.frame_count += used_frames
        kept = 0
        view_points = []
        for key, (center_sum, count, votes) in qualified.items():
            if votes < self.min_frame_votes:
                continue
            kept += 1
            view_points.append(center_sum / float(count))
            entry = self._cells.get(key)
            if entry is None:
                self._cells[key] = [center_sum, count, 1]
            else:
                entry[0] = entry[0] + center_sum
                entry[1] += count
                entry[2] += 1
        self._view_points = (np.asarray(view_points, dtype=float)
                             if view_points else np.zeros((0, 3), dtype=float))
        return kept

    def last_view_points(self):
        """Points of the most recent view, before cross-view merging."""
        return self._view_points.copy()

    def points(self, min_views=1):
        """Return ``(points, observations)`` for cells seen by N views."""
        min_views = int(min_views)
        if min_views < 1:
            raise ValueError("min_views至少为1")
        selected = [entry for entry in self._cells.values()
                    if entry[2] >= min_views]
        if not selected:
            return np.zeros((0, 3)), np.zeros(0, dtype=int)
        points = np.asarray(
            [entry[0] / float(entry[1]) for entry in selected], dtype=float)
        observations = np.asarray(
            [entry[2] for entry in selected], dtype=int)
        return points, observations

    def clear(self):
        self._cells.clear()
        self.view_count = 0
        self.frame_count = 0
        self._view_points = np.zeros((0, 3), dtype=float)

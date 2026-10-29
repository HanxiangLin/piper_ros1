#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure filters that turn one fused scan view into static obstacle-map points.

The collision map is fed to MoveIt's ``PointCloudOctomapUpdater``, so every
function here works on ``(N, 3)`` numpy arrays and never imports ROS.

Why the map is carved at all:

* the base keep-out cylinder removes the mounting surface / base body that the
  camera sees around the robot; a map cell overlapping ``link1`` would make
  every plan invalid before the arm even moves;
* the target cylinder is removed so the fingers are not blocked by the bottle's
  own voxels (the bottle stays a semantic collision object with an explicit
  finger contact permission);
* the support plane is removed because the table is represented by an exact
  collision box (``table_fit`` top surface) and a map cell 1 cm above the
  surface would otherwise collide with the released bottle (5 mm release
  clearance).
"""

import numpy as np


def _finite_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云必须是(N,3)数组")
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def downsample(points, voxel_m):
    """Average every occupied voxel into one point."""
    voxel_m = float(voxel_m)
    if not np.isfinite(voxel_m) or voxel_m <= 0.0:
        raise ValueError("体素尺寸必须为正有限值")
    points = _finite_points(points)
    if len(points) == 0:
        return points
    cells = np.floor(points / voxel_m).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.stack([
        np.bincount(inverse, weights=points[:, axis], minlength=inverse.max() + 1)
        for axis in range(3)], axis=1)
    counts = np.bincount(inverse, minlength=inverse.max() + 1)
    return sums / counts[:, None]


def carve_base_keepout(points, radius_m):
    """Drop everything inside a vertical cylinder around the base axis."""
    points = _finite_points(points)
    radius_m = float(radius_m)
    if not np.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("底座避让半径必须为非负有限值")
    if len(points) == 0 or radius_m == 0.0:
        return points, 0
    radial = np.linalg.norm(points[:, :2], axis=1)
    drop = radial <= radius_m
    return points[~drop], int(drop.sum())


def carve_cylinder(points, center, radius_m, height_m=None):
    """Drop a vertical cylinder; ``height_m=None`` means full height."""
    points = _finite_points(points)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("圆柱中心必须是有限的三维坐标")
    radius_m = float(radius_m)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("圆柱半径必须为正有限值")
    if len(points) == 0:
        return points, 0
    drop = np.linalg.norm(points[:, :2] - center[None, :2], axis=1) <= radius_m
    if height_m is not None:
        height_m = float(height_m)
        if not np.isfinite(height_m) or height_m <= 0.0:
            raise ValueError("圆柱高度必须为正有限值")
        vertical = (points[:, 2] >= center[2] - height_m / 2.0) & \
                   (points[:, 2] <= center[2] + height_m / 2.0)
        drop &= vertical
    return points[~drop], int(drop.sum())


def carve_support_band(points, table, thickness_m, xy_margin_m=0.0):
    """Drop support-plane points inside the table footprint.

    The table top is represented by an exact collision box, so the map must not
    contain the same surface again with occupancy padding (that is what would
    collide with the released bottle).  Only the footprint is carved: a flat
    object elsewhere at the same height stays in the map (conservative).
    """
    points = _finite_points(points)
    thickness_m = float(thickness_m)
    xy_margin_m = float(xy_margin_m)
    for name, value in (("support_carve", thickness_m),
                        ("support_xy_margin", xy_margin_m)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(name + "必须为非负有限值")
    if len(points) == 0 or thickness_m == 0.0 or table is None:
        return points, 0
    normal = np.asarray(table["plane_normal"], dtype=float)
    offset = float(table["plane_offset"])
    lower = np.asarray(table["bounds_min"], dtype=float)[:2] - xy_margin_m
    upper = np.asarray(table["bounds_max"], dtype=float)[:2] + xy_margin_m
    inside = np.all((points[:, :2] >= lower[None, :])
                    & (points[:, :2] <= upper[None, :]), axis=1)
    distance = np.abs(points @ normal + offset)
    drop = inside & (distance <= thickness_m)
    return points[~drop], int(drop.sum())


def to_optical(points, base_T_optical):
    """Express base_link points in the capture view's optical frame."""
    points = _finite_points(points)
    transform = np.asarray(base_T_optical, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("base_link->optical 变换无效")
    if len(points) == 0:
        return points
    rotation = transform[:3, :3]
    return (points - transform[:3, 3][None, :]) @ rotation


def build_map_view(points, table, target_center=None, target_radius=0.0,
                   target_height=0.0, base_keepout_m=0.16,
                   target_carve_m=0.02, support_carve_m=0.015,
                   support_xy_margin_m=0.05, target_extra_height_m=0.04,
                   voxel_m=0.015):
    """Return ``(map_points, stats)`` for one scan view (base_link frame)."""
    points = _finite_points(points)
    stats = {"input": int(len(points))}
    points, removed = carve_base_keepout(points, base_keepout_m)
    stats["base_keepout_removed"] = removed
    if target_center is not None and float(target_radius) > 0.0:
        height = float(target_height)
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("目标瓶高度必须为正有限值")
        points, removed = carve_cylinder(
            points, target_center, float(target_radius) + float(target_carve_m),
            height + float(target_extra_height_m))
        stats["target_removed"] = removed
    points, removed = carve_support_band(
        points, table, support_carve_m, support_xy_margin_m)
    stats["support_removed"] = removed
    before = len(points)
    points = downsample(points, voxel_m)
    stats["downsampled_removed"] = int(before - len(points))
    stats["published"] = int(len(points))
    return points, stats

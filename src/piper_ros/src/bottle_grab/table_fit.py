#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime table-plane fitting for the live scene scan.

The algorithm is the one used by the offline ``camera_table_scene.py collect``
tool, lifted into a plain module so the grasp program can build the table model
from the fused scan cloud instead of a hand-written JSON file.
"""

import math

import numpy as np


def _downsample(points, voxel_m):
    if len(points) == 0:
        return points
    cells = np.floor(points / voxel_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[indices]


def _inside(points, lower, upper):
    return np.all((points >= lower) & (points <= upper), axis=1)


def _check_inputs(points, roi_min, roi_max, plane_threshold_m, voxel_m,
                  max_tilt_deg, min_inliers, min_inlier_ratio,
                  ransac_iterations):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云必须是(N,3)数组")
    lower = np.asarray(roi_min, dtype=float)
    upper = np.asarray(roi_max, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(upper <= lower)):
        raise ValueError("扫描ROI无效")
    if not np.isfinite(plane_threshold_m) or plane_threshold_m <= 0.0:
        raise ValueError("桌面平面阈值必须为正")
    if not np.isfinite(voxel_m) or voxel_m <= 0.0:
        raise ValueError("桌面拟合体素必须为正")
    if not 0.0 < max_tilt_deg <= 10.0:
        raise ValueError("max_tilt_deg必须在(0,10]内")
    if int(min_inliers) < 3 or int(ransac_iterations) < 3:
        raise ValueError("min_inliers/ransac_iterations至少为3")
    if not 0.0 < min_inlier_ratio <= 1.0:
        raise ValueError("min_inlier_ratio必须在(0,1]内")
    return points, lower, upper


def _ransac_inliers(points, threshold_m, sample_cosine, iterations,
                    min_inliers, rng):
    """Largest inlier mask from near-horizontal 3-point RANSAC samples.

    ``sample_cosine`` filters the *raw three-point sample* normal.  That
    normal is very sensitive to depth noise, so callers may pass a loose
    value here and keep the strict tilt check for the SVD-refined plane.
    """
    best_count = 0
    best = None
    for _ in range(int(iterations)):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < sample_cosine:
            continue
        offset = -float(np.dot(normal, a))
        inliers = np.abs(points @ normal + offset) <= threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best = inliers
    if best is None or best_count < int(min_inliers):
        return None
    return best


def _refine_plane(points, inliers, threshold_m, min_inliers):
    """SVD-refine a RANSAC plane and reselect its inliers."""
    selected = points[inliers]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, center))
    mask = np.abs(points @ normal + offset) <= threshold_m
    if int(mask.sum()) < int(min_inliers):
        raise RuntimeError("桌面平面精修后点不足")
    return normal, offset, mask


def fit_table(points, roi_min, roi_max, plane_threshold_m=0.006,
              voxel_m=0.008, max_tilt_deg=3.0, min_inliers=300,
              min_inlier_ratio=0.25, ransac_iterations=200,
              edge_padding_m=0.03, support_depth_m=0.02,
              mask_padding_m=0.02, min_span_m=0.05, seed=42,
              max_points=30000):
    """Fit the support plane and return a table model dict.

    The returned dict uses the same schema as the offline scanner
    (``version/frame_id/object_id/bounds_*/plane_*``) so existing helpers such
    as ``table_surface_z`` keep working unchanged.
    """
    points, lower_roi, upper_roi = _check_inputs(
        points, roi_min, roi_max, plane_threshold_m, voxel_m, max_tilt_deg,
        min_inliers, min_inlier_ratio, ransac_iterations)
    for name, value in (("edge_padding_m", edge_padding_m),
                        ("support_depth_m", support_depth_m),
                        ("mask_padding_m", mask_padding_m)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(name + "必须为非负有限值")

    points = points[np.isfinite(points).all(axis=1)]
    points = points[_inside(points, lower_roi, upper_roi)]
    points = _downsample(points, voxel_m)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]
    if len(points) < int(min_inliers):
        raise RuntimeError(
            "扫描ROI内有效点不足(%d<%d)，检查相机视野和扫描视点"
            % (len(points), int(min_inliers)))

    strict_cosine = math.cos(math.radians(max_tilt_deg))
    # 三点随机样本的法向噪声很大：严格预筛经常选到坏样本，精修后反而
    # 报倾角超限。先按严格预筛拟合；只有精修失败时才用宽松预筛重试。
    # 最终仍然只接受精修倾角 <= max_tilt_deg 的平面，安全门没有放松。
    loose_tilt_deg = min(45.0, max_tilt_deg + 15.0)
    loose_cosine = math.cos(math.radians(loose_tilt_deg))
    failures = []
    accepted = None
    for label, sample_cosine in (("严格预筛", strict_cosine),
                                 ("宽松预筛", loose_cosine)):
        rng = np.random.default_rng(seed)
        inliers = _ransac_inliers(
            points, plane_threshold_m, sample_cosine,
            ransac_iterations, min_inliers, rng)
        if inliers is None:
            failures.append("%s: 未找到足够大的近水平桌面" % label)
            continue
        if inliers.mean() < min_inlier_ratio:
            failures.append(
                "%s: 桌面平面占比不足(%.2f<%.2f)"
                % (label, inliers.mean(), min_inlier_ratio))
            continue
        try:
            normal, offset, mask = _refine_plane(
                points, inliers, plane_threshold_m, min_inliers)
        except RuntimeError as error:
            failures.append("%s: %s" % (label, error))
            continue
        tilt_deg = math.degrees(math.acos(
            min(1.0, max(-1.0, float(normal[2])))))
        if tilt_deg > max_tilt_deg + 1e-6:
            failures.append(
                "%s: 精修倾角%.2f°>%.2f°(内点%d/%d)"
                % (label, tilt_deg, max_tilt_deg,
                   int(mask.sum()), len(points)))
            continue
        if mask.mean() < min_inlier_ratio:
            failures.append(
                "%s: 精修后占比不足(%.2f<%.2f)"
                % (label, mask.mean(), min_inlier_ratio))
            continue
        accepted = (normal, offset, mask)
        break
    if accepted is None:
        raise RuntimeError("桌面平面拟合失败: " + "; ".join(failures))

    normal, offset, mask = accepted
    selected = points[mask]

    span = np.ptp(selected[:, :2], axis=0)
    if np.any(span < min_span_m):
        raise RuntimeError("桌面可见范围太小(%.3fm)，重新扫描" % float(span.min()))

    bounds_min = selected.min(axis=0)
    bounds_max = selected.max(axis=0)
    bounds_min[:2] -= edge_padding_m
    bounds_max[:2] += edge_padding_m
    bounds_min[2] -= support_depth_m + mask_padding_m
    bounds_max[2] += mask_padding_m
    return {
        "version": 1,
        "frame_id": "base_link",
        "object_id": "table",
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "plane_normal": normal.tolist(),
        "plane_offset": float(offset),
        "inlier_count": int(len(selected)),
        "sample_count": int(len(points)),
        "voxel_m": float(voxel_m),
        "note": "运行时扫描拟合的局部支撑桌面包络",
    }


def validate_model(model):
    """Raise ValueError unless ``model`` is a usable table model."""
    if not isinstance(model, dict):
        raise ValueError("桌面模型必须是字典")
    if (model.get("version") != 1 or model.get("frame_id") != "base_link"
            or model.get("object_id") != "table"):
        raise ValueError("桌面模型版本、frame或ID无效")
    lower = np.asarray(model["bounds_min"], dtype=float)
    upper = np.asarray(model["bounds_max"], dtype=float)
    normal = np.asarray(model["plane_normal"], dtype=float)
    offset = float(model["plane_offset"])
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower)):
        raise ValueError("桌面模型边界无效")
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isfinite(offset)
            or abs(np.linalg.norm(normal) - 1.0) > 1e-3
            or normal[2] < math.cos(math.radians(10.0))):
        raise ValueError("桌面平面参数无效")
    return model


def surface_z(model, x, y):
    """Return the table plane height at ``(x, y)`` in base_link."""
    normal = np.asarray(model["plane_normal"], dtype=float)
    if abs(normal[2]) < 1e-6:
        raise ValueError("桌面法向几乎水平，无法求表面高度")
    return -(normal[0] * x + normal[1] * y
             + float(model["plane_offset"])) / normal[2]

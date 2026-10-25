#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn a fused scan cloud into one target bottle plus explicit obstacles.

Pure numpy / no ROS.  The scene the program has to handle is: one bottle, one
printed card standing on the table, plus the table itself.  Everything that is
not the target bottle becomes a named collision object, so the planner can tell
"avoid that" from "grasp this" instead of treating the whole occupancy map as
one opaque obstacle.

Classification uses geometry only:

* a bottle is a vertical cylinder whose fitted radius and height match the
  configured bottle (``bottle_radius`` / ``bottle_height``);
* a printed card is a thin plate: most points lie on one plane and its in-plane
  extent is much larger than the bottle diameter, so it can never be mistaken
  for the target even when it shows the same colour photo.
"""

import numpy as np

OBSTACLE_PREFIX = "scan_obstacle_"

# Half of the 26-neighbourhood; every pair is visited exactly once.
_NEIGHBORS = (
    (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0),
    (-1, -1, 1), (0, -1, 1), (1, -1, 1), (-1, 0, 1),
    (0, 0, 1), (1, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1),
)


# ----------------------------------------------------------------------
# Segmentation and clustering.
# ----------------------------------------------------------------------
def segment(points, table, clearance=0.008, footprint_margin=0.10):
    """Keep points above the table plane and inside the table footprint."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros((0, 3))
    normal = np.asarray(table["plane_normal"], dtype=float)
    offset = float(table["plane_offset"])
    lower = np.asarray(table["bounds_min"], dtype=float) - [
        footprint_margin, footprint_margin, 0.0]
    upper = np.asarray(table["bounds_max"], dtype=float) + [
        footprint_margin, footprint_margin, 0.0]
    above = points @ normal + offset > float(clearance)
    inside = np.all((points[:, :2] >= lower[:2])
                    & (points[:, :2] <= upper[:2]), axis=1)
    return points[above & inside]


def cluster(points, voxel_m=0.012, min_points=40):
    """Grid connected-component clustering without scipy/sklearn."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return []
    if not np.isfinite(voxel_m) or voxel_m <= 0.0:
        raise ValueError("聚类体素必须为正")
    index = np.floor(points / float(voxel_m)).astype(np.int64)
    index -= index.min(axis=0)
    dims = index.max(axis=0) + 1
    keys = (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    _, first = np.unique(inverse, return_index=True)
    cells = index[first]
    parent = np.arange(unique.size, dtype=np.int64)

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for offset in _NEIGHBORS:
        neighbor = cells + np.asarray(offset, dtype=np.int64)
        valid = np.all((neighbor >= 0) & (neighbor < dims), axis=1)
        if not valid.any():
            continue
        neighbor_keys = (neighbor[:, 0] * dims[1] + neighbor[:, 1]) * dims[2] \
            + neighbor[:, 2]
        position = np.clip(np.searchsorted(unique, neighbor_keys),
                           0, unique.size - 1)
        matched = valid & (unique[position] == neighbor_keys)
        for left, right in zip(
                np.nonzero(matched)[0].tolist(), position[matched].tolist()):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

    labels = np.asarray([find(node) for node in range(unique.size)],
                        dtype=np.int64)
    point_labels = labels[inverse]
    groups = []
    for label in np.unique(point_labels):
        part = points[point_labels == label]
        if len(part) >= int(min_points):
            groups.append(part)
    groups.sort(key=len, reverse=True)
    return groups


# ----------------------------------------------------------------------
# Shape description.
# ----------------------------------------------------------------------
def _fit_circle_xy(xy):
    """Algebraic (Kasa) circle fit; robust enough for a half-visible bottle."""
    x, y = xy[:, 0], xy[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = x * x + y * y
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    cx, cy, constant = solution
    squared = constant + cx * cx + cy * cy
    if not np.isfinite(squared) or squared <= 0.0:
        return None
    center = np.array([cx, cy], dtype=float)
    radius = float(np.sqrt(squared))
    residual = float(np.sqrt(np.mean(
        (np.linalg.norm(xy - center, axis=1) - radius) ** 2)))
    return center, radius, residual


def describe(points, table, planarity_m=0.012):
    """Return geometric features used for classification and obstacle specs."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("簇点数不足，无法描述")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal, axis_u, axis_v = vt[2], vt[0], vt[1]
    # SVD does not guarantee a right-handed basis; a reflected box pose would
    # make MoveIt reject the collision object.
    if np.linalg.det(np.column_stack((axis_u, axis_v, normal))) < 0.0:
        axis_v = -axis_v
    distance = centered @ normal
    rms = float(np.sqrt(np.mean(distance ** 2)))
    planarity_ratio = float(np.mean(np.abs(distance) <= planarity_m))
    proj_u = centered @ axis_u
    proj_v = centered @ axis_v
    # 用 1%/99% 分位而不是 min/max：边缘几个离群点会把薄板撑大 1~2cm。
    u_low, u_high = np.percentile(proj_u, [1.0, 99.0])
    v_low, v_high = np.percentile(proj_v, [1.0, 99.0])
    extent_u = float(u_high - u_low)
    extent_v = float(v_high - v_low)
    center_plane = (centroid
                    + 0.5 * (u_high + u_low) * axis_u
                    + 0.5 * (v_high + v_low) * axis_v)

    circle = _fit_circle_xy(points[:, :2])
    aabb_min = points.min(axis=0)
    aabb_max = points.max(axis=0)

    model = {
        "points": int(len(points)),
        "centroid": centroid,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "height": float(aabb_max[2] - aabb_min[2]),
        "plane": {
            "normal": normal, "u": axis_u, "v": axis_v,
            "center": center_plane,
            "extent_u": extent_u, "extent_v": extent_v,
            "thickness": float(2.0 * np.percentile(np.abs(distance), 90)),
            "rms": rms, "ratio": planarity_ratio,
        },
        "planar": bool(rms <= planarity_m and planarity_ratio >= 0.85),
        "circle": None if circle is None else {
            "center": circle[0], "radius": circle[1], "rms": circle[2]},
        "table_z": float(_table_z(table, center_plane[0], center_plane[1])),
        "top_height": float(aabb_max[2]
                            - _table_z(table, center_plane[0], center_plane[1])),
        "kind": "unknown",
    }
    return model


def _table_z(table, x, y):
    normal = np.asarray(table["plane_normal"], dtype=float)
    return -(normal[0] * x + normal[1] * y
             + float(table["plane_offset"])) / normal[2]


def classify(description, config):
    """Label a cluster as ``card``, ``cylinder`` or ``unknown``."""
    plane = description["plane"]
    circle = description["circle"]
    planarity_m = float(config.scan_card_planarity_m)
    small_extent = min(plane["extent_u"], plane["extent_v"])
    is_plate = (
        plane["rms"] <= planarity_m
        and plane["ratio"] >= 0.85
        and small_extent >= float(config.scan_card_min_extent)
        and plane["thickness"] <= float(config.scan_card_max_thickness)
        # A bottle shell also has one almost-planar direction; a real plate is
        # flat compared with its own size.
        and plane["thickness"] <= 0.3 * small_extent)
    if is_plate:
        return "card"

    if circle is not None:
        radius_tolerance = float(config.scan_cylinder_radius_tol)
        height_tolerance = float(config.scan_cylinder_height_tol)
        radius_ok = abs(circle["radius"] - config.bottle_radius) \
            <= radius_tolerance * config.bottle_radius
        height_ok = (abs(description["height"] - config.bottle_height)
                     <= height_tolerance * config.bottle_height)
        if radius_ok and height_ok:
            return "cylinder"
    return "unknown"


# ----------------------------------------------------------------------
# Target selection.
# ----------------------------------------------------------------------
def _surface_distance(point, points, sample=1500):
    point = np.asarray(point, dtype=float)
    if len(points) > sample:
        step = max(1, len(points) // sample)
        points = points[::step]
    return float(np.min(np.linalg.norm(points - point, axis=1)))


def associate_detections(detections, clusters):
    """Map cluster index -> colour label of the nearest YOLO detection."""
    labels = {}
    if not detections:
        return labels
    for detection in detections:
        surface = detection.get("surface_base")
        if surface is None:
            continue
        distances = [_surface_distance(surface, part) for part in clusters]
        index = int(np.argmin(distances))
        if distances[index] <= 0.03 and index not in labels:
            labels[index] = detection.get("label")
    return labels


def choose_target(descriptions, config):
    """Return the index of the single cylinder-shaped cluster."""
    candidates = [index for index, item in enumerate(descriptions)
                  if item["kind"] == "cylinder"]
    if config.target_label is not None and candidates:
        matched = [index for index in candidates
                   if descriptions[index].get("label") == config.target_label]
        if matched:
            candidates = matched
    if len(candidates) == 1:
        return candidates[0]
    summary = "; ".join(
        "簇%d kind=%s 半径=%.3f 高=%.3f 平面度=%.2f"
        % (index, item["kind"],
           -1.0 if item["circle"] is None else item["circle"]["radius"],
           item["height"], item["plane"]["ratio"])
        for index, item in enumerate(descriptions))
    if not candidates:
        raise RuntimeError("未找到圆柱形目标瓶: " + summary)
    raise RuntimeError("存在多个圆柱候选，无法确定目标: " + summary)


def nearest_cluster(point, clusters):
    """Return the index of the cluster closest to a base_link point."""
    distances = [_surface_distance(point, part) for part in clusters]
    if not distances:
        raise RuntimeError("没有可匹配的点云簇")
    return int(np.argmin(distances))


# ----------------------------------------------------------------------
# Geometry of the locked card / target (used to carve the fused cloud).
# ----------------------------------------------------------------------
def card_pose_and_dims(card, margin=0.0):
    """Return the oriented box (pose, dims) of a scanned card model."""
    normal = np.asarray(card["normal"], dtype=float)
    axis_u = np.asarray(card.get("u", [0.0, 1.0, 0.0]), dtype=float)
    axis_v = np.asarray(card.get("v", [0.0, 0.0, 1.0]), dtype=float)
    if np.linalg.det(np.column_stack((axis_u, axis_v, normal))) < 0.0:
        axis_v = -axis_v
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((axis_u, axis_v, normal))
    pose[:3, 3] = np.asarray(card["center"], dtype=float)
    dims = (float(card["width"]) + 2.0 * margin,
            float(card["length"]) + 2.0 * margin,
            max(float(card["thickness"]), 0.004) + 2.0 * margin)
    return pose, dims


def _outside_box(points, pose, dims):
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    outside = np.abs(local) - 0.5 * np.asarray(dims, dtype=float)
    return np.linalg.norm(np.maximum(outside, 0.0), axis=1) > 0.0


def _outside_cylinder(points, center, radius, height):
    radial = np.linalg.norm(points[:, :2] - center[None, :2], axis=1) - radius
    vertical = np.abs(points[:, 2] - center[2]) - height / 2.0
    inside = (radial <= 0.0) & (vertical <= 0.0)
    return ~inside


def _inside_box(point, pose, dims):
    local = pose[:3, :3].T @ (np.asarray(point, dtype=float) - pose[:3, 3])
    return bool(np.all(np.abs(local) <= 0.5 * np.asarray(dims, dtype=float)))


def refine_target_center(points, previous, table, config):
    """Refine the target centre from points around the known cylinder."""
    previous = np.asarray(previous, dtype=float)
    if previous.shape != (3,):
        raise ValueError("目标瓶中心无效")
    radius = float(config.bottle_radius)
    table_z = _table_z(table, previous[0], previous[1])
    select = (
        (np.linalg.norm(points[:, :2] - previous[None, :2], axis=1)
         <= radius + 0.03)
        & (points[:, 2] >= table_z)
        & (points[:, 2] <= table_z + float(config.bottle_height) + 0.03))
    selected = points[select]
    if len(selected) < 60:
        return np.array([previous[0], previous[1],
                         table_z + float(config.bottle_height) / 2.0])
    circle = _fit_circle_xy(selected[:, :2])
    if (circle is None
            or abs(circle[1] - radius) > 0.5 * radius
            or not np.isfinite(circle[0]).all()):
        center_xy = previous[:2]
    else:
        center_xy = circle[0]
    if float(np.linalg.norm(center_xy - previous[:2])) > 0.05:
        raise RuntimeError("目标瓶位置在多视点融合中跳变超过5cm，重新扫描")
    return np.array([center_xy[0], center_xy[1],
                     table_z + float(config.bottle_height) / 2.0])


def _target_dict(center, config, measured_radius=None, measured_height=None,
                 points=0, label=None):
    center = np.asarray(center, dtype=float)
    return {
        "center": center.copy(),
        "radius": float(config.bottle_radius),
        "height": float(config.bottle_height),
        "measured_radius": measured_radius,
        "measured_height": measured_height,
        "points": int(points),
        "label": label,
    }


def _card_dict(description):
    plane = description["plane"]
    return {
        "center": np.asarray(plane["center"], dtype=float).copy(),
        "normal": np.asarray(plane["normal"], dtype=float).copy(),
        "u": np.asarray(plane["u"], dtype=float).copy(),
        "v": np.asarray(plane["v"], dtype=float).copy(),
        "width": float(plane["extent_u"]),
        "length": float(plane["extent_v"]),
        "thickness": float(plane["thickness"]),
        "points": description["points"],
    }


def _describe_groups(groups, table, config, labels=None):
    """Describe + classify clusters, dropping table-surface noise."""
    descriptions = []
    minimum = float(getattr(config, "scan_min_obstacle_height", 0.0))
    for index, group in enumerate(groups):
        description = describe(group, table, config.scan_card_planarity_m)
        if description["top_height"] < minimum:
            continue
        description["kind"] = classify(description, config)
        description["label"] = (labels or {}).get(index)
        descriptions.append(description)
    return descriptions


# ----------------------------------------------------------------------
# Collision-object specs.
# ----------------------------------------------------------------------
def _box_pose(center, axis_u, axis_v, normal):
    rotation = np.column_stack((axis_u, axis_v, normal))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(center, dtype=float)
    return pose


def spec_from_description(description, index, padding, use_measured_cylinder,
                          card_padding=None):
    """Build one ``{'id','kind','dims','pose'}`` collision-object spec.

    Only a cluster that was classified as a bottle may use the measured circle:
    an ``unknown`` blob's circle fit is meaningless (a plate can "fit" a 16 cm
    circle) and would produce a huge phantom cylinder.
    """
    name = "%s%d" % (OBSTACLE_PREFIX, index)
    kind = description["kind"]
    plane = description["plane"]
    if kind == "card":
        pad = padding if card_padding is None else float(card_padding)
        dims = (plane["extent_u"] + 2.0 * pad,
                plane["extent_v"] + 2.0 * pad,
                plane["thickness"] + 2.0 * pad)
        return {
            "id": name, "kind": "box", "dims": dims,
            "pose": _box_pose(plane["center"], plane["u"], plane["v"],
                              plane["normal"]),
            "points": description["points"], "label": "card",
            "center": np.asarray(plane["center"], dtype=float).copy(),
        }
    circle = description["circle"]
    if kind == "cylinder" and use_measured_cylinder and circle is not None:
        height = description["height"] + 2.0 * padding
        center = np.array([
            circle["center"][0], circle["center"][1],
            0.5 * (description["aabb_min"][2] + description["aabb_max"][2])])
        pose = np.eye(4)
        pose[:3, 3] = center
        return {
            "id": name, "kind": "cylinder",
            "dims": (circle["radius"] + padding, height),
            "pose": pose, "points": description["points"],
            "label": "cylinder", "center": center.copy(),
        }
    if description.get("planar"):
        # 薄板类未知障碍（支架、纸板、面板）：用斜的薄盒，别用大 AABB。
        dims = (plane["extent_u"] + 2.0 * padding,
                plane["extent_v"] + 2.0 * padding,
                plane["thickness"] + 2.0 * padding)
        return {
            "id": name, "kind": "box", "dims": dims,
            "pose": _box_pose(plane["center"], plane["u"], plane["v"],
                              plane["normal"]),
            "points": description["points"], "label": kind,
            "center": np.asarray(plane["center"], dtype=float).copy(),
        }
    lower = description["aabb_min"] - padding
    upper = description["aabb_max"] + padding
    pose = np.eye(4)
    pose[:3, 3] = 0.5 * (lower + upper)
    return {
        "id": name, "kind": "box",
        "dims": tuple((upper - lower).tolist()),
        "pose": pose, "points": description["points"], "label": kind,
        "center": pose[:3, 3].copy(),
    }


def is_card_stand(description, card, config):
    """True for the card's support/stand: a thin plate next to the card.

    It is only visible from the side views (it is hidden behind the card in the
    front view).  The operator asked to keep it out of the obstacle set: the
    bottle is released in front of the card and never reaches the stand.
    """
    if card is None or not description.get("planar"):
        return False
    limit = getattr(config, "scan_stand_max_distance", 0.25)
    thickness = getattr(config, "scan_stand_max_thickness", 0.03)
    if description["plane"]["thickness"] > float(thickness):
        return False
    distance = float(np.linalg.norm(
        description["centroid"] - np.asarray(card["center"], dtype=float)))
    return distance <= float(limit)


def card_spec(card, padding, name=OBSTACLE_PREFIX + "card"):
    """Stable collision spec for the scanned card (id does not depend on view)."""
    pose, dims = card_pose_and_dims(card, padding)
    return {"id": name, "kind": "box", "dims": dims, "pose": pose,
            "points": int(card.get("points", 0)), "label": "card",
            "center": np.asarray(card["center"], dtype=float).copy()}


def _summary(table, target, card, specs, stands=None):
    def number(value):
        if value is None or not np.isfinite(value):
            return "n/a"
        return "%.3f" % value

    return [
        "桌面平面 z=%.4f 范围 x[%.3f,%.3f] y[%.3f,%.3f]"
        % (_table_z(table, 0.0, 0.0),
           table["bounds_min"][0], table["bounds_max"][0],
           table["bounds_min"][1], table["bounds_max"][1]),
        "目标瓶 center=(%.4f, %.4f, %.4f) 实测半径=%s 实测高=%s 颜色=%s"
        % (target["center"][0], target["center"][1], target["center"][2],
           number(target["measured_radius"]),
           number(target["measured_height"]), target["label"]),
        "障碍物 %d 个: %s" % (len(specs), ", ".join(
            "%s(%s,%s,%d点)" % (item["id"], item["label"], item["kind"],
                                item["points"])
            for item in specs)),
        "卡片: " + ("未识别" if card is None else
                    "center=(%.3f,%.3f,%.3f) %.0fx%.0fcm 厚%.0fmm"
                    % (card["center"][0], card["center"][1], card["center"][2],
                       card["width"] * 100.0, card["length"] * 100.0,
                       card["thickness"] * 1000.0)),
        "已排除卡片支架 %d 个%s" % (len(stands or []), "" if not stands else
                                   ": " + ", ".join(
                                       "%.3fx%.3fcm@(%.2f,%.2f)"
                                       % (item["plane"]["extent_u"] * 100,
                                          item["plane"]["extent_v"] * 100,
                                          item["centroid"][0],
                                          item["centroid"][1])
                                       for item in stands)),
    ]


def build(points, table, config, target_center=None, detections=None,
          locked_card=None):
    """Segment, cluster, classify and return a scan obstacle map.

    ``locked_card`` is the card model decided from the first (front) view.  Once
    it exists, later views only refine the target centre and rebuild the
    *remaining* obstacles with the card and target carved out: the card must
    not be re-classified from a side view where background points can merge
    into it.
    """
    clearance = max(float(config.scan_table_clearance), 0.0)
    obstacle_points = segment(
        points, table, clearance=clearance,
        footprint_margin=float(config.scan_footprint_margin))
    if len(obstacle_points) == 0:
        raise RuntimeError("桌面以上没有点云，检查扫描视点和ROI")

    if locked_card is not None and target_center is not None:
        center = refine_target_center(
            obstacle_points, target_center, table, config)
        rest = _outside_cylinder(
            obstacle_points, center, float(config.bottle_radius) + 0.02,
            float(config.bottle_height) + 0.04)
        rest = rest & _outside_box(
            obstacle_points, *card_pose_and_dims(
                locked_card, float(getattr(config, "scan_card_carve_margin", 0.02))))
        remaining = obstacle_points[rest]
        groups = []
        if len(remaining):
            pose, dims = card_pose_and_dims(locked_card)
            for group in cluster(remaining, config.scan_cluster_voxel_m,
                                 config.scan_min_cluster_points):
                # 抠除后仍有残留点落在卡片包围盒里，直接丢弃，避免重复建模。
                if not _inside_box(group.mean(axis=0), pose, dims):
                    groups.append(group)
        descriptions = _describe_groups(groups, table, config)
        specs, stands = [], []
        for index, item in enumerate(descriptions):
            if is_card_stand(item, locked_card, config):
                stands.append(item)
                continue
            specs.append(spec_from_description(
                item, index, float(config.scan_obstacle_padding),
                config.scan_non_target_cylinder,
                card_padding=float(getattr(config, "scan_card_padding", None)
                                   or config.scan_obstacle_padding)))
        specs.append(card_spec(
            locked_card, float(getattr(config, "scan_card_padding", None)
                               or config.scan_obstacle_padding)))
        target = _target_dict(center, config)
        return {"target": target, "card": locked_card, "obstacles": specs,
                "clusters": descriptions}, _summary(
                    table, target, locked_card, specs, stands)

    groups = cluster(obstacle_points, config.scan_cluster_voxel_m,
                     config.scan_min_cluster_points)
    if not groups:
        raise RuntimeError("桌面以上点云聚类为空，降低scan_min_cluster_points")
    labels = associate_detections(detections, groups)
    descriptions = _describe_groups(groups, table, config, labels)
    if not descriptions:
        raise RuntimeError(
            "桌面以上只有低于%.0fmm的噪声，检查scan_min_obstacle_height"
            % (float(getattr(config, "scan_min_obstacle_height", 0.0)) * 1000))

    if target_center is None:
        target_index = choose_target(descriptions, config)
    else:
        target_index = None
        best = float("inf")
        for index, description in enumerate(descriptions):
            distance = float(np.linalg.norm(
                description["centroid"] - np.asarray(target_center)))
            if distance < best:
                best, target_index = distance, index
        if target_index is None or best > float(config.bottle_radius) + 0.06:
            raise RuntimeError(
                "目标瓶在多视点融合中丢失(最近簇%.3fm)，重新扫描" % best)

    target_description = descriptions[target_index]
    circle = target_description["circle"]
    if circle is None:
        raise RuntimeError("目标瓶簇没有有效圆柱拟合，重新扫描")
    center = np.array([
        circle["center"][0], circle["center"][1],
        _table_z(table, circle["center"][0], circle["center"][1])
        + float(config.bottle_height) / 2.0])
    target = _target_dict(
        center, config, measured_radius=float(circle["radius"]),
        measured_height=float(target_description["height"]),
        points=target_description["points"],
        label=target_description.get("label"))

    card_index = next(
        (index for index, item in enumerate(descriptions)
         if item["kind"] == "card" and index != target_index), None)
    card = None if card_index is None else _card_dict(descriptions[card_index])

    specs, stands = [], []
    for index, item in enumerate(descriptions):
        if index in (target_index, card_index):
            continue
        if is_card_stand(item, card, config):
            stands.append(item)
            continue
        specs.append(spec_from_description(
            item, index, float(config.scan_obstacle_padding),
            config.scan_non_target_cylinder,
            card_padding=float(getattr(config, "scan_card_padding", None)
                               or config.scan_obstacle_padding)))
    if card is not None:
        specs.append(card_spec(
            card, float(getattr(config, "scan_card_padding", None)
                        or config.scan_obstacle_padding)))

    return {"target": target, "card": card, "obstacles": specs,
            "clusters": descriptions}, _summary(
                table, target, card, specs, stands)

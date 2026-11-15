#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Free-space corridor check for grasp approach candidates.

Pure numpy.  MoveIt still performs the authoritative collision check; this
module only answers "which approach directions are already blocked by an
obstacle other than the target bottle", which makes failures explainable and
lets the task skip hopeless candidates before planning them.

Two independent sources are supported:

* ``filter_by_corridor`` - the semantic obstacle boxes of ``scan_mode``
  ``semantic``/``both``;
* ``filter_by_points`` - the downsampled map point cloud that feeds the
  OctoMap in ``scan_mode`` ``map`` (the default).  Without it the corridor
  check has nothing to look at and every direction is planned by MoveIt,
  which then burns ``planning_time`` per direction and only reports the
  opaque ``Fail: ABORTED: TIMED_OUT``.
"""

import numpy as np


def distance_to_box(point, pose, dims):
    """Signed distance from a point to an oriented box (negative = inside)."""
    point = np.asarray(point, dtype=float)
    pose = np.asarray(pose, dtype=float)
    half = 0.5 * np.asarray(dims, dtype=float)
    local = pose[:3, :3].T @ (point - pose[:3, 3])
    outside = np.abs(local) - half
    if np.all(outside <= 0.0):
        return float(np.max(outside))
    return float(np.linalg.norm(np.maximum(outside, 0.0)))


def distance_to_cylinder(point, pose, dims):
    """Signed distance from a point to a vertical cylinder."""
    point = np.asarray(point, dtype=float)
    pose = np.asarray(pose, dtype=float)
    radius, height = float(dims[0]), float(dims[1])
    delta = point - pose[:3, 3]
    radial = float(np.hypot(delta[0], delta[1])) - radius
    vertical = abs(float(delta[2])) - height / 2.0
    if radial > 0.0 and vertical > 0.0:
        return float(np.hypot(radial, vertical))
    if radial > 0.0:
        return radial
    if vertical > 0.0:
        return vertical
    return float(max(radial, vertical))


def distance_to_spec(point, spec):
    if spec["kind"] == "cylinder":
        return distance_to_cylinder(point, spec["pose"], spec["dims"])
    return distance_to_box(point, spec["pose"], spec["dims"])


def corridor_clearance(start, end, specs, samples=24):
    """Return the minimum clearance along the segment and the blocking id.

    ``specs`` must not contain the target bottle: the fingers necessarily come
    closer to it than any corridor radius.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("走廊端点必须是base_link三维点")
    count = max(int(samples), 2)
    fractions = np.linspace(0.0, 1.0, count)
    points = start[None, :] + fractions[:, None] * (end - start)[None, :]
    result = {"clearance": float("inf"), "blocking": None, "by_spec": {}}
    for spec in specs:
        if spec["kind"] == "cylinder":
            distances = [distance_to_cylinder(point, spec["pose"], spec["dims"])
                         for point in points]
        else:
            distances = [distance_to_box(point, spec["pose"], spec["dims"])
                         for point in points]
        minimum = float(min(distances))
        result["by_spec"][spec["id"]] = minimum
        if minimum < result["clearance"]:
            result["clearance"] = minimum
            result["blocking"] = spec["id"]
    return result


def filter_by_corridor(candidates, specs, radius, samples=24):
    """Split candidates into accepted ones and diagnostics.

    ``candidates`` is a list of dicts with ``label`` and the segment endpoints
    ``start`` / ``end``.  Candidates keep their original order.
    """
    accepted, rejected = [], []
    for candidate in candidates:
        check = corridor_clearance(
            candidate["start"], candidate["end"], specs, samples)
        candidate["clearance"] = check["clearance"]
        candidate["blocking"] = check["blocking"]
        if check["clearance"] >= float(radius):
            accepted.append(candidate)
        else:
            rejected.append(candidate)
    return accepted, rejected


def point_clearance(start, end, points, inflation=0.0):
    """Return ``(clearance, blocking_point)`` for a point set vs a segment.

    Every point is treated as a sphere of ``inflation`` radius, so callers can
    compare the result against the same radius used for semantic boxes and
    stay consistent with the padded voxels MoveIt gets from the OctoMap.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    points = np.asarray(points, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("走廊端点必须是base_link三维点")
    if points.size == 0:
        return float("inf"), None
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("地图点云必须是Nx3数组")
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared <= 1e-12:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        fractions = np.clip(
            (points - start) @ segment / length_squared, 0.0, 1.0)
        projections = start + fractions[:, None] * segment
        distances = np.linalg.norm(points - projections, axis=1)
    distances = distances - float(inflation)
    index = int(np.argmin(distances))
    return float(distances[index]), points[index]


def filter_by_points(candidates, points, radius, inflation=0.0,
                     forward_margin=0.0):
    """Split candidates by the clearance to the map point cloud.

    ``forward_margin`` extends the segment past its end: the fingers stick out
    further than ``link6_tool_xyz``, so a point just behind the grasp point can
    still be hit during the approach.
    """
    accepted, rejected = [], []
    for candidate in candidates:
        start = np.asarray(candidate["start"], dtype=float)
        end = np.asarray(candidate["end"], dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length > 1e-9 and forward_margin:
            end = end + direction / length * float(forward_margin)
        clearance, point = point_clearance(
            start, end, points, inflation)
        candidate["map_clearance"] = clearance
        candidate["map_blocking_point"] = point
        if clearance >= float(radius):
            accepted.append(candidate)
        else:
            rejected.append(candidate)
    return accepted, rejected


def sort_by_clearance(candidates):
    """Return candidates ordered from the most free approach to the tightest.

    ``filter_by_corridor``/``filter_by_points`` store the clearance on each
    candidate; ``inf`` means "nothing in the corridor" and keeps the original
    relative order thanks to the stable sort.
    """
    def key(candidate):
        return -float(candidate.get("map_clearance",
                                    candidate.get("clearance", 0.0)))
    return sorted(candidates, key=key)

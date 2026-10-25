#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Free-space corridor check for grasp approach candidates.

Pure numpy.  MoveIt still performs the authoritative collision check; this
module only answers "which approach directions are already blocked by an
obstacle other than the target bottle", which makes failures explainable and
lets the task skip hopeless candidates before planning them.
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

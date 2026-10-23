#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure vision helpers.

These functions only need cv2/numpy (plus an optional pyrealsense2 import
inside ``deproject``) and never touch a RealSense pipeline object, so they
can be unit-tested offline.
"""

import cv2
import numpy as np

# HSV color definitions for the three supported bottle/card colors.
COLOR_DEFS = {
    "bottle-y": {  # 橘色
        "hsv_lower": np.array([5, 80, 80]),
        "hsv_upper": np.array([25, 255, 255]),
        "bgr": (0, 140, 255),
    },
    "bottle-g": {  # 绿色
        "hsv_lower": np.array([35, 50, 50]),
        "hsv_upper": np.array([85, 255, 255]),
        "bgr": (0, 200, 0),
    },
    "bottle-b": {  # 黑色
        "hsv_lower": np.array([0, 0, 0]),
        "hsv_upper": np.array([180, 255, 60]),
        "bgr": (80, 80, 80),
    },
}

# Depth ROI / filtering defaults.
DEPTH_MIN = 0.1
DEPTH_MAX = 8.0
SHRINK_RATIO = 0.20
DEPTH_PERCENTILE = 20
COLOR_RATIO_THRES = 0.15


def classify_bottle_color(color_image, x1, y1, x2, y2):
    """根据检测框内 HSV 颜色占比，返回 bottle-y/bottle-g/bottle-b 或 None。"""
    h, w = color_image.shape[:2]
    rx1 = max(int(x1), 0)
    ry1 = max(int(y1), 0)
    rx2 = min(int(x2), w - 1)
    ry2 = min(int(y2), h - 1)

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = color_image[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return None

    priority_order = ["bottle-y", "bottle-g", "bottle-b"]
    claimed = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=bool)
    scores = {}

    for label in priority_order:
        cfg = COLOR_DEFS[label]
        mask = cv2.inRange(hsv, cfg["hsv_lower"], cfg["hsv_upper"])
        exclusive = (mask > 0) & (~claimed)
        scores[label] = np.count_nonzero(exclusive) / total
        claimed |= exclusive

    best_label = max(scores, key=scores.get)
    if scores[best_label] >= COLOR_RATIO_THRES:
        return best_label
    return None


def get_depth_percentile(depth_frame, raw_depth_frame, x1, y1, x2, y2, depth_scale):
    """在检测框中心区域取深度百分位，避免直接取中心点受空洞/反光影响。"""
    bw = x2 - x1
    bh = y2 - y1
    sx = int(bw * SHRINK_RATIO)
    sy = int(bh * SHRINK_RATIO)

    rx1 = max(x1 + sx, 0)
    ry1 = max(y1 + sy, 0)
    rx2 = min(x2 - sx, depth_frame.get_width() - 1)
    ry2 = min(y2 - sy, depth_frame.get_height() - 1)

    cx = int((x1 + x2) // 2)
    cy = int((y1 + y2) // 2)

    if rx2 <= rx1 or ry2 <= ry1:
        return depth_frame.get_distance(cx, cy)

    depth_image = np.asanyarray(raw_depth_frame.get_data())
    roi = depth_image[ry1:ry2, rx1:rx2].astype(np.float32) * depth_scale
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]

    if len(valid) == 0:
        return depth_frame.get_distance(cx, cy)

    return float(np.percentile(valid, DEPTH_PERCENTILE))


def deproject(intrinsics, px, py, depth_m):
    """De-project a pixel to a point in the camera optical frame."""
    # Imported lazily so this module can be imported without the SDK.
    import pyrealsense2 as rs
    return rs.rs2_deproject_pixel_to_point(
        intrinsics, [float(px), float(py)], float(depth_m))


def draw_detection(img, x1, y1, x2, y2, color_label, conf, dist_m, xyz):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bgr = COLOR_DEFS[color_label]["bgr"]

    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(img, f"{color_label} {conf:.2f} | {dist_m:.3f}m",
                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    if xyz is not None:
        X, Y, Z = xyz
        cv2.putText(img, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}",
                    (x1, min(y2 + 20, img.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def draw_rejected(img, x1, y1, x2, y2, reason):
    """Draw a rejected candidate so the debug window shows every YOLO box."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)
    cv2.putText(img, reason, (x1, max(int(y1) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

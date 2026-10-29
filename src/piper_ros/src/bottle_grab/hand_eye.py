#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eye-in-hand calibration and pixel-to-base_link geometry.

Loads the calibration JSON produced by the eye-in-hand calibration tool and
converts observed pixel/depth measurements into base_link target poses.
"""

import json
import math
from pathlib import Path

import numpy as np

from pose_utils import build_level_grasp_rotation, normalize_vector, validate_transform
from vision_utils import deproject


class HandEyeCalibration:
    """link6 -> color-optical transform and pixel/depth -> base_link helpers."""

    def __init__(self, calibration_path):
        calibration = json.loads(Path(calibration_path).expanduser().read_text())
        if (calibration.get("mode") != "eye_in_hand"
                or calibration.get("tf_parent") != "link6"):
            raise ValueError("手眼标定必须为eye_in_hand且安装父link为link6")
        self.T_link6_color_optical = np.asarray(calibration["ee_T_camera"], dtype=float)
        validate_transform(self.T_link6_color_optical, "手眼外参")
        self.frames = dict(calibration.get("frames") or {})
        link = calibration.get("camera_link_T_optical")
        self.T_camera_link_color_optical = (
            None if link is None else np.asarray(link, dtype=float))
        if self.T_camera_link_color_optical is not None:
            validate_transform(
                self.T_camera_link_color_optical, "camera_link->optical外参")

    def static_transforms(self):
        """Return ``[(parent, child, 4x4)]`` for link6->camera_link->optical.

        MoveIt's ``PointCloudOctomapUpdater`` uses the *point-cloud header
        frame* as the sensor frame, so the static octomap must receive clouds
        in the camera optical frame and needs this chain in TF.  Values come
        from the same calibration JSON the point cloud is built with, so the
        ray-casting origin and the cloud agree.
        """
        if self.T_camera_link_color_optical is None:
            raise RuntimeError(
                "标定文件缺少camera_link_T_optical，无法发布相机TF")
        ee = self.frames.get("ee", "link6")
        camera_link = self.frames.get("camera_link", "camera_link")
        optical = self.frames.get(
            "camera_optical", "camera_color_optical_frame")
        link6_T_camera_link = (
            self.T_link6_color_optical
            @ np.linalg.inv(self.T_camera_link_color_optical))
        return [
            (ee, camera_link, link6_T_camera_link),
            (camera_link, optical, self.T_camera_link_color_optical),
        ]

    def make_bottle_pose_in_base(self, bottle_xyz_cam, current_ee_pose_mat):
        """Convert an observed color-optical point to base_link.

        This is retained for diagnostics and for the fallback path.  The
        measured depth point is generally on the visible bottle surface, not
        on the bottle axis.  Grasp planning should therefore prefer
        make_bottle_center_pose_in_base().
        """
        T_cam_point = np.eye(4)
        T_cam_point[:3, 3] = np.asarray(bottle_xyz_cam, dtype=float)
        T_base_point = (
            current_ee_pose_mat @ self.T_link6_color_optical @ T_cam_point)
        result = np.eye(4)
        result[:3, 3] = T_base_point[:3, 3]
        result[:3, :3] = build_level_grasp_rotation(
            current_ee_pose_mat[:3, :3])
        return result

    def pixel_to_base(self, pixel, depth, current_ee_pose_mat, intrinsics):
        """De-project a pixel + depth measurement and convert it to base_link."""
        xyz_cam = deproject(intrinsics, pixel[0], pixel[1], depth)
        return self.make_bottle_pose_in_base(xyz_cam, current_ee_pose_mat)

    def make_bottle_center_pose_in_base(self, pixel, intrinsics,
                                        current_ee_pose_mat, table_model,
                                        center_height=None, bottle_height=None):
        """Estimate an object center from a pixel and the scanned table.

        The YOLO/depth point is the camera-facing surface.  For a standing
        bottle, the center lies on the ray through the detection pixel and at
        bottle_height/2 above the fitted table plane.  For a flat card the
        same ray is intersected with the table plane itself
        (``center_height=0``).  Solving that intersection removes the
        systematic ``front-surface`` offset without changing the
        link6-to-tool/TCP transform.
        """
        if table_model is None:
            raise RuntimeError("尚未加载桌面模型，无法反算物体中心")
        if center_height is None:
            if bottle_height is None:
                raise RuntimeError("尚未提供瓶高，无法反算物体中心")
            center_height = bottle_height / 2.0
        elif (not math.isfinite(center_height) or center_height < 0.0):
            raise RuntimeError("物体中心离桌面的高度无效")
        px, py = pixel
        ray_cam = np.asarray(deproject(intrinsics, px, py, 1.0), dtype=float)
        ray_norm = np.linalg.norm(ray_cam)
        if ray_cam.shape != (3,) or not np.isfinite(ray_cam).all() or ray_norm < 1e-9:
            raise RuntimeError("瓶子像素无法生成有效相机射线")
        ray_cam /= ray_norm

        T_base_camera = current_ee_pose_mat @ self.T_link6_color_optical
        origin = T_base_camera[:3, 3]
        direction = T_base_camera[:3, :3] @ ray_cam
        normal = np.asarray(table_model["plane_normal"], dtype=float)
        offset = float(table_model["plane_offset"])
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or not math.isfinite(offset) or abs(normal[2]) < 0.98):
            raise RuntimeError("桌面模型不适合反算瓶子中心")

        # table plane: n dot p + offset = 0.  The object center is
        # center_height above the table along base Z, hence
        # n dot center + offset = n_z * center_height.
        denominator = float(np.dot(normal, direction))
        numerator = (normal[2] * center_height
                     - (float(np.dot(normal, origin)) + offset))
        if abs(denominator) < 1e-6:
            raise RuntimeError("检测射线几乎平行于桌面")
        distance = numerator / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("物体中心射线与桌面高度的交点无效")
        center = origin + distance * direction
        if not np.isfinite(center).all():
            raise RuntimeError("反算得到的物体中心包含非有限值")

        result = np.eye(4)
        result[:3, 3] = center
        result[:3, :3] = build_level_grasp_rotation(
            current_ee_pose_mat[:3, :3])
        return result

    @staticmethod
    def held_bottle_contains(point_base, current_ee_pose_mat, T_link6_bottle,
                             bottle_height, bottle_radius, margin=0.02):
        """Return True when a base_link point lies inside the held bottle.

        The bottle in the gripper is a known cylinder (``T_link6_bottle``),
        so this filter does not depend on the table height or on the target
        object's height.  A plain Z threshold cannot be used during placement:
        a card standing on the table is several centimetres above the table
        plane, exactly like the gripper bottle.
        """
        if T_link6_bottle is None:
            return False
        bottle = current_ee_pose_mat @ T_link6_bottle
        center = bottle[:3, 3]
        axis = normalize_vector(bottle[:3, 2])
        if axis is None:
            return False
        offset = np.asarray(point_base, dtype=float) - center
        along = float(np.clip(np.dot(offset, axis),
                              -bottle_height / 2.0,
                              bottle_height / 2.0))
        radial = np.linalg.norm(offset - along * axis)
        return radial <= bottle_radius + margin

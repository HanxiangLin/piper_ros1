#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO bottle/card detector and multi-frame pose acquisition loop."""

import os

import cv2
import numpy as np
import rospy

from camera import apply_filters
from joint_feedback import ARM_JOINTS
from vision_utils import (
    DEPTH_MAX,
    DEPTH_MIN,
    classify_bottle_color,
    deproject,
    draw_detection,
    draw_rejected,
    get_depth_percentile,
)

IMG_SIZE = 640
CONF_THRES = 0.5


class BottleDetector:
    """Loads YOLO and turns color/depth frames into detection candidates."""

    def __init__(self, model_path, imgsz=IMG_SIZE, conf_thres=CONF_THRES):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.model = None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "当前Python环境缺少YOLO依赖: " + str(error)) from error
        if not os.path.isfile(self.model_path):
            raise ValueError("YOLO模型不存在: " + self.model_path)
        rospy.loginfo("正在加载YOLO瓶子检测模型: %s", self.model_path)
        self.model = YOLO(self.model_path)
        rospy.loginfo("YOLO模型加载完成")

    def detect(self, color_image, depth_frame, raw_depth_frame, depth_scale,
               conf_thres=None):
        """Run YOLO on one aligned frame pair and return candidate dicts.

        Each candidate contains the pixel box, color label (possibly None),
        confidence, box center and a depth percentile measurement.  Callers
        decide target-label filtering, deprojection and coordinate frames.
        """
        results = self.model.predict(
            source=color_image, imgsz=self.imgsz,
            conf=self.conf_thres if conf_thres is None else conf_thres,
            verbose=False)

        candidates = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            height, width = color_image.shape[:2]
            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = box.tolist()
                x1, x2 = np.clip([x1, x2], 0, width - 1)
                y1, y2 = np.clip([y1, y2], 0, height - 1)
                label = classify_bottle_color(
                    color_image, x1, y1, x2, y2)
                cx = int((x1 + x2) // 2)
                cy = int((y1 + y2) // 2)
                distance = get_depth_percentile(
                    depth_frame, raw_depth_frame, x1, y1, x2, y2,
                    depth_scale)
                candidates.append({
                    "box": (x1, y1, x2, y2),
                    "label": label,
                    "conf": float(confidence),
                    "center": (cx, cy),
                    "dist_m": float(distance),
                })
        return candidates

    def get_pose(self, camera, arm_controller, joint_feedback, state,
                 hand_eye, table_model, bottle_height, bottle_radius,
                 estimate_bottle_center_from_table, show_detection_window,
                 target_label=None, max_frames=60, center_height=None,
                 description="目标瓶子", use_surface_point=False,
                 conf_thres=None):
        """Detect one bottle/card and return its observed pose in base_link.

        ``center_height`` overrides the assumed height of the detected
        object's center above the table.  ``None`` keeps the standing-bottle
        assumption.  ``use_surface_point`` returns the measured depth point
        itself instead of intersecting the pixel ray with the table plane;
        use it for objects that stand above the table (for example a card
        leaning on the table), where the table-ray center is tens of
        centimetres off.
        """
        current_ee_pose_mat = arm_controller.current_ee_matrix()
        state.current_ee_pose_mat = current_ee_pose_mat
        reference_joints = joint_feedback._fresh_joint_positions()
        # 丢弃缓冲中的旧画面；SDK时间域不同，不将其毫秒时间戳冒充ROS时间。
        for _ in range(5):
            camera.frames(timeout_ms=5000)
        label_text = target_label if target_label is not None else "任意颜色目标"
        rospy.loginfo("正在检测%s: %s", description, label_text)
        best_candidate = None
        best_score = -1.0
        last_annotated = None

        for frame_index in range(max_frames):
            if rospy.is_shutdown():
                break
            try:
                frames = camera.frames(timeout_ms=5000)
            except RuntimeError as error:
                rospy.logwarn("RealSense等待图像超时: %s", error)
                continue
            measured = joint_feedback._fresh_joint_positions()
            if any(abs(measured[j] - reference_joints[j]) > 0.01
                   for j in ARM_JOINTS):
                raise RuntimeError("视觉采集期间机械臂发生移动；停止并重新定位")
            aligned = camera.align.process(frames)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                continue
            filtered_depth = apply_filters(
                raw_depth, camera.filters).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            annotated = color_image.copy()
            intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics
            candidates = self.detect(
                color_image, filtered_depth, raw_depth, camera.depth_scale,
                conf_thres)

            height, width = color_image.shape[:2]
            for candidate in candidates:
                x1, y1, x2, y2 = candidate["box"]
                color_label = candidate["label"]
                confidence = candidate["conf"]
                if color_label is None:
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "color? %.2f" % confidence)
                    continue
                if target_label is not None and color_label != target_label:
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "label=%s" % color_label)
                    continue
                cx, cy = candidate["center"]
                distance = candidate["dist_m"]
                if not DEPTH_MIN < distance < DEPTH_MAX:
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "depth=%.3f" % distance)
                    continue
                xyz_cam = deproject(intrinsics, cx, cy, distance)
                surface_pose = hand_eye.make_bottle_pose_in_base(
                    xyz_cam, current_ee_pose_mat)
                if use_surface_point:
                    # 立着/斜靠的物体直接使用深度表面点，不能和桌面求交。
                    T_base_bottle = surface_pose
                elif (estimate_bottle_center_from_table
                        or center_height is not None):
                    # 用像素射线+桌面求物体中心，而不是把深度表面点
                    # 的base_link X/Y直接当作物体中心。
                    T_base_bottle = hand_eye.make_bottle_center_pose_in_base(
                        (cx, cy), intrinsics, current_ee_pose_mat,
                        table_model, center_height, bottle_height)
                else:
                    T_base_bottle = surface_pose
                if hand_eye.held_bottle_contains(
                        surface_pose[2, 3], current_ee_pose_mat,
                        state.T_link6_bottle, bottle_height, bottle_radius):
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "held bottle")
                    rospy.loginfo_throttle(
                        2.0,
                        "忽略夹爪中的瓶子候选: %s pixel=%s depth=%.3fm",
                        color_label, (cx, cy), distance)
                    continue
                draw_detection(
                    annotated, x1, y1, x2, y2, color_label,
                    confidence, distance, xyz_cam)
                center_penalty = (
                    abs(cx - width / 2.0) / width
                    + abs(cy - height / 2.0) / height)
                score = confidence - 0.10 * center_penalty
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "label": color_label,
                        "conf": confidence,
                        "center": (cx, cy),
                        "dist_m": distance,
                        "xyz_cam": np.asarray(xyz_cam, dtype=float),
                        "surface_base": surface_pose[:3, 3].copy(),
                        "T_base_bottle": T_base_bottle,
                    }

            last_annotated = annotated
            if show_detection_window:
                try:
                    cv2.imshow("Bottle Detection For Grasp", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
            if best_candidate is not None and frame_index >= 10:
                candidate = best_candidate
                position = candidate["T_base_bottle"][:3, 3]
                surface = candidate["surface_base"]
                correction = position - surface
                rospy.loginfo(
                    "检测成功: %s conf=%.2f pixel=%s depth=%.4fm "
                    "surface_base=(%.4f, %.4f, %.4f) "
                    "target_center_base=(%.4f, %.4f, %.4f) "
                    "correction=(%.4f, %.4f, %.4f)%s",
                    candidate["label"], candidate["conf"], candidate["center"],
                    candidate["dist_m"], surface[0], surface[1], surface[2],
                    position[0], position[1], position[2],
                    correction[0], correction[1], correction[2],
                    " (depth surface)" if use_surface_point
                    else (" (table-ray center)"
                          if (estimate_bottle_center_from_table
                              or center_height is not None)
                          else " (depth surface fallback)"))
                return candidate["T_base_bottle"]

        if last_annotated is not None and show_detection_window:
            try:
                cv2.imshow("Bottle Detection For Grasp", last_annotated)
                cv2.waitKey(1)
            except cv2.error:
                pass
        rospy.logwarn("超时：未检测到%s: %s", description, label_text)
        return None

    def detect_scene(self, camera, arm_controller, joint_feedback, state,
                     hand_eye, table_model, bottle_height, bottle_radius,
                     show_detection_window, target_label=None, max_frames=20,
                     conf_thres=None):
        """Return every YOLO candidate at the current stationary pose.

        Unlike :meth:`get_pose` this keeps *all* objects instead of the single
        best one, because the scan needs to know about the printed card and any
        other obstacle as well as the target bottle.  Candidates are
        de-duplicated across frames by their base_link surface point.
        """
        current_ee_pose_mat = arm_controller.current_ee_matrix()
        state.current_ee_pose_mat = current_ee_pose_mat
        reference_joints = joint_feedback._fresh_joint_positions()
        for _ in range(5):
            camera.frames(timeout_ms=5000)

        found = {}
        last_annotated = None
        for frame_index in range(max_frames):
            if rospy.is_shutdown():
                break
            try:
                frames = camera.frames(timeout_ms=5000)
            except RuntimeError as error:
                rospy.logwarn("RealSense等待图像超时: %s", error)
                continue
            measured = joint_feedback._fresh_joint_positions()
            if any(abs(measured[name] - reference_joints[name]) > 0.01
                   for name in ARM_JOINTS):
                raise RuntimeError("视觉采集期间机械臂发生移动；停止并重新扫描")
            aligned = camera.align.process(frames)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                continue
            filtered_depth = apply_filters(
                raw_depth, camera.filters).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            annotated = color_image.copy()
            intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics
            candidates = self.detect(
                color_image, filtered_depth, raw_depth, camera.depth_scale,
                conf_thres)
            for candidate in candidates:
                x1, y1, x2, y2 = candidate["box"]
                label = candidate["label"]
                confidence = candidate["conf"]
                if label is None:
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "color? %.2f" % confidence)
                    continue
                if target_label is not None and label != target_label:
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "label=%s" % label)
                    continue
                cx, cy = candidate["center"]
                distance = candidate["dist_m"]
                if not DEPTH_MIN < distance < DEPTH_MAX:
                    draw_rejected(annotated, x1, y1, x2, y2,
                                  "depth=%.3f" % distance)
                    continue
                xyz_cam = deproject(intrinsics, cx, cy, distance)
                surface_pose = hand_eye.make_bottle_pose_in_base(
                    xyz_cam, current_ee_pose_mat)
                if hand_eye.held_bottle_contains(
                        surface_pose[2, 3], current_ee_pose_mat,
                        state.T_link6_bottle, bottle_height, bottle_radius):
                    continue
                try:
                    center_pose = hand_eye.make_bottle_center_pose_in_base(
                        (cx, cy), intrinsics, current_ee_pose_mat, table_model,
                        bottle_height=bottle_height)
                except RuntimeError:
                    center_pose = None
                draw_detection(annotated, x1, y1, x2, y2, label, confidence,
                               distance, xyz_cam)
                key = (label, round(float(surface_pose[0, 3]), 2),
                       round(float(surface_pose[1, 3]), 2))
                previous = found.get(key)
                if previous is None or confidence > previous["conf"]:
                    found[key] = {
                        "label": label,
                        "conf": float(confidence),
                        "box": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "dist_m": float(distance),
                        "surface_base": surface_pose[:3, 3].copy(),
                        "center_base": (None if center_pose is None
                                        else center_pose[:3, 3].copy()),
                    }
            last_annotated = annotated
            if show_detection_window:
                try:
                    cv2.imshow("Bottle Detection For Scan", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
            if frame_index >= 4 and found:
                break

        if last_annotated is not None and show_detection_window:
            try:
                cv2.imshow("Bottle Detection For Scan", last_annotated)
                cv2.waitKey(1)
            except cv2.error:
                pass
        result = sorted(found.values(), key=lambda item: -item["conf"])
        rospy.loginfo("扫描检测到 %d 个物体: %s", len(result), ", ".join(
            "%s(%.2f)" % (item["label"], item["conf"]) for item in result))
        return result

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live scene scan: move the camera over the workspace, freeze one snapshot.

Two stages, because the first motion has no map to protect it:

* stage A -- from the configured start pose (no motion if the arm is already
  there) capture several frames, fit the table, detect every object and build
  the first obstacle map.  The table and the obstacles are written into the
  planning scene before any lateral motion;
* stage B -- swing the wrist/shoulder by the configured offsets and capture
  again.  These motions are planned against the stage-A map, so the swing
  cannot hit an obstacle the first view already saw.

In ``scan_mode=map`` (default) the collision scene is:

* one *exact* support box (``table``, top surface = fitted plane),
* one semantic cylinder for the target bottle (fingers may touch it),
* the static OctoMap built from every scan view, which contains **everything
  else** the camera saw -- desk, walls, cabinets, other bottles.

The final snapshot is frozen into ``SceneManager``; afterwards every motion
verifies that the scene did not change.
"""

import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

import map_filter
import obstacle_model
from camera import apply_filters
from cloud_fusion import FusedCloud, crop_points, depth_points, transform_points
from grasp_corridor import distance_to_spec
from pose_utils import build_level_grasp_rotation
from table_fit import fit_table, surface_z

SCAN_MODES = ("map", "semantic", "both")
MAP_MODES = ("map", "both")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError("无法序列化的类型: " + type(value).__name__)


class ScanTask:
    """Owns the live scan and freezes the resulting planner scene."""

    def __init__(self, config, state, arm, gripper, scene, joint_feedback,
                 camera, hand_eye, detector):
        self.config = config
        self.state = state
        self.arm = arm
        self.gripper = gripper
        self.scene = scene
        self.joint_feedback = joint_feedback
        self.camera = camera
        self.hand_eye = hand_eye
        self.detector = detector
        if config.scan_mode not in SCAN_MODES:
            raise ValueError("scan_mode必须是map/semantic/both")
        self.map_mode = config.scan_mode in MAP_MODES
        self.semantic_obstacles = config.scan_mode in ("semantic", "both")
        self._cloud_pub = None
        if self.map_mode:
            self._cloud_pub = rospy.Publisher(
                config.scan_cloud_topic, PointCloud2, queue_size=1)
        self._map_views = []

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------
    def _view_offsets(self):
        """Return ``[(joint1_delta, joint2_delta)]`` with the start pose first."""
        views = [(0.0, 0.0)]
        for joint2 in self.config.scan_joint2_offsets:
            for joint1 in self.config.scan_joint1_offsets:
                if abs(float(joint1)) < 1e-9 and abs(float(joint2)) < 1e-9:
                    continue
                views.append((float(joint1), float(joint2)))
        return views

    def execute(self):
        rospy.loginfo(">>> 扫描：清场并移动到扫描起始位")
        self.joint_feedback.wait_for_joint_feedback()
        self.scene.reset_for_scan()
        self._map_views = []
        rospy.loginfo(">>> 张开夹爪")
        self.gripper.command(
            self.config.opening_open, wait_for_target=True, guarded=False)
        start = list(self.config.scan_start_joint_target)
        self.arm.move_joints(start, guarded=False)

        fused = FusedCloud(self.config.scan_voxel_m,
                           self.config.scan_min_frame_votes)
        views = self._view_offsets()
        detections = None
        target_center = None
        card = None
        summary = []
        images = []
        points = np.zeros((0, 3))
        observations = np.zeros(0, dtype=int)
        table = None

        for index, (offset1, offset2) in enumerate(views):
            joints = list(start)
            joints[0] += offset1
            joints[1] += offset2
            if index:
                rospy.loginfo(
                    ">>> 扫描视点%d: joint1偏移%+.3f joint2偏移%+.3f",
                    index, offset1, offset2)
                self.arm.move_joints(joints, guarded=False)
            frames, image, transform, stamp = self._capture(index)
            images.append(image)
            fused.add_view(frames)
            points, observations = fused.points(self.config.scan_min_views)
            if len(points) < self.config.scan_min_points:
                raise RuntimeError(
                    "视点%d融合点云过少(%d)，检查相机与ROI"
                    % (index, len(points)))

            focus = target_center
            if index == 0:
                # 没有可信桌面时也要先检测：YOLO 的表面点足以给桌面拟合做种子，
                # 否则大ROI里的地板/邻桌会赢掉RANSAC。
                detections = self._detect(None)
                focus = self._detection_focus(detections)
            previous_table = table
            table = self._fit_table(points, index, focus, previous_table)
            self.scene.set_table(table)

            if index == 0 or self.semantic_obstacles:
                built, lines = obstacle_model.build(
                    points, table, self.config,
                    target_center=target_center,
                    detections=detections if index == 0 else None,
                    locked_card=card)
                new_center = built["target"]["center"]
                new_card = built["card"]
                if self.map_mode:
                    # 语义障碍盒不加入场景，摘要里也不能再声称它们存在。
                    lines = [line for line in lines
                             if not line.startswith(
                                 ("障碍物", "已排除卡片支架"))]
                    lines.append("障碍物：扫描到的一切进入静态OctoMap（无语义盒）")
            else:
                # 地图模式后续视点只需要细化目标中心；不再重新聚类整个房间。
                new_center = self._refine_target(points, table, target_center)
                new_card = card
                lines = []

            if target_center is None:
                self.scene.add_bottle(
                    new_center, self.config.bottle_height,
                    self.config.bottle_radius, self.config.lift_distance,
                    self.config.lift_table_margin, self.state)
                rospy.loginfo(
                    ">>> 目标瓶已加入场景: %s",
                    lines[1] if len(lines) > 1 else " | ".join(lines))
            else:
                if float(np.linalg.norm(new_center - target_center)) > 0.05:
                    raise RuntimeError(
                        "多视点融合后目标瓶位置跳变%.3fm，重新扫描"
                        % float(np.linalg.norm(new_center - target_center)))
                if float(np.linalg.norm(new_center - target_center)) > 0.005:
                    self.scene.update_bottle(
                        new_center, self.config.bottle_height,
                        self.config.bottle_radius, self.config.lift_distance,
                        self.config.lift_table_margin, self.state)
            if lines:
                summary = lines
            target_center = new_center

            if index == 0:
                card = new_card
            elif new_card is not None and card is None:
                card = new_card

            if self.semantic_obstacles:
                self.scene.set_obstacles(built["obstacles"])
            if self.map_mode:
                self._publish_map_view(
                    fused.last_view_points(), transform, stamp, table,
                    target_center, index)

        if table is None:
            raise RuntimeError("扫描没有产生桌面模型")
        self._require_final_table_tilt(table)
        if self.config.enable_place and self.config.place_mode == "card" \
                and card is None:
            raise RuntimeError(
                "扫描未识别出卡片，无法执行放置；检查卡片是否被遮挡或换视点")
        if card is not None and self.config.enable_place \
                and self.config.place_mode == "card":
            self._check_card_clearance(card)
        if self.map_mode:
            self._verify_map()

        rotation = build_level_grasp_rotation(self.arm.current_ee_matrix()[:3, :3])
        target_pose = np.eye(4)
        target_pose[:3, :3] = rotation
        target_pose[:3, 3] = target_center

        result = {
            "table": table,
            "target_center": np.asarray(target_center, dtype=float),
            "target_pose": target_pose,
            "card": card,
            "obstacles": list(self.scene.obstacles),
            "detections": detections or [],
            "points": int(len(points)),
            "views": int(fused.view_count),
            "frames": int(fused.frame_count),
            "stamp": time.time(),
        }
        self.state.scan_result = result
        self.state.T_base_bottle = target_pose
        self.state.card_model = card
        self.scene.freeze(result)
        for line in summary:
            rospy.loginfo("扫描结果: %s", line)
        if self.map_mode:
            rospy.loginfo(
                "扫描结果: 障碍地图 %d个视点快照，已冻结为静态OctoMap",
                len(self._map_views))
        self._save(result, points, observations, images, summary)
        rospy.loginfo(
            ">>> 扫描完成: %d个视点 %d帧 %d个融合点 %d个语义障碍",
            result["views"], result["frames"], result["points"],
            len(result["obstacles"]))
        return result

    # ------------------------------------------------------------------
    # Capture.
    # ------------------------------------------------------------------
    def _capture(self, index):
        rospy.sleep(self.config.scan_settle_time)
        reference = self.joint_feedback._fresh_joint_positions()
        frames = []
        color_image = None
        for _ in range(self.config.scan_frames_per_view):
            frameset = self.camera.frames(timeout_ms=5000)
            aligned = self.camera.align.process(frameset)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                continue
            filtered = self.camera.filters
            depth = apply_filters(raw_depth, filtered).as_depth_frame()
            intrinsics = depth.profile.as_video_stream_profile().intrinsics
            points = depth_points(
                np.asanyarray(depth.get_data()), self.camera.depth_scale,
                intrinsics, stride=self.config.scan_stride,
                depth_min=self.config.scan_min_depth,
                depth_max=self.config.scan_max_range)
            frames.append(points)
            color_image = np.asanyarray(color_frame.get_data()).copy()
            rospy.sleep(self.config.scan_frame_interval)
        measured = self.joint_feedback._fresh_joint_positions()
        drift = max(abs(measured[name] - reference[name])
                    for name in reference)
        if drift > 0.01:
            raise RuntimeError(
                "视点%d采集期间机械臂移动%.4frad，重新扫描" % (index, drift))
        if not frames:
            raise RuntimeError("视点%d没有有效深度帧" % index)

        transform = self.arm.current_ee_matrix() @ self.hand_eye.T_link6_color_optical
        base_frames = []
        for points in frames:
            base = transform_points(points, transform)
            base = crop_points(base, self.config.scan_roi_min,
                               self.config.scan_roi_max)
            if len(base):
                base_frames.append(base)
        rospy.loginfo("视点%d采集%d帧，ROI内最多%d点",
                      index, len(base_frames),
                      max([len(item) for item in base_frames] or [0]))
        # 点云帧与相机位姿必须对应同一时刻；留50ms给robot_state_publisher。
        stamp = rospy.Time.now() - rospy.Duration(0.05)
        if stamp.to_sec() <= 0.0:
            stamp = rospy.Time.now()
        return base_frames, color_image, transform, stamp

    # ------------------------------------------------------------------
    # Detection / fitting / publishing.
    # ------------------------------------------------------------------
    def _detect(self, table):
        rospy.loginfo(">>> 检测场景中的全部物体")
        return self.detector.detect_scene(
            self.camera, self.arm, self.joint_feedback, self.state,
            self.hand_eye, table, self.config.bottle_height,
            self.config.bottle_radius, self.config.show_detection_window,
            max_frames=self.config.scan_detect_frames)

    @staticmethod
    def _detection_focus(detections):
        """Return the surface point of the most confident detection."""
        candidates = [item for item in (detections or [])
                      if item.get("surface_base") is not None]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: float(item.get("conf") or 0.0))
        return np.asarray(best["surface_base"], dtype=float)

    def _table_tilt_limit(self, index):
        """返回该视点/最终视点的桌面倾角门限。

        早期视点的单视角桌面常常比最终融合桌面更斜（局部纸板起伏/卡片边缘
        污染）；它只用于初始检测和障碍建模，真正冻结给抓取用的是最后
        一个视点的融合桌面。因此中间视点放宽 2°，最后一个视点仍用严格
        的 ``scan_table_max_tilt``。
        """
        strict = self.config.scan_table_max_tilt
        if index == len(self._view_offsets()) - 1:
            return strict
        return min(10.0, strict + 2.0)

    def _fit_table(self, points, index, focus=None, previous=None):
        """Fit this view's support plane, keeping the previous one if it drifts.

        A later view may look at the wall instead of the table; the local focus
        window then has no support points and the fit can fall back to the floor
        or another surface.  A plane that moves more than a few centimetres
        would silently corrupt the map carving and the grasp height, so the
        previous (view-0) table wins in that case.
        """
        tilt_limit = self._table_tilt_limit(index)
        try:
            table = fit_table(
                points,
                roi_min=self.config.scan_roi_min,
                roi_max=self.config.scan_roi_max,
                plane_threshold_m=self.config.scan_plane_threshold,
                voxel_m=self.config.scan_table_voxel,
                max_tilt_deg=tilt_limit,
                min_inliers=self.config.scan_table_min_inliers,
                min_inlier_ratio=self.config.scan_table_min_ratio,
                ransac_iterations=self.config.scan_table_iterations,
                edge_padding_m=self.config.scan_table_edge_padding,
                support_depth_m=self.config.scan_table_support_depth,
                mask_padding_m=self.config.scan_table_mask_padding,
                focus_xyz=focus,
                focus_radius_m=self.config.scan_table_focus_radius,
                focus_down_m=self.config.scan_table_focus_down,
                focus_up_m=self.config.scan_table_focus_up,
                footprint_radius_m=self.config.scan_table_footprint_radius,
                surface_epsilon_m=self.config.scan_table_surface_epsilon)
        except RuntimeError as error:
            if previous is None:
                raise
            rospy.logwarn(
                "视点%d桌面拟合失败(%s)，沿用上一视点桌面", index, error)
            return previous
        if previous is not None:
            probe = focus if focus is not None else np.asarray(
                previous["bounds_min"], dtype=float)
            x = float(probe[0])
            y = float(probe[1])
            new_z = surface_z(table, x, y)
            old_z = surface_z(previous, x, y)
            if abs(new_z - old_z) > float(self.config.scan_table_max_jump):
                rospy.logwarn(
                    "视点%d桌面高度跳变%.3fm(旧%.4f 新%.4f)，沿用上一视点桌面",
                    index, new_z - old_z, old_z, new_z)
                return previous
        normal = table["plane_normal"]
        tilt_deg = math.degrees(math.acos(
            min(1.0, max(-1.0, float(normal[2])))))
        rospy.loginfo(
            "桌面拟合: 视点%d 倾角%.2f°(门限%.2f°) 内点%d/%d "
            "footprint=%d 范围%s 顶面%.4f 法向[%.4f, %.4f, %.4f]",
            index, tilt_deg, tilt_limit,
            table["inlier_count"], table["sample_count"],
            table.get("footprint_count", table["inlier_count"]),
            table.get("fit_scope", "global"),
            table.get("surface_top", float("nan")),
            normal[0], normal[1], normal[2])
        return table

    def _require_final_table_tilt(self, table):
        """Re-apply the strict tilt limit to the table that will be frozen.

        Middle views use a 2-degree looser limit so a locally wavy first view
        does not abort the scan, and a failed later refit keeps the previous
        plane.  Neither may let a plane above ``scan_table_max_tilt`` reach the
        grasp: the grasp height is measured along base Z from that plane.
        """
        normal = np.asarray(table["plane_normal"], dtype=float)
        tilt = math.degrees(math.acos(
            min(1.0, max(-1.0, float(normal[2])))))
        limit = float(self.config.scan_table_max_tilt)
        if tilt > limit + 1e-6:
            raise RuntimeError(
                "最终桌面倾角%.2f°>%.2f°：抓取高度会沿倾斜面累积误差，"
                "先调平桌面/底座，不要靠放大阈值绕过" % (tilt, limit))

    def _refine_target(self, points, table, previous):
        """Refine the target centre on a later view without re-clustering."""
        if previous is None:
            raise RuntimeError("目标瓶尚未建立，无法细化")
        obstacle_points = obstacle_model.segment(
            points, table,
            clearance=max(float(self.config.scan_table_clearance), 0.0),
            footprint_margin=float(self.config.scan_footprint_margin))
        if len(obstacle_points) == 0:
            rospy.logwarn("视点障碍点为空，沿用上一视点目标位置")
            return np.asarray(previous, dtype=float)
        return obstacle_model.refine_target_center(
            obstacle_points, previous, table, self.config)

    def _publish_map_view(self, view_points, transform, stamp, table,
                          target_center, index):
        """Carve and publish one view into the static OctoMap.

        The cloud is published in the *camera optical* frame so MoveIt's
        ``PointCloudOctomapUpdater`` ray-casts from the real camera pose.  A
        fused cloud published in base_link would be ray-cast from the base
        origin and would erase real obstacles.
        """
        carved, stats = map_filter.build_map_view(
            view_points, table,
            target_center=target_center,
            target_radius=self.config.bottle_radius,
            target_height=self.config.bottle_height,
            base_keepout_m=self.config.scan_map_base_keepout_radius,
            target_carve_m=self.config.scan_map_target_carve,
            support_carve_m=self.config.scan_map_support_carve,
            support_xy_margin_m=self.config.scan_map_support_margin,
            target_extra_height_m=self.config.scan_map_target_height_margin,
            voxel_m=self.config.scan_map_cloud_voxel)
        optical = map_filter.to_optical(carved, transform)
        if stamp is None:
            stamp = rospy.Time.now()
        header = Header()
        header.frame_id = self.config.scan_camera_frame
        header.stamp = stamp
        before = self.scene.octomap_signature()
        self._cloud_pub.publish(
            point_cloud2.create_cloud_xyz32(header, optical.tolist()))
        after = self.scene.wait_octomap_quiet(
            self.config.scan_map_settle_quiet,
            self.config.scan_map_settle_timeout)
        if after == before:
            if index == 0:
                raise RuntimeError(
                    "发布点云后OctoMap未变化：检查 move_group 是否加载 "
                    "sensors_3d.yaml，以及 TF %s -> dummy_link 是否可达"
                    % self.config.scan_camera_frame)
            rospy.logwarn(
                "视点%d发布后OctoMap未变化：该视点可能被TF或更新速率丢弃",
                index)
        self._map_views.append({
            "index": int(index),
            "stamp": float(stamp.to_sec()),
            "frame": self.config.scan_camera_frame,
            "base_T_optical": np.asarray(transform, dtype=float),
            "points_base": np.asarray(carved, dtype=float),
            "stats": dict(stats),
        })
        rospy.loginfo(
            "视点%d障碍地图: 输入%d 底座挖除%d 目标挖除%d 支撑挖除%d "
            "降采样%d → 发布%d点",
            index, stats["input"], stats["base_keepout_removed"],
            stats.get("target_removed", 0), stats["support_removed"],
            stats["downsampled_removed"], stats["published"])

    def _verify_map(self):
        """Refuse to freeze an empty map or a self-colliding start state."""
        signature = self.scene.wait_octomap_quiet(
            self.config.scan_map_settle_quiet,
            self.config.scan_map_settle_timeout)
        snapshot = self.scene.octomap_snapshot()
        if snapshot is None or signature is None:
            raise RuntimeError(
                "静态障碍地图为空：检查 %s 是否被 "
                "PointCloudOctomapUpdater 消费（move_group 需带sensors）"
                % self.config.scan_cloud_topic)
        if snapshot["bytes"] < int(self.config.scan_map_min_bytes):
            raise RuntimeError(
                "障碍地图过小(%d字节<%d)，检查传感器TF与点云"
                % (snapshot["bytes"], int(self.config.scan_map_min_bytes)))
        rospy.loginfo(
            "障碍地图已建立: frame=%s 分辨率=%.3fm 序列化%d字节",
            snapshot["frame"], snapshot["resolution"], snapshot["bytes"])
        if self.config.scan_map_require_current_valid:
            self.scene.validate_current_state()
            rospy.loginfo("当前机械臂状态与障碍地图无碰撞")

    # ------------------------------------------------------------------
    def _check_card_clearance(self, card):
        center = np.asarray(card["center"], dtype=float)
        normal = np.asarray(card["normal"], dtype=float)
        horizontal = np.linalg.norm(normal[:2])
        if horizontal < 1e-3:
            raise RuntimeError("卡片法向几乎竖直，无法确定放置方向")
        direction = normal[:2] / horizontal
        if np.dot(direction, -center[:2]) < 0.0:
            direction = -direction
        bottle_xy = center[:2] + self.config.place_offset_from_card * direction
        table_z = self.scene.table_surface_z(bottle_xy[0], bottle_xy[1])
        required = (self.config.bottle_radius + card["thickness"] / 2.0
                    + self.config.scan_obstacle_padding
                    + self.config.place_card_clearance)
        if self.config.place_offset_from_card < required:
            raise RuntimeError(
                "place_offset_from_card=%.3f 太小，需至少%.3f才能避免碰卡片"
                % (self.config.place_offset_from_card, required))
        pose, dims = obstacle_model.card_pose_and_dims(card)
        measured = distance_to_spec(
            np.array([bottle_xy[0], bottle_xy[1],
                      table_z + self.config.bottle_height / 2.0]),
            {"kind": "box", "pose": pose, "dims": dims})
        rospy.loginfo(
            "放置校验: 松手点(%.3f,%.3f) 桌面z=%.3f 到卡片包围盒实测%.3fm"
            " (需>=%.3fm)",
            bottle_xy[0], bottle_xy[1], table_z, measured,
            self.config.bottle_radius + self.config.place_card_clearance)

    # ------------------------------------------------------------------
    # Artifacts.
    # ------------------------------------------------------------------
    def _save(self, result, points, observations, images, summary):
        root = Path(self.config.scan_artifact_root).expanduser()
        try:
            folder = root / time.strftime("scan_%Y%m%d_%H%M%S")
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "cloud.npz",
                                points=points, observations=observations)
            (folder / "table_model.json").write_text(json.dumps(
                result["table"], ensure_ascii=False, indent=2,
                default=_json_default))
            scene = {
                "target_center": result["target_center"].tolist(),
                "card": None,
                "obstacles": [],
                "detections": result["detections"],
                "views": result["views"],
                "frames": result["frames"],
                "points": result["points"],
            }
            if result["card"] is not None:
                scene["card"] = {
                    "center": result["card"]["center"].tolist(),
                    "normal": result["card"]["normal"].tolist(),
                    "width": result["card"]["width"],
                    "length": result["card"]["length"],
                    "thickness": result["card"]["thickness"],
                }
            for spec in result["obstacles"]:
                scene["obstacles"].append({
                    "id": spec["id"], "kind": spec["kind"],
                    "dims": list(spec["dims"]),
                    "pose": np.asarray(spec["pose"]).tolist(),
                    "points": spec["points"], "label": spec["label"],
                })
            (folder / "scene.json").write_text(json.dumps(
                scene, ensure_ascii=False, indent=2,
                default=_json_default))
            self._save_map(folder)
            (folder / "summary.txt").write_text(
                "\n".join(summary) + "\n", encoding="utf-8")
            for index, image in enumerate(images):
                if image is not None:
                    cv2.imwrite(str(folder / ("view_%d.png" % index)), image)
            rospy.loginfo("扫描快照已保存: %s", folder)
        except (OSError, ValueError, TypeError) as error:
            rospy.logwarn("扫描快照保存失败(不影响抓取): %s", error)

    def _save_map(self, folder):
        """Persist every published map view and the frozen octomap."""
        if not self._map_views:
            return
        map_dir = folder / "map"
        map_dir.mkdir(parents=True, exist_ok=True)
        meta = []
        for entry in self._map_views:
            name = "view_%02d" % entry["index"]
            np.savez_compressed(
                map_dir / (name + ".npz"),
                points_base=entry["points_base"],
                base_T_optical=entry["base_T_optical"],
                stamp=entry["stamp"], frame=entry["frame"])
            meta.append({
                "view": entry["index"], "stamp": entry["stamp"],
                "frame": entry["frame"], "stats": entry["stats"],
            })
        snapshot = self.scene.octomap_snapshot()
        if snapshot is not None:
            np.savez_compressed(
                map_dir / "octomap.npz", data=snapshot["data"],
                frame=snapshot["frame"], tree_id=snapshot["id"],
                resolution=snapshot["resolution"],
                binary=snapshot["binary"], sha256=snapshot["sha256"])
            meta.append({"octomap": {
                "frame": snapshot["frame"], "id": snapshot["id"],
                "resolution": snapshot["resolution"],
                "binary": snapshot["binary"], "bytes": snapshot["bytes"],
                "sha256": snapshot["sha256"]}})
            self._save_restorable_bag(map_dir)
        (map_dir / "meta.json").write_text(json.dumps(
            meta, ensure_ascii=False, indent=2, default=_json_default))

    def _save_restorable_bag(self, map_dir):
        """Best-effort snapshot in the format ``static_scene_snapshot.py`` loads.

        ``rosbag`` needs Cryptodome, which the Piper Conda interpreter lacks, so
        the existing helper is invoked with the system interpreter.  Failure
        only loses the restorable copy; the raw octree in ``octomap.npz`` is
        always written.
        """
        try:
            import rospkg
            script = Path(rospkg.RosPack().get_path("moveit_ctrl")) / \
                "scripts" / "static_scene_snapshot.py"
        except Exception as error:  # rospkg/package missing
            rospy.logwarn("无法定位static_scene_snapshot.py: %s", error)
            return
        if not script.is_file():
            rospy.logwarn("静态地图快照脚本不存在: %s", script)
            return
        output = map_dir / "octomap_snapshot.bag"
        try:
            reply = subprocess.run(
                ["/usr/bin/python3", str(script), "save",
                 "--output", str(output)],
                capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            rospy.logwarn("静态地图快照保存失败(仅诊断快照): %s", error)
            return
        if reply.returncode != 0:
            rospy.logwarn(
                "静态地图快照保存失败(仅诊断快照): %s",
                (reply.stderr or reply.stdout).strip())
        else:
            rospy.loginfo("静态地图快照(可恢复): %s", output)

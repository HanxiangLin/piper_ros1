#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning scene, ACM, OctoMap gate and table-model management."""

import copy
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    GetPlanningSceneRequest,
    GetStateValidityRequest,
)
from std_srvs.srv import Trigger

from pose_utils import pose_from_matrix, pose_matrix

TARGET_BOTTLE_ID = "target_bottle"


class SceneManager:
    """Owns the MoveIt planning scene, ACM, table model and scan gate."""

    def __init__(self, get_scene, apply_scene, check_validity, robot, tf_buffer,
                 table_model_path, scene, support_edge_margin=0.0,
                 target_bottle_id=TARGET_BOTTLE_ID):
        self.get_scene = get_scene
        self.apply_scene = apply_scene
        self.check_validity = check_validity
        self.robot = robot
        self.tf_buffer = tf_buffer
        self.table_model_path = table_model_path
        self.scene = scene
        self.support_edge_margin = support_edge_margin
        self.target_bottle_id = target_bottle_id

        self.table_model = None
        self.scan_signature = None
        self.acm_before_task = None

    def frozen_scene_signature(self):
        rospy.wait_for_service("/piper_static_scan_gate/status", timeout=5.0)
        response = rospy.ServiceProxy(
            "/piper_static_scan_gate/status", Trigger)()
        if not response.success:
            raise RuntimeError("扫描门状态服务返回失败: " + response.message)
        try:
            gate_status = json.loads(response.message)
        except (TypeError, ValueError) as error:
            raise RuntimeError("扫描门返回无效JSON") from error
        if gate_status.get("enabled") is not False:
            raise RuntimeError("扫描门未关闭，拒绝规划/执行")

        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.OCTOMAP |
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        scene = self.get_scene(request).scene
        outer = scene.world.octomap.header.frame_id
        octomap = scene.world.octomap.octomap
        inner = octomap.header.frame_id
        frame = (outer or inner).lstrip("/")
        if (not octomap.data or not np.isfinite(octomap.resolution)
                or octomap.resolution <= 0 or not octomap.id
                or (outer and inner and outer != inner)
                or frame != "dummy_link"):
            raise RuntimeError("环境OctoMap为空、无效或坐标系错误")
        objects = {obj.id: obj for obj in scene.world.collision_objects}
        if "table" not in objects:
            raise RuntimeError("自动table模型未加载")
        table = objects["table"]
        geometry = copy.deepcopy(table)
        geometry.header.seq = 0
        geometry.header.stamp = rospy.Time(0)
        stream = BytesIO()
        geometry.serialize(stream)
        payload = bytes((value & 255 for value in octomap.data))
        origin = scene.world.octomap.origin
        origin_values = (
            origin.position.x, origin.position.y, origin.position.z,
            origin.orientation.x, origin.orientation.y,
            origin.orientation.z, origin.orientation.w)
        signature = (
            frame, octomap.id, bool(octomap.binary),
            float(octomap.resolution), origin_values,
            hashlib.sha256(payload).hexdigest(),
            hashlib.sha256(stream.getvalue()).hexdigest())
        return signature, table

    def require_scan_scene(self):
        signature, table = self.frozen_scene_signature()
        model = json.loads(Path(self.table_model_path).read_text())
        if (model.get("version") != 1
                or model.get("frame_id") != "base_link"
                or model.get("object_id") != "table"):
            raise RuntimeError("桌面模型文件无效")
        lower = np.asarray(model["bounds_min"], dtype=float)
        upper = np.asarray(model["bounds_max"], dtype=float)
        normal = np.asarray(model["plane_normal"], dtype=float)
        offset = float(model["plane_offset"])
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite([lower, upper]).all()
                or np.any(upper <= lower)):
            raise RuntimeError("桌面模型边界无效")
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or not np.isfinite(offset)
                or abs(np.linalg.norm(normal) - 1.0) > 1e-3
                or normal[2] < np.cos(np.deg2rad(10.0))):
            raise RuntimeError("扫描桌面平面参数无效")
        if (len(table.primitives) != 1 or len(table.primitive_poses) != 1
                or table.primitives[0].type != 1
                or not np.allclose(
                    table.primitives[0].dimensions, upper - lower,
                    rtol=0.0, atol=1e-6)):
            raise RuntimeError("场景table尺寸与模型文件不一致")
        actual = pose_from_matrix(
            pose_matrix(table.pose)
            @ pose_matrix(table.primitive_poses[0]))
        actual.header = table.header
        if actual.header.frame_id.lstrip("/") != "base_link":
            actual = self.tf_buffer.transform(
                actual, "base_link", timeout=rospy.Duration(1.0))
        actual_matrix = pose_matrix(actual.pose)
        if (not np.allclose(
                actual_matrix[:3, 3], (lower + upper) / 2.0,
                rtol=0.0, atol=1e-4)
                or not np.allclose(
                    actual_matrix[:3, :3], np.eye(3),
                    rtol=0.0, atol=1e-4)):
            raise RuntimeError("场景table位姿与扫描文件不一致")

        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        scene = self.get_scene(request).scene
        acm = scene.allowed_collision_matrix
        names = list(acm.entry_names)
        defaults = dict(zip(
            acm.default_entry_names, acm.default_entry_values))
        for obstacle in ("table", "<octomap>"):
            for link in self.robot.get_link_names():
                if obstacle in names and link in names:
                    allowed = acm.entry_values[
                        names.index(obstacle)].enabled[names.index(link)]
                else:
                    allowed = defaults.get(obstacle, False) or defaults.get(link, False)
                if allowed:
                    raise RuntimeError(
                        "障碍与机器人碰撞被放开: {}/{}".format(obstacle, link))
        validity = GetStateValidityRequest()
        validity.robot_state = scene.robot_state
        validity.group_name = ""
        reply = self.check_validity(validity)
        if not reply.valid:
            pairs = [(item.contact_body_1, item.contact_body_2)
                     for item in reply.contacts]
            raise RuntimeError("当前状态无效/碰撞: " + str(pairs))
        self.table_model = model
        self.scan_signature = signature
        self.acm_before_task = self.get_acm()
        rospy.loginfo("冻结场景、table模型和当前机器人状态检查通过")

    def require_unchanged_scan(self):
        signature, _ = self.frozen_scene_signature()
        if signature != self.scan_signature:
            raise RuntimeError("环境OctoMap/table在任务中改变")

    def validate_robot_state(self, state):
        request = GetStateValidityRequest()
        request.robot_state = state
        request.group_name = ""
        reply = self.check_validity(request)
        if not reply.valid:
            pairs = [(item.contact_body_1, item.contact_body_2)
                     for item in reply.contacts]
            raise RuntimeError("机器人状态无效/碰撞: " + str(pairs))

    def get_acm(self):
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        return copy.deepcopy(
            self.get_scene(request).scene.allowed_collision_matrix)

    def apply_acm(self, acm):
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = acm
        if not self.apply_scene(scene).success:
            raise RuntimeError("MoveIt拒绝ACM更新")

    def set_contact_pairs(self, pairs, allowed):
        acm = self.get_acm()
        for first, second in pairs:
            for name in (first, second):
                if name not in acm.entry_names:
                    for row in acm.entry_values:
                        row.enabled.append(False)
                    acm.entry_names.append(name)
                    acm.entry_values.append(AllowedCollisionEntry(
                        enabled=[False] * len(acm.entry_names)))
            first_index = acm.entry_names.index(first)
            second_index = acm.entry_names.index(second)
            acm.entry_values[first_index].enabled[second_index] = bool(allowed)
            acm.entry_values[second_index].enabled[first_index] = bool(allowed)
        self.apply_acm(acm)

    def wait_object(self, name, known, attached, timeout=5.0):
        """Wait until the object matches the requested scene state.

        ``known`` or ``attached`` may be ``None`` when that part of the state
        is irrelevant.  ``remove_attached_object`` puts the object back into
        the world instead of deleting it, so its ``known`` state after a
        release is version dependent and must not be asserted.
        """
        deadline = time.monotonic() + timeout
        is_known = is_attached = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            is_known = name in self.scene.get_known_object_names()
            is_attached = bool(self.scene.get_attached_objects([name]))
            if ((known is None or is_known == known)
                    and (attached is None or is_attached == attached)):
                return
            rospy.sleep(0.05)
        raise RuntimeError(
            "场景对象未同步: %s (known=%s attached=%s)"
            % (name, is_known, is_attached))

    def table_surface_z(self, x, y):
        normal = np.asarray(self.table_model["plane_normal"], dtype=float)
        return -(normal[0] * x + normal[1] * y
                 + float(self.table_model["plane_offset"])) / normal[2]

    def require_support_footprint(self, center_xyz, radius):
        lower = np.asarray(self.table_model["bounds_min"], dtype=float)
        upper = np.asarray(self.table_model["bounds_max"], dtype=float)
        border = radius + self.scan_signature[3] + self.support_edge_margin
        xy = np.asarray(center_xyz[:2], dtype=float)
        if np.any(xy <= lower[:2] + border) or np.any(xy >= upper[:2] - border):
            raise RuntimeError("瓶子接触区域靠近table边界，拒绝抓取")

    def add_bottle(self, center_xyz, bottle_height, bottle_radius,
                   lift_distance, lift_table_margin, state):
        center_xyz = np.asarray(center_xyz, dtype=float)
        if center_xyz.shape != (3,) or not np.isfinite(center_xyz).all():
            raise ValueError("瓶子中心无效")
        self.require_support_footprint(center_xyz, bottle_radius)

        bottom_at_grasp = center_xyz[2] - bottle_height / 2.0
        envelope_top = self.table_model["bounds_max"][2]
        if (bottom_at_grasp + lift_distance
                <= envelope_top + lift_table_margin):
            raise ValueError(
                "lift_distance不足以让瓶底离开table碰撞盒: "
                "瓶底z=%.4f 提升后z=%.4f 盒顶z=%.4f 需要>%.4f；"
                "请增大lift_distance或检查机械臂垂直可达高度"
                % (bottom_at_grasp, bottom_at_grasp + lift_distance,
                   envelope_top, envelope_top + lift_table_margin))

        if (self.scene.get_attached_objects([self.target_bottle_id])
                or self.target_bottle_id in self.scene.get_known_object_names()):
            raise RuntimeError("已有target_bottle，请核对上次真实抓取状态后处理，拒绝覆盖")
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.orientation.w = 1.0
        (pose.pose.position.x, pose.pose.position.y,
         pose.pose.position.z) = center_xyz
        self.scene.add_cylinder(
            self.target_bottle_id, pose,
            height=bottle_height, radius=bottle_radius)
        self.wait_object(self.target_bottle_id, known=True, attached=False)
        state.bottle_center_at_grasp = center_xyz.copy()
        state.created_bottle = True
        self.set_contact_pairs(
            [(self.target_bottle_id, link)
             for link in self.robot.get_link_names()], False)
        return center_xyz.copy()

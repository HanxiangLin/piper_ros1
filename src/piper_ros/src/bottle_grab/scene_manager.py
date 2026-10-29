#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live planning-scene management built from one scan snapshot.

Every object in the scene is created here and can be rebuilt from the scan
result:

* ``table``            -- the fitted support box (top surface + padding);
* ``target_bottle``    -- one cylinder, allowed to collide with both fingers;
* ``scan_obstacle_N``  -- one collision object per non-target cluster.

``reset_for_scan`` throws the previous snapshot away (world objects, ACM pairs
and the MoveIt OctoMap) before the arm is allowed to move, and ``freeze`` stores
a signature of the finished snapshot so every later motion can prove that the
scene did not change behind the program's back.
"""

import copy
import hashlib
import time

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    GetPlanningSceneRequest,
    GetStateValidityRequest,
)
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty, Trigger

from obstacle_model import OBSTACLE_PREFIX
from pose_utils import pose_from_matrix
from table_fit import surface_z, validate_model

TARGET_BOTTLE_ID = "target_bottle"
TABLE_ID = "table"
OCTOMAP_ID = "<octomap>"


def _pose_key(pose):
    return tuple(round(float(value), 6) for value in (
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w))


class SceneManager:
    """Owns the MoveIt planning scene, ACM, table model and scan snapshot."""

    def __init__(self, get_scene, apply_scene, check_validity, robot, tf_buffer,
                 scene, support_edge_margin=0.02, voxel_size=0.02,
                 clear_octomap_service="/clear_octomap"):
        self.get_scene = get_scene
        self.apply_scene = apply_scene
        self.check_validity = check_validity
        self.robot = robot
        self.tf_buffer = tf_buffer
        self.scene = scene
        self.support_edge_margin = support_edge_margin
        self.voxel_size = float(voxel_size)
        self.clear_octomap_service = clear_octomap_service

        self.table_model = None
        self.obstacles = []
        self.scan_result = None
        self.signature = None
        self.acm_before_task = None

    # ------------------------------------------------------------------
    # Planning-scene queries.
    # ------------------------------------------------------------------
    def _get(self, components):
        request = GetPlanningSceneRequest()
        request.components.components = components
        return self.get_scene(request).scene

    def octomap_signature(self):
        scene = self._get(PlanningSceneComponents.OCTOMAP)
        octomap = scene.world.octomap.octomap
        if not octomap.data:
            return None
        payload = bytes((value & 255 for value in octomap.data))
        frame = (scene.world.octomap.header.frame_id
                 or octomap.header.frame_id).lstrip("/")
        return (frame, octomap.id, bool(octomap.binary),
                round(float(octomap.resolution), 6),
                hashlib.sha256(payload).hexdigest())

    def octomap_snapshot(self):
        """Return the serialized octree for the scan artifact snapshot."""
        scene = self._get(PlanningSceneComponents.OCTOMAP)
        octomap = scene.world.octomap.octomap
        if not octomap.data:
            return None
        payload = bytes((value & 255 for value in octomap.data))
        frame = (scene.world.octomap.header.frame_id
                 or octomap.header.frame_id).lstrip("/")
        return {
            "frame": frame,
            "id": octomap.id,
            "binary": bool(octomap.binary),
            "resolution": float(octomap.resolution),
            "data": np.frombuffer(payload, dtype=np.uint8).copy(),
            "bytes": int(len(payload)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def wait_octomap_quiet(self, quiet_s=1.0, timeout_s=20.0, poll_s=0.4):
        """Block until the octomap stops changing.

        The updater consumes published clouds asynchronously, so the next
        scan-view motion must not be planned before the current view is fully
        integrated.  Raises when the map keeps changing past ``timeout_s``.
        """
        quiet_s = float(quiet_s)
        timeout_s = float(timeout_s)
        if quiet_s < 0.0 or timeout_s <= 0.0:
            raise ValueError("OctoMap稳定等待参数无效")
        deadline = time.monotonic() + timeout_s
        last = object()
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            current = self.octomap_signature()
            if current != last:
                last = current
                last_change = time.monotonic()
            elif time.monotonic() - last_change >= quiet_s:
                return current
            rospy.sleep(poll_s)
        raise RuntimeError(
            "OctoMap在%.1fs内未稳定；检查PointCloudOctomapUpdater是否在消费%s"
            % (timeout_s, "点云"))

    def current_signature(self):
        """Hash of everything the planner is allowed to see right now."""
        scene = self._get(PlanningSceneComponents.OCTOMAP
                          | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        octomap = scene.world.octomap.octomap
        octomap_key = None
        if octomap.data:
            payload = bytes((value & 255 for value in octomap.data))
            frame = (scene.world.octomap.header.frame_id
                     or octomap.header.frame_id).lstrip("/")
            octomap_key = (frame, octomap.id, bool(octomap.binary),
                           round(float(octomap.resolution), 6),
                           hashlib.sha256(payload).hexdigest())
        world = []
        for obj in scene.world.collision_objects:
            world.append((
                obj.id,
                obj.header.frame_id.lstrip("/"),
                tuple(int(item.type) for item in obj.primitives),
                tuple(tuple(round(float(value), 6) for value in item.dimensions)
                      for item in obj.primitives),
                _pose_key(obj.pose),
                tuple(_pose_key(item) for item in obj.primitive_poses),
            ))
        return octomap_key, tuple(sorted(world))

    # ------------------------------------------------------------------
    # Reset / freeze.
    # ------------------------------------------------------------------
    def clear_octomap(self):
        if self.octomap_signature() is None:
            return
        try:
            rospy.wait_for_service(self.clear_octomap_service, timeout=5.0)
            rospy.ServiceProxy(self.clear_octomap_service, Empty)()
        except (rospy.ROSException, rospy.ServiceException) as error:
            raise RuntimeError(
                "无法清空残留OctoMap(%s)；拒绝在旧地图上继续" % error)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.octomap_signature() is None:
                return
            rospy.sleep(0.1)
        raise RuntimeError("清空OctoMap超时，仍存在残留体素")

    def stop_scan_gate(self):
        """Best-effort stop of the legacy static scan gate.

        The live scan no longer uses it, but a leftover enabled gate would
        republish the raw cloud into the OctoMap and fight the frozen scene.
        """
        name = "/piper_static_scan_gate/stop"
        try:
            rospy.wait_for_service(name, timeout=1.0)
        except rospy.ROSException:
            return False
        try:
            reply = rospy.ServiceProxy(name, Trigger)()
        except rospy.ServiceException as error:
            raise RuntimeError("无法关闭静态扫描门: %s" % error)
        rospy.logwarn("已关闭遗留的静态扫描门: %s", reply.message)
        return True

    def reset_for_scan(self):
        """Remove every object created by this program and empty the OctoMap."""
        self.stop_scan_gate()
        if self.scene.get_attached_objects([TARGET_BOTTLE_ID]):
            raise RuntimeError(
                "target_bottle仍附着在link6，核对上次真实抓取状态后处理")
        known = list(self.scene.get_known_object_names())
        ours = [name for name in known
                if name in (TABLE_ID, TARGET_BOTTLE_ID)
                or name.startswith(OBSTACLE_PREFIX)]
        unknown = sorted(set(known) - set(ours))
        if unknown:
            raise RuntimeError(
                "规划场景中存在未知对象，拒绝清场: " + ", ".join(unknown))
        if ours:
            self.apply_world([], ours)
        self.clear_octomap()
        self.reset_contact_pairs(
            list(ours) + [TABLE_ID, TARGET_BOTTLE_ID, OCTOMAP_ID])
        self.table_model = None
        self.obstacles = []
        self.scan_result = None
        self.signature = None
        self.acm_before_task = None
        rospy.loginfo("扫描前清场完成：世界对象、ACM接触放开、OctoMap均已复位")

    def freeze(self, scan_result):
        """Store the finished snapshot and its signature."""
        self.scan_result = scan_result
        self.table_model = scan_result["table"]
        self.obstacles = list(scan_result["obstacles"])
        self.signature = self.current_signature()
        self.acm_before_task = self.get_acm()
        rospy.loginfo(
            "场景已冻结：桌面+目标瓶+%d个语义障碍；OctoMap=%s",
            len(self.obstacles),
            "非空" if self.octomap_signature() is not None else "空")

    def refresh_signature(self):
        """Accept the current scene as the new reference.

        Used only right after the program itself attaches/detaches/re-adds the
        target bottle, i.e. after a change it made on purpose.
        """
        if self.signature is not None:
            self.signature = self.current_signature()

    def require_unchanged(self):
        if self.signature is None:
            raise RuntimeError("场景尚未冻结，拒绝规划或执行")
        if self.current_signature() != self.signature:
            raise RuntimeError("环境/规划场景在任务中改变，拒绝继续")

    # ------------------------------------------------------------------
    # World objects.
    # ------------------------------------------------------------------
    def apply_world(self, objects, remove_ids):
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = list(objects)
        for name in remove_ids:
            removal = CollisionObject()
            removal.header.frame_id = "base_link"
            removal.id = name
            removal.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(removal)
        if not self.apply_scene(scene).success:
            raise RuntimeError("MoveIt拒绝场景更新")
        stored = {obj.id: obj
                  for obj in self._get(
                      PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                  ).world.collision_objects}
        missing = [name for name in remove_ids if name in stored]
        if missing:
            raise RuntimeError("场景对象删除未生效: " + ", ".join(missing))
        return stored

    @staticmethod
    def _primitive_object(name, primitive_type, dimensions, pose, frame="base_link"):
        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.id = name
        obj.operation = CollisionObject.ADD
        obj.pose.orientation.w = 1.0
        primitive = SolidPrimitive()
        primitive.type = primitive_type
        primitive.dimensions = [float(value) for value in dimensions]
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        return obj

    def set_table(self, model):
        validate_model(model)
        lower = np.asarray(model["bounds_min"], dtype=float)
        upper = np.asarray(model["bounds_max"], dtype=float)
        pose = Pose()
        (pose.position.x, pose.position.y, pose.position.z) = (
            (lower + upper) / 2.0).tolist()
        pose.orientation.w = 1.0
        obj = self._primitive_object(
            TABLE_ID, SolidPrimitive.BOX, (upper - lower).tolist(), pose)
        stored = self.apply_world([obj], [])
        table = stored.get(TABLE_ID)
        if (table is None or len(table.primitives) != 1
                or table.primitives[0].type != SolidPrimitive.BOX
                or not np.allclose(table.primitives[0].dimensions,
                                   upper - lower, rtol=0.0, atol=1e-6)):
            raise RuntimeError("场景table几何与扫描模型不一致")
        self.table_model = model
        rospy.loginfo(
            "桌面已载入: x[%.3f,%.3f] y[%.3f,%.3f] 表面z=%.4f 顶面=%.4f",
            lower[0], upper[0], lower[1], upper[1],
            surface_z(model, 0.5 * (lower[0] + upper[0]),
                      0.5 * (lower[1] + upper[1])),
            upper[2])

    def set_obstacles(self, specs):
        objects = []
        for spec in specs:
            primitive_type = (SolidPrimitive.CYLINDER
                              if spec["kind"] == "cylinder"
                              else SolidPrimitive.BOX)
            pose = Pose()
            matrix = np.asarray(spec["pose"], dtype=float)
            matrix_pose = pose_from_matrix(matrix)
            pose.position = matrix_pose.pose.position
            pose.orientation = matrix_pose.pose.orientation
            objects.append(self._primitive_object(
                spec["id"], primitive_type, spec["dims"], pose))
        new_ids = {spec["id"] for spec in specs}
        stale = [spec["id"] for spec in self.obstacles
                 if spec["id"] not in new_ids]
        stored = self.apply_world(objects, stale)
        for spec in specs:
            obj = stored.get(spec["id"])
            if (obj is None or not obj.primitives
                    or not np.allclose(obj.primitives[0].dimensions,
                                       spec["dims"], rtol=0.0, atol=1e-6)):
                raise RuntimeError("障碍物写入场景失败: " + spec["id"])
        self.obstacles = list(specs)

    def remove_obstacles(self):
        if self.obstacles:
            self.apply_world([], [spec["id"] for spec in self.obstacles])
        self.obstacles = []

    # ------------------------------------------------------------------
    # ACM.
    # ------------------------------------------------------------------
    def get_acm(self):
        return copy.deepcopy(
            self._get(PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                      ).allowed_collision_matrix)

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

    def reset_contact_pairs(self, names):
        """Disable every ACM pair that mentions one of ``names``."""
        acm = self.get_acm()
        changed = False
        for name in names:
            if name not in acm.entry_names:
                continue
            index = acm.entry_names.index(name)
            for other in range(len(acm.entry_names)):
                if acm.entry_values[index].enabled[other]:
                    acm.entry_values[index].enabled[other] = False
                    acm.entry_values[other].enabled[index] = False
                    changed = True
        if changed:
            self.apply_acm(acm)
        return changed

    # ------------------------------------------------------------------
    # Geometry helpers.
    # ------------------------------------------------------------------
    def table_surface_z(self, x, y):
        if self.table_model is None:
            raise RuntimeError("尚未载入桌面模型")
        return surface_z(self.table_model, x, y)

    def require_support_footprint(self, center_xyz, radius):
        lower = np.asarray(self.table_model["bounds_min"], dtype=float)
        upper = np.asarray(self.table_model["bounds_max"], dtype=float)
        border = radius + self.voxel_size + self.support_edge_margin
        xy = np.asarray(center_xyz[:2], dtype=float)
        if np.any(xy <= lower[:2] + border) or np.any(xy >= upper[:2] - border):
            raise RuntimeError(
                "物体接触区域靠近table边界，拒绝抓取/放置: "
                "xy=(%.3f,%.3f) 边界余量需%.3f"
                % (xy[0], xy[1], border))

    def add_bottle(self, center_xyz, bottle_height, bottle_radius,
                   lift_distance, lift_table_margin, state, replace=False):
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

        attached = self.scene.get_attached_objects([TARGET_BOTTLE_ID])
        known = TARGET_BOTTLE_ID in self.scene.get_known_object_names()
        if (attached or known) and not replace:
            raise RuntimeError(
                "已有target_bottle，请核对上次真实抓取状态后处理，拒绝覆盖")

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = center_xyz.tolist()
        pose.orientation.w = 1.0
        obj = self._primitive_object(
            TARGET_BOTTLE_ID, SolidPrimitive.CYLINDER,
            (bottle_height, bottle_radius), pose)
        stored = self.apply_world([obj], [])
        if TARGET_BOTTLE_ID not in stored:
            raise RuntimeError("target_bottle未同步到规划场景")
        state.bottle_center_at_grasp = center_xyz.copy()
        state.created_bottle = True
        self.set_contact_pairs(
            [(TARGET_BOTTLE_ID, link) for link in self.robot.get_link_names()],
            False)
        return center_xyz.copy()

    def update_bottle(self, center_xyz, bottle_height, bottle_radius,
                      lift_distance, lift_table_margin, state):
        """Re-place the target cylinder after a refined multi-view estimate."""
        return self.add_bottle(
            center_xyz, bottle_height, bottle_radius, lift_distance,
            lift_table_margin, state, replace=True)

    def wait_object(self, name, known, attached, timeout=5.0):
        """Wait until the object matches the requested scene state."""
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

    def inspect_robot_state(self, robot_state):
        """Return ``(valid, contact_pairs)`` without raising.

        Diagnostics use this to report which link/object pair blocks a
        candidate; motion paths keep using :meth:`validate_robot_state`.
        """
        request = GetStateValidityRequest()
        request.robot_state = robot_state
        request.group_name = ""
        reply = self.check_validity(request)
        pairs = [(item.contact_body_1, item.contact_body_2)
                 for item in reply.contacts]
        return bool(reply.valid), pairs

    def validate_robot_state(self, robot_state):
        valid, pairs = self.inspect_robot_state(robot_state)
        if not valid:
            raise RuntimeError("机器人状态无效/碰撞: " + str(pairs))

    def validate_current_state(self):
        """Raise unless the *current* robot state is collision free.

        Called right after the static obstacle map is built: if the arm's own
        base/fingers already overlap the map, every later plan would be
        rejected, so this must fail early and loudly.
        """
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        self.validate_robot_state(self.get_scene(request).scene.robot_state)

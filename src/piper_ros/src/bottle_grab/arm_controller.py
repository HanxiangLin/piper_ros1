#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic MoveGroup wrapper: planning, validation and execution."""

import copy
import math
import xml.etree.ElementTree as ET

import numpy as np
import rospy
from moveit_commander.exception import MoveItCommanderException
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import (
    GetPlanningSceneRequest,
    GetPositionIK,
    GetPositionIKRequest,
)

from joint_feedback import ARM_JOINTS
from pose_utils import pose_from_matrix, pose_matrix


class ArmController:
    """Thin, task-agnostic wrapper around MoveGroupCommander for the arm."""

    def __init__(self, move_group, robot, scene_manager, joint_feedback,
                 display_pub, state, confirm_grasp,
                 velocity_scale, acceleration_scale,
                 cartesian_joint_step_limit):
        self.arm = move_group
        self.robot = robot
        self.scene_manager = scene_manager
        self.joint_feedback = joint_feedback
        self.display_pub = display_pub
        self.state = state
        self.confirm_grasp = confirm_grasp
        self.velocity_scale = velocity_scale
        self.acceleration_scale = acceleration_scale
        self.cartesian_joint_step_limit = cartesian_joint_step_limit
        self._ik_proxy = None
        self._limits = None

    def current_ee_matrix(self):
        """Return the current link6 pose in base_link as a 4x4 matrix."""
        pose = self.arm.get_current_pose("link6")
        if pose.header.frame_id.lstrip("/") != "base_link":
            pose = self.scene_manager.tf_buffer.transform(
                pose, "base_link", timeout=rospy.Duration(1.0))
        return pose_matrix(pose.pose)

    @staticmethod
    def unpack_plan(result):
        """Normalize the moveit_commander RobotTrajectory result."""
        if isinstance(result, tuple):
            if not result[0]:
                raise RuntimeError("MoveIt规划失败")
            result = result[1]
        if result is None or not result.joint_trajectory.points:
            raise RuntimeError("MoveIt返回空轨迹")
        return result

    def _ik_service(self):
        if self._ik_proxy is None:
            rospy.wait_for_service("/compute_ik", timeout=5.0)
            self._ik_proxy = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        return self._ik_proxy

    def _full_robot_state(self):
        """Return the planning scene's complete robot state (with fingers)."""
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        return copy.deepcopy(
            self.scene_manager.get_scene(request).scene.robot_state)

    def _joint_limits(self):
        """Parse ``/robot_description`` once and return joint limit pairs."""
        if self._limits is None:
            limits = {}
            try:
                text = rospy.get_param("/robot_description")
                root = ET.fromstring(text)
                for joint in root.findall("joint"):
                    name = joint.get("name")
                    limit = joint.find("limit")
                    if name is None or limit is None:
                        continue
                    lower = limit.get("lower")
                    upper = limit.get("upper")
                    if lower is not None and upper is not None:
                        limits[name] = (float(lower), float(upper))
            except Exception as error:  # noqa: BLE001 - diagnostics only
                rospy.logwarn("读取URDF关节限位失败: %s", error)
            self._limits = limits
        return self._limits

    def _joint_margin_text(self, names, values):
        """Format joint values and their distance to the nearest limit."""
        limits = self._joint_limits()
        parts = []
        for name, value in zip(names, values):
            value = float(value)
            if name not in limits:
                parts.append("%s=%.3f" % (name, value))
                continue
            lower, upper = limits[name]
            margin = min(value - lower, upper - value)
            parts.append("%s=%.3f(余%.3f)" % (name, value, margin))
        return " ".join(parts)

    def joint_limit_margins(self, values):
        """Return each joint's distance to its nearest limit (rad)."""
        limits = self._joint_limits()
        margins = []
        for name, value in zip(ARM_JOINTS, values):
            if name not in limits:
                margins.append(float("inf"))
                continue
            lower, upper = limits[name]
            value = float(value)
            margins.append(min(value - lower, upper - value))
        return margins

    def plan_pose_nearest(self, matrix, eef_link="link6"):
        """Plan to a pose through an IK joint goal seeded by the current state.

        ``set_pose_target`` only stores a pose constraint: OMPL then samples
        the goal with its own IK seeds and can pick a far-away elbow/wrist
        branch, which explains the occasional "reachable but strange" posture.

        ``set_joint_value_target`` solves one IK from the current state and
        passes the resulting joint values to the planner, so the goal stays on
        the nearest branch.  That IK solution is not collision checked, hence
        the trajectory is validated here and the original pose target is used
        as a fallback.
        """
        self.arm.set_start_state_to_current_state()
        ik_error = None
        try:
            self.arm.set_joint_value_target(
                pose_from_matrix(matrix), eef_link)
            target = self.arm.get_joint_value_target()
            if (isinstance(target, (list, tuple))
                    and len(target) == len(ARM_JOINTS)):
                rospy.loginfo(
                    "就近IK关节目标: %s",
                    self._joint_margin_text(ARM_JOINTS, target))
            planned = self.unpack_plan(self.arm.plan())
            self.validate_trajectory(planned)
            return planned
        except (MoveItCommanderException, RuntimeError) as error:
            ik_error = error
            rospy.logwarn(
                "就近IK关节目标不可用(%s)，回退pose目标", error)
            self.arm.clear_pose_targets()
            self.arm.set_start_state_to_current_state()
            try:
                self.arm.set_pose_target(pose_from_matrix(matrix), eef_link)
                planned = self.unpack_plan(self.arm.plan())
                self.validate_trajectory(planned)
                return planned
            except (MoveItCommanderException, RuntimeError) as fallback:
                raise RuntimeError(
                    "就近IK失败(%s)，pose回退也失败(%s)"
                    % (ik_error, fallback))

    def diagnose_cartesian(self, matrix, fraction=None, eef_step=0.003):
        """Explain why a straight-line approach was truncated.

        Repeats the same request with collision checking disabled.  A complete
        path then proves the line is kinematically reachable and the original
        truncation was caused by a collision; an incomplete path points at
        IK/joint-limit trouble.  The collision-free path is scanned for the
        first invalid state so the blocking pair is reported.

        Must be called while ``considered_start_state_`` still holds the
        virtual pre-grasp endpoint and the relaxed ACM is still applied.
        """
        original = "n/a" if fraction is None else "%.3f" % fraction
        try:
            trajectory, free_fraction = self.arm.compute_cartesian_path(
                [pose_from_matrix(matrix).pose], eef_step=eef_step,
                avoid_collisions=False)
        except Exception as error:  # noqa: BLE001 - diagnostic must not raise
            reason = "直线段复算异常: %s" % error
            rospy.logwarn("直线接近诊断(原始%s): %s", original, reason)
            return reason

        points = list(trajectory.joint_trajectory.points)
        if (not points or not np.isfinite(free_fraction)
                or free_fraction < 0.999999):
            free_text = ("nan" if not np.isfinite(free_fraction)
                         else "%.3f" % free_fraction)
            if np.isfinite(free_fraction) and free_fraction < 0.0:
                reason = "无碰撞复算服务失败(比例%s)" % free_text
            else:
                reason = ("无碰撞复算仍只完成%s，属于IK/关节限位问题"
                          % free_text)
            if points:
                reason += "；最后可达点 " + self._joint_margin_text(
                    trajectory.joint_trajectory.joint_names,
                    points[-1].positions)
            rospy.logwarn("直线接近诊断(原始%s): %s", original, reason)
            return reason

        inspect = getattr(self.scene_manager, "inspect_robot_state", None)
        if inspect is None:
            reason = ("无碰撞复算通过，但场景接口不支持碰撞详情")
            rospy.logwarn("直线接近诊断(原始%s): %s", original, reason)
            return reason
        state = self._full_robot_state()
        names = list(state.joint_state.name)
        index = {name: number for number, name in enumerate(names)}
        joint_names = list(trajectory.joint_trajectory.joint_names)
        if any(name not in index for name in joint_names):
            reason = ("无碰撞复算通过，但轨迹关节不在完整机器人状态中")
            rospy.logwarn("直线接近诊断(原始%s): %s", original, reason)
            return reason
        positions = list(state.joint_state.position)
        first_bad = None
        for number, point in enumerate(points):
            for name, value in zip(joint_names, point.positions):
                positions[index[name]] = float(value)
            state.joint_state.position = positions
            valid, pairs = inspect(state)
            if not valid:
                first_bad = (number, list(pairs))
                break
        if first_bad is None:
            reason = ("无碰撞复算通过，但未找到无效点；"
                      "可能是复算期间场景/状态变化")
        else:
            number, pairs = first_bad
            reason = "第%d/%d步发生碰撞: %s" % (
                number + 1, len(points), pairs)
        rospy.logwarn("直线接近诊断(原始%s): %s", original, reason)
        return reason

    def diagnose_pose(self, matrix, description, timeout=0.5, seeds=8):
        """Explain why a pose cannot be reached; return ``None`` when it can.

        ``/compute_ik`` solves from a *single* seed, and KDL often fails on a
        legal but tightly folded pose.  OMPL's pose-goal sampling uses random
        seeds, so a single-seed failure is not proof.  Try several seeds and
        only report a reason when none of them yields a collision-free
        solution.  Callers use this as a diagnosis after planning failed, never
        as a gate before planning.
        """
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        state = copy.deepcopy(
            self.scene_manager.get_scene(request).scene.robot_state)
        names = list(state.joint_state.name)
        base = list(state.joint_state.position)
        arm_names = [name for name in names if name in ARM_JOINTS]
        if len(arm_names) != 6:
            return "规划场景缺少六轴关节，无法判断"

        candidates = [list(base)]
        zeros = list(base)
        for name in arm_names:
            zeros[names.index(name)] = 0.0
        candidates.append(zeros)
        for name in arm_names:
            for delta in (0.5, -0.5):
                values = list(base)
                values[names.index(name)] += delta
                candidates.append(values)
        rng = np.random.default_rng(7)
        for _ in range(max(int(seeds) - len(candidates), 0)):
            values = list(base)
            for name in arm_names:
                values[names.index(name)] = float(rng.uniform(-1.5, 1.5))
            candidates.append(values)

        last_reason = None
        for values in candidates:
            seed = copy.deepcopy(state)
            seed.joint_state.position = values
            target = GetPositionIKRequest()
            target.ik_request.group_name = "arm"
            target.ik_request.robot_state = seed
            target.ik_request.ik_link_name = "link6"
            target.ik_request.pose_stamped = pose_from_matrix(matrix)
            target.ik_request.timeout = rospy.Duration(timeout)
            try:
                response = self._ik_service()(target)
            except rospy.ServiceException as error:
                return "IK服务不可用: %s" % error
            if response.error_code.val != 1:
                last_reason = "KDL无IK解(错误码%d)" % response.error_code.val
                continue
            solved = copy.deepcopy(state)
            positions = list(base)
            for name, value in zip(response.solution.joint_state.name,
                                   response.solution.joint_state.position):
                if name in names:
                    positions[names.index(name)] = value
            solved.joint_state.position = positions
            try:
                self.scene_manager.validate_robot_state(solved)
            except RuntimeError as error:
                last_reason = str(error)
                continue
            return None
        reason = "%s，已尝试%d组初值" % (
            last_reason or "无解", len(candidates))
        rospy.logwarn("%s不可达/碰撞: %s", description, reason)
        return reason

    def validate_trajectory(self, trajectory):
        """Collision-check the trajectory against the full robot state."""
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        state = copy.deepcopy(
            self.scene_manager.get_scene(request).scene.robot_state)
        names = list(state.joint_state.name)
        index = {name: position for position, name in enumerate(names)}
        joint_names = list(trajectory.joint_trajectory.joint_names)
        if not joint_names or any(name not in index for name in joint_names):
            raise RuntimeError("轨迹关节为空或不在完整机器人状态中")
        previous = None
        for point in trajectory.joint_trajectory.points:
            goal = np.asarray(point.positions, dtype=float)
            if len(goal) != len(joint_names) or not np.isfinite(goal).all():
                raise RuntimeError("轨迹关节数不匹配或包含非有限值")
            # 返回点之间按每关节最多0.03rad细分，补充全机器人/附着物检查。
            count = 1 if previous is None else max(
                1, int(np.ceil(np.max(abs(goal - previous)) / 0.03)))
            for alpha in np.linspace(1.0 / count, 1.0, count):
                values = goal if previous is None else previous + (goal - previous) * alpha
                positions = list(state.joint_state.position)
                for name, value in zip(joint_names, values):
                    positions[index[name]] = float(value)
                state.joint_state.position = positions
                self.scene_manager.validate_robot_state(state)
            previous = goal

    def check_trajectory_start(self, trajectory):
        """Reject a trajectory whose first point disagrees with the real state.

        This mirrors MoveIt's own start-point check (0.01 rad) but reports the
        offending joint and deviation instead of only ``CONTROL_FAILED``.
        """
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("MoveIt返回空轨迹")
        start = self.joint_feedback._fresh_joint_positions()
        names = list(trajectory.joint_trajectory.joint_names)
        if any(name not in start for name in names):
            raise RuntimeError("轨迹关节不在/joint_states中")
        first = dict(zip(
            names, trajectory.joint_trajectory.points[0].positions))
        deviation = max(abs(first[name] - start[name]) for name in names)
        if deviation > 0.010:
            worst = max(names, key=lambda name: abs(first[name] - start[name]))
            raise RuntimeError(
                "轨迹起点与实机状态相差 %.4f rad(%s)，拒绝执行"
                % (deviation, worst))

    def finish_motion(self, trajectory, guarded=True):
        """Display and execute a planned trajectory.

        ``guarded=False`` is only used by the scan phase, where the planner
        scene is *supposed* to change between motions.
        """
        if guarded:
            self.scene_manager.require_unchanged()
        self.joint_feedback._fresh_joint_positions()
        self.validate_trajectory(trajectory)
        display = DisplayTrajectory()
        display.trajectory_start = self.robot.get_current_state()
        display.trajectory.append(trajectory)
        self.display_pub.publish(display)
        if guarded:
            self.scene_manager.require_unchanged()
        self.joint_feedback._fresh_joint_positions()
        if self.state.grasp_confirmed:
            self.confirm_grasp()
        self.check_trajectory_start(trajectory)
        if not self.arm.execute(trajectory, wait=True):
            raise RuntimeError("MoveIt/FollowJointTrajectory执行失败")
        # execute(wait=True)已在目标点结束；这里不再重复stop()，
        # 避免取消已完成goal导致actionlib晚到状态回调报警。
        final = dict(zip(
            self.arm.get_active_joints(), self.arm.get_current_joint_values()))
        names = trajectory.joint_trajectory.joint_names
        goals = trajectory.joint_trajectory.points[-1].positions
        error = max(abs(final[name] - goal)
                    for name, goal in zip(names, goals))
        if error > 0.05:
            raise RuntimeError(
                "执行后最大关节误差过大: {:.4f}rad".format(error))

    def move_joints(self, values, guarded=True):
        if len(values) != 6:
            raise ValueError("六轴目标必须包含6个数")
        self.arm.set_start_state_to_current_state()
        self.arm.set_joint_value_target(dict(zip(ARM_JOINTS, values)))
        try:
            self.finish_motion(self.unpack_plan(self.arm.plan()), guarded)
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def move_pose(self, matrix, guarded=True):
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(pose_from_matrix(matrix), "link6")
        try:
            self.finish_motion(self.unpack_plan(self.arm.plan()), guarded)
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def move_cartesian(self, matrix, guarded=True, allow_plan=False):
        """Straight-line move; ``allow_plan`` falls back to OMPL when blocked.

        A truncated cartesian path means the *line* is blocked, not that the
        goal is unreachable.  The place descent uses ``allow_plan=True`` so a
        large card cannot make placement fail just because the bottle would
        sweep its edge.
        """
        self.arm.set_start_state_to_current_state()
        trajectory, fraction = self.arm.compute_cartesian_path(
            [pose_from_matrix(matrix).pose], eef_step=0.003,
            avoid_collisions=True)
        if not math.isfinite(fraction):
            raise RuntimeError("直线路径完成比例无效")
        if fraction < 0.999999 or not trajectory.joint_trajectory.points:
            if not allow_plan:
                raise RuntimeError(
                    "直线路径受阻或不可达，完成比例 {:.6f}".format(fraction))
            rospy.logwarn(
                "直线路径只完成%.1f%%，改用MoveIt规划到同一目标位",
                fraction * 100.0)
            self.arm.clear_pose_targets()
            self.move_pose(matrix, guarded)
            return
        limit = self.cartesian_joint_step_limit
        previous = dict(zip(
            self.arm.get_active_joints(), self.arm.get_current_joint_values()))
        for point in trajectory.joint_trajectory.points:
            current = dict(zip(
                trajectory.joint_trajectory.joint_names, point.positions))
            if any(abs(value - previous[name]) > limit
                   for name, value in current.items()):
                raise RuntimeError("笛卡尔轨迹关节跳变过大")
            previous = current
        trajectory = self.arm.retime_trajectory(
            self.robot.get_current_state(), trajectory,
            velocity_scaling_factor=self.velocity_scale,
            acceleration_scaling_factor=self.acceleration_scale)
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("轨迹时间参数化失败")
        times = [point.time_from_start.to_sec()
                 for point in trajectory.joint_trajectory.points]
        if times[-1] <= 0 or any(second <= first
                                 for first, second in zip(times, times[1:])):
            raise RuntimeError("轨迹时间无效")
        try:
            self.finish_motion(trajectory, guarded)
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

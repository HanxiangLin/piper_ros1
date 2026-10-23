#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic MoveGroup wrapper: planning, validation and execution."""

import copy
import math

import numpy as np
import rospy
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningSceneRequest

from joint_feedback import ARM_JOINTS
from pose_utils import pose_from_matrix, pose_matrix


class PlanningOnlyStop(RuntimeError):
    """Stop the sequential task after displaying its first planned motion."""


class ArmController:
    """Thin, task-agnostic wrapper around MoveGroupCommander for the arm."""

    def __init__(self, move_group, robot, scene_manager, joint_feedback,
                 display_pub, execute_motion, state, confirm_grasp,
                 velocity_scale, acceleration_scale,
                 cartesian_joint_step_limit):
        self.arm = move_group
        self.robot = robot
        self.scene_manager = scene_manager
        self.joint_feedback = joint_feedback
        self.display_pub = display_pub
        self.execute_motion = execute_motion
        self.state = state
        self.confirm_grasp = confirm_grasp
        self.velocity_scale = velocity_scale
        self.acceleration_scale = acceleration_scale
        self.cartesian_joint_step_limit = cartesian_joint_step_limit

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

    def finish_motion(self, trajectory):
        """Display and (when enabled) execute a planned trajectory."""
        self.scene_manager.require_unchanged_scan()
        self.joint_feedback._fresh_joint_positions()
        self.validate_trajectory(trajectory)
        display = DisplayTrajectory()
        display.trajectory_start = self.robot.get_current_state()
        display.trajectory.append(trajectory)
        self.display_pub.publish(display)
        if not self.execute_motion:
            rospy.sleep(1.0)
            raise PlanningOnlyStop(
                "仅规划模式：第一段轨迹已显示，未发送运动命令")
        self.scene_manager.require_unchanged_scan()
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

    def move_joints(self, values):
        if len(values) != 6:
            raise ValueError("六轴目标必须包含6个数")
        self.arm.set_start_state_to_current_state()
        self.arm.set_joint_value_target(dict(zip(ARM_JOINTS, values)))
        try:
            self.finish_motion(self.unpack_plan(self.arm.plan()))
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def move_pose(self, matrix):
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(pose_from_matrix(matrix), "link6")
        try:
            self.finish_motion(self.unpack_plan(self.arm.plan()))
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def move_cartesian(self, matrix):
        self.arm.set_start_state_to_current_state()
        trajectory, fraction = self.arm.compute_cartesian_path(
            [pose_from_matrix(matrix).pose], eef_step=0.003,
            avoid_collisions=True)
        if (not math.isfinite(fraction) or fraction < 0.999999
                or not trajectory.joint_trajectory.points):
            raise RuntimeError(
                "直线路径受阻或不可达，完成比例 {:.6f}".format(fraction))
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
            self.finish_motion(trajectory)
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

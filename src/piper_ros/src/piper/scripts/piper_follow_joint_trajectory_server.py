#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adapt ROS FollowJointTrajectory goals to the Piper JointState command API.

The same executable can serve either the six arm joints or the MoveIt ``joint7``
gripper joint. It streams interpolated position commands and uses real
``joint_states`` feedback to decide whether a goal has actually completed.
"""

import math
import threading
import time

import actionlib
import rospy
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
ARM_POSITION_LIMITS = {
    "joint1": (-2.618, 2.618),
    "joint2": (0.0, 3.14),
    "joint3": (-2.967, 0.0),
    "joint4": (-1.745, 1.745),
    "joint5": (-1.22, 1.22),
    "joint6": (-2.0944, 2.0944),
    "joint7": (0.0, 0.035),
}
GRIPPER_JOINTS = ("joint7",)


class PiperFollowJointTrajectoryServer:
    """A FollowJointTrajectory Action server backed by Piper's ROS driver."""

    def __init__(self):
        self.joints = tuple(rospy.get_param("~joints", list(ARM_JOINTS)))
        self.action_name = rospy.get_param(
            "~action_name", "/arm_controllers/follow_joint_trajectory"
        )
        self.command_topic = rospy.resolve_name(
            rospy.get_param("~command_topic", "/joint_ctrl_single")
        )
        self.feedback_topic = rospy.resolve_name(
            rospy.get_param("~feedback_topic", "/joint_states")
        )
        self.control_rate = float(rospy.get_param("~control_rate", 50.0))
        self.feedback_timeout = float(rospy.get_param("~feedback_timeout", 1.0))
        self.default_path_tolerance = float(
            rospy.get_param("~default_path_tolerance", 0.50)
        )
        self.default_goal_tolerance = float(
            rospy.get_param("~default_goal_tolerance", 0.03)
        )
        self.default_goal_time_tolerance = float(
            rospy.get_param("~default_goal_time_tolerance", 5.0)
        )
        self.start_tolerance = float(rospy.get_param("~start_tolerance", 0.10))
        self.settle_cycles = int(rospy.get_param("~settle_cycles", 5))
        configured_limits = rospy.get_param("~joint_position_limits", {})
        self.position_limits = {}
        for name in self.joints:
            limits = configured_limits.get(name, ARM_POSITION_LIMITS.get(name))
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise ValueError("关节 {} 没有有效的位置限制".format(name))
            self.position_limits[name] = (float(limits[0]), float(limits[1]))

        self._validate_parameters()
        self._state_lock = threading.Lock()
        self._actual_positions = None
        self._actual_velocities = None
        self._last_feedback_wall_time = None

        self._command_publisher = rospy.Publisher(
            self.command_topic, JointState, queue_size=1, tcp_nodelay=True
        )
        self._state_subscriber = rospy.Subscriber(
            self.feedback_topic,
            JointState,
            self._joint_state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._server = actionlib.SimpleActionServer(
            self.action_name,
            FollowJointTrajectoryAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo(
            "Piper FollowJointTrajectory 服务端已启动：%s；命令=%s，反馈=%s",
            self.action_name,
            self.command_topic,
            self.feedback_topic,
        )

    def _validate_parameters(self):
        if self.joints not in (ARM_JOINTS, GRIPPER_JOINTS):
            raise ValueError(
                "~joints 只支持六轴机械臂 {} 或夹爪 {}".format(
                    ARM_JOINTS, GRIPPER_JOINTS
                )
            )
        if self.command_topic == self.feedback_topic:
            raise ValueError("命令话题和反馈话题不能相同")
        if self.control_rate <= 0.0:
            raise ValueError("~control_rate 必须大于 0")
        if self.feedback_timeout <= 0.0:
            raise ValueError("~feedback_timeout 必须大于 0")
        if self.default_path_tolerance <= 0.0:
            raise ValueError("~default_path_tolerance 必须大于 0")
        if self.default_goal_tolerance <= 0.0:
            raise ValueError("~default_goal_tolerance 必须大于 0")
        if self.default_goal_time_tolerance < 0.0:
            raise ValueError("~default_goal_time_tolerance 不能小于 0")
        if self.start_tolerance <= 0.0:
            raise ValueError("~start_tolerance 必须大于 0")
        if self.settle_cycles < 1:
            raise ValueError("~settle_cycles 必须至少为 1")

    def _joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            rospy.logwarn_throttle(2.0, "忽略 name/position 长度不一致的 JointState")
            return

        positions = dict(zip(message.name, message.position))
        if any(name not in positions for name in self.joints):
            return

        ordered_positions = [positions[name] for name in self.joints]
        if not all(math.isfinite(value) for value in ordered_positions):
            rospy.logwarn_throttle(2.0, "忽略包含非有限关节位置的 JointState")
            return

        velocities = dict(zip(message.name, message.velocity))
        ordered_velocities = [velocities.get(name, 0.0) for name in self.joints]

        with self._state_lock:
            self._actual_positions = ordered_positions
            self._actual_velocities = ordered_velocities
            self._last_feedback_wall_time = time.monotonic()

    def _read_actual_state(self):
        with self._state_lock:
            if self._actual_positions is None:
                return None, None, None
            return (
                list(self._actual_positions),
                list(self._actual_velocities),
                self._last_feedback_wall_time,
            )

    def _wait_for_initial_feedback(self):
        deadline = time.monotonic() + self.feedback_timeout
        rate = rospy.Rate(min(self.control_rate, 100.0))
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            positions, velocities, stamp = self._read_actual_state()
            if positions is not None and time.monotonic() - stamp <= self.feedback_timeout:
                return positions, velocities
            if self._server.is_preempt_requested():
                return None, None
            rate.sleep()
        return None, None

    def _validate_goal(self, goal):
        trajectory = goal.trajectory
        names = list(trajectory.joint_names)

        if len(names) != len(set(names)) or set(names) != set(self.joints):
            return (
                FollowJointTrajectoryResult.INVALID_JOINTS,
                "轨迹必须且只能包含关节：{}".format(", ".join(self.joints)),
            )
        if not trajectory.points:
            return FollowJointTrajectoryResult.INVALID_GOAL, "轨迹不包含任何点"

        previous_time = -1.0
        for index, point in enumerate(trajectory.points):
            point_time = point.time_from_start.to_sec()
            if point_time < 0.0 or point_time <= previous_time:
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 time_from_start 必须严格递增（错误点 {}）".format(index),
                )
            if len(point.positions) != len(names):
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 {} 的 positions 数量错误".format(index),
                )
            if point.velocities and len(point.velocities) != len(names):
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 {} 的 velocities 数量错误".format(index),
                )
            if point.accelerations and len(point.accelerations) != len(names):
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 {} 的 accelerations 数量错误".format(index),
                )
            if point.effort and len(point.effort) != len(names):
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 {} 的 effort 数量错误".format(index),
                )
            values = (
                list(point.positions)
                + list(point.velocities)
                + list(point.accelerations)
                + list(point.effort)
            )
            if not all(math.isfinite(value) for value in values):
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 {} 包含非有限数值".format(index),
                )
            named_positions = dict(zip(names, point.positions))
            outside_limits = [
                name
                for name in self.joints
                if not self.position_limits[name][0]
                <= named_positions[name]
                <= self.position_limits[name][1]
            ]
            if outside_limits:
                return (
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹点 {} 超出关节位置限制：{}".format(
                        index, ", ".join(outside_limits)
                    ),
                )
            previous_time = point_time

        if not trajectory.header.stamp.is_zero():
            if trajectory.header.stamp < rospy.Time.now():
                return (
                    FollowJointTrajectoryResult.OLD_HEADER_TIMESTAMP,
                    "轨迹的 header.stamp 已经过期",
                )

        known_joints = set(self.joints)
        for tolerance in list(goal.path_tolerance) + list(goal.goal_tolerance):
            if tolerance.name and tolerance.name not in known_joints:
                return (
                    FollowJointTrajectoryResult.INVALID_JOINTS,
                    "容差包含未知关节 {}".format(tolerance.name),
                )

        return FollowJointTrajectoryResult.SUCCESSFUL, ""

    def _ordered_points(self, trajectory):
        indices = [trajectory.joint_names.index(name) for name in self.joints]
        ordered = []
        for source in trajectory.points:
            point = JointTrajectoryPoint()
            point.positions = [source.positions[index] for index in indices]
            if source.velocities:
                point.velocities = [source.velocities[index] for index in indices]
            if source.accelerations:
                point.accelerations = [source.accelerations[index] for index in indices]
            point.time_from_start = source.time_from_start
            ordered.append(point)
        return ordered

    def _position_tolerances(self, tolerances, default):
        values = {name: default for name in self.joints}
        for tolerance in tolerances:
            if not tolerance.name:
                continue
            if tolerance.position == -1.0:
                values[tolerance.name] = float("inf")
            elif tolerance.position > 0.0:
                values[tolerance.name] = tolerance.position
        return [values[name] for name in self.joints]

    @staticmethod
    def _interpolate(previous, following, elapsed):
        start_time = previous.time_from_start.to_sec()
        end_time = following.time_from_start.to_sec()
        duration = end_time - start_time
        if duration <= 0.0:
            ratio = 1.0
        else:
            ratio = max(0.0, min(1.0, (elapsed - start_time) / duration))

        positions = [
            start + ratio * (end - start)
            for start, end in zip(previous.positions, following.positions)
        ]
        if following.velocities and previous.velocities:
            velocities = [
                start + ratio * (end - start)
                for start, end in zip(previous.velocities, following.velocities)
            ]
        elif duration > 0.0:
            velocities = [
                (end - start) / duration
                for start, end in zip(previous.positions, following.positions)
            ]
        else:
            velocities = [0.0] * len(positions)
        return positions, velocities

    def _publish_command(self, positions):
        command = JointState()
        command.header.stamp = rospy.Time.now()
        command.name = list(self.joints)
        command.position = list(positions)
        # Piper's existing driver selects its configured safe speed when velocity is empty.
        command.velocity = []
        command.effort = []
        self._command_publisher.publish(command)

    def _publish_feedback(self, desired_positions, desired_velocities, actual, velocity):
        feedback = FollowJointTrajectoryFeedback()
        feedback.header.stamp = rospy.Time.now()
        feedback.joint_names = list(self.joints)
        feedback.desired.positions = list(desired_positions)
        feedback.desired.velocities = list(desired_velocities)
        feedback.actual.positions = list(actual)
        feedback.actual.velocities = list(velocity)
        feedback.error.positions = [
            target - current for target, current in zip(desired_positions, actual)
        ]
        feedback.error.velocities = [
            target - current for target, current in zip(desired_velocities, velocity)
        ]
        self._server.publish_feedback(feedback)

    def _hold_current_position(self):
        actual, _, _ = self._read_actual_state()
        if actual is not None:
            self._publish_command(actual)

    def _abort(self, error_code, message):
        self._hold_current_position()
        result = FollowJointTrajectoryResult()
        result.error_code = error_code
        result.error_string = message
        rospy.logerr("FollowJointTrajectory 失败：%s", message)
        self._server.set_aborted(result, message)

    def _preempt(self, message="轨迹被取消"):
        self._hold_current_position()
        result = FollowJointTrajectoryResult()
        result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
        result.error_string = message
        rospy.logwarn(message)
        self._server.set_preempted(result, message)

    def _fresh_actual_state(self):
        actual, velocity, stamp = self._read_actual_state()
        if actual is None or time.monotonic() - stamp > self.feedback_timeout:
            return None, None
        return actual, velocity

    def _execute(self, goal):
        error_code, error_message = self._validate_goal(goal)
        if error_code != FollowJointTrajectoryResult.SUCCESSFUL:
            self._abort(error_code, error_message)
            return

        actual, actual_velocity = self._wait_for_initial_feedback()
        if self._server.is_preempt_requested():
            self._preempt()
            return
        if actual is None:
            self._abort(
                FollowJointTrajectoryResult.INVALID_GOAL,
                "没有收到新鲜的实机关节反馈：{}".format(self.feedback_topic),
            )
            return

        trajectory = goal.trajectory
        points = self._ordered_points(trajectory)
        if points[0].time_from_start.to_sec() <= 0.05:
            start_error = max(
                abs(target - current)
                for target, current in zip(points[0].positions, actual)
            )
            if start_error > self.start_tolerance:
                self._abort(
                    FollowJointTrajectoryResult.INVALID_GOAL,
                    "轨迹起点与实机当前状态相差 {:.4f} rad，拒绝跳变".format(
                        start_error
                    ),
                )
                return
        path_tolerances = self._position_tolerances(
            goal.path_tolerance, self.default_path_tolerance
        )
        goal_tolerances = self._position_tolerances(
            goal.goal_tolerance, self.default_goal_tolerance
        )

        initial = JointTrajectoryPoint()
        initial.positions = list(actual)
        initial.velocities = list(actual_velocity)
        initial.time_from_start = rospy.Duration(0.0)

        if not trajectory.header.stamp.is_zero():
            rate = rospy.Rate(self.control_rate)
            while rospy.Time.now() < trajectory.header.stamp and not rospy.is_shutdown():
                if self._server.is_preempt_requested():
                    self._preempt()
                    return
                rate.sleep()

        rospy.loginfo(
            "开始执行轨迹：%d 个点，计划时长 %.3f s",
            len(points),
            points[-1].time_from_start.to_sec(),
        )
        start_wall_time = time.monotonic()
        point_index = 0
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._preempt()
                return

            elapsed = time.monotonic() - start_wall_time
            if elapsed >= points[-1].time_from_start.to_sec():
                break
            while (
                point_index < len(points) - 1
                and elapsed > points[point_index].time_from_start.to_sec()
            ):
                point_index += 1

            following = points[point_index]
            previous = initial if point_index == 0 else points[point_index - 1]
            desired, desired_velocity = self._interpolate(previous, following, elapsed)
            self._publish_command(desired)

            actual, actual_velocity = self._fresh_actual_state()
            if actual is None:
                self._abort(
                    FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED,
                    "实机关节反馈中断超过 {:.2f} s".format(self.feedback_timeout),
                )
                return
            self._publish_feedback(desired, desired_velocity, actual, actual_velocity)
            errors = [abs(target - value) for target, value in zip(desired, actual)]
            violated = [
                self.joints[index]
                for index, error in enumerate(errors)
                if error > path_tolerances[index]
            ]
            if violated:
                self._abort(
                    FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED,
                    "路径跟踪误差超限：{}，最大误差 {:.4f} rad".format(
                        ", ".join(violated), max(errors)
                    ),
                )
                return
            rate.sleep()

        final_positions = list(points[-1].positions)
        final_velocities = (
            list(points[-1].velocities)
            if points[-1].velocities
            else [0.0] * len(self.joints)
        )
        requested_goal_time = goal.goal_time_tolerance.to_sec()
        settle_time = (
            requested_goal_time
            if requested_goal_time > 0.0
            else self.default_goal_time_tolerance
        )
        settle_deadline = time.monotonic() + settle_time
        consecutive_successes = 0

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._preempt()
                return

            self._publish_command(final_positions)
            actual, actual_velocity = self._fresh_actual_state()
            if actual is None:
                self._abort(
                    FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED,
                    "等待终点时实机关节反馈中断",
                )
                return

            self._publish_feedback(
                final_positions, final_velocities, actual, actual_velocity
            )
            errors = [
                abs(target - value) for target, value in zip(final_positions, actual)
            ]
            if all(
                error <= tolerance
                for error, tolerance in zip(errors, goal_tolerances)
            ):
                consecutive_successes += 1
                if consecutive_successes >= self.settle_cycles:
                    result = FollowJointTrajectoryResult()
                    result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
                    result.error_string = "实机已到达轨迹终点"
                    rospy.loginfo(
                        "轨迹执行完成，最大终点误差 %.4f rad", max(errors)
                    )
                    self._server.set_succeeded(result, result.error_string)
                    return
            else:
                consecutive_successes = 0

            if time.monotonic() > settle_deadline:
                self._abort(
                    FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED,
                    "终点到位超时，最大误差 {:.4f} rad，容差最大值 {:.4f} rad".format(
                        max(errors), max(goal_tolerances)
                    ),
                )
                return
            rate.sleep()

        self._preempt("ROS 正在关闭，轨迹终止")


def main():
    rospy.init_node("piper_follow_joint_trajectory_server")
    PiperFollowJointTrajectoryServer()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, ValueError) as error:
        rospy.logerr("Piper FollowJointTrajectory 服务端终止：%s", error)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plan and execute three sequential Piper arm motions through MoveIt.

Each call to ``execute(..., wait=True)`` waits for MoveIt's execution result.
With ``piper_real.launch`` that result comes from the Piper
FollowJointTrajectory server, which reports success only after measured joints
have reached the goal tolerance.  There is no time-delay sequencing here.
"""

import math
import sys

import moveit_commander
import rospy
from moveit_commander.exception import MoveItCommanderException
from sensor_msgs.msg import JointState


def offset_inside_limits(value, amount, lower, upper, margin=0.03):
    """Choose a +amount or -amount target that remains inside joint limits."""
    if value + amount <= upper - margin:
        return value + amount
    if value - amount >= lower + margin:
        return value - amount
    raise RuntimeError(
        "当前关节位置 {:.4f} rad 附近没有足够空间移动 {:.4f} rad".format(
            value, amount
        )
    )


def unpack_plan(plan_result):
    """Support both current and older MoveIt Commander plan() return formats."""
    if isinstance(plan_result, tuple):
        planning_success, trajectory = plan_result[0], plan_result[1]
        if not planning_success:
            return None
    else:
        trajectory = plan_result
    if not trajectory.joint_trajectory.points:
        return None
    return trajectory


def execute_segment(arm, target, segment_number):
    arm.set_start_state_to_current_state()
    arm.set_joint_value_target(target)
    trajectory = unpack_plan(arm.plan())
    if trajectory is None:
        raise RuntimeError("第 {} 段规划失败，未向实机发送轨迹".format(segment_number))

    duration = trajectory.joint_trajectory.points[-1].time_from_start.to_sec()
    rospy.loginfo(
        "第 %d 段规划完成：%d 个轨迹点，预计 %.2f s；开始等待实机执行结果",
        segment_number,
        len(trajectory.joint_trajectory.points),
        duration,
    )

    # This blocks on MoveIt's ExecuteTrajectory Action.  MoveIt in turn blocks
    # on /arm_controllers/follow_joint_trajectory and receives its final result.
    succeeded = arm.execute(trajectory, wait=True)
    arm.stop()
    if not succeeded:
        raise RuntimeError(
            "第 {} 段执行失败或被取消，后续运动不会执行".format(segment_number)
        )

    measured = arm.get_current_joint_values()
    max_error = max(abs(goal - actual) for goal, actual in zip(target, measured))
    rospy.loginfo(
        "第 %d 段收到 SUCCESSFUL 完成反馈，当前最大关节误差 %.4f rad",
        segment_number,
        max_error,
    )


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_three_segments", anonymous=True)
    arm = None

    try:
        execute = bool(rospy.get_param("~execute", False))
        delta = float(rospy.get_param("~delta", 0.08))
        velocity_scale = float(rospy.get_param("~velocity_scale", 0.05))
        acceleration_scale = float(rospy.get_param("~acceleration_scale", 0.05))
        planning_time = float(rospy.get_param("~planning_time", 8.0))

        if not 0.02 <= delta <= 0.20:
            raise ValueError("~delta 必须在 0.02～0.20 rad 之间")
        if not 0.0 < velocity_scale <= 0.20:
            raise ValueError("~velocity_scale 必须在 (0, 0.20] 内")
        if not 0.0 < acceleration_scale <= 0.20:
            raise ValueError("~acceleration_scale 必须在 (0, 0.20] 内")
        if not execute:
            rospy.logwarn(
                "安全锁定：没有执行运动。确认机械臂周围安全后增加 _execute:=true"
            )
            return 0

        rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
        robot = moveit_commander.RobotCommander()
        arm = moveit_commander.MoveGroupCommander("arm")
        arm.allow_replanning(False)
        arm.set_planning_time(planning_time)
        arm.set_num_planning_attempts(5)
        arm.set_max_velocity_scaling_factor(velocity_scale)
        arm.set_max_acceleration_scaling_factor(acceleration_scale)
        arm.set_goal_joint_tolerance(0.03)

        initial = arm.get_current_joint_values()
        active_joints = arm.get_active_joints()
        if len(initial) != 6 or len(active_joints) != 6:
            raise RuntimeError(
                "arm 规划组应包含 6 个活动关节，当前得到 {} 个".format(len(initial))
            )
        if not all(math.isfinite(value) for value in initial):
            raise RuntimeError("当前关节反馈包含非有限数值")

        first = list(initial)
        joint1_bounds = robot.get_joint(active_joints[0]).bounds()
        first[0] = offset_inside_limits(
            first[0], delta, joint1_bounds[0], joint1_bounds[1]
        )

        second = list(first)
        joint4_bounds = robot.get_joint(active_joints[3]).bounds()
        second[3] = offset_inside_limits(
            second[3], delta, joint4_bounds[0], joint4_bounds[1]
        )

        targets = (first, second, list(initial))
        rospy.logwarn(
            "即将执行三段实机运动：joint1 小幅运动、joint4 小幅运动、返回初始关节位姿。"
            "任一段没有返回 SUCCESSFUL 都会立即终止。"
        )

        for segment_number, target in enumerate(targets, start=1):
            execute_segment(arm, target, segment_number)

        rospy.loginfo("三段运动全部完成")
        return 0
    except (MoveItCommanderException, rospy.ROSException, RuntimeError, ValueError) as error:
        rospy.logerr("三段运动脚本终止：%s", error)
        return 1
    finally:
        if arm is not None:
            arm.stop()
            arm.clear_pose_targets()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass


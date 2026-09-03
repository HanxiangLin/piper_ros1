#!/usr/bin/env python3

"""使用 MoveIt 生成并可选执行一条末端笛卡尔圆轨迹。

当前末端位置就是圆的起点，整个圆保持当前末端姿态不变。默认只规划并在
RViz 中显示；只有显式设置 ``_execute:=true`` 才会向控制器发送轨迹。
"""

import copy
import math
import sys
import time

import moveit_commander
import moveit_msgs.msg
import rospy
from moveit_commander.exception import MoveItCommanderException
from sensor_msgs.msg import JointState


SUPPORTED_PLANES = ("xy", "xz", "yz")


def read_parameters():
    """读取并检查适合首次实机测试的保守参数。"""
    radius = float(rospy.get_param("~radius", 0.01))
    plane = str(rospy.get_param("~plane", "xy")).lower()
    point_count = int(rospy.get_param("~points", 72))
    eef_step = float(rospy.get_param("~eef_step", 0.002))
    velocity_scale = float(rospy.get_param("~velocity_scale", 0.05))
    acceleration_scale = float(rospy.get_param("~acceleration_scale", 0.05))
    min_fraction = float(rospy.get_param("~min_fraction", 0.999))
    max_joint_step = float(rospy.get_param("~max_joint_step", 0.15))
    execute = bool(rospy.get_param("~execute", False))
    require_hardware_feedback = bool(
        rospy.get_param("~require_hardware_feedback", execute)
    )
    hardware_feedback_topic = str(
        rospy.get_param("~hardware_feedback_topic", "/joint_states")
    )
    hardware_tolerance = float(rospy.get_param("~hardware_tolerance", 0.05))

    if not 0.005 <= radius <= 0.10:
        raise ValueError("~radius 必须在 0.005～0.10 m 之间")
    if plane not in SUPPORTED_PLANES:
        raise ValueError("~plane 必须是 xy、xz 或 yz")
    if not 12 <= point_count <= 720:
        raise ValueError("~points 必须在 12～720 之间")
    if not 0.0005 <= eef_step <= 0.01:
        raise ValueError("~eef_step 必须在 0.0005～0.01 m 之间")
    if not 0.0 < velocity_scale <= 0.20:
        raise ValueError("实机安全限制：~velocity_scale 必须在 (0, 0.20] 内")
    if not 0.0 < acceleration_scale <= 0.20:
        raise ValueError("实机安全限制：~acceleration_scale 必须在 (0, 0.20] 内")
    if not 0.95 <= min_fraction <= 1.0:
        raise ValueError("~min_fraction 必须在 0.95～1.0 之间")
    if not 0.01 <= max_joint_step <= 0.50:
        raise ValueError("~max_joint_step 必须在 0.01～0.50 rad 之间")
    if not 0.005 <= hardware_tolerance <= 0.20:
        raise ValueError("~hardware_tolerance 必须在 0.005～0.20 rad 之间")

    return {
        "radius": radius,
        "plane": plane,
        "point_count": point_count,
        "eef_step": eef_step,
        "velocity_scale": velocity_scale,
        "acceleration_scale": acceleration_scale,
        "min_fraction": min_fraction,
        "max_joint_step": max_joint_step,
        "execute": execute,
        "require_hardware_feedback": require_hardware_feedback,
        "hardware_feedback_topic": hardware_feedback_topic,
        "hardware_tolerance": hardware_tolerance,
    }


def make_circle_waypoints(start_pose, radius, plane, point_count):
    """以当前位姿为起点生成一圈位姿点，姿态始终保持不变。"""
    waypoints = []

    if plane == "xy":
        # 当前点是圆在 +X 方向的边界点，圆心向基座方向偏移一个半径。
        center_x = start_pose.position.x - radius
        center_y = start_pose.position.y
    elif plane == "xz":
        # 当前点是圆的最低点，轨迹不会低于当前 Z 高度。
        center_x = start_pose.position.x
        center_z = start_pose.position.z + radius
    else:  # yz
        center_y = start_pose.position.y
        center_z = start_pose.position.z + radius

    for index in range(1, point_count + 1):
        angle = 2.0 * math.pi * index / point_count
        waypoint = copy.deepcopy(start_pose)

        if plane == "xy":
            waypoint.position.x = center_x + radius * math.cos(angle)
            waypoint.position.y = center_y + radius * math.sin(angle)
        elif plane == "xz":
            waypoint.position.x = center_x + radius * math.sin(angle)
            waypoint.position.z = center_z - radius * math.cos(angle)
        else:  # yz
            waypoint.position.y = center_y + radius * math.sin(angle)
            waypoint.position.z = center_z - radius * math.cos(angle)

        waypoints.append(waypoint)

    return waypoints


def largest_joint_step(trajectory, current_joints):
    """返回轨迹相邻点之间最大的单关节跳变量。"""
    points = trajectory.joint_trajectory.points
    previous = list(current_joints)
    largest = 0.0

    for point in points:
        if len(point.positions) != len(previous):
            raise RuntimeError("轨迹关节数量与 arm 规划组不一致")
        largest = max(
            largest,
            max(abs(current - old) for current, old in zip(point.positions, previous)),
        )
        previous = list(point.positions)

    return largest


def publish_trajectory(display_publisher, start_state, trajectory):
    display = moveit_msgs.msg.DisplayTrajectory()
    display.trajectory_start = start_state
    display.trajectory.append(trajectory)
    display_publisher.publish(display)


def positions_from_message(message, joint_names):
    """按指定关节顺序从 JointState 中取出位置。"""
    positions = dict(zip(message.name, message.position))
    missing = [name for name in joint_names if name not in positions]
    if missing:
        raise RuntimeError("硬件反馈缺少关节：{}".format(", ".join(missing)))
    return [positions[name] for name in joint_names]


def wait_for_hardware_target(topic, joint_names, target, tolerance, timeout):
    """等待硬件反馈接近目标，返回最后一次最大关节误差。"""
    deadline = time.monotonic() + timeout
    last_error = None

    while not rospy.is_shutdown() and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            message = rospy.wait_for_message(
                topic, JointState, timeout=max(0.05, min(0.5, remaining))
            )
        except rospy.ROSException:
            continue

        actual = positions_from_message(message, joint_names)
        last_error = max(abs(goal - value) for goal, value in zip(target, actual))
        if last_error <= tolerance:
            return last_error

    if last_error is None:
        raise RuntimeError("在 {} 上未收到硬件关节反馈".format(topic))
    raise RuntimeError(
        "硬件未在 {:.1f} 秒内到位，最大关节误差 {:.4f} rad".format(
            timeout, last_error
        )
    )


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_draw_circle", anonymous=True)

    arm = None
    try:
        params = read_parameters()

        # 等待 MoveIt 的当前状态监视器收到关节数据，避免用全零假状态规划。
        rospy.wait_for_message("/joint_states", JointState, timeout=5.0)

        arm = moveit_commander.MoveGroupCommander("arm")
        arm.allow_replanning(False)
        arm.set_max_velocity_scaling_factor(params["velocity_scale"])
        arm.set_max_acceleration_scaling_factor(params["acceleration_scale"])
        arm.set_start_state_to_current_state()

        current_joints = arm.get_current_joint_values()
        if len(current_joints) != 6:
            raise RuntimeError(
                "arm 规划组应包含 6 个关节，实际读取到 {} 个".format(
                    len(current_joints)
                )
            )

        start_state = arm.get_current_state()
        start_pose = arm.get_current_pose().pose
        waypoints = make_circle_waypoints(
            start_pose,
            params["radius"],
            params["plane"],
            params["point_count"],
        )

        rospy.loginfo(
            "开始规划：平面=%s，半径=%.3f m，离散点=%d，起点=(%.4f, %.4f, %.4f)",
            params["plane"],
            params["radius"],
            params["point_count"],
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z,
        )

        # 该接口逐段调用逆解并进行碰撞检查；它不是 OMPL 随机采样规划。
        trajectory, fraction = arm.compute_cartesian_path(
            waypoints,
            eef_step=params["eef_step"],
            avoid_collisions=True,
        )

        if fraction < params["min_fraction"]:
            raise RuntimeError(
                "圆轨迹只规划出 {:.1f}%，低于要求的 {:.1f}%，拒绝执行".format(
                    fraction * 100.0, params["min_fraction"] * 100.0
                )
            )
        if len(trajectory.joint_trajectory.points) < 2:
            raise RuntimeError("规划结果没有足够的轨迹点")

        max_step = largest_joint_step(trajectory, current_joints)
        if max_step > params["max_joint_step"]:
            raise RuntimeError(
                "检测到关节跳变 {:.4f} rad，超过限制 {:.4f} rad，拒绝执行".format(
                    max_step, params["max_joint_step"]
                )
            )

        trajectory = arm.retime_trajectory(
            start_state,
            trajectory,
            velocity_scaling_factor=params["velocity_scale"],
            acceleration_scaling_factor=params["acceleration_scale"],
            algorithm="iterative_time_parameterization",
        )
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("轨迹时间参数化失败")

        duration = trajectory.joint_trajectory.points[-1].time_from_start.to_sec()
        display_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=1,
            latch=True,
        )
        rospy.sleep(0.5)
        publish_trajectory(display_publisher, start_state, trajectory)

        rospy.loginfo(
            "圆轨迹规划完成：fraction=%.3f，轨迹点=%d，时长=%.2f s，最大关节步长=%.4f rad",
            fraction,
            len(trajectory.joint_trajectory.points),
            duration,
            max_step,
        )

        if not params["execute"]:
            rospy.logwarn(
                "当前为只规划模式，没有向机械臂发送命令。确认 RViz 和现场安全后，"
                "使用 _execute:=true 执行。"
            )
            rospy.sleep(1.0)
            return 0

        active_joints = arm.get_active_joints()
        if params["require_hardware_feedback"]:
            start_error = wait_for_hardware_target(
                params["hardware_feedback_topic"],
                active_joints,
                current_joints,
                params["hardware_tolerance"],
                timeout=2.0,
            )
            rospy.loginfo("MoveIt 与硬件起始状态一致，最大误差 %.4f rad", start_error)
        else:
            rospy.logwarn("已关闭真实硬件反馈校验，仅应在 Gazebo 中使用。")

        rospy.logwarn("3 秒后执行实机圆轨迹；如有异常请立即急停或按 Ctrl-C。")
        rospy.sleep(3.0)
        if rospy.is_shutdown():
            return 1

        success = arm.execute(trajectory, wait=True)
        arm.stop()
        if not success:
            rospy.logerr("MoveIt 轨迹执行失败")
            return 1

        if params["require_hardware_feedback"]:
            final_point = trajectory.joint_trajectory.points[-1]
            final_error = wait_for_hardware_target(
                params["hardware_feedback_topic"],
                trajectory.joint_trajectory.joint_names,
                final_point.positions,
                params["hardware_tolerance"],
                timeout=5.0,
            )
            rospy.loginfo("硬件终点校验通过，最大关节误差 %.4f rad", final_error)

        rospy.loginfo("圆轨迹执行完成")
        return 0
    except (MoveItCommanderException, rospy.ROSException, RuntimeError, ValueError) as exc:
        rospy.logerr("画圆脚本终止：%s", exc)
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

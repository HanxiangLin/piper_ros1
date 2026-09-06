#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan, display, and optionally execute an OMPL goal against a frozen OctoMap."""

import copy
import json
import math
import sys

from piper_static_scene.octomap import describe_octomap, octomap_signature

MIN_TRANSLATION_M = 0.005
MAX_TRANSLATION_M = 0.50


def unpack_plan(plan_result):
    """Support MoveIt Commander return formats used across ROS Noetic builds."""
    if isinstance(plan_result, tuple):
        if not plan_result[0]:
            return None
        trajectory = plan_result[1]
    else:
        trajectory = plan_result
    if trajectory is None or not trajectory.joint_trajectory.points:
        return None
    return trajectory


def apply_translation_delta(pose_stamped, dx, dy, dz):
    result = copy.deepcopy(pose_stamped)
    result.pose.position.x += dx
    result.pose.position.y += dy
    result.pose.position.z += dz
    return result


def read_parameters(rospy):
    values = {
        "dx": float(rospy.get_param("~dx", 0.0)),
        "dy": float(rospy.get_param("~dy", 0.0)),
        "dz": float(rospy.get_param("~dz", 0.0)),
        "planning_time": float(rospy.get_param("~planning_time", 10.0)),
        "planning_attempts": int(rospy.get_param("~planning_attempts", 8)),
        "velocity_scale": float(rospy.get_param("~velocity_scale", 0.03)),
        "acceleration_scale": float(rospy.get_param("~acceleration_scale", 0.03)),
        "planner_id": str(rospy.get_param("~planner_id", "RRTConnect")),
        "execute": bool(rospy.get_param("~execute", False)),
        "acknowledge_missing_camera_collision": bool(
            rospy.get_param("~acknowledge_missing_camera_collision", False)),
        "expected_map_frame": str(rospy.get_param("~expected_map_frame", "dummy_link")),
        "map_service": str(rospy.get_param("~map_service", "/get_planning_scene")),
        "gate_status_service": str(rospy.get_param(
            "~gate_status_service", "/piper_static_scan_gate/status")),
    }
    numeric = [values[key] for key in (
        "dx", "dy", "dz", "planning_time", "velocity_scale", "acceleration_scale")]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("规划参数包含非有限数值")
    distance = math.sqrt(values["dx"] ** 2 + values["dy"] ** 2 + values["dz"] ** 2)
    if not MIN_TRANSLATION_M <= distance <= MAX_TRANSLATION_M:
        raise ValueError("末端相对平移距离必须在 {:.3f}～{:.2f} m 之间".format(
            MIN_TRANSLATION_M, MAX_TRANSLATION_M))
    if not 2.0 <= values["planning_time"] <= 30.0:
        raise ValueError("~planning_time 必须在 2～30 秒之间")
    if not 1 <= values["planning_attempts"] <= 30:
        raise ValueError("~planning_attempts 必须在 1～30 之间")
    for key in ("velocity_scale", "acceleration_scale"):
        if not 0.0 < values[key] <= 0.10:
            raise ValueError("~{} 必须在 (0, 0.10] 内".format(key))
    if not values["planner_id"]:
        raise ValueError("~planner_id 不能为空")
    if values["execute"] and not values["acknowledge_missing_camera_collision"]:
        raise ValueError(
            "当前没有相机/支架碰撞模型；实机执行还需设置 "
            "_acknowledge_missing_camera_collision:=true")
    return values


def query_octomap(rospy, service_name, timeout=5.0):
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
    rospy.wait_for_service(service_name, timeout=timeout)
    request = GetPlanningSceneRequest()
    request.components.components = PlanningSceneComponents.OCTOMAP
    return rospy.ServiceProxy(service_name, GetPlanningScene)(request).scene


def require_frozen_gate(rospy, service_name, timeout=5.0):
    from std_srvs.srv import Trigger
    rospy.wait_for_service(service_name, timeout=timeout)
    response = rospy.ServiceProxy(service_name, Trigger)()
    if not response.success:
        raise RuntimeError("扫描门状态服务返回失败：{}".format(response.message))
    try:
        status = json.loads(response.message)
    except (TypeError, ValueError) as error:
        raise RuntimeError("扫描门返回了无效状态") from error
    if status.get("enabled") is not False:
        raise RuntimeError("扫描门仍处于开启状态；先 stop 冻结地图")
    return status


def publish_display(rospy, start_state, trajectory):
    import moveit_msgs.msg
    display = moveit_msgs.msg.DisplayTrajectory()
    display.trajectory_start = start_state
    display.trajectory.append(trajectory)
    publisher = rospy.Publisher(
        "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory,
        queue_size=1, latch=True)
    rospy.sleep(0.4)
    publisher.publish(display)
    return publisher


def max_goal_error(arm, trajectory):
    names = trajectory.joint_trajectory.joint_names
    positions = trajectory.joint_trajectory.points[-1].positions
    measured_by_name = dict(zip(arm.get_active_joints(), arm.get_current_joint_values()))
    return max(abs(goal - measured_by_name[name]) for name, goal in zip(names, positions))


def main():
    import moveit_commander
    import rospy
    from moveit_commander.exception import MoveItCommanderException
    from sensor_msgs.msg import JointState

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_static_scene_goal", anonymous=True)
    arm = None
    display_publisher = None
    try:
        params = read_parameters(rospy)
        gate = require_frozen_gate(rospy, params["gate_status_service"])
        scene_before = query_octomap(rospy, params["map_service"])
        report = describe_octomap(scene_before)
        if not report["present"]:
            raise RuntimeError("MoveIt 当前 OctoMap 为空，拒绝规划")
        if report["frame"] != params["expected_map_frame"]:
            raise RuntimeError("OctoMap frame={}，期望 {}".format(
                report["frame"], params["expected_map_frame"]))
        signature = octomap_signature(scene_before)
        rospy.loginfo(
            "冻结地图有效：frame=%s，resolution=%.3f m，序列化字节=%d，扫描帧=%d",
            report["frame"], report["resolution_m"], report["serialized_bytes"],
            gate.get("session_forwarded", 0))

        rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
        arm = moveit_commander.MoveGroupCommander("arm")
        arm.allow_replanning(False)
        arm.set_planner_id(params["planner_id"])
        arm.set_planning_time(params["planning_time"])
        arm.set_num_planning_attempts(params["planning_attempts"])
        arm.set_max_velocity_scaling_factor(params["velocity_scale"])
        arm.set_max_acceleration_scaling_factor(params["acceleration_scale"])
        arm.set_goal_position_tolerance(0.005)
        arm.set_goal_orientation_tolerance(0.03)
        arm.set_start_state_to_current_state()

        start_state = arm.get_current_state()
        start_pose = arm.get_current_pose()
        target = apply_translation_delta(
            start_pose, params["dx"], params["dy"], params["dz"])
        arm.set_pose_target(target)
        rospy.loginfo(
            "OMPL目标（坐标系 %s）：起点=(%.4f, %.4f, %.4f)，目标=(%.4f, %.4f, %.4f)",
            start_pose.header.frame_id, start_pose.pose.position.x,
            start_pose.pose.position.y, start_pose.pose.position.z,
            target.pose.position.x, target.pose.position.y, target.pose.position.z)

        trajectory = unpack_plan(arm.plan())
        if trajectory is None:
            raise RuntimeError("OMPL规划失败：目标可能不可达、位于障碍物中或没有无碰撞通道")
        scene_after = query_octomap(rospy, params["map_service"])
        if octomap_signature(scene_after) != signature:
            raise RuntimeError("规划期间 OctoMap 发生变化，拒绝使用该轨迹")

        duration = trajectory.joint_trajectory.points[-1].time_from_start.to_sec()
        display_publisher = publish_display(rospy, start_state, trajectory)
        rospy.loginfo("规划成功：planner=%s，轨迹点=%d，预计时长=%.2f s",
                      params["planner_id"], len(trajectory.joint_trajectory.points), duration)

        if not params["execute"]:
            rospy.logwarn("当前为只规划模式，没有发送运动命令；请在RViz检查显示轨迹。")
            rospy.sleep(1.0)
            return 0

        rospy.logwarn(
            "相机和支架没有碰撞模型。确认地图、真实路径、急停及低速设置后，"
            "在本终端输入 EXECUTE；其他输入均取消。")
        try:
            confirmation = input("执行确认> ").strip()
        except EOFError:
            confirmation = ""
        if confirmation != "EXECUTE":
            rospy.logwarn("未收到精确确认，轨迹没有执行")
            return 2
        require_frozen_gate(rospy, params["gate_status_service"])
        if octomap_signature(query_octomap(rospy, params["map_service"])) != signature:
            raise RuntimeError("确认期间 OctoMap 发生变化，拒绝执行")

        succeeded = arm.execute(trajectory, wait=True)
        arm.stop()
        if not succeeded:
            raise RuntimeError("MoveIt/FollowJointTrajectory 返回执行失败或取消")
        error = max_goal_error(arm, trajectory)
        rospy.loginfo("收到SUCCESSFUL执行反馈；当前最大关节目标误差 %.4f rad", error)
        return 0
    except (MoveItCommanderException, rospy.ROSException, RuntimeError,
            ValueError, KeyError) as error:
        rospy.logerr("静态场景目标规划终止：%s", error)
        return 1
    finally:
        if arm is not None:
            arm.stop()
            arm.clear_pose_targets()
        # Keep the latched publisher alive until shutdown after the final log.
        _ = display_publisher
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass

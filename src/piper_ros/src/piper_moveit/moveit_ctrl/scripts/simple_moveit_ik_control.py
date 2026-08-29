#!/usr/bin/env python3

"""通过末端位姿目标调用 MoveIt 逆运动学并控制 Piper。"""

import sys

import moveit_commander
import rospy


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("simple_moveit_ik_control", anonymous=True)

    arm = None
    try:
        arm = moveit_commander.MoveGroupCommander("arm")
        arm.set_max_velocity_scaling_factor(0.15)
        arm.set_max_acceleration_scaling_factor(0.15)
        arm.set_planning_time(5.0)
        arm.set_goal_position_tolerance(0.005)
        arm.set_goal_orientation_tolerance(0.01)

        # 在当前末端位姿基础上沿规划坐标系 Z 轴抬高 3 cm。
        target_pose = arm.get_current_pose()
        target_pose.pose.position.z += 0.03

        rospy.loginfo("规划坐标系：%s", arm.get_planning_frame())
        rospy.loginfo("末端连杆：%s", arm.get_end_effector_link())
        rospy.loginfo(
            "目标位置：x=%.4f, y=%.4f, z=%.4f",
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
        )

        arm.set_start_state_to_current_state()

        # 设置末端位姿目标后，MoveIt 会调用 kinematics.yaml 中配置的
        # KDL 逆解插件求目标关节角，再由规划器生成关节轨迹。
        arm.set_pose_target(target_pose)
        success = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()

        if not success:
            rospy.logerr("MoveIt 逆解、规划或执行失败")
            return 1

        rospy.loginfo("运动完成，当前末端位姿：%s", arm.get_current_pose().pose)
        return 0
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

#!/usr/bin/env python3

"""使用 MoveIt 控制 Piper 机械臂的最小示例。"""

import sys

import moveit_commander
import rospy


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("simple_moveit_control", anonymous=True)

    try:
        arm = moveit_commander.MoveGroupCommander("arm")

        # 速度和加速度缩放范围为 0~1，首次测试建议使用较小值。
        arm.set_max_velocity_scaling_factor(0.2)
        arm.set_max_acceleration_scaling_factor(0.2)
        arm.set_planning_time(5.0)

        # 保持其他关节不变，只把 joint1 移动到 0.2 rad。
        joint_goal = arm.get_current_joint_values()
        if len(joint_goal) != 6:
            raise RuntimeError(
                "arm 规划组应包含 6 个关节，实际读取到 {} 个".format(len(joint_goal))
            )

        joint_goal[0] = 0.2
        arm.set_start_state_to_current_state()
        arm.set_joint_value_target(joint_goal)

        # go() 会让 MoveIt 规划轨迹并等待轨迹执行完成。
        success = arm.go(wait=True)
        arm.stop()

        if not success:
            rospy.logerr("MoveIt 规划或执行失败")
            return 1

        rospy.loginfo("运动完成，当前关节角：%s", arm.get_current_joint_values())
        return 0
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass

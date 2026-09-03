cmake_minimum_required(VERSION 3.0.2)
project(moveit_ctrl)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  moveit_commander
  moveit_msgs
  roscpp
  moveit_ros_planning_interface
  sensor_msgs
  message_generation
  rospy
)

# 添加服务文件
add_service_files(
  FILES
  JointMoveitCtrl.srv  # 这里是你的自定义服务文件
)

# 生成消息和服务文件
generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS moveit_msgs roscpp rospy moveit_ros_planning_interface sensor_msgs message_runtime
)

# 包含头文件
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 安装 Python 脚本
catkin_install_python(PROGRAMS
  scripts/joint_moveit_ctrl_server.py
  scripts/simple_moveit_control.py
  scripts/simple_moveit_ik_control.py
  scripts/moveit_draw_circle.py
  scripts/moveit_three_segments.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

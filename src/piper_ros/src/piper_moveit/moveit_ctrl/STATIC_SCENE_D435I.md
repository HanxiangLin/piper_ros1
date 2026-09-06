# D435i eye-in-hand 静态预扫描与 MoveIt OctoMap

本流程在手眼标定独立验证通过、`camera_extrinsics.launch` 已生成后使用。它不会自动控制机械臂，
不会持续实时避障。操作者明确开启扫描时，RealSense 点云才进入 MoveIt；明确停止后，
MoveIt 内部 OctoMap 保持冻结，供后续 OMPL 规划做碰撞检查。

## 1. 已实现的链路

```text
RealSense PointCloud2（带原始时间戳和 optical frame）
  → static_scan_gate.py（默认关闭，不改变点云 header）
  → /piper/static_scan/points
  → occupancy_map_monitor/PointCloudOctomapUpdater
  → base_link 中 0.02 m 分辨率的 MoveIt OctoMap
  → Planning Scene 碰撞检查
```

配置位置：

- `piper_with_gripper_moveit/config/sensors_3d.yaml`：点云更新器、2 m最大量程、3 cm自体过滤膨胀。
- `piper_with_gripper_moveit/launch/sensor_manager.launch.xml`：`base_link`地图坐标、2 cm体素。
- `moveit_ctrl/launch/static_scene_mapping.launch`：只启动扫描门；不启动相机、MoveIt或机械臂。
- `static_scan_control.py`：`clear/start/stop/status`。
- `static_scene_snapshot.py`：保存、恢复 MoveIt 内部 OctoMap。

当前相机和安装支架尚未加入机械臂碰撞模型。扫描功能可以验证，但在实机自动规划执行前，
必须根据实际支架和相机尺寸添加碰撞几何与合理余量。不要用猜测尺寸替代实测。

## 2. 启动顺序

修改传感器配置后，旧的 `move_group` 不会热加载，必须停止原来的 `piper_real.launch`，
在确认急停可用、机械臂可以重新使能后手动重新启动。每个终端先运行：

```bash
source /opt/ros/noetic/setup.bash
source /home/hank/piper_ws/devel/setup.bash
```

终端一重新启动实机与 MoveIt：

```bash
roslaunch piper_with_gripper_moveit piper_real.launch \
  can_port:=can0 auto_enable:=true
```

终端二启动 D435i 点云。相机驱动已经运行时不要重复启动，应先停止旧相机 launch：

```bash
roslaunch moveit_ctrl eye_in_hand_camera.launch enable_pointcloud:=true
```

终端三加载已经通过独立验证的 eye-in-hand 外参：

```bash
export PIPER_CALIB_RUN=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02
roslaunch "$PIPER_CALIB_RUN/camera_extrinsics.launch"
```

终端四启动扫描门。该节点启动后仍为关闭状态：

```bash
roslaunch moveit_ctrl static_scene_mapping.launch
```

如果实际点云话题不是默认 `/camera/depth/color/points`，先查看：

```bash
rostopic list | grep points
```

然后明确传入，例如：

```bash
roslaunch moveit_ctrl static_scene_mapping.launch \
  input_cloud:=/实际的点云话题
```

检查 MoveIt 与扫描门：

```bash
rosservice list | grep -E 'clear_octomap|piper_static_scan_gate'
rosrun moveit_ctrl static_scan_control.py status
```

扫描未开始时，状态中的 `enabled` 应为 `false`。如果 `received` 始终为0，输入点云话题不对
或相机没有发布点云。`forwarded` 为累计转发数量，`session_forwarded` 是本次 start 后的数量。

## 3. 建立一张新地图

先让机械臂位于已确认安全、能看到工作区的初始姿态。预扫描前没有完整障碍地图，
所以所有扫描换姿必须由操作者确认真实空间安全；不能把未知区域当作无障碍。

按顺序执行：

```bash
rosrun moveit_ctrl static_scan_control.py stop
rosrun moveit_ctrl static_scan_control.py clear
rosrun moveit_ctrl static_scan_control.py start
```

`clear` 只清除 MoveIt 内存中的 OctoMap，不删除磁盘文件，也不会启动或移动机械臂。
`start` 不会自动清图，防止误操作毁掉已有地图。

打开扫描后，使用已经验证的 RViz/MoveIt控制方式低速到达多个安全视角。每个姿态必须先收到
`FollowJointTrajectory` 成功反馈并停稳，再保持观察数秒。尽量覆盖工作区的前后、左右、上下表面，
但不进入未确认安全的区域。不要移动桌面和障碍物。

点云的原始时间戳和 frame 会被保留，MoveIt 使用对应时刻的 TF 转到 `base_link`。
扫描门拒绝零时间戳、过旧、超前或无 frame 的点云，不使用“最新 TF”冒充采集时刻。

扫描过程中可检查：

```bash
rosrun moveit_ctrl static_scan_control.py status
rostopic hz /piper/static_scan/points
```

扫描完成后立即冻结：

```bash
rosrun moveit_ctrl static_scan_control.py stop
rosrun moveit_ctrl static_scan_control.py status
```

确认 `enabled=false` 且 `session_forwarded` 大于0。停止后即使原始相机继续发布，MoveIt也不会
接收新的扫描点云；现有OctoMap保持在当前 `move_group` 内存中，直到清空或重启。

## 4. 在 RViz 检查 Planning Scene

在 RViz 中设置：

```text
Global Options → Fixed Frame = base_link
MotionPlanning → Scene Geometry → Show Scene Geometry = true
```

检查 OctoMap 体素：

- 桌面、墙、纸箱等固定障碍位置正确，没有明显双层重影。
- 机械臂自身没有被大量写入地图；`/piper/static_scan/filtered_points` 可辅助检查自体过滤。
- 工作区获得足够多视角覆盖，遮挡和相机视野外的区域不能假设为空闲。
- 透明、镜面、黑色吸光和过远物体可能漏测，应添加几何碰撞物或禁止进入相关区域。

如果地图错误，执行 `stop`、`clear`，修正问题后重新扫描。不要在错误地图上规划实机运动。

## 5. 保存和恢复静态 OctoMap

停止扫描后保存。工具拒绝覆盖已有快照：

```bash
export PIPER_STATIC_MAP=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02/static_scene.bag
export PIPER_SCENE_PKG="$(rospack find moveit_ctrl)"

/usr/bin/python3 "$PIPER_SCENE_PKG/scripts/static_scene_snapshot.py" save \
  --output "$PIPER_STATIC_MAP"
```

快照只保存 `base_link` 中的 OctoMap，不保存机器人关节状态，也不替代普通 CollisionObject。

重启 MoveIt 后，先保证扫描门处于关闭状态，再恢复。`--replace` 是必须的显式确认，
它表示先清空当前 MoveIt OctoMap，再加载快照：

```bash
rosrun moveit_ctrl static_scan_control.py stop

/usr/bin/python3 "$PIPER_SCENE_PKG/scripts/static_scene_snapshot.py" load \
  --input "$PIPER_STATIC_MAP" \
  --expected-frame base_link \
  --replace
```

加载后重新在 RViz 检查 Scene Geometry。环境、机械臂基座或相机安装发生变化时，旧快照失效，
不得继续用于实机避障；重新清图和扫描。

## 6. 规划与执行边界

先只点击 RViz 的 `Plan`，检查完整轨迹和 Planning Scene，确认后才允许低速执行。
OMPL的普通关节/位姿目标规划会针对当前OctoMap寻找无碰撞路径；
`compute_cartesian_path(..., avoid_collisions=True)`只检查笛卡尔插值路径，遇到障碍通常截短或失败，
不会主动绕弯。

冻结地图不是实时避障：扫描后移动、新增的物体、人员和漏测区域不会自动更新。
OctoMap的2 cm体素与3 cm过滤膨胀只是起始配置，不是安全认证。实际安全余量还要结合标定误差、
深度噪声、机械臂跟踪误差、负载变形和任务风险设置。

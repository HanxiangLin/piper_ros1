# D435i eye-in-hand 静态预扫描与 MoveIt OctoMap

本流程在手眼标定独立验证通过、`camera_extrinsics.launch` 已生成后使用。它不会自动控制机械臂，
不会持续实时避障。操作者明确开启扫描时，RealSense 点云才进入 MoveIt；明确停止后，
MoveIt 内部 OctoMap 保持冻结，供后续 OMPL 规划做碰撞检查。

## 0. 操作路线图

只按当前所处状态选择一条路线：

```text
还没有地图
  → 第2节启动四组节点
  → 第3节 clear / start / 多视角扫描 / stop
  → 第4节检查地图
  → 第5节保存

刚刚保存成功，move_group 没有重启
  → 不要 load
  → 第5.1节检查当前内存地图
  → 第6节只做规划验证

以后重启了 move_group
  → 不必重新扫描（环境和安装完全没变时）
  → 第5.2节从 static_scene.bag 恢复
  → 第5.1节再次检查
  → 第6节只做规划验证
```

`save`只是复制当前地图到磁盘，不会从MoveIt删除地图。因此刚刚保存后直接继续检查和规划，
不能马上执行`load`；`load --replace`用于未来重启后恢复，会先清空当时的内存地图。

## 1. 已实现的链路

```text
RealSense PointCloud2（带原始时间戳和 optical frame）
  → static_scan_gate.py（默认关闭，不改变点云 header）
  → /piper/static_scan/points
  → occupancy_map_monitor/PointCloudOctomapUpdater
  → MoveIt规划根 dummy_link 中 0.02 m 分辨率的OctoMap
  → Planning Scene 碰撞检查
```

配置位置：

- `piper_with_gripper_moveit/config/sensors_3d.yaml`：点云更新器、2 m最大量程、3 cm自体过滤膨胀。
- `piper_with_gripper_moveit/launch/sensor_manager.launch.xml`：`dummy_link`地图坐标、2 cm体素。
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

终端二启动 D435i 点云。相机驱动已经运行时不要重复启动，应先停止旧相机 launch。
作用是：启动RealSense D435i驱动，并按照标定和静态扫描所需的分辨率发布彩色图、深度图、相机内参及相机内部TF。

```bash
roslaunch moveit_ctrl eye_in_hand_camera.launch enable_pointcloud:=true
```

终端三加载已经通过独立验证的 eye-in-hand 外参。
注：该launch文件位于普通目录或动态生成目录所以需要"roslaunch /完整路径/文件.launch"来启动。
作用：发布"link6 → camera_link"的固定 TF 

```bash
export PIPER_CALIB_RUN=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02
roslaunch "$PIPER_CALIB_RUN/camera_extrinsics.launch"
```

终端四启动扫描门。该节点启动后仍为关闭状态。
作用：这个launch不是“地图程序”，而是MoveIt静态建图过程中的“受控点云阀门”：打开时积累地图，关闭时冻结地图。

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

点云的原始时间戳和 frame 会被保留，MoveIt 使用对应时刻的 TF 转到规划根 `dummy_link`。
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

## 5. 保存、检查和恢复静态 OctoMap

停止扫描后保存。工具拒绝覆盖已有快照。
注：
【1】 /usr/bin/python3 表示使用ROS Noetic系统Python
【2】 如果 static_scene.bag 已经存在，必须换一个文件名，例如：
export PIPER_STATIC_MAP=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02/static_scene_02.bag
作用：保存扫描融合完成后的最终OctoMap快照。

```bash
export PIPER_STATIC_MAP=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02/static_scene.bag
export PIPER_SCENE_PKG="$(rospack find moveit_ctrl)"

/usr/bin/python3 "$PIPER_SCENE_PKG/scripts/static_scene_snapshot.py" save \
  --output "$PIPER_STATIC_MAP"
```

快照只保存MoveIt规划根`dummy_link`中的OctoMap，不保存机器人关节状态，也不替代普通CollisionObject。
Piper URDF通过零平移、零旋转的固定关节连接`dummy_link → base_link`，两者坐标值完全重合；
RViz继续使用`base_link`作为Fixed Frame即可。

### 5.1 情况一* 刚刚保存成功：检查当前内存地图

保持原来的 `move_group` 运行，扫描门保持关闭。此时不运行 `load`，只读检查：

```bash
export PIPER_SCENE_PKG="$(rospack find moveit_ctrl)"

rosrun moveit_ctrl static_scan_control.py status

/usr/bin/python3 "$PIPER_SCENE_PKG/scripts/static_scene_snapshot.py" inspect
```

第一条应显示 `enabled=false`。第二条应明确显示：

```text
MoveIt OctoMap：非空
frame: dummy_link
resolution: 0.020 m
```

`serialized bytes`只说明快照数据非空，不是障碍体素数量。若显示空地图，不进入第6节；重新检查
`session_forwarded`、点云话题、MoveIt启动日志和RViz Scene Geometry。

然后在RViz设置：

```text
Global Options → Fixed Frame = base_link
MotionPlanning → Scene Geometry → Show Scene Geometry = true
MotionPlanning → Planning Request → Planning Group = arm
MotionPlanning → Planning Request → Start State = Current
```

地图应与刚才扫描时相同。这里看到的是MoveIt内部Planning Scene，不是单独添加的实时PointCloud2。

### 5.2 情况二* 仅在以后重启 MoveIt 后恢复

重启 MoveIt 后，先保证扫描门处于关闭状态，再恢复。`--replace` 是必须的显式确认，
它表示先清空当前 MoveIt OctoMap，再加载快照：

```bash
rosrun moveit_ctrl static_scan_control.py stop

export PIPER_SCENE_PKG="$(rospack find moveit_ctrl)"
export PIPER_STATIC_MAP=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02/static_scene.bag
/usr/bin/python3 "$PIPER_SCENE_PKG/scripts/static_scene_snapshot.py" load \
  --input "$PIPER_STATIC_MAP" \
  --expected-frame dummy_link \
  --replace
```

加载后重新在 RViz 检查 Scene Geometry。环境、机械臂基座或相机安装发生变化时，旧快照失效，
不得继续用于实机避障；重新清图和扫描。

## 6. 规划与执行边界

### 6.1 当前只做规划，不执行

保持扫描门 `enabled=false`，不要关闭当前 `move_group`。在RViz的MotionPlanning面板中：

1. `Planning Group`选择`arm`，`Start State`选择`Current`。
2. 用末端交互标记选择一个明确位于空闲区、机械臂可达的目标。
3. 点击`Plan`，不要点击`Plan & Execute`，也不要点击`Execute`。
4. 展开显示轨迹，逐段观察机器人模型、末端和各连杆是否穿过OctoMap体素。
5. 再选择一个位于已扫描障碍物内部的目标，只点击`Plan`；它应规划失败。
6. 选择障碍物另一侧的可达自由目标；如果存在足够间隙，OMPL应产生绕行轨迹，否则应失败。

目标在自由区仍失败不一定是地图故障，也可能是IK不可达、关节限位或没有足够绕行空间。
目标在障碍物内部却成功，或显示轨迹穿过体素，禁止实机执行并检查Planning Scene。

### 6.2 实机执行前仍缺少一项

当前D435i和实际安装支架还没有加入机器人碰撞模型。在提供并配置相机/支架相对`link6`的
实际包络尺寸和位置之前，只做上述`Plan`检查，不进入实机自动执行。否则MoveIt可能成功避开裸机械臂，
但相机或支架撞上障碍物。

OMPL的普通关节/位姿目标规划会针对当前OctoMap寻找无碰撞路径；
`compute_cartesian_path(..., avoid_collisions=True)`只检查笛卡尔插值路径，遇到障碍通常截短或失败，
不会主动绕弯。

冻结地图不是实时避障：扫描后移动、新增的物体、人员和漏测区域不会自动更新。
OctoMap的2 cm体素与3 cm过滤膨胀只是起始配置，不是安全认证。实际安全余量还要结合标定误差、
深度噪声、机械臂跟踪误差、负载变形和任务风险设置。

## 7. RViz规划验证通过后：用脚本提交目标

`moveit_static_scene_goal.py`使用OMPL针对当前冻结OctoMap规划一个末端平移目标，保持末端姿态不变。
它先检查地图非空、frame为MoveIt规划根`dummy_link`、扫描门已经关闭，并在规划前后比较地图哈希。
默认只规划并把同一条轨迹发布到RViz，不向机械臂发送命令。

先选择一个已经在RViz验证过的相对位移。例如沿MoveIt规划坐标系X方向移动3 cm：

```bash
rosrun moveit_ctrl moveit_static_scene_goal.py \
  _dx:=0.03 _dy:=0.0 _dz:=0.0 \
  _execute:=false
```

终端会打印实际规划坐标系、当前末端位置、目标位置、规划器、轨迹点和预计时长。
在RViz观察`Display Planned Path`。如果障碍物阻断直线路径但存在绕行空间，RRTConnect可以给出绕行；
目标不可达、位于障碍物中或不存在无碰撞通道时，脚本应失败且不会运动。

相对位移长度限制为5 mm～50 cm，因此可以使用`_dx:=0.30`进行30 cm的只规划避障测试。
`dx/dy/dz`使用终端打印的MoveIt规划坐标系轴，不能把相机光学轴当作它们。较大的位移并不保证
目标可达；仍须确保目标和完整轨迹都位于已经扫描、现实中确认安全的区域。

你已经明确暂时不加入相机/支架碰撞模型。如果仍要执行同一轮新规划，必须同时打开两项显式开关：

```bash
rosrun moveit_ctrl moveit_static_scene_goal.py \
  _dx:=0.03 _dy:=0.0 _dz:=0.0 \
  _velocity_scale:=0.03 _acceleration_scale:=0.03 \
  _execute:=true \
  _acknowledge_missing_camera_collision:=true
```

脚本规划并在RViz显示后会停在终端等待。检查现场、OctoMap和这一次显示的轨迹；只有输入完全一致的：

```text
EXECUTE
```

才会调用`arm.execute(trajectory, wait=True)`。它等待MoveIt ExecuteTrajectory Action，进而等待
Piper `FollowJointTrajectory`服务端的完成结果；没有用延时冒充完成反馈。确认等待期间如果扫描门开启
或OctoMap内容改变，执行会被拒绝。

即使提供确认参数，缺少相机/支架碰撞模型的风险仍然存在：MoveIt只保证当前机器人模型和OctoMap
之间的碰撞检查，不会凭确认参数获得相机外形。第一次只用低速、小位移、急停可用并由人员全程观察。

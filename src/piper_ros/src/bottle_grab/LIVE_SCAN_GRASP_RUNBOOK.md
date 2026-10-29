# 实时扫描抓取 —— 命令行操作手册

配套说明看 [`LIVE_SCAN_GRASP.md`](LIVE_SCAN_GRASP.md)（流程、参数含义、失败模式）。
本文件只给可以直接复制执行的命令。

---

## 0. 每个新终端先执行的公共设置

```bash
export WS=/home/hank/piper_ws
export BG=$WS/src/piper_ros/src/bottle_grab
export PY=$WS/.venv-vision/bin/python            
export PARAMS=$BG/moveit_grab_params.yaml

source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash
```

> 不要用 `/usr/bin/python3` 或 conda base 跑抓取脚本：前者没有 `pyrealsense2/ultralytics`，
> 后者没有 `rospy`。`.venv-vision` 是当前唯一三者齐全的解释器（已用它跑通 54 个离线测试）。
> `bottle_grab` 不是 catkin 包，只能用 `python moveit_grab.py` 的方式启动，不能 `rosrun`。

---

## 1. 开机前物理检查

- 急停按钮在手可及范围内。
- 机械臂停在**低姿态、安全位**（程序第一段运动没有任何地图保护，是从当前位抬到扫描起始位）。
- 瓶子和卡片都放在桌面上，**中心离桌边 ≥ 6cm**，两者间距约 45cm。
- 拍摄视野内不要有反光镜面、透明物、悬空的架子（相机/支架没有碰撞模型，见说明文档第 6 节）。
- 扫描期间不要有人或物进出工作区。

---

## 2. CAN 接口

```bash
ip -details link show can0
```

只有在 `can0` 不存在时才需要激活（确认系统只识别到一个 Piper 的 USB-CAN 适配器）：

```bash
bash $WS/src/piper_ros/can_activate.sh can0 1000000
ip -details link show can0        # 应显示 state UP
```

---

## 3. 终端 1：机械臂 + MoveIt（不启动相机）

```bash
export WS=/home/hank/piper_ws
source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash

roslaunch piper_with_gripper_moveit piper_real.launch \
  can_port:=can0 auto_enable:=false use_rviz:=true
```

- `auto_enable:=false` 不会自动请求电机使能。需要运动前先使能：

  ```bash
  rosservice call /enable_srv "{enable_request: true}"
  ```

  确认方式：`rostopic echo -n1 /joint_states` 的数值会跟随手动推动机械臂变化。
- 等到日志出现 `You can start planning now!`。
- 如果刚改过 `piper_with_gripper_moveit/config/kinematics.yaml`，必须重启
  `move_group`；启动后确认真实生效值是 50ms：

  ```bash
  rosparam get /robot_description_kinematics/arm/kinematics_solver_timeout
  # 期望输出 0.05；若仍是 0.005，说明加载的是别的 config 或没重启。
  ```
- **静态障碍地图依赖 `move_group` 里的 OctoMap 监控**。`sensors_3d.yaml`
  （量程 2m / 更新 10Hz / 自体过滤 3cm）和 `sensor_manager.launch.xml`（体素 2cm）
  改动后**必须重启 `piper_real.launch`**，不会热加载。启动后确认：

  ```bash
  rosparam get /move_group/octomap_resolution       # 期望 0.02
  rosservice list | grep -E 'clear_octomap|get_planning_scene'
  ```

  如果 `move_group` 启动时没加载 sensors（`allow_trajectory_execution=false`），
  扫描会在第一视点报“发布点云后OctoMap未变化”。
- **不要**再启动 `eye_in_hand_camera.launch`、`realsense2_camera`、`static_scene_mapping.launch`：
  新流程自己用 SDK 打开 D435i，相机节点占用设备会直接报错。相机
  `link6 → camera_link → camera_color_optical_frame` 静态 TF 由程序按标定文件
  自动发布（如果 TF 里已有则沿用），所以也不需要单独启动
  `camera_extrinsics.launch`。

---

## 4. 终端 2：接口自检（跑之前 30 秒做完）

```bash
export WS=/home/hank/piper_ws
export BG=$WS/src/piper_ros/src/bottle_grab
export PY=$WS/.venv-vision/bin/python
source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash

# 4.1 相机不能被 ROS 节点占用（输出应为空）
rosnode list | grep -E "realsense|rs_camera|camera$"

# 4.2 关键服务齐全（应列出 6 行）
rosservice list | grep -E "^/(get_planning_scene|apply_planning_scene|check_state_validity|gripper_srv|clear_octomap|enable_srv)$"

# 4.3 关节反馈完整（应有 joint1..joint8）
rostopic echo -n1 /joint_states | head -25

# 4.4 相机能被 SDK 打开（应打印 Intel RealSense D435i）
$PY -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().query_devices()])"

# 4.5 遗留扫描门必须是关闭的（没装该节点也算正常）
rosrun moveit_ctrl static_scan_control.py status 2>/dev/null || echo "扫描门节点未运行（正常）"

# 4.6 参数文件包含新键
grep -E "scan_mode|scan_camera_frame|scan_joint1_offsets|scan_joint2_offsets|scan_map_base_keepout_radius|place_offset_from_card|home_joint_target" $PARAMS

# 4.7 IK超时生效值（A阶段诊断项，必须是0.05）
rosparam get /robot_description_kinematics/arm/kinematics_solver_timeout

# 4.8 成功率测试阶段先确认 dry_run 为 true
grep -E "^dry_run:" $PARAMS
```

如果 4.1 有输出，先关掉那个终端的相机节点；如果是别的用户启动的：

```bash
rosnode kill /camera/realsense2_camera    # 节点名以 rosnode list 的实际输出为准
```

---

## 5. 终端 3：正式运行

```bash
export WS=/home/hank/piper_ws
export BG=$WS/src/piper_ros/src/bottle_grab
export PY=$WS/.venv-vision/bin/python
source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash

cd $BG
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml
```

启动即开始动作：清场 → 抬臂 → 5 个偏航视点扫描（每视点直接建静态障碍地图）
→ 冻结 → 抓取 → 放置 → 回零。
预计抓取开始前 60~90 秒（每个视点约 6~8 秒 + 地图稳定 1~2 秒），放置再加 20~30 秒。

扫描阶段的碰撞场景现在是：

| 对象 | 来源 | 说明 |
|---|---|---|
| `table` | 拟合平面 | 顶面**精确**等于支撑平面（不再抬 2cm） |
| `target_bottle` | 圆柱拟合 | 指尖可以接触；其余一律禁止 |
| `<octomap>` | 每视点发布 | 桌面/墙/柜子/**其它瓶子**等扫描到的一切 |

即“除目标瓶和支撑面外，全部是障碍”。旧的语义障碍盒（`scan_obstacle_*`）默认不再创建。

### 5.1 临时覆盖单个参数（不改文件）

rospy 会解析命令行的 `_参数:=值`，优先级高于 `_params_file`：

```bash
# 只用三个偏航视点，扫描更快（瓶子两侧仍能看到时）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_joint1_offsets:="[0.0, 0.4, -0.4]"

# 加一档俯仰，覆盖更高/更矮的障碍（先空跑确认方向）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_joint2_offsets:="[0.0, -0.25]"

# 只抓不放，抓完抬升即停（此时不会回零）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _enable_place:=false

# 卡片判别放宽（卡片只露出一部分时）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_card_min_extent:=0.08

# 撒手点再远 2cm（卡片更大/更斜时）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _place_offset_from_card:=0.13

# 卡片盒跟着实物收紧 / 放宽（默认 5mm 外扩）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_card_padding:=0.008

# 接近走廊放宽（瓶子和卡片靠得比较近时）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_corridor_radius:=0.04

# 接近方向候选：相对“基座->瓶子”径向，0=正对瓶子，正值绕base Z逆时针
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _approach_yaw_offsets_deg:="[0.0, 15.0, -15.0]"

# 放置方向候选：相对“基座->放置点”径向；0=径向
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _place_yaw_offsets_deg:="[0.0, 15.0, -15.0, 30.0, -30.0]"

# 地图把抓取通道封死时：先缩小支撑面挖除厚度/加大底座余量排查
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_map_support_carve:=0.020

# 回滚到旧语义障碍（只用于对比，不建地图）
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _scan_mode:=semantic

# 关掉 OpenCV 检测窗口
$PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml \
  _show_detection_window:=false
```

如果不想开 RViz，终端 1 改用：

```bash
roslaunch piper_with_gripper_moveit piper_real.launch \
  can_port:=can0 auto_enable:=false use_rviz:=false
```

### 5.2 A 阶段：IK / 接近成功率测试（不动机器人）

`dry_run` 只跳过抓取执行，扫描阶段仍会走到全部视点（并建图）。固定瓶子位置后连续跑
5~10 次：

```bash
cd $BG
for i in $(seq 1 10); do
  echo "===== run $i ====="
  $PY moveit_grab.py _params_file:=$BG/moveit_grab_params.yaml _dry_run:=true
done
```

每次重点看：

- `就近IK关节目标: ...`：每个候选各自的关节目标及限位余量；
- `候选通过: 偏转... 代价...`：该候选的预抓取 + 直线接近都通过，
  `0度` 现在是真正的“基座→瓶子”径向接近；
- `选择候选: ... 有效候选N/M`：程序规划完所有候选后，按代价选出的最终解；
  其余有效候选的代价也会打印，便于判断姿态是否自然；
- `直线接近诊断(原始...): ...`：失败时区分“第N步发生碰撞”还是
  “IK/关节限位问题”；
- `>>> dry_run: ...`：确认停在执行前。

判定标准：

- 10 次里 ≥8 次出现“选择候选”且出现 `dry_run` → A 阶段有效，再进入 B；
- 反复报“IK/关节限位问题” → 优先考虑 B（成对 IK 选优）或 TRAC-IK；
- 报“第N步发生碰撞” → 先查桌面碰撞盒、瓶子 ACM 和 cartesian_joint_step_limit，
  不要继续调 IK。

测完把参数文件里的 `dry_run` 改回 `false`，或正式运行时不要加 `_dry_run:=true`。

---

## 6. 正常运行的日志顺序

```text
RealSense初始化完成；depth scale=...
正在加载YOLO瓶子检测模型: ...
YOLO模型加载完成
扫描前清场完成：世界对象、ACM接触放开、OctoMap均已复位
>>> 张开夹爪
>>> 扫描：清场并移动到扫描起始位
视点0采集4帧，ROI内最多NNNNN点
>>> 检测场景中的全部物体
扫描检测到 2 个物体: bottle-b(0.9x), bottle-g(0.8x)
桌面拟合: 视点0 倾角x.xx°(门限5.00°) 内点NNNN/NNNN footprint=NNNN 范围focus 法向[..., ..., ...]
桌面已载入: x[...] y[...] 表面z=... 顶面=...
>>> 目标瓶已加入场景: 目标瓶 center=(...) ...
视点0障碍地图: 输入NNNN 底座挖除NN 目标挖除NN 支撑挖除NNN 降采样NNNN → 发布NNNN点
>>> 扫描视点1: joint1偏移+0.393 joint2偏移+0.000
...
>>> 扫描视点4: joint1偏移-0.785 joint2偏移+0.000
障碍地图已建立: frame=dummy_link 分辨率=0.020m 序列化NNNNNN字节
当前机械臂状态与障碍地图无碰撞
扫描结果: 桌面平面 z=... / 目标瓶 ... / 障碍物 0 个 / 卡片 ...
场景已冻结：桌面+目标瓶+0个语义障碍；OctoMap=非空
>>> 开始MoveIt视觉抓取任务
...
```

如果 `move_group` 没有消费点云，第一视点就会报
`发布点云后OctoMap未变化`，不要继续调抓取参数。

出现 `扫描未识别出卡片`、`未找到圆柱形目标瓶`、`没有可行的接近走廊` 时，程序在**任何运动之前**
就退出（扫描阶段除外），不会残留夹持状态。

---

## 7. 运行时盯 RViz

```text
Global Options → Fixed Frame = base_link
MotionPlanning → Scene Geometry → Show Scene Geometry = true
MotionPlanning → Planning Request → Planning Group = arm
Display → MotionPlanning → Planned Path → Show Trail = true
```

应该看到：

| 对象 | 形态 |
|---|---|
| `table` | 一个扁盒，**上表面精确等于拟合的支撑平面**（不再抬高 2cm） |
| `target_bottle` | 一个圆柱，中心在瓶子轴线上；指尖允许接触 |
| `<octomap>` | 彩色体素：桌面以外的墙、柜子、显示器、其它瓶子等**全部**扫描到的障碍 |

默认不再有 `scan_obstacle_*`（`scan_mode=semantic` 回滚时才会出现）。

如果 `target_bottle` 套在卡片上、或地图把瓶子/夹爪也包进去了，立刻 Ctrl-C，
看第 8 节的地图快照再调参数。

---

## 8. 扫描产物检查

每次运行都会落盘，失败后先看这里，不要靠猜。

```bash
export WS=/home/hank/piper_ws
export PY=$WS/.venv-vision/bin/python

d=$(ls -dt $WS/calibration/bottle_grasp_scan/scan_2*/ 2>/dev/null | head -1)   # 新目录是 scan_<日期>_<时间>
echo "$d"

cat "$d/summary.txt"

python3 -m json.tool "$d/table_model.json" | head -30
python3 -m json.tool "$d/scene.json" | head -80

# 每个视点实际发布给OctoMap的点（已挖掉底座/目标/支撑面）
ls "$d/map"
cat "$d/map/meta.json"
# 可恢复快照（与 static_scene_snapshot.py load 兼容）
ls -lh "$d/map/octomap_snapshot.bag"

$PY - <<PY
import numpy as np
data = np.load("$d/cloud.npz")
points, views = data["points"], data["observations"]
print("点数:", points.shape)
print("范围 min:", points.min(0).round(3), "max:", points.max(0).round(3))
print("每个体素被几个视点看到:", np.bincount(views))
view0 = np.load("$d/map/view_00.npz")
print("视点0发布点数:", len(view0["points_base"]))
print("视点0挖除统计在 meta.json；octomap.npz 里是冻结后的序列化八叉树")
PY
```

看彩色帧（每个视点一张）：

```bash
xdg-open "$d/view_0.png"
```

判读要点：

- `summary.txt` 里的桌面 z 是**桌面中心**处的平面高度，应和实际桌面相差 < 1cm。
- `table_model.json` 的 `fit_scope` 应为 `focus`；如果是 `global`，说明没拿到目标瓶
  检测点，桌面拟合退化成了全局拟合（大 ROI 下可能选到地板）。
- `bounds_max[2] - surface_top` 应约等于 `scan_table_surface_epsilon`（默认 2mm），
  不再包含 2cm 膨胀。
- `scene.json` 的 `card.center` 应在卡片位置，`card.width/length` 约 0.20/0.15。
- `map/meta.json` 里每个视点都应有 `published > 0`，且 `octomap.bytes` 不应小于
  `scan_map_min_bytes`。
- `target_center` 应与瓶轴重合；如果偏向相机一侧，说明把可见表面当成了轴线。
- 默认 `scene.json` 的 `obstacles` 为空——障碍现在全在 OctoMap 里。

---

## 9. 空桌面自检（验证相机 / 桌面拟合，不抓取）

第一次上机建议先不放瓶子：

```bash
cd $BG
$PY moveit_grab.py _params_file:=$PARAMS
```

预期在“检测场景中的全部物体”之后报：

```text
RuntimeError: 未找到圆柱形目标瓶: 簇0 kind=unknown ...
```

看到 `桌面已载入: ... 表面z=... 顶面=...` 与现场一致、`view_0.png` 能清楚拍到桌面，
就说明相机、深度、TF、清场、桌面拟合都正常。**注意：没有目标瓶时程序在发布第一幅
障碍地图之前就中止了**，所以这一步不验证 OctoMap 链路。此时**没有任何抓取动作**。

要验证“扫描 + 建图 + 规划”整条链路，放上瓶子和卡片后用 dry_run：

```bash
cd $BG
$PY moveit_grab.py _params_file:=$PARAMS _dry_run:=true
```

预期看到 5 个视点的 `视点N障碍地图: ...`、`障碍地图已建立`、
`当前机械臂状态与障碍地图无碰撞`，然后所有接近候选规划通过、日志出现
`dry_run: 预抓取与直线接近校验通过` 并退出（不会动夹爪、不会抬升）。

验证完再按第 5 节正式运行。

---

## 10. 常见报错与处理

| 报错 | 处理 |
|---|---|
| `RealSense启动失败；运行本脚本前停止ROS相机launch以释放设备` | 第 4.1 节检查并 `rosnode kill` 相机节点 |
| `缺少实测任务参数: ~scan_start_joint_target` | 参数文件没加载：确认 `_params_file:=` 路径存在 |
| `无法清空残留OctoMap` | 手动 `rosservice call /clear_octomap`；仍失败就重启 `piper_real.launch` |
| `规划场景中存在未知对象，拒绝清场` | 上一次手工在 RViz 加过东西；RViz 里删掉，或重启 `move_group` |
| `相机静态TF不可用` / `静态障碍地图需要相机TF` | 标定文件缺 `camera_link_T_optical`，或 TF 里已有冲突的 `link6→camera_link`；检查 `result.json` 与是否有别的节点在发相机TF |
| `发布点云后OctoMap未变化` | `move_group` 没加载 `sensors_3d.yaml`（需 `allow_trajectory_execution=true`），或 `camera_color_optical_frame → dummy_link` 不可达；重启 `piper_real.launch` 并确认 `rosservice list` 有 `clear_octomap` |
| `OctoMap在xxs内未稳定` | 点云更新器进程卡住/一直在收数据；`rostopic hz /piper/static_scan/points` 只应有扫描时发布，检查是否有遗留扫描门在转发 |
| `障碍地图为空` / `障碍地图过小` | 同上；另外确认 `scan_roi_min/max` 没把视野全裁掉 |
| `机器人状态无效/碰撞: [('link1','<octomap>')]`（建图后自检） | 底座余量不足：把 `_scan_map_base_keepout_radius` 从 0.16 加到 0.18~0.20；仍失败说明相机看到了机械臂自身，需在扫描位避开 |
| `扫描ROI内有效点不足` / `桌面平面占比不足` | 视点没看到桌面，或 ROI 混入地面/墙：调 `_scan_roi_min:="[0.2,-0.35,0.12]"` 之类 |
| `桌面平面拟合失败: 严格预筛: ... 宽松预筛: 精修倾角x.xx°>3.00°(内点N/M)` | 已自动用宽松预筛重试过。中间视点的门限是 `scan_table_max_tilt + 2°`，**最后一个视点**仍是严格的 `scan_table_max_tilt`；如果失败发生在最后视点，说明冻结给抓取的融合桌面确实倾斜，先调平桌面/机器人底座，不要靠调阈值绕过。看 `桌面拟合: 视点N 倾角...(门限...)` 即可定位 |
| `桌面拟合` 日志里 `范围global`（而不是 `范围focus`） | 第一视点没拿到瓶子检测点，RANSAC 退化到全局；大 ROI 下可能拟合到地板。先解决 YOLO 检测 |
| `未找到圆柱形目标瓶` | 点太少/聚类太碎：`_scan_min_cluster_points:=20`；或 `_scan_min_depth:=0.20` |
| `存在多个圆柱候选且无法用检测唯一确定目标` | 桌上有多个瓶子且YOLO分数接近：加 `_target_bottle_label:=bottle-g`（y/g/b），或移走多余瓶子 |
| `扫描未识别出卡片` | `_scan_card_min_extent:=0.08`，或把卡片立正一点；卡片必须是立在支撑面上（`scan_card_max_top_height`） |
| `目标瓶在多视点融合中丢失` | 某个视点看不到瓶子：`_scan_joint1_offsets:="[0.0,0.35,-0.35]"` |
| `物体接触区域靠近table边界` | 松手点在**已观测到的**支撑面之外（现在 footprint 以瓶子为中心 0.6m）：缩小 `_place_offset_from_card`，或把扫描视点/ROI 调到能看到卡片外侧桌面 |
| `没有可行的接近走廊` | 日志里会写“被哪个 id 挡住、净距多少”；调 `_scan_corridor_radius:=0.04` |
| `没有有效的预抓取/直线接近候选: ...；预抓取位..度不可达/碰撞: KDL无IK解` | 只有**预抓取规划本身**失败时才出现：14 组初值都解不出，多半是真的不可达/关节限位。换 `_approach_yaw_offsets_deg`，或考虑 B 阶段/TRAC-IK |
| `直线接近诊断(原始0.xxx): 第N/M步发生碰撞: [('link6','table')]` | 直线段本身 IK 可达，是碰撞：按接触对检查桌面碰撞盒、瓶子 ACM、`_cartesian_joint_step_limit`，不要继续调 IK |
| `直线接近诊断(原始0.xxx): 无碰撞复算仍只完成0.xxx，属于IK/关节限位问题` | 关掉碰撞检查也走不完：是 KDL/关节限位/奇异。先看日志里的 `余`（限位余量）；最紧的通常是 joint5。可试 `_grasp_height_from_table:=0.09` 或进入 B 阶段 |
| `MoveIt规划失败；预放置位不可达/碰撞: ...` | 预放置位被挡：加大 `_place_offset_from_card`，或检查 `scan_obstacle_card` 是否与实物一致 |
| `直线路径只完成..%，改用MoveIt规划到同一目标位`（WARN） | 正常兜底，不是错误：下降直线被卡片/桌面截断时自动改走规划路径，松手位姿不变 |
| `直线路径受阻或不可达，完成比例 ...；放置位不可达/碰撞: ...` | 连规划都到不了松手位：加大 `_place_offset_from_card`（每次 +0.02），或降低 `_place_approach_height` |
| `多视点融合后目标瓶位置跳变` | 视点间看到的是不同物体：检查是否有人/物移动，或减少视点 |
| `环境/规划场景在任务中改变` | 有人动了场景，或遗留扫描门在发点云：`rosrun moveit_ctrl static_scan_control.py stop` |
| `夹持开口/负载未达到实测阈值` | 瓶子位置偏了或抓取高度不对；程序停住不提升，先人工核对再复位 |
| `lift_distance不足以让瓶底离开table碰撞盒` | 桌面变高：增大 `_lift_distance:=0.05`（确认垂直可达） |

排查完，最省事的复位是重启 `piper_real.launch`（场景对象和 OctoMap 都在 move_group 内存里）。

---

## 11. 失败后的手动复位

程序**不会**自动松爪、detach 或回零。确认现场安全后按需执行：

```bash
# 张开夹爪（瓶子会掉落，下面必须有人/软垫）
rosservice call /gripper_srv "{gripper_angle: 0.07, gripper_effort: 1.0, gripper_code: 1, set_zero: 0}"

# 机械臂回零位（会运动！先确认路径）
rosservice call /go_zero_srv "{is_mit_mode: false}"

# 清空 OctoMap（不会移动机械臂）
rosservice call /clear_octomap
```

如果程序在“已附着瓶子”状态下退出，场景里会保留 AttachedCollisionObject，
直接重启程序会被 `target_bottle仍附着在link6` 拒绝。此时：

```bash
# 最干净的做法：重启 move_group（场景只在内存里）
# 终端1 Ctrl-C 后重新 roslaunch piper_real.launch ...
```

---

## 12. 改完代码后的回归测试（不需要 ROS master）

```bash
export WS=/home/hank/piper_ws
export PY=$WS/.venv-vision/bin/python
source /opt/ros/noetic/setup.bash
source $WS/devel/setup.bash

cd $WS/src/piper_ros/src/bottle_grab
$PY -m unittest discover -s tests -v
```

纯几何部分不依赖 ROS，可以单独快速跑：

```bash
$PY -m unittest tests.test_scan_geometry -v
```

---

## 13. 收尾

```bash
# 抓取程序：Ctrl-C
# 它会 stop 相机、关闭 joint_states 订阅；不会自动回零
# piper_real.launch：Ctrl-C
```

结束后确认：

- 机械臂停在零位或安全位；
- 夹爪里没有夹着东西；
- 桌面上的瓶子位置就是程序放下的位置（世界模型里也保留这个圆柱）。

---

## 14. 本次已知边界（贴在这里提醒操作者）

- 第一段“当前位 → 扫描起始位”是 OMPL 规划，**没有地图保护**。
- 障碍地图是**静态快照**：只包含扫描时各视点实际看到的面。遮挡区、背面、2m 以外、
  以及扫描结束之后才出现/移动的障碍都**不会被避开**。
- 扫描时进入视野的人手/工具/移动物会作为障碍永久留在快照里（不会自动消失）；
  重新跑一次扫描才能清除。
- 底座附近 0.16m 圆柱被挖掉（底座自身半径 13cm + 3cm 余量）；相机/支架没有碰撞几何。
- 手眼标定残差仍有 ~12mm，地图里的障碍位置和桌面高度就带着约 1cm 的误差。
- 默认 `scan_mode=map` 不再生成 `scan_obstacle_*` 语义盒；卡片只作为**放置几何模型**，
  它的碰撞由 OctoMap 负责。
- 支撑面用“顶面精确的 table 盒”表示，桌面顶面 ±1.5cm 的体素从地图里挖掉；
  贴桌面放置的薄物体（<1.5cm）会因此看不到，需要时用 `scan_map_support_carve` 调整。

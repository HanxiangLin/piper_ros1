# 瓶子抓取接入 MoveIt：桌面避障与目标接触

核对日期：2026-10-13。

分析的原脚本：`/home/hank/competition_ws/piper_ws/src/piper_ros/src/xiaosai/grab2016_5_28_3.py`。
使用的 MoveIt 工作空间：`/home/hank/piper_ws`，ROS1 Noetic。

本文提供修改步骤和可复制的核心函数，没有改动原比赛脚本，也没有执行实机运动。
代码中的桌面、瓶子、工具变换、模型路径和夹持判断参数需要实测；这些片段不是已经验证过的实机成品。

## 1. 要实现的行为

目标检测 → 在 base_link 下生成抓取目标 → MoveIt 对桌面和固定障碍规划 →
到预抓取位 → 允许两根夹指接触指定瓶子 → 碰撞检查下短距离直线接近 →
闭合并确认夹持 → 把瓶子附着到 link6 → 竖直抬起 → 携带瓶子规划返回/放置。

桌面与机器人各 link 始终参与碰撞检查。只有目标瓶子与实际接触它的夹指允许接触。

当前脚本的 `move_to_target_smooth()` 只是直线位置插值和姿态 Slerp，发布 `/pin_pos_cmd`；
它没有请求 MoveIt。向 `/joint_states` 发布初始姿态的代码也绕过规划。
因此，单独启动 MoveIt 或仅增加桌面模型都不会改变这些旧动作。

还有一个物理限制：最终抓取姿态如果让夹指穿入桌面，规划器只能拒绝。
需要提高真实抓取点、改变水平接近方向、改变夹爪姿态，或者先调整瓶子的摆放。

## 2. 选择一个 ROS 工作空间和一种相机使用方式

比赛工作空间中未找到当前实机避障入口 `piper_real.launch`。
推荐复制抓取脚本到已有避障工作空间，避免复制整套旧驱动覆盖新驱动。

```bash
mkdir -p /home/hank/piper_ws/src/piper_ros/src/bottle_grab
cp -n /home/hank/competition_ws/piper_ws/src/piper_ros/src/xiaosai/grab2016_5_28_3.py \
  /home/hank/piper_ws/src/piper_ros/src/bottle_grab/grab2016_5_28_3_moveit.py
```

如果该目标文件已经存在，`cp -n` 不覆盖；先查看它，再决定使用另一文件名。
后续修改的是新文件。原文件保留为备份。
每个终端使用同一个 ROS 环境；当前显示 `(base)` 时先退出 Conda：

```bash
conda deactivate
source /opt/ros/noetic/setup.bash
source /home/hank/piper_ws/devel/setup.bash
rospack find piper_with_gripper_moveit
rospack find piper
rospack find moveit_ctrl
```

输出都应指向 `/home/hank/piper_ws/src/...`。不要在这些终端再次 source 比赛工作空间。

原脚本通过 `rs.pipeline().start()` 直接占用相机，有两种可行方式：

- 保留直接 SDK 检测：先用 ROS 相机完成静态扫描，stop 冻结地图，然后在相机 launch 所在终端 Ctrl+C
  停止 ROS 相机、确认设备释放，再运行抓取脚本。保留 MoveIt、实机驱动和外参发布节点。
  仅关闭扫描门不会释放相机。脚本使用已验证的 link6 到 color optical frame 外参。
- 后续改成 ROS 图像输入：彩色图、对齐深度图和 CameraInfo 用时间同步方式接收，使用图像时间戳的 TF。
  删除直接 SDK 开流代码，ROS 相机可以继续运行，但抓取期间扫描门仍应关闭。

最少改动可先使用第一种方式。不要同时让两个程序打开同一相机。
原脚本 `MODEL_PATH` 指向 `/home/user/...`；本工作空间存在
`/home/hank/piper_ws/src/piper_ros/src/bottle_grab/bottle.pt`，但要确认它的类别和实际模型确实匹配。
必须用能够同时导入 MoveIt 和当前 YOLO/RealSense 依赖的 Python 环境。

## 3. 桌面建模，避免把目标瓶子扫描成障碍

推荐分两步接入：第一版把桌面和抓取区内的全部固定障碍建成命名几何对象，
使用现有 MoveIt 规划/执行链，在这一区域不依赖 OctoMap 的桌面体素。
这样支撑桌面、目标瓶子都能单独管理接触权限。
如果当前已经加载 OctoMap，关闭扫描门并清空前，先确认原地图内所有相关固定障碍
都有完整的几何替代；清空是清掉整张地图，不是只删瓶子，不能因此丢失其他障碍。

第二版再组合点云地图：移走所有待抓瓶子，扫描其余固定环境，stop 冻结地图，
然后把瓶子放回。桌面抓取区应由命名 box 可靠覆盖，进入地图前分割掉该区桌面点，
避免同一支撑面同时以 box 和不可单独管理接触的体素表示。
如果分割还不能可靠实现，先使用覆盖完整环境的几何对象版本。
对每个瓶子建立独立几何模型，只有当前目标获得夹指接触权限。

已有地图如果包含瓶子，单独给 `target_bottle` 设置 ACM 不会消除瓶子原位置的 OctoMap 体素。
那些体素属于 `<octomap>`，不属于命名对象。已有 `static_scene.bag` 不能未经检查直接复用。

如果瓶子不能移走，需在送入 OctoMap 前分割目标点云，或精确清理属于目标的体素。
清理范围不能挖掉桌面/桌边，普通 `PlanningSceneInterface.remove_world_object()` 不能删除 OctoMap 的局部区域。
不要设置“夹爪可以碰整个 OctoMap”，也不要关闭全部碰撞检查。

除了点云地图，建议增加有限尺寸桌面 box，使桌边和桌板厚度更可靠。
桌面 box 至少覆盖真实台面和桌板厚度；桌腿、横梁、挡板、桌裙按实际几何另加 box。
不能只添加一个无厚度平面而忽略桌边下方的结构。

所有桌面尺寸都以 base_link 表示。下例假设桌子边缘平行于基坐标系 X/Y；
如果桌子旋转，先用实测桌子坐标系与 TF 给 box 设置正确姿态。

```python
# 加入 PiperVisionController 类；单位 m。
def add_table(self, x_min, x_max, y_min, y_max, z_bottom, z_top,
              margin_xy, margin_z):
    if not (x_max > x_min and y_max > y_min and z_top > z_bottom):
        raise ValueError("桌面边界无效")
    if margin_xy < 0 or margin_z < 0:
        raise ValueError("桌面余量不能为负")
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.orientation.w = 1.0
    pose.pose.position.x = (x_min + x_max) / 2.0
    pose.pose.position.y = (y_min + y_max) / 2.0
    # 上下各膨胀 margin_z，因此中心保持在原桌板中心。
    pose.pose.position.z = (z_bottom + z_top) / 2.0
    self.scene.add_box(
        "table", pose,
        size=(x_max - x_min + 2 * margin_xy,
              y_max - y_min + 2 * margin_xy,
              z_top - z_bottom + 2 * margin_z))
    self.wait_object("table", known=True, attached=False)
```

余量根据外参误差、桌面测量误差、模型误差和轨迹跟踪误差选择；不能把任意固定厘米数当成保证。
如果机械臂固定在同一张桌上，box 不得错误地包住实际底座安装体积。
按安装结构建模，必要时只为真实固定接触的底座 link 与桌子建立窄范围例外；
手臂、手腕、夹指不能获得桌面碰撞例外。

## 4. 检查机器人碰撞模型

当前加载模型为：
`piper_description/urdf/piper_description.urdf`。
当前 SRDF：`piper_with_gripper_moveit/config/piper_description.srdf`。

- `arm` 是 base_link → link6；规划末端明确设为 `link6`。
- 两根夹指为 `link7`、`link8`；夹爪基座是 `gripper_base`。
- 三者当前都有 collision mesh。与桌子的检测仍要用已知碰撞状态验证，不能只看 RViz 的 visual。
- 相机、实际支架、加长夹指、线缆固定件需有相符的 collision 或附着碰撞包络。
- 反馈是 joint7=q、joint8=-q。joint7 范围 0～0.035 m，完整开口量约为 2q。

当前 URDF 的 joint8 没有 mimic。若用 MoveIt 的 gripper 规划开闭，
建议在 joint8 内添加以下标记，使预测的两根夹指对称运动；以正在加载的模型为准。

```xml
<mimic joint="joint7" multiplier="-1" offset="0"/>
```

本机驱动也发布 joint8=-joint7。修改 URDF 后重启相关 MoveIt/robot_state_publisher 节点，
检查运行时 robot_description 与 joint7/joint8 的模型、反馈一致。
不是改了磁盘文件就会自动热加载。

更完整的 SRDF 可把 gripper 声明为 end_effector，但本方案明确用 link6 规划，
并显式设置 `touch_links=["link7", "link8"]`，不依赖默认 end_effector 名称。
夹爪基座只有确实参与允许接触时才额外列入，不能放开整个机械臂与瓶子的碰撞。

## 5. 替换底层运动接口

新增 import：

```python
import copy
import sys
import moveit_commander
import tf2_ros
import tf2_geometry_msgs  # 注册 PoseStamped 的 TF 转换。
from moveit_msgs.msg import (
    DisplayTrajectory, PlanningScene, PlanningSceneComponents,
    AllowedCollisionEntry,
)
from moveit_msgs.srv import (
    GetPlanningScene, GetPlanningSceneRequest, ApplyPlanningScene,
    GetStateValidity, GetStateValidityRequest,
)
from piper_msgs.srv import Gripper
```

在 `rospy.init_node()` 之前调用：

```python
moveit_commander.roscpp_initialize(sys.argv)
```

删除发布 `/pin_pos_cmd`、发布 `/joint_states` 的 publisher；所有旧的 `publish()` 控制调用都替换。
删除等待 `/end_pose` 的代码。本方案不依赖旧 Pinocchio 节点。
初始化改为以下结构；原相机检测初始化仍保留：

```python
self.tf_buffer = tf2_ros.Buffer()
self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
self.robot = moveit_commander.RobotCommander()
self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
self.arm = moveit_commander.MoveGroupCommander("arm")
self.arm.set_pose_reference_frame("base_link")
self.arm.set_end_effector_link("link6")
self.arm.set_planner_id("RRTConnect")
self.arm.set_planning_time(10.0)
self.arm.set_num_planning_attempts(8)
self.arm.set_max_velocity_scaling_factor(0.03)
self.arm.set_max_acceleration_scaling_factor(0.03)
self.arm.set_goal_position_tolerance(0.003)
self.arm.set_goal_orientation_tolerance(0.03)
self.arm.allow_replanning(False)

# 默认只展示第一段；规划预览不改变真实机器人状态。
self.execute_motion = bool(rospy.get_param("~execute", False))
self.display_pub = rospy.Publisher(
    "/move_group/display_planned_path", DisplayTrajectory,
    queue_size=1, latch=True)
for name in ("/get_planning_scene", "/apply_planning_scene",
             "/check_state_validity"):
    rospy.wait_for_service(name, timeout=10.0)
self.get_scene = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)
self.apply_scene = rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)
self.check_validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
```

增加这些核心函数：

```python
class PlanningOnlyStop(RuntimeError):
    pass

# 以下函数加入 PiperVisionController 类。
def current_ee_matrix(self):
    pose = self.arm.get_current_pose("link6")
    if pose.header.frame_id.lstrip("/") != "base_link":
        pose = self.tf_buffer.transform(
            pose, "base_link", timeout=rospy.Duration(1.0))
    p = pose.pose
    T = tf_trans.quaternion_matrix([
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T

def pose_from_matrix(self, T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("目标矩阵无效")
    p = PoseStamped()
    p.header.frame_id = "base_link"
    p.pose.position.x, p.pose.position.y, p.pose.position.z = T[:3, 3]
    q = tf_trans.quaternion_from_matrix(T)
    (p.pose.orientation.x, p.pose.orientation.y,
     p.pose.orientation.z, p.pose.orientation.w) = q
    return p

def unpack_plan(self, result):
    if isinstance(result, tuple):
        if not result[0]:
            raise RuntimeError("MoveIt 规划失败")
        result = result[1]
    if result is None or not result.joint_trajectory.points:
        raise RuntimeError("MoveIt 返回空轨迹")
    return result

def validate_trajectory(self, traj):
    # 全机器人状态检查，同时带上已附着瓶子。
    req = GetPlanningSceneRequest()
    req.components.components = (
        PlanningSceneComponents.ROBOT_STATE |
        PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
    state = copy.deepcopy(self.get_scene(req).scene.robot_state)
    names = list(state.joint_state.name)
    index = {name: i for i, name in enumerate(names)}
    for point in traj.joint_trajectory.points:
        positions = list(state.joint_state.position)
        for name, value in zip(traj.joint_trajectory.joint_names, point.positions):
            if name not in index:
                raise RuntimeError("状态缺少关节 " + name)
            positions[index[name]] = value
        # 若轨迹包含夹爪关节，维持当前模型的对称关系。
        if "joint7" in traj.joint_trajectory.joint_names and "joint8" in index:
            positions[index["joint8"]] = -positions[index["joint7"]]
        state.joint_state.position = positions
        request = GetStateValidityRequest()
        request.robot_state = state
        request.group_name = ""  # 检查全机器人，而不是只列出六轴 link。
        reply = self.check_validity(request)
        if not reply.valid:
            pairs = [(c.contact_body_1, c.contact_body_2) for c in reply.contacts]
            raise RuntimeError("轨迹状态无效/碰撞: " + str(pairs))
    # 此附加检查检查返回轨迹采样点；点间检查依赖 MoveIt 规划插值分辨率。

def finish_motion(self, traj):
    self.validate_trajectory(traj)
    display = DisplayTrajectory()
    display.trajectory_start = self.robot.get_current_state()
    display.trajectory.append(traj)
    self.display_pub.publish(display)
    if not self.execute_motion:
        # 首段预览后停止任务；不能把未执行终点当作下一段真实起点。
        rospy.sleep(1.0)
        raise PlanningOnlyStop("仅规划，任务在第一段展示后停止")
    if not self.arm.execute(traj, wait=True):
        raise RuntimeError("MoveIt/FollowJointTrajectory 执行失败")
    self.arm.stop()

def move_joints(self, values):
    if len(values) != 6:
        raise ValueError("这里只接受六轴目标，不包含夹爪")
    self.arm.set_start_state_to_current_state()
    self.arm.set_joint_value_target(dict(zip(
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], values)))
    try:
        self.finish_motion(self.unpack_plan(self.arm.plan()))
    finally:
        self.arm.stop()
        self.arm.clear_pose_targets()

def move_pose(self, T):
    self.arm.set_start_state_to_current_state()
    self.arm.set_pose_target(self.pose_from_matrix(T), "link6")
    try:
        self.finish_motion(self.unpack_plan(self.arm.plan()))
    finally:
        self.arm.stop()
        self.arm.clear_pose_targets()

def move_cartesian(self, T):
    self.arm.set_start_state_to_current_state()
    # 本机安装版本签名没有 jump_threshold 参数。
    traj, fraction = self.arm.compute_cartesian_path(
        [self.pose_from_matrix(T).pose], eef_step=0.003,
        avoid_collisions=True)
    if fraction < 0.999999 or not traj.joint_trajectory.points:
        raise RuntimeError("直线段受阻/不可达，完成比例 %.6f" % fraction)
    # 显式检查关节跳变；上限需结合实际关节速度和采样距离验证。
    limit = float(rospy.get_param("~cartesian_joint_step_limit", 0.15))
    previous = dict(zip(self.arm.get_active_joints(),
                        self.arm.get_current_joint_values()))
    for point in traj.joint_trajectory.points:
        current = dict(zip(traj.joint_trajectory.joint_names, point.positions))
        if any(abs(value - previous[name]) > limit for name, value in current.items()):
            raise RuntimeError("笛卡尔轨迹关节跳变过大")
        previous = current
    traj = self.arm.retime_trajectory(
        self.robot.get_current_state(), traj,
        velocity_scaling_factor=0.03,
        acceleration_scaling_factor=0.03)
    if not traj.joint_trajectory.points:
        raise RuntimeError("轨迹时间参数化失败")
    times = [p.time_from_start.to_sec() for p in traj.joint_trajectory.points]
    if times[-1] <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError("轨迹时间无效")
    try:
        self.finish_motion(traj)
    finally:
        self.arm.stop()
        self.arm.clear_pose_targets()
```

`move_pose()` 用于去预抓取位、返回、转运等需要绕行的动作。
`move_cartesian()` 只用于已确认有直线通道的短距离接近、竖直提升、退出。
直线段受阻时停止或重新选择预抓取方向；不能关掉碰撞检查强行执行。
3% 是关节速度/加速度缩放比例，不等于旧脚本的 `v=0.02 m/s`。

每次动作前还应确认实机关节反馈新鲜、扫描门关闭、桌面/瓶子模型已同步，且没有其他程序改场景或发运动命令。
上述函数假设单个任务控制器独占运动和场景更新。

## 6. 核对手眼变换和 TCP，不直接反转旧矩阵

用 MoveIt/TF 获取的 `base_link → link6` 替换旧 `/end_pose`。
`make_bottle_pose_in_base()` 内应使用拍照停稳时保存的矩阵：

```python
# get_bottle_pose() 开始时，确认机器人停稳，记录拍照姿态。
self.current_ee_pose_mat = self.current_ee_matrix()
# 原 make_bottle_pose_in_base() 的检测变换可继续使用这个固定快照。
```

原 `T_EE_TO_CAM` 必须实际表示 `link6 → camera_color_optical_frame`。
RealSense deproject 输出的是光学坐标，不是相机外壳 `camera_link` 坐标。
从本工作空间已经验证的标定结果核对矩阵，不能因为旧任务能运行就直接认定与新模型完全一致。
图像采集期间机械臂保持静止；ROS 图像方案应进一步使用采集时刻 TF。

工具变换定义应明确为：

```text
T_base_link6_target = T_base_tool_target × inverse(T_link6_tool)
```

原变量 `T_EE_TO_TOOL` 的平移是 -0.14 m，实际使用方式却像 tool→link6 的补偿。
当前 URDF 的指根位于 link6 正 Z 附近（joint7 原点 z=0.1358 m）。
因此不能只按变量名把旧 -0.14 矩阵再次求逆。
实测真正 TCP 后，定义一个含义明确的 `T_LINK6_TOOL`，后续统一使用它的逆：

```python
# T_LINK6_TOOL 必须实测；TCP 的定义应是实际两夹指抓取中心。
T_base_tool_grasp = T_base_bottle @ T_BOTTLE_TO_GRASP
T_grasp_ee = T_base_tool_grasp @ np.linalg.inv(T_LINK6_TOOL)
```

YOLO 框中心 + 深度百分位得到的是观测表面附近点，未必是瓶子轴线/几何中心。
用已知瓶子尺寸和桌面高度确定圆柱模型中心；抓取高度单独选取，
必要时补偿表面到轴线的水平偏差。不要把框中心点直接当圆柱中心、底面和抓取点三者。

## 7. 创建命名瓶子和只针对它的接触权限

添加场景同步函数：

```python
def wait_object(self, name, known, attached, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        is_known = name in self.scene.get_known_object_names()
        is_attached = bool(self.scene.get_attached_objects([name]))
        if is_known == known and is_attached == attached:
            return
        rospy.sleep(0.05)
    raise RuntimeError("场景对象未同步: " + name)

def add_bottle(self, center_xyz, height, radius):
    if height <= 0 or radius <= 0:
        raise ValueError("瓶子尺寸无效")
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.orientation.w = 1.0  # 瓶子直立，圆柱轴沿 base Z。
    (pose.pose.position.x, pose.pose.position.y,
     pose.pose.position.z) = center_xyz
    self.scene.add_cylinder("target_bottle", pose, height=height, radius=radius)
    self.wait_object("target_bottle", known=True, attached=False)
```

直立瓶子圆柱中心 z = 实际桌面 z + 瓶高/2。
半径可以保守膨胀；底面不要随意向下膨胀，以免制造额外桌面穿透。
有把手、瓶盖突出部分时使用组合几何，而不是假设单圆柱足够。

ACM 修改必须读取并保留已有矩阵，成对对称地修改。下面只改指定 pair，
已有机器人自碰撞配置不会被一个全新矩阵清空：

```python
def get_acm(self):
    req = GetPlanningSceneRequest()
    req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    return copy.deepcopy(self.get_scene(req).scene.allowed_collision_matrix)

def apply_acm(self, acm):
    diff = PlanningScene()
    diff.is_diff = True
    diff.robot_state.is_diff = True
    diff.allowed_collision_matrix = acm
    if not self.apply_scene(diff).success:
        raise RuntimeError("ACM 更新失败")

def set_contact_pairs(self, pairs, allowed):
    acm = self.get_acm()
    for first, second in pairs:
        for name in (first, second):
            if name not in acm.entry_names:
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_names.append(name)
                acm.entry_values.append(AllowedCollisionEntry(
                    enabled=[False] * len(acm.entry_names)))
        i = acm.entry_names.index(first)
        j = acm.entry_names.index(second)
        acm.entry_values[i].enabled[j] = bool(allowed)
        acm.entry_values[j].enabled[i] = bool(allowed)
    self.apply_acm(acm)
```

第一次建立任务场景后保存 `self.acm_before_task = self.get_acm()`。
正常结束再恢复。新对象对应的新条目初始化为禁止；原矩阵的旧条目和默认条目保留。
如果系统依赖针对“所有新对象”的 default allow 规则，需要审查并收紧该规则；
不能让 table 因默认权限而失去碰撞检查。

预抓取位之前，瓶子与夹指也禁止碰撞；到了预抓取位，才调用：

```python
self.set_contact_pairs([
    ("target_bottle", "link7"),
    ("target_bottle", "link8"),
], True)
```

检查 ACM 中 `table ↔ link7/link8/gripper_base/link6` 没有被放开。
也不要设置某个夹指的 default_entry 为 True；那会允许它碰其他障碍。

## 8. 夹爪开闭从旧 PosCmd 拆开

原 `gripper_val=100/200` 不能直接移植。当前 SDK/驱动接口以米表示完整开口量。
joint7=0.035 m 对应约 0.07 m 完整开口。

MoveIt 夹爪组适合空载张开到已知位置；带物体闭合到零位可能因瓶子阻挡而无法满足到位容差。
可用当前驱动的 `/gripper_srv` 力度限制闭合，手臂仍通过 MoveIt 运动。
以下是接口用法，服务返回 True 仅表示驱动接收了请求：

```python
# 首次使用前：
rospy.wait_for_service("/gripper_srv", timeout=5.0)
self.gripper_service = rospy.ServiceProxy("/gripper_srv", Gripper)

def command_gripper(self, opening_m, effort):
    if not self.execute_motion:
        raise PlanningOnlyStop("仅规划模式不发送夹爪命令")
    if not (0.0 <= opening_m <= 0.07 and 0.5 <= effort <= 2.0):
        raise ValueError("夹爪开口/力度参数越界")
    result = self.gripper_service(
        gripper_angle=opening_m, gripper_effort=effort,
        gripper_code=1, set_zero=0)
    if not result.status:
        raise RuntimeError("夹爪驱动请求失败")
```

这个服务绕过 MoveIt 的夹爪轨迹检查，因此发送前必须验证开闭扫掠：
取完整当前 RobotState（含附着物），按开口变化采样 joint7=q、joint8=-q，
用第5节的 `/check_state_validity` 检查每个样本，group_name 为空。
瓶子接触只按当前抓取阶段放开，桌面仍禁止碰撞。
若不实现这项扫掠检查，改用 MoveIt gripper 规划到经验证的目标开口。

张开后等待 `/joint_states` 中测得的 2×joint7 接近目标，再开始手臂运动。
闭合后需要测量开口、夹持力度/稳定状态，并结合视觉或任务传感器确认夹持。
确定“实物已抓住”后才 attach；不能仅靠 time.sleep() 或服务成功返回值判断抓取成功。
具体成功阈值根据实际瓶子直径、可压缩程度、夹指结构和驱动反馈校验后设定。

## 9. 将原任务按阶段替换

先完成上述几何和接口初始化，再改 `execute_vision_grasp()`。

| 原动作 | 新动作 | 碰撞要求 |
|---|---|---|
| 发布拍照 JointState | `move_joints([1.521, 0.403, -0.521, 0, 0.808, 0])` | 桌面/固定场景已经加载 |
| 检测瓶子 | 停稳后取拍照时 link6 位姿；建立独立圆柱 | 识别点、模型中心、抓取点分别处理 |
| 到预抓取位 | `move_pose(T_pre_ee)` | OMPL 绕桌边，禁止接触瓶子 |
| 前探4 cm | 接触权限后 `move_cartesian(T_grasp_ee)` | 全程夹爪与桌面禁止碰撞 |
| 闭合 | 独立夹爪控制，等待实际夹持确认 | 开闭扫掠也做碰撞检测 |
| 提升15 cm | attach 后 `move_cartesian(T_lift_ee)` | 携带瓶子，临时处理瓶底与支撑桌面的接触 |
| 返回零位/转运 | `move_joints(...)` 或 `move_pose(...)` | 瓶子保持附着；新目标也必须可行 |
| 放置 | 预放置 → 短距离下降/释放 → detach → 退出 | 场景中增加放下的瓶子，夹指退出后取消接触权限 |

不要继续调用旧的 `move_to_target_smooth()`，也不要保留初始、返回和放置阶段的 `joint_pub.publish()`。
执行失败立即停止后续阶段，禁止自动退回旧 PosCmd 控制。

去拍照位时还没有本次目标位置。如果瓶子或其他活动物可能处在这段路径上，
应先在当前安全停稳姿态检测，或预先用保守几何覆盖已知摆放区域、验证初始路径。
不能把尚未定位的物体默认当成不存在。

原抓取姿态固定 grasp X=-base Z，原 grasp Y/Z 由当前姿态投影生成，并非总是 base Y/base X。
所以预抓取点要按实际 grasp Z 接近轴推算，不能照注释默认其等于全局 X：

```python
# 以下 dist 必须在瓶子、桌边、机械臂可达性下选择，不是通用参数。
T_pre_tool = T_base_tool_grasp.copy()
approach_axis = T_base_tool_grasp[:3, 2]
T_pre_tool[:3, 3] -= approach_axis * pregrasp_distance
T_pre_ee = T_pre_tool @ np.linalg.inv(T_LINK6_TOOL)

T_lift_ee = T_grasp_ee.copy()
T_lift_ee[2, 3] += lift_distance  # base Z 竖直提升，姿态保持。
```

根据桌边和瓶子位置，生成多个水平接近方向：面向瓶子的外侧、左右侧等。
对每个方向先验证最终抓取状态，然后验证短直线段及可达预抓取路径，成功才选择该候选。
最终夹爪最低点至少高于实测桌面加所选误差余量。
TCP 高于桌面不能代替整个夹爪 collision 的检查。

如果要求全程保持夹爪水平，需要增加 OrientationConstraint 并验证可行性；
仅设置水平目标姿态不能保证 OMPL 中途也保持水平。
夹持转运也应根据瓶子是否允许倾斜选择约束，防止绕行中甩动或滑落。

## 10. 抓住后附着，并处理支撑接触

夹持确认之后：

```python
self.scene.attach_object(
    "target_bottle", link="link6", touch_links=["link7", "link8"])
self.wait_object("target_bottle", known=False, attached=True)
```

附着的是实际瓶子几何，姿态需对应真实抓住的位置。若接近时瓶子滑动，先更新模型。
不能在尚未抓住时 attach：否则模型会提前随机器人移动。

瓶子此时仍在桌上。桌面膨胀、几何相切和数值误差可能让附着瓶底与桌面被判为碰撞。
可只在第一段竖直抬起中允许该瓶子与支撑桌面接触：

```python
# 保存当前阶段的 ACM，提升后只还原本阶段权限。
acm_before_lift = self.get_acm()
self.set_contact_pairs([("target_bottle", "table")], True)
self.move_cartesian(T_lift_ee)
# 此时必须确认瓶底已离开膨胀桌面（并达到误差余量）。
self.apply_acm(acm_before_lift)
# 之后立即用 /check_state_validity 验证当前全机器人状态，再继续转运。
```

只能在姿态固定、沿 base Z 向上的提起段放开这一个 pair；
不能借这个例外水平穿过桌板，也不能放开任意机器人 link 与 table 的碰撞。
如果支撑桌面仍以 OctoMap 体素表示，还可能有瓶底对 `<octomap>` 的接触：
优先把抓取区域支撑几何用实测 box 独立表示，并确保瓶子之外的体素准确，
或采用专门支持支撑接触的抓放实现；不要因此放开整个瓶子与所有 OctoMap 障碍。
对本方案，最易调试的初始版本是该桌子使用 box、其他固定障碍使用 box/无目标物的冻结地图。

放置时先规划到瓶底位于桌面以上的预放置位。
如果最终下降到支撑面，需要同样只临时处理 bottle/table 接触。
松开后 `remove_attached_object("link6", name="target_bottle")`，
查询状态并把世界对象的位置校正为实测放置位置。
夹指退出前保留 bottle/finger 例外；退出成功后禁止接触并恢复任务前 ACM。
不能先删除场景中所有瓶子再撤退。

如果中途失败：停止运动，保留与真实夹持状态相符的 attached/world 模型，记录当前权限，
不要在异常清理中自动松爪、detach、回零或删除真实障碍。

## 11. 启动顺序与完整验证

以下命令是修改之后由操作者运行的步骤，本次没有运行。

### A. 场景准备阶段

第一版若全部使用几何对象，加载所有测量好的障碍后即可验证，
无需启动相机点云或扫描门；如果旧 OctoMap 仍在，明确停止更新并在几何替代完整后清空。
下面是组合点云地图的扩展流程，抓取区桌面点的分割和 box 替代须先完成。

启动实机 MoveIt 前确认机械臂初始状态、环境、急停和模型。
这个 launch 默认会尝试使能机械臂；它不是纯离线查看模型。

```bash
roslaunch piper_with_gripper_moveit piper_real.launch can_port:=can0 auto_enable:=true
```

其他终端分别启动相机、已验证的外参和扫描门：

```bash
roslaunch moveit_ctrl eye_in_hand_camera.launch enable_pointcloud:=true
roslaunch /home/hank/piper_ws/calibration/eye_in_hand_d435i/session_02/camera_extrinsics.launch
roslaunch moveit_ctrl static_scene_mapping.launch
```

移走待抓瓶子并确认已知安全视角后，依次 clear/start、多视角扫描、stop：

```bash
rosrun moveit_ctrl static_scan_control.py stop
rosrun moveit_ctrl static_scan_control.py clear
rosrun moveit_ctrl static_scan_control.py start
# 在已确认安全的多个视角停稳采集；这些命令不会自动扫描移动机械臂。
rosrun moveit_ctrl static_scan_control.py stop
rosrun moveit_ctrl static_scan_control.py status
```

RViz 中检查桌面、桌边和固定障碍，确认没有待抓瓶子体素；
检查抓取区支撑面已由 box 覆盖，并没有会阻碍带瓶提升的重复桌面体素。
status 应为 enabled=false；将瓶子放回后不再 start。
保留直接 SDK 检测时，接着停止 ROS 相机 launch，确认设备释放。

### B. 接口和模型检查

```bash
rostopic info /joint_states
rostopic echo -n 1 /joint_states
rosnode info /piper_ctrl_single_node
rosservice list
rosparam get /move_group/arm/longest_valid_segment_fraction
```

`/joint_states` 的真实反馈应由驱动发布，不应出现抓取脚本或旧 Pinocchio 控制节点发布伪反馈。
最后一个参数的命名以实际命名空间为准；当前配置文件给 arm 设置的是 0.005。
桌边很薄时，应评估 OMPL 离散碰撞检查的采样分辨率，不能只提高地图分辨率。

验证 `/arm_controllers/follow_joint_trajectory`、`/gripper_srv`、
`/get_planning_scene`、`/apply_planning_scene`、`/check_state_validity` 都存在。
URDF 的 link7/link8、TCP、相机外参和真实实物必须一致。

### C. 先预览，再分段执行

```bash
/usr/bin/python3 /home/hank/piper_ws/src/piper_ros/src/bottle_grab/grab2016_5_28_3_moveit.py _execute:=false
```

按第5节代码，此命令只展示第一段规划后停止。
应在 `run()` 中捕获 `PlanningOnlyStop` 并记录日志，在 finally 中保持原相机释放逻辑并 shutdown MoveIt。
初始化相机失败时也要处理 pipeline 未创建的情况。

若要检查完整未来序列，需要在离线机器人/虚拟状态中逐段更新 RobotState、夹爪和附着物，
不能对真实机器人连续做 plan 但不 execute，然后假装每一段都从前一段终点开始。
可使用独立 ROS master 的 fake `demo.launch`，以记录的目标和同一场景检查路径；不要与实机反馈混用。

至少验证以下结果：

1. 故意给出一个夹指穿桌的目标，`/check_state_validity` 报出 table/link7 或 table/link8，规划拒绝。
2. 合法目标有可执行轨迹；绕行时手腕、夹爪、相机都在模型内，桌边没有被忽略。
3. 靠近瓶子时只有 target_bottle/link7、target_bottle/link8 允许接触。
4. 最终抓取状态和短直线段完整有效；fraction 不完整时任务终止，不闭爪。
5. 真正抓住后，场景中可见 AttachedCollisionObject，抬起/返回轨迹检查包含瓶子。
6. 提升结束后撤回 bottle/table 支撑例外，并验证完整当前状态。
7. MoveIt/action、夹爪确认或 TF/反馈任一失败，后续状态不再执行。
8. 放置、撤退、返回也走 MoveIt，场景中的新瓶子位置同步。

逐段确认后才用 `_execute:=true` 进行低速实机验证；它会发运动命令和夹爪命令。
新脚本中的等待应依赖到位/抓取反馈，而不是沿用原固定 sleep 时间。

## 12. 本机查证依据

- 原比赛脚本：构造函数第263行附近；旧运动函数第323行；任务状态序列第513行。
- `piper_with_gripper_moveit/launch/piper_real.launch`：实机 trajectory action、命令反馈分离。
- `piper_with_gripper_moveit/config/piper_description.srdf`：arm、gripper、open/close。
- `piper_description/urdf/piper_description.urdf`：夹指 collision、joint7/joint8。
- `piper/scripts/piper_ctrl_single_node.py`：第419行夹爪话题；第472行夹爪服务；实测反馈 q/-q。
- `piper_msgs/srv/Gripper.srv`：完整开口量和力度请求字段。
- 本机 `/opt/ros/noetic/lib/python3/dist-packages/moveit_commander/move_group.py`：
  `compute_cartesian_path(waypoints, eef_step, avoid_collisions=True, path_constraints=None)`；
  `retime_trajectory()` 的速度和加速度参数。
- 同目录 `planning_scene_interface.py`：同步场景接口、attach_object、add_cylinder。

接口核对来自当前本机文件。没有做目标桌面测量、真实 TCP/夹指包络标定、完整脚本改造或实机路径验证。

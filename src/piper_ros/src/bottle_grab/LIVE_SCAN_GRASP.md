# 实时扫描 + 静态障碍地图 + 抓取放置

`moveit_grab.py` 的当前版本在启动后自己完成“扫视 — 建静态障碍地图 — 冻结 — 避障 — 抓取 —
放置”，不再需要离线 `table_model.json`，也不需要手工开关静态扫描门。

默认 `scan_mode=map`：**除目标瓶和支撑面外，扫描到的一切都是障碍**（桌面、墙、柜子、
显示器、其他瓶子……），由 MoveIt 内部的 OctoMap 表示。

## 1. 启动前提

1. `move_group` / 机械臂驱动 / 夹爪服务已经在运行（`piper_real.launch`），并且
   `move_group` 加载了 `sensors_3d.yaml`（`allow_trajectory_execution=true`）。
   改过 `sensors_3d.yaml`（量程2m/10Hz/自体过滤3cm）或 `sensor_manager.launch.xml`
   （体素2cm）必须重启 `piper_real.launch`。
2. **ROS 相机节点必须停止**：程序自己用 `pyrealsense2` 打开 D435i，否则设备被占用会直接报错。
   相机 `link6→camera_link→camera_color_optical_frame` 静态 TF 由程序按标定文件自动发布。
3. 机械臂停在操作者确认安全的位置。第一段“当前位 → `scan_start_joint_target`”没有地图保护
   （OMPL 随机规划，理论上可能先下沉再抬升），必须人工确认该路径不会碰到任何东西。
4. 扫描期间工作区保持静止（人和物都不要进出视野）。

## 2. 一次运行的顺序

```text
清场：删 table/target_bottle/scan_obstacle_*、清 OctoMap、复位 ACM、关闭遗留扫描门
张开夹爪（原地）
OMPL 运动到 scan_start_joint_target
视点0：停稳 → 取4帧 → 逐帧投票去噪 → 融合
       → YOLO识别全部物体（先用表面点给桌面拟合做种子）
       → 目标瓶附近的局部RANSAC拟合桌面 → 桌面盒进场景
       → 点云聚类/形状判别 → 目标瓶进场景、其余不再建语义盒
       → 挖掉底座圆柱/目标瓶/支撑面 → 以optical frame发布该视点 → 等OctoMap稳定
视点1..4：joint1 ±22.5°/±45° → 同上（换姿轨迹已在上一视点地图上规划）
         后续视点只精修目标瓶中心，不重新聚类整个房间
建图自检：OctoMap非空、字节数达标、当前机械臂状态与地图无碰撞
冻结：场景签名 = OctoMap + 全部世界对象（桌面/目标瓶）
抓取：接近走廊预筛 → 预抓取规划 → 虚拟起点验证直线接近 → 执行 → 夹紧确认 → 附着 → 抬升
回零：夹着瓶子先回 home_joint_target = [0,0,0,0,0,0]
放置：按扫描到的卡面法向，在卡面前 place_offset_from_card 处、桌面+间隙高度松手
      下降优先走直线；直线被截断时自动退回 OMPL 规划到同一目标位
回零：放置完成后再回零位
```

每一步运动前都会重新比对场景签名；签名变化立即中止，不自动回零、不自动松爪。

## 3. 碰撞场景

| 对象 | 来源 | 接触 |
|---|---|---|
| `table` | 局部RANSAC平面 + footprint | 精确顶面；瓶子↔桌面放行（松手/抬升） |
| `target_bottle` | 圆柱拟合 | 指尖 link7/link8 放行，其余一律检查 |
| `<octomap>` | 每视点发布的点云 | 全部检查（机器人本体、手指都不得碰撞） |

支撑面方案（b2）：桌面顶面 ±`scan_map_support_carve`（1.5cm）、在桌面 footprint+5cm 内的
体素从地图挖掉，由“顶面精确”的 table 盒代替；桌腿/侧面/其他高度的一切仍在图里。
底座半径 13cm + 3cm 余量（`scan_map_base_keepout_radius=0.16`）的圆柱从地图挖掉，
否则 `link1` 一开始就与地图自碰撞。

## 4. 关键参数

| 参数 | 含义 |
|---|---|
| `scan_start_joint_target` | 第一个视点的关节角，必须能同时看见瓶子和卡片 |
| `scan_joint1_offsets` | 相对起始位的 joint1 偏移，默认 5 个偏航角（±22.5°/±45°） |
| `scan_joint2_offsets` | 相对起始位的 joint2 偏移（俯仰档位），默认只扫一档 |
| `scan_mode` | `map`（默认）/ `semantic`（旧语义盒，回滚用）/ `both`（调试，双重膨胀） |
| `scan_max_range` | 传感器有效深度，默认 2.0m（也是 OctoMap 的量程） |
| `scan_roi_min/max` | base_link 下参与建图/拟合的点云范围；不再等于“桌面范围” |
| `scan_table_focus_radius/down/up` | 桌面拟合焦点窗口（以目标瓶表面点为种子） |
| `scan_table_footprint_radius` | 桌面盒/支撑检查的footprint半径（默认 0.6m） |
| `scan_table_surface_epsilon` | 桌面盒顶面比拟合平面高出的量（默认 2mm） |
| `scan_map_cloud_voxel` | 发布给 OctoMap 前的降采样体素（默认 1.5cm） |
| `scan_map_base_keepout_radius` | 底座避让圆柱半径（默认 0.16 = 13cm + 3cm） |
| `scan_map_target_carve` | 目标瓶挖空半径余量（默认 2cm） |
| `scan_map_support_carve` | 支撑面挖除厚度（默认 1.5cm） |
| `scan_map_settle_quiet/timeout` | 每视点发布后等地图稳定的静默时间/超时 |
| `scan_map_require_current_valid` | 建图后强制校验当前机械臂状态无碰撞 |
| `scan_card_max_top_height` | 卡片簇顶离桌面高度上限，防止大ROI里的墙面板被当卡片 |
| `scan_corridor_radius` | 接近走廊净距；小于该值的接近方向直接跳过并报出阻挡物 |
| `place_offset_from_card` | 松手点距卡面中心的法向距离，默认 0.11 |
| `home_joint_target` | 放置完成后的回零目标 |

目标瓶与卡片的区分**只用几何**：卡片是薄板（平面度高、双向展开大、厚度远小于尺寸），
瓶子是圆柱（拟合半径/高度与参数一致）。多个瓶子时，颜色标签优先，其次用最自信的
YOLO 检测就近匹配；两者都分不清就直接中止，不猜。

## 5. 每次扫描的落盘产物

`scan_artifact_root/scan_<时间戳>/`：

- `cloud.npz`：融合点云 `points` 与每个点的 `observations`（被几个视点看到）
- `table_model.json`：本次拟合的桌面（含 `fit_scope`/`surface_top`/`footprint_count`）
- `scene.json`：目标瓶中心、卡片模型、YOLO检测结果（`obstacles` 默认是空的）
- `summary.txt`：人读摘要
- `view_<n>.png`：每个视点的彩色图
- `map/view_<n>.npz`：该视点**实际发布给OctoMap**的点（base_link）+ `base_T_optical`
- `map/meta.json`：每视点挖除统计、OctoMap 的 frame/分辨率/字节数/sha256
- `map/octomap.npz`：冻结后的序列化八叉树（原始数据）
- `map/octomap_snapshot.bag`：可被 `static_scene_snapshot.py load` 恢复的快照（尽力生成）

## 6. 常见失败与含义

| 报错 | 原因 / 处理 |
|---|---|
| `发布点云后OctoMap未变化` | `move_group` 没加载 sensors，或相机 optical frame 的 TF 不可达 |
| `障碍地图为空/过小` | 同上；检查 `scan_roi_min/max` 与 `scan_max_range` |
| `OctoMap在xxs内未稳定` | 有别的节点在持续往 `/piper/static_scan/points` 发点云 |
| `机器人状态无效/碰撞: [('link1','<octomap>')]` | 底座余量不足，加大 `scan_map_base_keepout_radius` |
| `未找到圆柱形目标瓶` | 瓶子被遮挡、点太少、聚类太碎（调 `scan_min_cluster_points`） |
| `存在多个圆柱候选且无法用检测唯一确定目标` | 多瓶且分数接近；用 `target_bottle_label` 或用颜色 |
| `扫描未识别出卡片` | 卡片被遮挡或只露出一部分（`scan_card_min_extent` 太大） |
| `目标瓶在多视点融合中丢失` | 某个视点看不到瓶子，换 `scan_joint1_offsets` |
| `没有可行的接近走廊` | 目标瓶被其他障碍包围；日志会给出阻挡物 id 和净距 |
| `环境/规划场景在任务中改变` | 有人动了场景、或遗留扫描门又在发点云 |

## 7. 已知边界（本次明确的取舍）

- **静态地图**：只有扫描时各视点看到的面是障碍；遮挡区、背面、2m 以外、扫描后移动的
  东西都不会被避开。
- **扫描时进入视野的人手/工具会永久留在快照里**，需要重新扫描才能清除。
- **相机/支架没有几何模型**；底座用 `scan_map_base_keepout_radius` 近似。
- **第一段运动无地图保护**。
- **手眼标定残差约 12mm**，地图中的障碍位置/桌面高度就带约 1cm 误差。
- 贴桌面放置的薄物体（<1.5cm）会被支撑面挖除，可能看不到。

## 8. 命令行操作

逐步命令、参数临时覆盖、日志对照、报错处理和复位方式见
[`LIVE_SCAN_GRASP_RUNBOOK.md`](LIVE_SCAN_GRASP_RUNBOOK.md)。

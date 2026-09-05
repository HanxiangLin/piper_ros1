# Piper + 外部固定 RealSense D435i：eye-to-hand 标定

本工具用于求“固定相机相对于机械臂基座的外参”，供后续深度点云建图使用。
相机固定在机械臂外部；打印标定板刚性固定在末端。改变机械臂位姿，采集多组
“基座到末端 TF”和“彩色光学坐标系到标定板位姿”。不需要预先测量标定板到末端的安装偏移。

采样脚本只订阅实机状态/图像、发布检测预览并写采样文件，不发送机械臂运动命令。
求解和验证脚本完全离线，不启动 ROS 节点。通过独立验证后才生成供你手动启动的静态 TF launch。

## 1. 已准备的文件

| 文件 | 作用 |
| --- | --- |
| `calibration/board/board.pdf` | A4 实际尺寸打印板，优先打印这个 PDF |
| `calibration/board/board.png` | 预览图，不能保证打印物理尺寸 |
| `calibration/board/board.json` | 板尺寸、字典和标记 ID，必须与实物一致 |
| `scripts/eye_to_hand_make_board.py` | 需要不同尺寸时生成新板 |
| `launch/eye_to_hand_camera.launch` | 启动 D435i 彩色/深度流和内部 TF；不启动机械臂 |
| `scripts/eye_to_hand_collect.py` | 手动触发、停稳检查、图像时间戳 TF 配对、持久化采样 |
| `scripts/eye_to_hand_solve.py` | `solve` 求解；`validate` 独立验证并生成外参 TF 文件 |

本机系统 Python/OpenCV 4.2 和 piper Conda/OpenCV 4.8 均有 ArUco、calibrateHandEye。
下文统一显式使用 `/usr/bin/python3`，避免 Conda base 没有 OpenCV/ROS 依赖的问题。
使用 ArUco 多标记定位，不需要 YOLO，也不需要 IMU。

## 2. 准备与安装标定板

默认板为 `DICT_4X4_50`，4列×3行，标记 ID 0–11，每个黑色标记的外边长 35 mm，
相邻标记净间距 8 mm，整体标记区域 164×121 mm。

1. 打印 `calibration/board/board.pdf`，选择 **A4、实际大小/100%**，关闭“适合页面”。
2. 用尺确认 PDF 下方标尺实际为 100 mm，单个标记外边长为 35 mm。
3. 平整贴在轻质、刚性的板材上，避免弯曲、反光、胶带遮住标记。
4. 用牢固支架固定在末端或夹爪上。允许未知安装偏移，但整个训练和独立验证期间必须不变。
5. 如果夹爪夹住板，采样期间不要开合夹爪、重新夹持或让板滑动。
6. 相机和基座全程固定。标定板必须随末端移动，不能也固定在桌上。

板坐标原点位于标记区域左上角，X 向右、Y 向下、Z 指向纸内。ID 对应关系固定，
不会因为转动纸板而更换坐标原点。默认至少4个标记清晰可见才能采样，建议尽量看见全部12个。

如需其他尺寸，先生成新板并测量，再在所有采样命令中使用新 JSON：

```bash
/usr/bin/python3 "$(rospack find moveit_ctrl)/scripts/eye_to_hand_make_board.py" \
  --output-dir /home/hank/piper_ws/calibration/custom_board \
  --marker-length 0.030 --separation 0.007
```

该示例生成30 mm标记、7 mm间距的新板；不要拿新 JSON 配旧纸板。生成器拒绝覆盖已有输出。

## 3. 加载环境

每个使用 ROS 的终端都执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/hank/piper_ws/devel/setup.bash
```

源码修改后如需重新注册脚本/模块，在工作空间编译：

```bash
cd /home/hank/piper_ws
catkin build moveit_ctrl --no-deps
source devel/setup.bash
```

本工具运行时依赖 `rospy`、`sensor_msgs`、`cv_bridge`、`tf`、`tf2_ros`、NumPy、
带 ArUco 的 OpenCV 和 `realsense2_camera`，已在 package.xml 声明。

## 4. 启动实机反馈和 RViz（终端一）

沿用已配置好的真实机械臂启动方式。它会连接 CAN 并自动使能机械臂：

```bash
roslaunch piper_with_gripper_moveit piper_real.launch \
  can_port:=can0 auto_enable:=true
```

如果实机驱动/MoveIt 已运行，不要重复启动。不能使用 fake controller 或 Gazebo 的关节状态做实机外参标定。
检查：

```bash
rostopic info /joint_states
rosrun tf tf_echo base_link link6
```

`/joint_states` 应来自 Piper 实机反馈，不能是旧抓取脚本向同名话题发布的目标角度。
改变实际关节位置时，`base_link -> link6` 的 TF 应相应变化。

默认采样使用 `base_link` 和 `link6`。如果安装板的刚性末端 link 不同，用 `--ee-frame` 明确指定；
整个数据集保持一致。后续 MoveIt 根 `dummy_link` 与 `base_link` 的固定连接不影响本结果使用。

## 5. 启动 D435i（终端二）

关闭之前直接占用相机的 `pyrealsense2` 抓取程序，再启动：

```bash
roslaunch moveit_ctrl eye_to_hand_camera.launch
```

默认彩色1280×720、30 FPS；深度848×480、30 FPS；保留相机内部 TF，关闭IMU，标定时不发布点云。
有多台相机时用 `serial_no:=设备序列号` 选择同一台 D435i。

默认采样话题：

```text
/camera/color/image_raw
/camera/color/camera_info
```

采样脚本从 CameraInfo 读取内参和畸变，不写死焦距。只使用原始彩色图像，
不要换成裁剪/缩放后的图像，或把深度 CameraInfo 配给彩色图像。
图像和 CameraInfo 的 frame_id 必须一致。采样期间不改变相机分辨率、内参或安装位置。

此时尚未有基座到相机的 TF 是正常的；采集只需要两条独立的链：

```text
base_link → 机械臂各关节 → link6
camera_link → RealSense 自带内部 TF → camera_color_optical_frame
```

不要为了连通两棵 TF 树而临时发布一个猜测的 base→camera 变换。

## 6. 采集训练位姿（终端三）

选择本次标定的专用目录，下一次重新标定换 session_02 等新目录：

```bash
export PIPER_CALIB_PKG="$(rospack find moveit_ctrl)"
export PIPER_CALIB_RUN=/home/hank/piper_ws/calibration/eye_to_hand_d435i/session_01
mkdir -p "$PIPER_CALIB_RUN"

/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_to_hand_collect.py" \
  --board "$PIPER_CALIB_PKG/calibration/board/board.json" \
  --output "$PIPER_CALIB_RUN/train.json"
```

在 RViz 中点击 Add → Image，将 Image Topic 选为 `/eye_to_hand/preview`，观察板上的标记检测。
也可以同时添加原始彩色图像。预览图像仍在相机坐标系中，显示图像不需要已完成外参。

终端交互：

| 输入 | 行为 |
| --- | --- |
| Enter 或 `s` | 触发一次新的静止观测，成功后保存一个位姿样本 |
| `p` | 显示可见标记/当前等待原因和已保存数量 |
| `u` | 撤销最后一个样本，原始图像仍保留以便检查 |
| `q` | 退出，每个成功样本已经独立保存 |

更换位姿通过现有 RViz 的 Plan/Execute 或既有示教方式完成；采样程序不控制运动。
在 RViz 中先设低速（例如速度/加速度比例0.05），在已确认有空间容纳机械臂及标定板的区域换姿。
每次先检查规划、执行并停稳，再回采样终端按 Enter。标定板属于新增末端体积，
当前 URDF 不包含它，不能仅凭裸机械臂的碰撞检查判断标定换姿是否有足够空间。

建议训练采20–25个不同位姿，程序最低要求12个。一个按键内部采多帧用于停稳检查，
它们最终只算一个位姿，不是一次按键就收集了多个标定姿态。

采样姿态需要：

- 标定板在图像左/中/右、上/中/下以及稍远/稍近位置均有覆盖。
- 绕至少两个不平行方向改变姿态；在可见和可达范围内，例如分别尝试约±15–30°的倾斜。
- 组合位置和角度变化，避免只平移，或所有样本都绕一个轴转动。
- 不需要标定板绕轴转满一圈，也不应为了凑角度去接近关节极限。

示意采样顺序可以是：中心附近5组不同倾斜 → 左右位置各4组并改变倾斜 →
上下/远近再6组。位置范围以实际工作空间为准，代码没有写入未经现场确认的运动目标。

默认采样质量门槛：至少8个不同图像时间戳、跨度至少0.7 s；相邻有效帧间隔≤0.35 s；
末端位置变化≤2 mm、姿态变化≤0.5°，六关节变化≤0.005 rad；标定板检测位姿抖动≤4 mm/1°；
所选帧重投影 RMS≤1.5 px。使用图像时刻的 TF，失败时不会偷偷回退到“最新 TF”。
这些是采集质量门槛，不是机械臂运动完成信号或最终绝对精度指标。

每个样本保存：图像、精确配对的两组位姿、关节状态、时间戳、检测到的标记ID/角点、
内参与内部相机TF元数据、稳定性统计。记录于 `train.json` 和 `train_images/`。

中断后继续同一组训练可加 `--resume`，必须保持原来的板安装、相机位置和参数不变。
否则另建数据集；不能把重新夹持板前后的样本混在一起。

## 7. 离线求解

采样终端输入 `q` 后执行：

```bash
/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_to_hand_solve.py" solve \
  --dataset "$PIPER_CALIB_RUN/train.json" \
  --output "$PIPER_CALIB_RUN/result.json"
```

程序使用 OpenCV PARK 方法。变换约定 `a_T_b` 表示把 b 中的坐标转换到 a：

```text
B_i = base_T_ee[i]             （TF反馈）
C_i = camera_optical_T_board[i]（ArUco + PnP）
X   = base_T_camera_optical    （待求固定外参）
Y   = ee_T_board               （待求固定安装偏移）

B_i · Y = X · C_i
```

向 `calibrateHandEye` 传入 `inv(B_i)` 和 `C_i`，得到 X；没有将普通 eye-in-hand 的输出方向直接照搬。
求解器会检查两轴旋转是否充分，并输出逐样本及总体平移/旋转残差，不静默删掉离群样本。

`result.json` 中的 `base_T_camera` 专指彩色光学坐标系，另有供发布使用的 `base_T_camera_link`。
训练拟合成功暂不生成 TF；独立验证通过后才导出 launch。

## 8. 独立新位姿验证

保持板的夹持/支架和相机固定不变，再采5–8个**没有用来训练的新位姿**：

```bash
/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_to_hand_collect.py" \
  --board "$PIPER_CALIB_PKG/calibration/board/board.json" \
  --output "$PIPER_CALIB_RUN/validation_samples.json"
```

退出后运行：

```bash
/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_to_hand_solve.py" validate \
  --dataset "$PIPER_CALIB_RUN/validation_samples.json" \
  --calibration "$PIPER_CALIB_RUN/result.json" \
  --output "$PIPER_CALIB_RUN/validation_report.json" \
  --launch-output "$PIPER_CALIB_RUN/camera_extrinsics.launch"
```

验证固定训练得到的 X 和 Y，不重新拟合。默认要求每个验证样本的一致性误差≤10 mm且≤2°。
这只是初步验收门槛，可用 `--max-translation`（米）、`--max-rotation`（度）按任务收紧；
不能把这两个数字当作点云绝对精度或实机避障间隙。
验证会拒绝复制训练样本作为验证数据。失败时保存报告、退出码2，不生成 launch。

输出文件均不自动覆盖旧文件。如果补采后重算，使用 result_v2.json / validation_report_v2.json 等新名字。
检查失败样本的原图、打印尺寸、松动、时钟和姿态分布，而不是单纯放宽门槛。

## 9. 启用标定 TF，并检查深度点云

独立验证通过后，确认没有其他节点给 `camera_link` 发布父变换，手动运行：

```bash
roslaunch "$PIPER_CALIB_RUN/camera_extrinsics.launch"
```

实际发布的是：

```text
base_link → camera_link → RealSense 内部 TF → 彩色/深度 optical frame
```

使用记录的驱动内部 TF，转换公式为：

```text
base_T_camera_link = base_T_camera_optical · inv(camera_link_T_camera_optical)
```

因此不会给 `camera_color_optical_frame` 再添加一个与驱动冲突的父坐标系。
TF launch 必须保持运行；重启后也需重新启动它。不要同时运行两个不同标定结果。

检查：

```bash
rosrun tf tf_echo base_link camera_link
rosrun tf tf_echo base_link camera_color_optical_frame
```

如果要检查深度点云，先退出原来的相机 launch，再启动同一台 D435i：

```bash
roslaunch moveit_ctrl eye_to_hand_camera.launch enable_pointcloud:=true
```

在 RViz 设置 Fixed Frame 为 `base_link`，添加 PointCloud2，选 `/camera/depth/color/points`。
对照真实桌面和一个已测量位置/尺寸的固定盒子检查点云位置，再换几个机械臂姿态检查。
此时尚未配置 OctoMap 建图，RViz 点云显示本身不会使 MoveIt 自动避障；外参是下一步建图的基础。

这种基于机器人运动学的独立验证不能排除打印缩放、内参误差等共模系统偏差，
所以已知几何尺寸/位置的额外核对很有价值。相机或机械臂基座重新安装后应重新标定。
标定结束、完成验证后可以移除末端标定板；它不需要一直留在机械臂上。

## 10. 离线测试与参考

在包目录执行，不需要相机、CAN 或 ROS Master：

```bash
cd /home/hank/piper_ws/src/piper_ros/src/piper_moveit/moveit_ctrl
/usr/bin/python3 -m unittest discover -s tests -p 'test_eye_to_hand_*.py' -v
```

测试覆盖合成位姿的外参方向、噪声与退化运动、GridBoard投影/遮挡/错误ID、
打印板物理尺寸、静止采样筛选、独立验证及TF导出。
软件测试不能代替现场相机和机械臂的标定验收。

- [OpenCV hand-eye calibration API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)：求解公式与方法定义。
- [MoveIt 手眼标定教程](https://moveit.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html)：基座/末端/光学坐标系含义和多方向旋转采样原则；其示例是 eye-in-hand，本工具实现的是固定外部相机 eye-to-hand。
- [RealSense ROS1 驱动](https://github.com/realsenseai/realsense-ros/tree/ros1-legacy)：D435i 图像、CameraInfo 与相机内部 TF。

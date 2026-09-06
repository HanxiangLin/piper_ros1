# Piper + 末端安装 RealSense D435i：eye-in-hand 标定

本流程适用于：**D435i 随机械臂末端运动，标定板固定在桌面，机械臂基座固定。**
求出相机相对于安装 link 的固定外参，供后续把深度点云转换到机械臂基座坐标系使用。
不需要预先知道标定板在桌面上的精确位置，也不需要 YOLO、IMU 或手量相机安装偏移。

与旧工具的区别：

| 项目 | eye-to-hand（旧） | eye-in-hand（本流程） |
| --- | --- | --- |
| 相机 | 固定在机械臂外部 | 刚性安装在末端，随末端运动 |
| 标定板 | 固定在末端，随末端运动 | 固定在桌面，不随末端运动 |
| 求解外参 | `base_T_camera` | `ee_T_camera` |
| 导出 TF | `base_link → camera_link` | `安装 link → camera_link` |
| 采样/求解入口 | `eye_to_hand_*.py` | `eye_in_hand_*.py` |

旧工具和旧数据保留。两种模式不能混用数据，也不能仅重命名旧 JSON 后使用。
新的入口共享已有板检测、时间戳配对和停稳检查；求解数学方向与 TF 父节点按模式明确区分。

## 1. 安装相机和放置标定板

1. 将 D435i 用刚性支架固定在末端。镜头能看到工作区，不被夹爪或线缆遮挡。
   不能仅手持相机，也不要固定在会随夹爪开合改变位置的手指上。
2. 确认相机实际与哪个 URDF link 刚性连接。默认 `--ee-frame link6`，只适用于相机与 `link6`
   之间没有相对运动的安装。如果支架固定在 `link5`，所有训练、验证采样均用 `--ee-frame link5`。
   “靠近末端”不等于“一定属于 link6”；选错 link 会使相机外参不再恒定。
3. USB 线在支架上做应力释放，给关节活动留出余量，避免拉扯相机、缠绕机械臂或限制转动。
   检查相机和支架的重量、安装强度、末端负载及运动空间是否适合实机。
4. 继续使用 `calibration/board/board.pdf`：A4、实际大小/100% 打印，关闭“适合页面”。
   用尺检查 100 mm 标尺、单个黑色标记外边长 35 mm。默认标记净间距 8 mm，4列×3行。
5. 将纸平整贴在刚性、不反光的板材上，固定在桌面或稳定支架上。可以平放或倾斜，
   以相机从多个安全位姿都能看清为准；不要手持，不要让纸板弯曲或在桌面滑动。
6. **整个训练及独立验证期间，板相对于机械臂基座不动，相机相对于安装 link 不动。**
   若移动板、基座或重新安装相机，应新建会话，重新采样。标定完成后才可移走板进行环境扫描。

优先让全板可见，至少要清晰看到4个板上标记。让标记足够大、画面不过曝、无明显反光，
不要把相机贴得过近导致失焦，也不要始终从同一角度正对板。

当前默认机械臂 URDF **没有因本次适配而加入 D435i、支架或线缆的碰撞几何**。
标定 TF 不会自动补充碰撞模型。换姿前必须确认实际空间，并在后续依赖自动避障前，
按实际尺寸和安装方式补充相机/支架的 URDF collision 或 AttachedCollisionObject。
不要直接套用安装位置不匹配的相机 URDF。

## 2. 环境和启动

每个 ROS 终端执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/hank/piper_ws/devel/setup.bash
```

下文明确使用 `/usr/bin/python3`，避免 Conda base 缺少 OpenCV 的问题。
本机已有 ROS、NumPy、带 ArUco/calibrateHandEye 的 OpenCV、cv_bridge、tf/tf2_ros、
realsense2_camera；本次适配不引入新的第三方依赖。
源码修改后需重新注册脚本时，在上述环境运行：

```bash
cd /home/hank/piper_ws
catkin build moveit_ctrl --no-deps
source devel/setup.bash
```

### 终端一：真实机械臂反馈和 MoveIt

如果已经运行，不要重复启动。下面的启动文件会连接 CAN，并自动使能机械臂；
仅在相机安装牢固、急停可用、工作区清空并确认可使能后手动运行：

```bash
roslaunch piper_with_gripper_moveit piper_real.launch \
  can_port:=can0 auto_enable:=true
```

只允许用实机测量反馈采样，不使用 Gazebo/fake controller 的关节状态：

```bash
rostopic info /joint_states
rosrun tf tf_echo base_link link6
```

第二条中的 `link6` 要与实际安装 link 一致。`/joint_states` 应由实机驱动发布，
不是旧脚本发布的目标角度。真实关节位置改变时，基座到安装 link 的 TF 应相应改变。

### 终端二：相机

```bash
roslaunch moveit_ctrl eye_in_hand_camera.launch
```

新 launch 复用已验证的 D435i 流配置：彩色1280×720@30、深度848×480@30、内部 TF 开启、
IMU 和点云关闭。标定使用原始彩色图像及其 CameraInfo，无需深度和 YOLO。
如果旧的 `eye_to_hand_camera.launch` 正在运行，它提供的相机数据同样可用，**不要重复启动**。
多相机时加 `serial_no:=设备序列号` 选择同一台相机。

若此前启动过标定生成的 **`base_link → camera_link` 外参 TF**，在改成末端相机前，
由你停止那个旧外参发布节点/launch。不要停止 RealSense 驱动自身的内部 TF。
本工具不会自动停止任何节点，也不会临时发布猜测外参。

未标定时两条 TF 链分开是正常现象，采集只分别查询它们：

```text
base_link → 各机械臂关节 → 安装 link（默认 link6）
camera_link → RealSense 内部 TF → camera_color_optical_frame
```

## 3. 采集训练数据

在终端三加载 ROS 环境后，设置本次会话目录；重新标定时改成 `session_02` 等新目录。
`PIPER_CALIB_EE` 必须按实际安装位置设置，后续训练和验证使用相同值：

```bash
export PIPER_CALIB_PKG="$(rospack find moveit_ctrl)"
export PIPER_CALIB_RUN=/home/hank/piper_ws/calibration/eye_in_hand_d435i/session_01
export PIPER_CALIB_EE=link6
mkdir -p "$PIPER_CALIB_RUN"

/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_in_hand_collect.py" \
  --board "$PIPER_CALIB_PKG/calibration/board/board.json" \
  --ee-frame "$PIPER_CALIB_EE" \
  --output "$PIPER_CALIB_RUN/train.json"
```

在 RViz 添加 Image，话题选 `/eye_in_hand/preview`。采样程序只发布检测预览，
**不规划路径、不发布运动命令、不自动移动机械臂**。

通过现有 RViz/控制界面，在确认安全的范围低速改变末端位置和姿态，每次执行完成、
停稳后，在采样终端按 Enter 或 `s` 保存一组。`p` 看状态，`u` 撤销末组但保留图片，`q` 退出。

建议20–25个不同训练位姿，程序最低12个。采样要求：

- 保持板固定，移动的是相机和末端。包含左/右、上/下、远/近的可见位置。
- 改变相机朝向，包含绕至少两个不平行轴的旋转；不能只平移或只绕一个轴转。
  在视野和实机安全允许的范围形成明显的姿态差异，不必也不应强行到达不可达姿态。
- 避免所有图像都正对板或都集中在同一个画面区域，也避免极端斜视和模糊图像。
- 每按一次采样，只得到一个位姿样本：内部至少8帧、持续至少0.7秒的检查用于判断停稳，
  不是把同一姿态的多帧当成多个标定姿态。

脚本按图像时间戳查询 `base → 安装 link`，配对同帧的板位姿；不退回“最新 TF”凑数据。
同时检查图像/反馈新鲜度、关节稳定性、板检测稳定性、重投影误差和近重复位姿。
它不能识别你实际把相机安装在哪个 link，也不能保证机器人绝对定位精度。

每个成功样本保存到 JSON，并保留对应原图。中断后，确认相机/板/基座完全未动、
设置一致，才可给同一命令加 `--resume` 续采。更换模式或硬件布置必须新建数据集。

## 4. 离线求解

采足训练位姿后按 `q`，在同一终端运行：

```bash
/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_in_hand_solve.py" solve \
  --dataset "$PIPER_CALIB_RUN/train.json" \
  --output "$PIPER_CALIB_RUN/result.json"
```

输出包括 `ee_T_camera`、`ee_T_camera_link`、`base_T_target`、训练残差及姿态多样性检查。
`camera` 在矩阵键中指板检测使用的 **color optical frame**，不是相机外壳原点。
单位为米；`a_T_b` 表示把 b 坐标系中的点转换到 a 坐标系。

求解关系为：

```text
base_T_ee[i] × ee_T_camera × camera_T_target[i] = base_T_target（固定）
```

OpenCV `calibrateHandEye(..., method=CALIB_HAND_EYE_PARK)` 在本模式直接使用 `base_T_ee`，
返回 `ee_T_camera`，不对它取逆。旧 eye-to-hand 模式则对输入的 `base_T_ee` 取逆；
只改脚本名字、TF 父节点或把旧结果取逆都不能完成两种模式切换。

此步骤不生成 TF launch，更不会发布 TF。不要把训练残差小当成已完成实物精度验证。

## 5. 另采新位姿并验证

不要移动桌面上的板，也不要松动相机安装。换5–8个未用于训练的新末端位姿，另存文件：

```bash
/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_in_hand_collect.py" \
  --board "$PIPER_CALIB_PKG/calibration/board/board.json" \
  --ee-frame "$PIPER_CALIB_EE" \
  --output "$PIPER_CALIB_RUN/validation.json"
```

退出采样后验证：

```bash
/usr/bin/python3 "$PIPER_CALIB_PKG/scripts/eye_in_hand_solve.py" validate \
  --dataset "$PIPER_CALIB_RUN/validation.json" \
  --calibration "$PIPER_CALIB_RUN/result.json" \
  --output "$PIPER_CALIB_RUN/validation_report.json" \
  --launch-output "$PIPER_CALIB_RUN/camera_extrinsics.launch"
```

验证固定使用训练得到的相机外参和板在基座中的位姿，**不在验证数据上重新拟合板位置**，
因此板在训练和验证之间被挪动不会被重新求平均掩盖。复用训练样本、模式/坐标/相机配置
不一致会被拒绝。默认每个样本的位置一致性误差最大10 mm、角度最大2°；可用
`--max-translation`（米）、`--max-rotation`（度）设定更适合任务的阈值。

默认阈值只是入门的一致性门槛，不是避障间隙、机器人绝对精度或相机精度保证。
若不通过，检查安装、图像、板尺寸、选定 link 和采样姿态，不应仅放宽阈值来获得通过。
失败时保存逐样本报告但不生成 launch；重新运行应选择新的输出文件名，工具拒绝覆盖结果。
`result.json` 保留为未经验证的原始拟合记录；是否通过以带结果文件哈希的验证报告为准。

## 6. 手动加载验证通过的 TF

确认没有任何其他节点/URDF 已在发布 `camera_link` 的父变换，再手动启动：

```bash
roslaunch "$PIPER_CALIB_RUN/camera_extrinsics.launch"
```

默认形成：

```text
base_link → 各机械臂关节 → link6
                             └→ camera_link → camera_color_optical_frame
                                            → camera_depth_optical_frame
```

静态的是 **安装 link 到 camera_link**；相机相对于基座仍然随机械臂运动。
工具先计算 `ee_T_camera_link = ee_T_camera × inverse(camera_link_T_optical)`，
再导出安装 link 到 camera_link 的 TF，不覆盖驱动拥有的 optical frame 内部变换。
若 `--ee-frame link5`，导出的父节点就是 `link5`。

在另一个已加载环境的终端检查：

```bash
rosrun tf tf_echo link6 camera_link
rosrun tf tf_echo base_link camera_color_optical_frame
```

第一条中的 link 应与实际安装一致；它的外参恒定。第二条应随着实机关节反馈变化。
一次运行一条检查命令，Ctrl+C 结束后再运行下一条。
重新安装相机、改变安装 link 或支架滑动后必须重新标定，不能继续使用旧外参。

## 7. 与后续静态环境扫描的衔接

手眼外参和点云稳定性检查通过后，按照
[静态预扫描与 MoveIt OctoMap 指南](STATIC_SCENE_D435I.md) 启动受控扫描门、建立并冻结地图。

标定成功后，相机点云可沿下面的时间相关变换转换到基座：

```text
base_T_camera(t) = base_T_ee(t) × ee_T_camera
```

若点云 frame 是深度 optical frame，应使用它自己的 TF 链，不能直接当成彩色 optical frame。
多视角预扫描需要把每帧点云按**采集时间戳**变换到固定的 `base_link` 再融合，
剔除机械臂/夹爪/相机自身等不应成为静态环境的点，最后生成并固定 MoveIt Planning Scene/OctoMap。
不能把相机局部坐标中的点直接累积，也不能一直用标定时某个固定的 `base_T_camera`。

本次只完成 eye-in-hand 标定工具适配，**未实现自动扫描、地图融合或自动避障**。
后续使用静态地图规划时，还需要实际相机/支架碰撞模型、环境覆盖检查及合适的碰撞余量；
未被相机看到、透明/反光导致漏测的区域和扫描后新增的障碍物不会因此自动得到保护。

参考：[MoveIt 官方手眼标定教程](https://moveit.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html)。

## 8. 离线检查

下面只做软件测试，不启动 ROS 节点，不使能或移动机械臂：

```bash
cd /home/hank/piper_ws
/usr/bin/python3 -m unittest discover \
  -s src/piper_ros/src/piper_moveit/moveit_ctrl/tests -p 'test_eye*.py' -v
```

测试覆盖两种模式的外参方向、带噪声数据、退化姿态、板移动/相机松动残差、
跨模式拒绝、采样元数据、独立验证、TF 安装父节点和 optical→camera_link 转换。
这些合成数据检查不能替代实际安装后的独立采样验证。

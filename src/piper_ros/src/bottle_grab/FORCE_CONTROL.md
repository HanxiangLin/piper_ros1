# RPPS-200 力控抓取说明

本文对应 `moveit_grab.py` 的 **单侧 RPPS-200、0–20 N、默认 8 N、闭环 + 运输保压** 版本。
不接传感器时把 `force_control` 设为 `false`，程序回到原来的“固定开口 + `gripper_effort`”流程。

## 1. 它到底控制什么

- 控制量：传感器一侧的**法向力** `F = Σ(p_i × A_taxel)`，单位 N。
- 不是 `gripper_effort`。`gripper_effort` 仍作为驱动器力矩上限（0.5–2.0），
  力的大小由“逐步闭合 + 传感器反馈”决定。
- 传感器只贴一侧时，该侧法向力就等于夹持力（另一侧法向力在准静态下相等），
  所以不需要乘 2。

## 2. 接线与依赖

传感器是 USB 虚拟串口，Linux 下通常出现为：

```text
/dev/ttyACM0      # CDC
/dev/ttyUSB0      # CH340/CP210x 等
```

参数默认 **2000000 baud、8N1、无流控**。先确认用户有串口权限：

```bash
sudo usermod -aG dialout $USER   # 重新登录后生效
ls -l /dev/ttyACM* /dev/ttyUSB*
```

安装依赖（机器人上的 Python 3）：

```bash
python3 -m pip install pyserial
```

注意：厂商 Windows 上位机不要和 ROS 同时占用同一个串口。

## 3. 先不启动机械臂，验证传感器

```bash
cd ~/piper_ws/src/piper_ros/src/bottle_grab
python3 rpps200_monitor.py --list
python3 rpps200_monitor.py --port /dev/ttyACM0
# 或自动扫描：
python3 rpps200_monitor.py
```

正常应看到 `fps≈100`，空载 `F≈0 N`。把已知重量（例如 500 g 砝码 + 平板）
均匀压在感应区，读数应接近 `4.9 N`；否则用比例系数修正：

```text
rpps_force_scale = 已知重量(N) / 显示的力(N)
```

默认标定文件：

```text
calibration/rpps200/rpps200_10x20_calibration.csv
```

它是厂商“查表”导出：逐测点 ADC→kPa，已去掉 15 kPa 显示偏移。
换传感器/重新标定后，用 `rpps_calibration` 指向新文件。

## 4. 关键参数

| 参数 | 默认 | 含义 |
|---|---:|---|
| `force_control` | `true` | 是否启用传感器闭环；`false` 回退旧流程 |
| `grasp_force_n` | `8.0` | 目标单侧法向力，N（0.5–20） |
| `grasp_force_tolerance_n` | `1.0` | 目标力容差，N |
| `grasp_force_max_n` | `12.0` | 安全上限，超过会回退/中止 |
| `grasp_force_min_n` | `4.0` | 保压下限，低于会补力/中止 |
| `rpps_port` | `""` | 空=自动扫描；也可写 `/dev/...` |
| `rpps_baud` | `2000000` | 串口波特率 |
| `rpps_calibration` | 仓库内 CSV | 厂商查表路径 |
| `rpps_taxel_area_mm2` | `2.25` | 单点面积；30×15 mm / 200 点，节距 1.5 mm |
| `rpps_force_scale` | `1.0` | 力比例系数，用已知重量校准 |
| `rpps_sensor_timeout` | `0.5` | 超过多久没新帧算故障，s |
| `rpps_tare_frames` | `30` | 零点标定帧数 |
| `force_step_coarse_m` | `0.001` | 接触前粗闭合步长，m |
| `force_step_fine_m` | `0.0002` | 接触后精闭合步长，m |
| `force_settle_time` | `0.10` | 每步等待，s |
| `force_max_squeeze_m` | `0.008` | 相对接触点的最大压缩量，m |
| `force_hold_period` | `0.15` | 运输保压周期，s |
| `force_hold_step_m` | `0.0001` | 保压单次调节量，m |
| `force_hold_deadband_n` | `0.5` | 保压死区，N |

> 8 N 只是初始值：0.5 L 满瓶、表面光滑或夹指磨损时，可能需要 12–15 N 才不打滑；
> 薄壁/易碎瓶则先降到 5 N 并用 `dry_run` 之外的低速试抓。

示例：命令行临时把目标改成 12 N（私有参数优先于 YAML）：

```bash
python3 moveit_grab.py _params_file:=/home/hank/piper_ws/src/piper_ros/src/bottle_grab/moveit_grab_params.yaml _grasp_force_n:=12.0
```

## 5. 闭环流程

1. **零点标定**：扫描已经把夹爪张开；在接近瓶子、尚未接触时采集
   `rpps_tare_frames` 帧，逐测点取 ADC 中位数作为基线。夹爪内有物体时
   标定会不准确，程序会打印标定前力提示。
2. **粗闭合**：从当前开口按 `force_step_coarse_m` 逐步闭合，直到
   `F ≥ contact_n`（目标力的 10%，最多 0.5 N）。
3. **精闭合**：以接触开口为基准，最多再压缩 `force_max_squeeze_m`；
   按 `force_step_fine_m` 闭合到目标带下沿。
4. **微调**：在 `[target-tolerance, target+tolerance]` 内连续稳定
   `stable_cycles` 次；超上限会先回退，仍超限则中止；到行程边界仍不达标也中止。
5. **确认**：`confirm()` 在力控模式下检查开口带 + 实测力稳定，不再依赖
   `joint7` 的 effort 阈值。
6. **运输保压**：附着后启动后台线程，每 `force_hold_period` 检查一次；
   力低于目标死区就闭合 `force_hold_step_m`，高于就张开；超上限/低于下限
   或传感器超时立即置故障并停止保压。
7. **松手**：`place_task` 先 `assert_force_healthy()`，再停止保压，最后张开。
   任何力控故障都会在抬升/放置/松手前中止，**保持当前夹持，不自动松开**。

## 6. 故障与安全

- 传感器打不开、没有新帧、帧校验持续失败 → 启动或抓取阶段直接报错；
  运行中置 `force_fault`，后续 `assert_force_healthy()` 抛异常，程序不抬升/不松手。
- 力超过 `grasp_force_max_n` → 先尝试张开回退；无法回退则中止。
- 闭合到最小开口仍未接触 → 自动张回 `gripper_opening_open` 后中止。
- 夹爪开口/力反馈丢失 → 中止，保持最后一条夹爪指令。
- 需要人工松开时（确认瓶子不会掉）：

```bash
rosservice call /gripper_srv "{gripper_angle: 0.07, gripper_effort: 1.0, gripper_code: 1, set_zero: 0}"
```

## 7. 调参建议

- **达不到目标力**：先确认 `gripper_effort` 不过低（可到 2.0），再增大
  `force_max_squeeze_m`；仍达不到说明瓶子太硬/接触太少，检查传感器是否贴紧。
- **一闭合就超上限**：减小 `force_step_coarse_m` / `force_step_fine_m`，
  降低 `gripper_effort`，或增大 `force_settle_time` 让力稳定后再判断。
- **保压来回抖**：增大 `force_hold_deadband_n`，减小 `force_hold_step_m`，
  或增大 `force_hold_period`。
- **空载力不为 0**：先跑 monitor 看 `F`；确认夹爪内无物后重新标定。
  若某个测点噪声大，可提高 `rpps_tare_frames`；机械安装预紧会在标定中被扣掉。
- **力读数整体偏大/偏小**：用已知重量修正 `rpps_force_scale`；
  不要改 `rpps_taxel_area_mm2` 来“凑”力，除非确认传感器感应区尺寸不同。

## 8. 离线测试

不需要 ROS、不需要传感器：

```bash
cd ~/piper_ws/src/piper_ros/src/bottle_grab
python3 -m unittest discover -s tests -p 'test_rpps200.py' -v
python3 -m unittest discover -s tests -p 'test_force_control.py' -v
```

## 9. 已知限制

- 只支持 10×20 / 200 点 RPPS-200；CSV 的 `Rows/Cols` 必须是 10/20。
- 只按单侧法向力控制；不估计摩擦系数、不检测滑移。
- 压力标定的低端每通道阈值不同，很轻的接触可能读 0；目标力 5 N 以上一般没问题。
- 2 Mbaud 对 USB 线较长/干扰敏感；出现大量校验错误时先换短屏蔽线。

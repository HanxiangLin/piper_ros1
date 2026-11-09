# RPPS-200 10x20 标定表

- 文件：`rpps200_10x20_calibration.csv`
- 来源：厂商 RPPS-200“校准版”资料中的
  `1794233190351353查表_10x20_20261104_03.csv`
- 传感器：RPPS-200，10×20 = 200 测点，感应区 30×15 mm，节距 1.5 mm
- 标定日期：2026-11-04
- 文件 SHA256：`13483B347A159FCD159D2EA3588ACED93E5DCE04AF2F219080C4318A8B4ECC73`

CSV 第一行是厂商元数据（`Rows=10, Cols=20, KpaCorrected=0,
DisplayPressureOffsetKpa=15.0`）；第二行是表头；之后每行是施加压力（kPa）
对应的 200 个原始 ADC。驱动按厂商 `AdcKpaLookup` 的规则：

1. 显示压力 = `max(0, 文件kPa - 15)`；
2. 每个测点单独做 ADC→kPa 分段线性插值；
3. 低于第一个标定点时按原点线性外推，高于最后一个点时取最后一个值。

单点面积按 `30 mm × 15 mm / 200 = 2.25 mm²` 计算总法向力：

```text
F[N] = Σ p_i[kPa] × 1000 × 2.25e-6 × force_scale
```

如果换传感器或重新标定，请把新 CSV 放到本目录，并把
`moveit_grab_params.yaml` 的 `rpps_calibration` 指向它；不要手工改这个文件。

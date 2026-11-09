#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone RPPS-200 monitor: verify port, frame rate, tare and force.

Run this on the ROS Linux machine *before* starting MoveIt; it does not need
ROS and does not touch the arm::

    python3 rpps200_monitor.py --port /dev/ttyACM0
    python3 rpps200_monitor.py                 # auto-detect /dev/ttyACM* etc.
    python3 rpps200_monitor.py --list

The printed force is the sum of all 200 taxel pressures times the nominal
taxel area.  Put a known weight on a flat plate covering the sensor to check
``--scale``: scale = known_weight_N / displayed_force_N.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from rpps200 import (
    DEFAULT_AREA_MM2,
    DEFAULT_BAUD,
    POINTS,
    RPPS200Calibration,
    RPPS200Sensor,
    SensorError,
    candidate_serial_ports,
)

DEFAULT_CALIBRATION = str(
    Path(__file__).resolve().parents[4]
    / "calibration/rpps200/rpps200_10x20_calibration.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RPPS-200 10x20 串口/标定/力显示工具（不启动ROS）")
    parser.add_argument("--port", default="",
                        help="串口设备；空=自动扫描 /dev/ttyACM*、/dev/ttyUSB*")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help="波特率，默认2000000")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION,
                        help="厂商查表CSV路径")
    parser.add_argument("--area-mm2", type=float, default=DEFAULT_AREA_MM2,
                        help="单个测点面积(mm^2)，默认2.25")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="力比例系数，默认1.0")
    parser.add_argument("--contact-kpa", type=float, default=1.0,
                        help="接触点判定阈值(kPa)")
    parser.add_argument("--tare-frames", type=int, default=30,
                        help="零点标定帧数，默认30")
    parser.add_argument("--no-tare", action="store_true",
                        help="不执行零点标定")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行秒数，0=一直运行")
    parser.add_argument("--csv", default="",
                        help="可选：把力/压力写入CSV")
    parser.add_argument("--list", action="store_true",
                        help="只列出候选串口并退出")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        ports = candidate_serial_ports()
        print("候选串口:")
        for port in ports:
            print("  " + port)
        if not ports:
            print("  （无；检查USB线、lsusb、dmesg | tail）")
        return 0

    if not os.path.isfile(args.calibration):
        print("标定文件不存在: %s" % args.calibration, file=sys.stderr)
        return 2
    calibration = RPPS200Calibration.load(args.calibration)
    print("标定: " + calibration.summary())
    print("测点数: %d；单点面积: %.3f mm^2；力比例: %.3f"
          % (POINTS, args.area_mm2, args.scale))

    sensor = RPPS200Sensor(
        calibration, port=args.port, baud=args.baud,
        area_mm2=args.area_mm2, force_scale=args.scale,
        contact_kpa=args.contact_kpa)
    try:
        port, baud = sensor.start()
    except SensorError as error:
        print("打开RPPS-200失败: %s" % error, file=sys.stderr)
        print("候选串口: %s" % (candidate_serial_ports() or "无"),
              file=sys.stderr)
        return 3
    print("已连接: %s @ %d 8N1" % (port, baud))

    writer = None
    csv_stream = None
    try:
        if not args.no_tare:
            time.sleep(0.2)
            pre_force = sensor.tare(frames=max(3, args.tare_frames))
            print("零点标定完成；标定前力 %.2f N（空载应接近0）" % pre_force)
        if args.csv:
            csv_stream = open(args.csv, "w", newline="",
                              encoding="utf-8-sig")
            writer = csv.writer(csv_stream)
            writer.writerow(
                ["timestamp", "force_n", "max_kpa", "contact_count"]
                + ["r%dc%d" % (index // 20 + 1, index % 20 + 1)
                   for index in range(POINTS)])
        started = time.monotonic()
        last_print = 0.0
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            reading = sensor.latest(max_age=1.0)
            now = time.monotonic()
            if reading is None:
                if sensor.last_error:
                    print("传感器错误: %s" % sensor.last_error,
                          file=sys.stderr)
                    return 4
                time.sleep(0.05)
                continue
            if now - last_print >= 0.2:
                last_print = now
                print("t=%5.1fs  F=%6.2f N  max=%7.1f kPa  接触=%3d/200  "
                      "fps=%5.1f"
                      % (now - started, reading.force_n, reading.max_kpa,
                         reading.contact_count, sensor.fps()))
            if writer is not None:
                writer.writerow(
                    ["%.6f" % time.time(), "%.4f" % reading.force_n,
                     "%.3f" % reading.max_kpa, reading.contact_count]
                    + ["%.2f" % value for value in reading.kpa])
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n已停止")
    except SensorError as error:
        print("传感器错误: %s" % error, file=sys.stderr)
        return 4
    finally:
        sensor.stop()
        if csv_stream is not None:
            csv_stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

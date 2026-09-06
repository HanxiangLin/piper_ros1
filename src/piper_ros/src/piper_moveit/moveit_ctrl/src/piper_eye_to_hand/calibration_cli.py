#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline hand-eye solve and independent validation; never publishes TF."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

from piper_eye_to_hand.board import Board
from piper_eye_to_hand.geometry import (
    evaluate_samples, invert_transform, quaternion_xyzw, solve_eye_to_hand,
    evaluate_eye_in_hand_samples, solve_eye_in_hand,
)
from piper_eye_to_hand.sampling import (
    MODES, atomic_write_json, check_mode, load_dataset, mode_format, utc_now,
)


def fingerprint(sample):
    matrices = [np.asarray(sample[key], dtype=float).round(10).tolist()
                for key in ("base_T_ee", "camera_T_target")]
    return hashlib.sha256(json.dumps(matrices, allow_nan=False).encode("utf-8")).hexdigest()


def checked_dataset(path, mode="eye_to_hand"):
    data = load_dataset(path, expected_mode=mode)
    Board(data["board"])
    frames = data["frames"]
    required = ("base", "ee", "camera_optical", "camera_link")
    for key in required:
        if not isinstance(frames.get(key), str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_/]*", frames[key]):
            raise ValueError("无效 TF 坐标系：{}".format(key))
    if len({frames[key] for key in required}) != 4:
        raise ValueError("四个坐标系必须互不相同；camera_link 不能设为 optical frame")
    invert_transform(data["camera_link_T_optical"])
    k = np.asarray(data["camera_info"]["K"], dtype=float)
    d = np.asarray(data["camera_info"]["D"], dtype=float)
    if k.size != 9 or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError("相机内参无效")
    ids, hashes = set(), set()
    for sample in data["samples"]:
        invert_transform(sample["base_T_ee"])
        invert_transform(sample["camera_T_target"])
        if not isinstance(sample.get("id"), str) or not sample["id"]:
            raise ValueError("样本缺少 id")
        stamp = float(sample["stamp"])
        if not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("样本时间戳无效")
        digest = fingerprint(sample)
        if sample["id"] in ids or digest in hashes:
            raise ValueError("数据集中存在重复样本")
        ids.add(sample["id"])
        hashes.add(digest)
    return data


def prepare_output(path, inputs=()):
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("输出已存在，选择新文件名以保留之前结果：{}".format(path))
    if any(path.resolve() == Path(item).expanduser().resolve() for item in inputs):
        raise ValueError("输出不能覆盖输入文件")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def print_residuals(label, report):
    print("{}：{} 个样本，平移 RMS {:.2f} mm / 最大 {:.2f} mm；旋转 RMS {:.3f}° / 最大 {:.3f}°".format(
        label, report["sample_count"], report["translation"]["rms"] * 1000,
        report["translation"]["max"] * 1000, report["rotation"]["rms"], report["rotation"]["max"]))


def solve_command(args):
    data = checked_dataset(args.dataset, mode=args.mode)
    output = prepare_output(args.output, [args.dataset])
    eye_in_hand = args.mode == "eye_in_hand"
    solve_function = solve_eye_in_hand if eye_in_hand else solve_eye_to_hand
    camera_key = "ee_T_camera" if eye_in_hand else "base_T_camera"
    link_key = "ee_T_camera_link" if eye_in_hand else "base_T_camera_link"
    result = solve_function(data["samples"], min_samples=12)
    link_transform = (np.asarray(result[camera_key]) @
                      invert_transform(data["camera_link_T_optical"]))
    # Validate and normalize interpretation before writing a result.
    quaternion_xyzw(link_transform)
    result.update({
        "format": mode_format(args.mode, "calibration"), "mode": args.mode,
        "created_at": utc_now(), "validated": False,
        "transform_convention": "a_T_b maps coordinates from b into a; translation is metres",
        "frames": data["frames"], "board": data["board"], "camera_info": data["camera_info"],
        "camera_link_T_optical": data["camera_link_T_optical"],
        link_key: link_transform.tolist(),
        "tf_parent": data["frames"]["ee" if eye_in_hand else "base"],
        "training_samples": [{"id": sample["id"], "stamp": sample["stamp"],
                              "sha256": fingerprint(sample), "base_T_ee": sample["base_T_ee"]}
                             for sample in data["samples"]],
        "source_dataset": str(Path(args.dataset).expanduser().resolve()),
        "note": "训练拟合结果；必须另采新位姿验证。残差不等于绝对定位精度。",
    })
    atomic_write_json(output, result)
    print_residuals("训练拟合", result["training_residuals"])
    print("已保存 {}；尚未生成或发布 TF。".format(output))
    if eye_in_hand:
        print("保持桌面标定板位置、相机相对末端安装不变，换至少5个新末端位姿后执行 validate。")
    else:
        print("保持标定板安装和相机位置不变，另外采集至少5个新位姿后执行 validate。")
    return 0


def validation_report(data, calibration, max_translation, max_rotation, mode="eye_to_hand"):
    check_mode(data, mode)
    check_mode(calibration, mode, "calibration")
    if len(data["samples"]) < 5:
        raise ValueError("独立验证至少需要5个新位姿")
    for key in ("frames", "board", "camera_info"):
        if data[key] != calibration[key]:
            raise ValueError("训练与验证 {} 不一致".format(key))
    if not np.allclose(data["camera_link_T_optical"], calibration["camera_link_T_optical"],
                       atol=1e-7, rtol=0):
        raise ValueError("训练与验证的相机内部 TF 不一致")
    train_ids = {row["id"] for row in calibration["training_samples"]}
    train_hashes = {row["sha256"] for row in calibration["training_samples"]}
    train_stamps = {row["stamp"] for row in calibration["training_samples"]}
    for sample in data["samples"]:
        if (sample["id"] in train_ids or sample["stamp"] in train_stamps
                or fingerprint(sample) in train_hashes):
            raise ValueError("验证数据包含训练样本；请另采5–8个新位姿，不要重命名复制训练文件")
    if mode == "eye_in_hand":
        report = evaluate_eye_in_hand_samples(
            data["samples"], calibration["ee_T_camera"], calibration["base_T_target"])
    else:
        report = evaluate_samples(data["samples"], calibration["base_T_camera"], calibration["ee_T_target"])
    report.update({
        "format": mode_format(mode, "validation"), "mode": mode, "created_at": utc_now(),
        "thresholds": {"max_translation_m": max_translation, "max_rotation_deg": max_rotation},
        "passed": (report["translation"]["max"] <= max_translation
                   and report["rotation"]["max"] <= max_rotation),
        "note": "独立位姿一致性验证；不检测共模系统误差，不等于避障安全间隙或绝对精度保证。",
    })
    return report


def static_launch(calibration, mode="eye_to_hand"):
    check_mode(calibration, mode, "calibration")
    # Publish to camera_link, not the optical frame owned by the RealSense driver.
    # Eye-in-hand's parent MUST move with the physical mounting link.
    camera_key = "ee_T_camera" if mode == "eye_in_hand" else "base_T_camera"
    parent_key = "ee" if mode == "eye_in_hand" else "base"
    transform = (np.asarray(calibration[camera_key], dtype=float) @
                 invert_transform(calibration["camera_link_T_optical"]))
    quat = quaternion_xyzw(transform)
    values = transform[:3, 3].tolist() + quat
    frames = calibration["frames"]
    for key in (parent_key, "camera_link"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_/]*", frames[key]):
            raise ValueError("无效静态 TF frame")
    if frames[parent_key] == frames["camera_link"]:
        raise ValueError("静态 TF 的父子 frame 不能相同")
    args = " ".join("{:.12g}".format(value) for value in values)
    args += " {} {}".format(frames[parent_key], frames["camera_link"])
    root = ET.Element("launch")
    root.append(ET.Comment("Generated after independent validation. Run only one parent transform for camera_link."))
    ET.SubElement(root, "node", {"pkg": "tf2_ros", "type": "static_transform_publisher",
                                "name": "piper_{}_camera_tf".format(mode), "args": args})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def validate_command(args):
    data = checked_dataset(args.dataset, mode=args.mode)
    with open(args.calibration, encoding="utf-8") as handle:
        calibration = json.load(handle)
    output = prepare_output(args.output, [args.dataset, args.calibration])
    launch_output = prepare_output(args.launch_output, [args.dataset, args.calibration, args.output])
    if output == launch_output:
        raise ValueError("报告与 launch 必须使用不同路径")
    report = validation_report(data, calibration, args.max_translation, args.max_rotation, mode=args.mode)
    report["calibration_file"] = str(Path(args.calibration).expanduser().resolve())
    report["calibration_sha256"] = hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()
    report["validation_dataset"] = str(Path(args.dataset).expanduser().resolve())
    atomic_write_json(output, report)
    print_residuals("独立验证", report)
    if not report["passed"]:
        print("验证未通过；已保存逐样本误差报告，没有生成 TF launch。", file=sys.stderr)
        return 2
    content = static_launch(calibration, mode=args.mode)
    with launch_output.open("xb") as handle:
        handle.write(content)
        handle.write(b"\n")
    print("验证通过，已生成 {}。尚未启动任何 TF 节点。".format(launch_output))
    print("请先确认没有其他节点发布同一个 camera_link 的父变换，再手动 roslaunch 此文件。")
    return 0


def main(argv=None, mode="eye_to_hand"):
    if mode not in MODES:
        raise ValueError("未知标定模式")
    parser = argparse.ArgumentParser(description="{}: {}".format(mode, __doc__))
    sub = parser.add_subparsers(dest="command", required=True)
    solve = sub.add_parser("solve", help="用至少12个训练位姿求外参，建议20–25")
    solve.add_argument("--dataset", required=True)
    solve.add_argument("--output", required=True)
    validate = sub.add_parser("validate", help="用至少5个独立新位姿验证；通过才生成TF launch")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--calibration", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--launch-output", required=True)
    validate.add_argument("--max-translation", type=float, default=0.01, help="默认最大一致性误差0.01米")
    validate.add_argument("--max-rotation", type=float, default=2.0, help="默认最大一致性角误差2度")
    args = parser.parse_args(argv)
    args.mode = mode
    try:
        if args.command == "validate":
            if not all(math.isfinite(value) and value > 0 for value in (args.max_translation, args.max_rotation)):
                raise ValueError("验证阈值必须是有限正数")
            return validate_command(args)
        return solve_command(args)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print("标定失败：{}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

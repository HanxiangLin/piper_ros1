"""ROS-independent stationary-burst checks and atomic dataset persistence."""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

FORMAT = "piper_eye_to_hand.samples.v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pose_distance(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if (first.shape != (4, 4) or second.shape != (4, 4)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("位姿必须是有限的 4×4 矩阵")
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = cv2.Rodrigues(first[:3, :3].T @ second[:3, :3])[0]
    return translation, math.degrees(float(np.linalg.norm(rotation)))


def choose_stationary_frame(frames, min_frames=8, min_duration=0.7,
                            max_robot_translation=0.002,
                            max_robot_rotation_deg=0.5,
                            max_target_translation=0.004,
                            max_target_rotation_deg=1.0,
                            max_joint_span=0.005):
    """Reject moving, duplicated or jittery bursts; keep one real paired frame.

    This is an acquisition stability test, not a guarantee of robot accuracy.
    Choosing a measured pair avoids independently averaging camera and arm poses.
    """
    if len(frames) < min_frames:
        raise ValueError("有效帧不足：{} / {}".format(len(frames), min_frames))
    stamps = [float(frame["stamp"]) for frame in frames]
    if not all(math.isfinite(value) for value in stamps):
        raise ValueError("时间戳包含非有限数值")
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("图像时间戳重复或倒退，请重新采样")
    if any(b - a > 0.35 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("有效图像间隔超过0.35秒，请保持标定板持续可见后重新采样")
    duration = stamps[-1] - stamps[0]
    if duration < min_duration:
        raise ValueError("稳定观测时间不足 {:.2f} s".format(min_duration))

    # Check all pairs, rather than only distances to the first frame.
    maxima = {"robot_translation_m": 0.0, "robot_rotation_deg": 0.0,
              "target_translation_m": 0.0, "target_rotation_deg": 0.0}
    for i, first in enumerate(frames):
        for second in frames[i + 1:]:
            for name, key in (("robot", "base_T_ee"), ("target", "camera_T_target")):
                trans, angle = pose_distance(first[key], second[key])
                maxima[name + "_translation_m"] = max(maxima[name + "_translation_m"], trans)
                maxima[name + "_rotation_deg"] = max(maxima[name + "_rotation_deg"], angle)
    thresholds = {"robot_translation_m": max_robot_translation,
                  "robot_rotation_deg": max_robot_rotation_deg,
                  "target_translation_m": max_target_translation,
                  "target_rotation_deg": max_target_rotation_deg}
    for key, limit in thresholds.items():
        if maxima[key] > limit:
            raise ValueError("机械臂未停稳或标定板检测抖动：{}={:.5f}，限值={:.5f}".format(
                key, maxima[key], limit))

    joints = np.asarray([frame["joint_positions"] for frame in frames], dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 6 or not np.isfinite(joints).all():
        raise ValueError("需要六个有限的机械臂关节位置")
    joint_span = float(np.max(np.ptp(joints, axis=0)))
    if joint_span > max_joint_span:
        raise ValueError("采样期间关节仍在运动：最大变化 {:.5f} rad".format(joint_span))
    errors = [float(frame["reprojection_error_px"]) for frame in frames]
    if not all(math.isfinite(value) and value >= 0 for value in errors):
        raise ValueError("无效重投影误差")
    chosen = frames[int(np.argmin(errors))]
    quality = dict(maxima, joint_span_rad=joint_span,
                   frame_count=len(frames), duration_s=duration)
    return chosen, quality


def is_duplicate_pose(samples, candidate, min_translation=0.008, min_rotation_deg=4.0):
    for sample in samples:
        trans, angle = pose_distance(sample["base_T_ee"], candidate["base_T_ee"])
        if trans < min_translation and angle < min_rotation_deg:
            return True
    return False


def atomic_write_json(path, data):
    """Replace this tool's dataset atomically; callers decide overwrite policy."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("拒绝写入符号链接：{}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                         prefix=".eye_to_hand_", suffix=".json", delete=False) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
        temporary = None
    finally:
        if temporary is not None and os.path.isfile(temporary):
            os.unlink(temporary)


def load_dataset(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("samples"), list):
        raise ValueError("不是 eye-to-hand 采样文件")
    for field in ("board", "frames", "camera_info", "camera_link_T_optical"):
        if field not in data:
            raise ValueError("采样文件缺少必要字段：{}".format(field))
    return data

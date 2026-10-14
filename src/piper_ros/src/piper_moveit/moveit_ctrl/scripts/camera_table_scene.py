#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""相机估计局部支撑桌面，加载table，并转发其余点云；不控制机械臂。"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPlanningSceneRequest
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from shape_msgs.msg import SolidPrimitive


def numeric(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " 必须填有限数字，不能保留MEASURE")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(label + " 包含非有限数值")
    return value


def numbers(value, count, label):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(label + " 长度错误")
    return np.array([numeric(v, label) for v in value], dtype=float)


def check_config(config):
    lo = numbers(config["roi_min"], 3, "roi_min")
    hi = numbers(config["roi_max"], 3, "roi_max")
    if np.any(hi <= lo):
        raise ValueError("roi_max必须大于roi_min")
    for key in ("support_depth_m", "plane_threshold_m", "voxel_m", "max_age_s"):
        if numeric(config[key], key) <= 0:
            raise ValueError(key + " 必须大于0")
    for key in ("edge_padding_m", "mask_padding_m"):
        if numeric(config[key], key) < 0:
            raise ValueError(key + " 不能为负")
    if not 0 < numeric(config["max_tilt_deg"], "max_tilt_deg") <= 10:
        raise ValueError("此示例仅处理近水平桌面，max_tilt_deg必须在(0,10]内")
    if not 0 < numeric(config["min_inlier_ratio"], "min_inlier_ratio") <= 1:
        raise ValueError("min_inlier_ratio必须在(0,1]内")
    for key in ("min_inliers", "ransac_iterations"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(key + " 必须是至少3的整数")
    if not isinstance(config["input_cloud"], str) or not config["input_cloud"]:
        raise ValueError("input_cloud为空")


def inside(points, lower, upper):
    return np.all((points >= lower) & (points <= upper), axis=1)


def xyz_to_base(points, transform):
    q = transform.transform.rotation
    matrix = tf_trans.quaternion_matrix([q.x, q.y, q.z, q.w])
    t = transform.transform.translation
    return points @ matrix[:3, :3].T + [t.x, t.y, t.z]


def downsample(points, voxel_m):
    if not len(points):
        return points
    cells = np.floor(points / voxel_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[indices]


def fit_table(points, config):
    check_config(config)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[inside(points, np.array(config["roi_min"]),
                           np.array(config["roi_max"]))]
    points = downsample(points, config["voxel_m"])
    rng = np.random.default_rng(42)
    if len(points) > 30000:
        points = points[rng.choice(len(points), 30000, replace=False)]
    if len(points) < config["min_inliers"]:
        raise RuntimeError("ROI内点不足；检查TF、ROI和扫描覆盖")
    threshold = config["plane_threshold_m"]
    cosine = math.cos(math.radians(config["max_tilt_deg"]))
    best = None
    for _ in range(config["ransac_iterations"]):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < cosine:
            continue
        offset = -np.dot(normal, a)
        inliers = np.abs(points @ normal + offset) <= threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < config["min_inliers"]:
        raise RuntimeError("未找到足够大的近水平平面")
    if best.mean() < config["min_inlier_ratio"]:
        raise RuntimeError("平面占比不足；缩小到真实桌面ROI或改善扫描")
    selected = points[best]
    center = selected.mean(axis=0)
    _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < cosine:
        raise RuntimeError("拟合后桌面倾角超限")
    offset = -np.dot(normal, center)
    selected = points[np.abs(points @ normal + offset) <= threshold]
    if len(selected) < config["min_inliers"]:
        raise RuntimeError("精修后平面点不足")
    if len(selected) / len(points) < config["min_inlier_ratio"]:
        raise RuntimeError("精修后平面占比不足")
    span = np.ptp(selected[:, :2], axis=0)
    if np.any(span < 0.05):
        raise RuntimeError("桌面可见范围太小，重新采集")
    lower = selected.min(axis=0)
    upper = selected.max(axis=0)
    lower[:2] -= config["edge_padding_m"]
    upper[:2] += config["edge_padding_m"]
    lower[2] -= config["support_depth_m"] + config["mask_padding_m"]
    upper[2] += config["mask_padding_m"]
    return {
        "version": 1, "frame_id": "base_link", "object_id": "table",
        "bounds_min": lower.tolist(), "bounds_max": upper.tolist(),
        "plane_normal": normal.tolist(), "plane_offset": float(offset),
        "inlier_count": len(selected), "sample_count": len(points),
        "note": "局部支撑桌面包络；其余桌面/结构仍由OctoMap表示",
    }


def read_model(path):
    model = json.loads(Path(path).read_text())
    if (model.get("version") != 1 or model.get("frame_id") != "base_link"
            or model.get("object_id") != "table"):
        raise ValueError("桌面模型版本、frame或ID无效")
    lower = numbers(model["bounds_min"], 3, "bounds_min")
    upper = numbers(model["bounds_max"], 3, "bounds_max")
    if np.any(upper <= lower):
        raise ValueError("桌面模型边界无效")
    normal = numbers(model["plane_normal"], 3, "plane_normal")
    if abs(np.linalg.norm(normal) - 1) > 1e-3 or normal[2] <= 0:
        raise ValueError("桌面平面法向无效")
    numeric(model["plane_offset"], "plane_offset")
    return model, lower, upper


def make_table(lower, upper):
    obj = CollisionObject()
    obj.id = "table"
    obj.header.frame_id = "base_link"
    obj.pose.orientation.w = 1.0
    obj.operation = CollisionObject.ADD
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = (upper - lower).tolist()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (lower + upper) / 2
    pose.orientation.w = 1.0
    obj.primitives = [primitive]
    obj.primitive_poses = [pose]
    return obj


def load_table(path):
    _, lower, upper = read_model(path)
    for name in ("/apply_planning_scene", "/get_planning_scene"):
        rospy.wait_for_service(name, timeout=10)
    diff = PlanningScene()
    diff.is_diff = True
    diff.robot_state.is_diff = True
    diff.world.collision_objects = [make_table(lower, upper)]
    if not rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)(diff).success:
        raise RuntimeError("MoveIt拒绝桌面模型")
    request = GetPlanningSceneRequest()
    request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
    scene = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)(request).scene
    objects = {obj.id: obj for obj in scene.world.collision_objects}
    if "table" not in objects:
        raise RuntimeError("table未同步到MoveIt")
    stored = objects["table"]
    if (len(stored.primitives) != 1 or stored.primitives[0].type != SolidPrimitive.BOX
            or not np.allclose(stored.primitives[0].dimensions, upper - lower,
                               rtol=0, atol=1e-6)):
        raise RuntimeError("table几何与模型文件不一致")
    rospy.loginfo("table已加载；在RViz核对位置和覆盖范围，没有发送运动命令")


class CloudProcessor:
    def __init__(self, config, mode, model_path=None):
        self.config = config
        self.mode = mode
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.lock = threading.Lock()
        self.samples = []
        self.collecting = True
        self.last_sample = -1.0
        if mode == "filter":
            _, self.lower, self.upper = read_model(model_path)
            self.publisher = rospy.Publisher(
                "/piper/table_scan/environment_points", PointCloud2, queue_size=1)
        self.subscriber = rospy.Subscriber(
            config["input_cloud"], PointCloud2, self.on_cloud, queue_size=1,
            buff_size=32 * 1024 * 1024)

    def on_cloud(self, message):
        try:
            age = (rospy.Time.now() - message.header.stamp).to_sec()
            if (not message.header.frame_id or message.header.stamp.to_sec() <= 0
                    or age < -0.1 or age > self.config["max_age_s"]):
                raise RuntimeError("点云无frame/时间戳无效/过旧")
            if self.mode == "collect" and time.monotonic() - self.last_sample < 0.5:
                return
            transform = self.buffer.lookup_transform(
                "base_link", message.header.frame_id, message.header.stamp,
                rospy.Duration(0.15))
            source = np.array(list(pc2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True)),
                dtype=float).reshape(-1, 3)
            source = source[np.isfinite(source).all(axis=1)]
            base = xyz_to_base(source, transform)
            if self.mode == "collect":
                selected = base[inside(base, np.array(self.config["roi_min"]),
                                      np.array(self.config["roi_max"]))]
                selected = downsample(selected, self.config["voxel_m"])
                # 每帧限制内存；最终还会再体素降采样。
                if len(selected) > 5000:
                    selected = selected[np.linspace(0, len(selected) - 1,
                                                     5000, dtype=int)]
                with self.lock:
                    if self.collecting and len(selected):
                        self.samples.append(selected)
                        self.last_sample = time.monotonic()
            else:
                # 只去掉与命名table几何完全重合的空间，几何仍阻挡机器人。
                keep = ~inside(base, self.lower, self.upper)
                if (rospy.Time.now() - message.header.stamp).to_sec() > self.config["max_age_s"]:
                    raise RuntimeError("处理后点云过旧，丢弃这一帧")
                # 必须保留传感器frame与采集时间；不能改成base_link或重新打时间戳。
                output = pc2.create_cloud_xyz32(copy.deepcopy(message.header),
                                               source[keep].tolist())
                self.publisher.publish(output)
        except (RuntimeError, ValueError, tf2_ros.TransformException) as error:
            rospy.logwarn_throttle(2.0, "相机桌面处理跳过一帧: %s", error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "load", "filter"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("camera_table_scene", anonymous=True)
    if args.command == "load":
        load_table(args.model)
        return
    config = rospy.get_param("/camera_table_scan")
    check_config(config)
    if args.command == "filter":
        processor = CloudProcessor(config, "filter", args.model)
        rospy.loginfo("桌面过滤启动，保持原始sensor frame和采集时间戳")
        rospy.spin()
        return
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 300:
        raise ValueError("采集时间必须在1～300秒内")
    path = Path(args.model).expanduser().absolute()
    if path.exists():
        raise ValueError("模型文件已存在，换新文件名避免覆盖")
    processor = CloudProcessor(config, "collect")
    rospy.loginfo("开始采集%.1f秒；仅在已确认安全的视角停稳采集", args.seconds)
    deadline = time.monotonic() + args.seconds
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        rospy.sleep(0.1)
    if rospy.is_shutdown():
        raise RuntimeError("采集被取消，未保存模型")
    processor.subscriber.unregister()
    with processor.lock:
        processor.collecting = False
        samples = list(processor.samples)
    if not samples:
        raise RuntimeError("没有有效ROI点云；检查相机、时间戳、TF和ROI")
    model = fit_table(np.vstack(samples), config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(model, stream, ensure_ascii=False, indent=2, allow_nan=False)
    rospy.loginfo("桌面模型已保存: %s；平面点=%d，需在RViz核对", path, model["inlier_count"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, RuntimeError, OSError, rospy.ROSException) as error:
        rospy.logerr("相机桌面流程失败: %s", error)
        sys.exit(1)

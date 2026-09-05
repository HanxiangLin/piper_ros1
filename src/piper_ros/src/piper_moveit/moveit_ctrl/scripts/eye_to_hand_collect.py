#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect stationary eye-to-hand pose pairs. Never commands robot motion."""

import argparse
import collections
import json
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

from piper_eye_to_hand.board import Board
from piper_eye_to_hand.sampling import (
    FORMAT, atomic_write_json, choose_stationary_frame, is_duplicate_pose,
    load_dataset, utc_now,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, help="打印板对应的 board.json")
    parser.add_argument("--output", required=True, help="本次训练或独立验证的样本 JSON")
    parser.add_argument("--resume", action="store_true", help="继续采样到已有文件，必须板/坐标/相机配置一致")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--ee-frame", default="link6", help="与标定板刚性连接的末端 link")
    parser.add_argument("--camera-link", default="camera_link", help="RealSense 驱动 TF 树的根，不是 optical frame")
    parser.add_argument("--preview-topic", default="/eye_to_hand/preview")
    parser.add_argument("--max-reprojection-error", type=float, default=1.5, help="每帧最大 RMS 像素误差")
    parser.add_argument("--capture-timeout", type=float, default=8.0)
    args = parser.parse_args(argv)
    if (not np.isfinite(args.max_reprojection_error) or args.max_reprojection_error <= 0
            or not np.isfinite(args.capture_timeout) or args.capture_timeout < 2.0):
        parser.error("误差阈值必须为正，capture-timeout 必须至少 2 秒")
    for name in (args.base_frame, args.ee_frame, args.camera_link):
        if not name or name.startswith("/") or any(c.isspace() for c in name):
            parser.error("TF frame 应非空、不含空白且不以 / 开头")
    if len({args.base_frame, args.ee_frame, args.camera_link}) != 3:
        parser.error("base、ee 和 camera_link 必须是不同坐标系")
    return args


def transform_matrix(transform):
    # Import after ROS initialization so --help works without a ROS environment.
    from tf.transformations import quaternion_matrix
    q = transform.rotation
    values = np.array([q.x, q.y, q.z, q.w], dtype=float)
    if not np.isfinite(values).all() or abs(np.linalg.norm(values) - 1.0) > 0.01:
        raise ValueError("TF 含无效四元数")
    matrix = quaternion_matrix(values)
    p = transform.translation
    matrix[:3, 3] = [p.x, p.y, p.z]
    if not np.isfinite(matrix).all():
        raise ValueError("TF 含非有限位置")
    return matrix.tolist()


class Collector:
    def __init__(self, args):
        import rospy
        import tf2_ros
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image, JointState

        self.rospy = rospy
        self.tf2_ros = tf2_ros
        self.args = args
        self.board = Board(args.board)
        self.output = Path(args.output).expanduser().absolute()
        if self.output.is_symlink():
            raise ValueError("采样输出不能是符号链接")
        if self.output.exists() and not args.resume:
            raise ValueError("输出已存在；继续采样请加 --resume，独立验证请换一个文件名")
        if args.resume and not self.output.is_file():
            raise ValueError("--resume 指定的采样文件不存在")
        self.dataset = load_dataset(self.output) if args.resume else None
        if self.dataset and self.dataset["board"] != self.board.config:
            raise ValueError("已有样本与指定标定板配置不同")
        if rospy.get_param("/use_sim_time", False):
            raise ValueError("当前开启 /use_sim_time；请使用实机驱动和真实反馈采样")

        self.lock = threading.Lock()
        self.frames = collections.deque(maxlen=40)
        self.info = None
        self.joints = None
        self.last_stamp = None
        self.last_process = 0.0
        self.status = "等待 CameraInfo、图像和实机关节反馈"
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.preview = rospy.Publisher(args.preview_topic, Image, queue_size=1)
        self.info_sub = rospy.Subscriber(args.camera_info_topic, CameraInfo, self.on_info, queue_size=1)
        self.joint_sub = rospy.Subscriber(args.joint_topic, JointState, self.on_joints, queue_size=1)
        self.image_sub = rospy.Subscriber(args.image_topic, Image, self.on_image,
                                          queue_size=1, buff_size=8 * 1024 * 1024)

    def on_info(self, message):
        with self.lock:
            self.info = message

    def on_joints(self, message):
        with self.lock:
            self.joints = (message, time.monotonic())

    def set_status(self, message):
        with self.lock:
            self.status = message

    def on_image(self, message):
        import cv2
        from cv_bridge import CvBridgeError
        rospy = self.rospy
        if time.monotonic() - self.last_process < 0.09:
            return
        self.last_process = time.monotonic()
        try:
            with self.lock:
                info, joint_entry = self.info, self.joints
            if info is None or joint_entry is None:
                raise ValueError("等待 CameraInfo 和 /joint_states")
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            if not message.header.frame_id or message.header.frame_id != info.header.frame_id:
                raise ValueError("图像与 CameraInfo 的 frame_id 不一致或为空")
            if (image.shape[1], image.shape[0]) != (info.width, info.height):
                raise ValueError("图像大小与 CameraInfo 不一致；不要给图像缩放或裁剪")
            if info.distortion_model not in ("plumb_bob", "rational_polynomial"):
                raise ValueError("仅支持原始彩色图像及 plumb_bob/rational_polynomial CameraInfo")
            if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
                raise ValueError("不支持带 binning/ROI 的图像，使用原始 color/image_raw")
            k = np.asarray(info.K, dtype=float).reshape(3, 3)
            d = np.asarray(info.D, dtype=float)
            if not np.isfinite(k).all() or not np.isfinite(d).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError("CameraInfo 内参无效")
            stamp = message.header.stamp
            age = (rospy.Time.now() - stamp).to_sec()
            if stamp.to_sec() <= 0 or age > 0.7 or age < -0.1:
                raise ValueError("图像时间戳无效/过旧/超前；检查相机时钟")
            joint_message, joint_received = joint_entry
            if time.monotonic() - joint_received > 0.5:
                raise ValueError("实机关节反馈已停止更新")
            if abs((joint_message.header.stamp - stamp).to_sec()) > 0.5:
                raise ValueError("关节反馈与图像时间不接近")
            positions = dict(zip(joint_message.name, joint_message.position))
            arm_positions = [float(positions["joint{}".format(i)]) for i in range(1, 7)]
            if not np.isfinite(arm_positions).all():
                raise ValueError("实机关节状态无效")
            result = self.board.detect(image, k, d)
            marked = self.board.annotate(image, result)
            if self.preview.get_num_connections():
                preview_message = self.bridge.cv2_to_imgmsg(marked, encoding="bgr8")
                preview_message.header = message.header
                self.preview.publish(preview_message)
            if result is None:
                raise ValueError("未检测到足够的有效标记，请让至少4个板上标记清晰可见")
            if result["reprojection_error_px"] > self.args.max_reprojection_error:
                raise ValueError("重投影误差过大：{:.2f}px".format(result["reprojection_error_px"]))
            # Lookup at the IMAGE timestamp. Do not silently fall back to latest TF.
            base_ee = self.tf_buffer.lookup_transform(
                self.args.base_frame, self.args.ee_frame, stamp, rospy.Duration(0.25))
            link_optical = self.tf_buffer.lookup_transform(
                self.args.camera_link, message.header.frame_id, stamp, rospy.Duration(0.25))
            candidate = {
                "stamp": stamp.to_sec(), "received_monotonic": time.monotonic(),
                "base_T_ee": transform_matrix(base_ee.transform),
                "camera_T_target": result["camera_T_target"],
                "camera_link_T_optical": transform_matrix(link_optical.transform),
                "camera_info": {"K": list(info.K), "D": list(info.D),
                                "width": info.width, "height": info.height,
                                "distortion_model": info.distortion_model},
                "frames": {"base": self.args.base_frame, "ee": self.args.ee_frame,
                           "camera_optical": message.header.frame_id,
                           "camera_link": self.args.camera_link},
                "joint_positions": arm_positions,
                "reprojection_error_px": float(result["reprojection_error_px"]),
                "marker_count": result["marker_count"], "marker_ids": result["marker_ids"],
                "image_points": result["image_points"], "object_points": result["object_points"],
                "image": image,
            }
            with self.lock:
                if self.last_stamp is not None and stamp.to_sec() <= self.last_stamp:
                    self.frames.clear()
                    self.status = "图像时间戳重复/倒退，等待新帧"
                    self.last_stamp = stamp.to_sec()
                    return
                self.last_stamp = stamp.to_sec()
                self.frames.append(candidate)
                self.status = "可见标记 {} 个；重投影误差 {:.3f}px".format(
                    result["marker_count"], result["reprojection_error_px"])
        except (ValueError, KeyError, CvBridgeError, cv2.error,
                self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException,
                self.tf2_ros.ExtrapolationException) as error:
            self.set_status(str(error))

    def check_metadata(self, candidate):
        if self.dataset is None:
            self.dataset = {
                "format": FORMAT, "created_at": utc_now(), "updated_at": utc_now(),
                "session_id": str(uuid.uuid4()), "board": self.board.config,
                "frames": candidate["frames"], "camera_info": candidate["camera_info"],
                "camera_link_T_optical": candidate["camera_link_T_optical"], "samples": [],
            }
        if candidate["frames"] != self.dataset["frames"]:
            raise ValueError("采样期间坐标系发生变化，请使用新的数据集")
        if candidate["camera_info"] != self.dataset["camera_info"]:
            raise ValueError("相机内参/分辨率发生变化，禁止混入原数据集")
        if not np.allclose(candidate["camera_link_T_optical"], self.dataset["camera_link_T_optical"],
                           atol=1e-7, rtol=0):
            raise ValueError("相机内部 TF 发生变化，禁止混入原数据集")

    def capture(self):
        import cv2
        rospy = self.rospy
        start = time.monotonic()
        start_stamp = rospy.Time.now().to_sec()
        print("开始采样，请保持机械臂、夹爪、标定板和相机静止……", flush=True)
        last_error = "没有有效图像"
        while not rospy.is_shutdown() and time.monotonic() - start < self.args.capture_timeout:
            with self.lock:
                frames = [frame for frame in self.frames
                          if frame["received_monotonic"] >= start and frame["stamp"] >= start_stamp]
                status = self.status
            if len(frames) >= 8 and frames[-1]["stamp"] - frames[0]["stamp"] >= 0.7:
                # A rejection requires another explicit capture; do not silently
                # search the moving burst for a conveniently quiet sub-window.
                chosen, quality = choose_stationary_frame(frames[:12])
                if time.monotonic() - frames[-1]["received_monotonic"] > 0.5:
                    raise ValueError("采样图像流已经停止，请重新采样")
                for frame in frames[:12]:
                    self.check_metadata(frame)
                if is_duplicate_pose(self.dataset["samples"], chosen):
                    raise ValueError("位姿与已有样本过近；请改变位置至少8mm或姿态至少4°")
                sample_id = str(uuid.uuid4())
                image_dir = self.output.parent / (self.output.stem + "_images")
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / (sample_id + ".png")
                if not cv2.imwrite(str(image_path), chosen["image"]):
                    raise ValueError("无法保存采样图像")
                sample = {key: value for key, value in chosen.items()
                          if key not in ("image", "received_monotonic", "camera_info", "frames", "camera_link_T_optical")}
                sample.update(id=sample_id, captured_at=utc_now(), quality=quality,
                              image_file=str(image_path.relative_to(self.output.parent)))
                self.dataset["samples"].append(sample)
                self.dataset["updated_at"] = utc_now()
                atomic_write_json(self.output, self.dataset)
                print("已保存第 {} 个位姿：{}；RMS {:.3f}px".format(
                    len(self.dataset["samples"]), self.output, sample["reprojection_error_px"]), flush=True)
                return
            last_error = status
            time.sleep(0.05)
        raise ValueError("采样超时：{}".format(last_error))

    def run(self):
        print("\n本工具不发送运动命令。用 RViz/现有控制界面换姿，停稳后再采样。")
        print("在 RViz 添加 Image，选择 {} 查看标记检测。".format(self.args.preview_topic))
        print("建议训练20–25个不同位姿；独立验证另建文件采5–8个新位姿。")
        print("Enter/s：采一组；p：状态；u：撤销末组（保留图像）；q：退出。\n")
        while not self.rospy.is_shutdown():
            try:
                command = input("标定> ").strip().lower()
            except EOFError:
                break
            if command == "q":
                break
            if command in ("", "s"):
                try:
                    self.capture()
                except ValueError as error:
                    print("本次未保存：{}".format(error), flush=True)
            elif command == "p":
                with self.lock:
                    print(self.status)
                print("已保存 {} 个位姿".format(len(self.dataset["samples"]) if self.dataset else 0))
            elif command == "u":
                if self.dataset and self.dataset["samples"]:
                    removed = self.dataset["samples"].pop()
                    self.dataset["updated_at"] = utc_now()
                    atomic_write_json(self.output, self.dataset)
                    print("已从样本列表撤销 {}；原图仍保留。".format(removed["id"]))
                else:
                    print("没有可撤销的样本")
            else:
                print("使用 Enter/s、p、u 或 q")


def main():
    # Do not require rospy for CLI --help; roslaunch remapping is otherwise stripped.
    args = parse_args([arg for arg in sys.argv[1:] if ":=" not in arg])
    import rospy
    rospy.init_node("eye_to_hand_collect", anonymous=True)
    try:
        Collector(args).run()
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        rospy.logerr("标定采样终止：%s", error)
        return 1
    finally:
        rospy.signal_shutdown("采样结束")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass

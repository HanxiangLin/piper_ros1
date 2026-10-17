#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import rospy
import tf.transformations as tf_trans
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped TF conversion
import tf2_ros
from geometry_msgs.msg import PoseStamped
import moveit_commander
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    DisplayTrajectory,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetPlanningScene,
    GetPlanningSceneRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from piper_msgs.srv import Gripper
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

# ===============================
# 瓶子检测参数
# ===============================
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().with_name("bottle.pt"))
DEFAULT_CALIBRATION_PATH = str(
    Path(__file__).resolve().parents[4]
    / "calibration/eye_in_hand_d435i/session_02/result.json")
rs = None  # 在实际启用相机时导入，离线检查不初始化SDK/YOLO。
CONF_THRES = 0.5
IMG_SIZE = 640

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
FPS = 30

DEPTH_MIN = 0.1
DEPTH_MAX = 8.0
SHRINK_RATIO = 0.20
DEPTH_PERCENTILE = 20
COLOR_RATIO_THRES = 0.15

# None 表示抓取任意检测到的瓶子；也可以改成 "bottle-y"、"bottle-g"、"bottle-b"
TARGET_BOTTLE_LABEL = None

# 是否显示检测窗口。
SHOW_DETECTION_WINDOW = True

COLOR_DEFS = {
    "bottle-y": {  # 橘色
        "hsv_lower": np.array([5, 80, 80]),
        "hsv_upper": np.array([25, 255, 255]),
        "bgr": (0, 140, 255),
    },
    "bottle-g": {  # 绿色
        "hsv_lower": np.array([35, 50, 50]),
        "hsv_upper": np.array([85, 255, 255]),
        "bgr": (0, 200, 0),
    },
    "bottle-b": {  # 黑色
        "hsv_lower": np.array([0, 0, 0]),
        "hsv_upper": np.array([180, 255, 60]),
        "bgr": (80, 80, 80),
    },
}

# TCP必须通过~link6_tool_xyz配置；含义为link6坐标系中的实际抓取中心。
# 相机变换从~hand_eye_calibration读取link6->彩色光学坐标系矩阵。
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["joint7", "joint8"]
TARGET_BOTTLE_ID = "target_bottle"


class PlanningOnlyStop(RuntimeError):
    """Stop the sequential task after displaying its first planned motion."""


def normalize_vector(v, eps=1e-9):
    """向量归一化，避免除零。"""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def build_level_grasp_rotation(reference_R=None):
    """
    构造“只约束夹爪水平”的目标旋转矩阵。

    约束条件：
        夹爪/抓取坐标系 X 轴 = 基坐标系 -Z 方向
        即 grasp_x = [0, 0, -1]

    不再强制：
        grasp_z = base_x 或 grasp_y = base_y

    做法：
        1. 固定 grasp_x 为 [0, 0, -1]，保证夹爪 X 轴与基坐标系 Z 轴反向重合；
        2. 从当前末端姿态 reference_R 中取一个水平面内的参考方向，
           尽量保留当前末端在水平面内的朝向；
        3. 用右手系补全 grasp_y、grasp_z。

    返回：
        R_base_grasp，3x3 旋转矩阵。矩阵三列分别是：
        grasp_x、grasp_y、grasp_z 在基坐标系下的方向。
    """
    # 只强制这一条：夹爪 X 轴朝向基坐标系 -Z。
    grasp_x = np.array([0.0, 0.0, -1.0], dtype=float)

    candidate_refs = []
    if reference_R is not None:
        reference_R = np.array(reference_R, dtype=float)
        # 优先沿用当前末端的 Z 轴作为水平面内朝向参考。
        # 如果当前 Z 轴接近竖直，投影后会退化，再尝试当前 Y/X 轴。
        candidate_refs.extend([
            reference_R[0:3, 2],
            reference_R[0:3, 1],
            reference_R[0:3, 0],
        ])

    # 最后的兜底方向。只有在当前末端几个轴都不适合时才会用到。
    candidate_refs.extend([
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ])

    grasp_z = None
    for ref in candidate_refs:
        # 去掉 ref 在 grasp_x 方向上的分量，只保留水平面内分量。
        z_candidate = ref - np.dot(ref, grasp_x) * grasp_x
        z_candidate = normalize_vector(z_candidate)
        if z_candidate is not None:
            grasp_z = z_candidate
            break

    if grasp_z is None:
        # 理论上不会走到这里，只是为了安全。
        grasp_z = np.array([1.0, 0.0, 0.0], dtype=float)

    # 右手系要求：grasp_x × grasp_y = grasp_z
    # 因此 grasp_y = grasp_z × grasp_x
    grasp_y = np.cross(grasp_z, grasp_x)
    grasp_y = normalize_vector(grasp_y)

    R_base_grasp = np.column_stack((grasp_x, grasp_y, grasp_z))
    return R_base_grasp


def classify_bottle_color(color_image, x1, y1, x2, y2):
    """根据检测框内 HSV 颜色占比，返回 bottle-y/bottle-g/bottle-b 或 None。"""
    h, w = color_image.shape[:2]
    rx1 = max(int(x1), 0)
    ry1 = max(int(y1), 0)
    rx2 = min(int(x2), w - 1)
    ry2 = min(int(y2), h - 1)

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = color_image[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return None

    priority_order = ["bottle-y", "bottle-g", "bottle-b"]
    claimed = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=bool)
    scores = {}

    for label in priority_order:
        cfg = COLOR_DEFS[label]
        mask = cv2.inRange(hsv, cfg["hsv_lower"], cfg["hsv_upper"])
        exclusive = (mask > 0) & (~claimed)
        scores[label] = np.count_nonzero(exclusive) / total
        claimed |= exclusive

    best_label = max(scores, key=scores.get)
    if scores[best_label] >= COLOR_RATIO_THRES:
        return best_label
    return None


def build_filters():
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 1)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)

    return [decimation, spatial, temporal, hole_filling]


def apply_filters(depth_frame, filters):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


def get_depth_percentile(depth_frame, raw_depth_frame, x1, y1, x2, y2, depth_scale):
    """在检测框中心区域取深度百分位，避免直接取中心点受空洞/反光影响。"""
    bw = x2 - x1
    bh = y2 - y1
    sx = int(bw * SHRINK_RATIO)
    sy = int(bh * SHRINK_RATIO)

    rx1 = max(x1 + sx, 0)
    ry1 = max(y1 + sy, 0)
    rx2 = min(x2 - sx, depth_frame.get_width() - 1)
    ry2 = min(y2 - sy, depth_frame.get_height() - 1)

    cx = int((x1 + x2) // 2)
    cy = int((y1 + y2) // 2)

    if rx2 <= rx1 or ry2 <= ry1:
        return depth_frame.get_distance(cx, cy)

    depth_image = np.asanyarray(raw_depth_frame.get_data())
    roi = depth_image[ry1:ry2, rx1:rx2].astype(np.float32) * depth_scale
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]

    if len(valid) == 0:
        return depth_frame.get_distance(cx, cy)

    return float(np.percentile(valid, DEPTH_PERCENTILE))


def deproject(intrinsics, px, py, depth_m):
    return rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)], float(depth_m))


def draw_detection(img, x1, y1, x2, y2, color_label, conf, dist_m, xyz):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bgr = COLOR_DEFS[color_label]["bgr"]

    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(img, f"{color_label} {conf:.2f} | {dist_m:.3f}m",
                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    if xyz is not None:
        X, Y, Z = xyz
        cv2.putText(img, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}",
                    (x1, min(y2 + 20, img.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


class PiperVisionController:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("piper_bottle_vision_grasp", anonymous=True)

        self.pipeline = None
        self.model = None
        self.current_ee_pose_mat = np.eye(4)
        self.grasp_confirmed = False
        self._joint_lock = threading.Lock()
        self._joint_positions = None
        self._joint_feedback_time = None
        self._joint_message_stamp = None
        self._joint_efforts = {}
        self._gripper_service_sent = False
        self._created_bottle = False
        self.arm = None
        self.joint_sub = None

        # 可选YAML与命令行私有参数合并，命令行优先；便于直接python运行。
        params_file = rospy.get_param("~params_file", "")
        if params_file:
            import yaml
            settings = yaml.safe_load(Path(params_file).expanduser().read_text())
            if not isinstance(settings, dict):
                raise ValueError("~params_file必须包含参数字典")
            for key, value in settings.items():
                if not isinstance(key, str) or not key or "/" in key or key.startswith("~"):
                    raise ValueError("参数文件键必须是不带路径的私有参数名")
                # ROS1参数服务器不能存储YAML null；可选空值按未设置处理。
                if value is None:
                    continue
                if not rospy.has_param("~" + key):
                    rospy.set_param("~" + key, value)
        execute = rospy.get_param("~execute", False)
        if not isinstance(execute, bool):
            raise ValueError("~execute必须为true或false")
        self.execute_motion = execute
        required = (
            "table_model", "support_edge_margin", "lift_clearance",
            "bottle_height", "bottle_radius", "grasp_height_from_table",
            "pregrasp_distance", "lift_distance", "link6_tool_xyz",
            "grasp_opening_min", "grasp_opening_max", "grasp_effort_min")
        missing = ["~" + key for key in required if not rospy.has_param("~" + key)]
        if missing:
            raise ValueError("缺少实测任务参数: " + ", ".join(missing))
        self.model_path = str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH))
        self.table_model_path = str(rospy.get_param("~table_model"))
        calibration_path = rospy.get_param("~hand_eye_calibration", DEFAULT_CALIBRATION_PATH)
        calibration = json.loads(Path(calibration_path).expanduser().read_text())
        if calibration.get("mode") != "eye_in_hand" or calibration.get("tf_parent") != "link6":
            raise ValueError("手眼标定必须为eye_in_hand且安装父link为link6")
        self.T_link6_color_optical = np.asarray(calibration["ee_T_camera"], dtype=float)
        self._validate_transform(self.T_link6_color_optical, "手眼外参")
        self.target_label = rospy.get_param("~target_bottle_label", TARGET_BOTTLE_LABEL)
        if self.target_label not in (None, "bottle-y", "bottle-g", "bottle-b"):
            raise ValueError("~target_bottle_label必须是bottle-y/g/b或null")
        self.show_detection_window = bool(rospy.get_param("~show_detection_window", SHOW_DETECTION_WINDOW))
        self.support_edge_margin = self._finite_float(
            "~support_edge_margin", minimum=0.0)
        self.lift_clearance = self._finite_float(
            "~lift_clearance", minimum=0.0, strict_min=True)
        self.bottle_height = self._finite_float(
            "~bottle_height", minimum=0.0, strict_min=True)
        self.bottle_radius = self._finite_float(
            "~bottle_radius", minimum=0.0, strict_min=True)
        self.grasp_height = self._finite_float(
            "~grasp_height_from_table", minimum=0.0, strict_min=True)
        if self.grasp_height >= self.bottle_height:
            raise ValueError("~grasp_height_from_table 必须小于瓶高")
        self.pregrasp_distance = self._finite_float(
            "~pregrasp_distance", minimum=0.0, strict_min=True)
        self.lift_distance = self._finite_float(
            "~lift_distance", minimum=0.0, strict_min=True)
        candidate_yaws = rospy.get_param("~approach_yaw_offsets_deg", [0.0, 30.0, -30.0])
        if not isinstance(candidate_yaws, list) or not candidate_yaws or len(candidate_yaws) > 12:
            raise ValueError("~approach_yaw_offsets_deg需要1～12个方向")
        self.approach_yaws = self._numeric_list(
            "~approach_yaw_offsets_deg", candidate_yaws, len(candidate_yaws))

        self.opening_open = self._finite_float(
            "~gripper_opening_open", 0.07, 0.0, 0.07)
        self.opening_close_command = self._finite_float(
            "~gripper_opening_close_command", 0.0, 0.0, 0.07)
        self.grasp_opening_min = self._finite_float(
            "~grasp_opening_min", minimum=0.0, strict_min=True)
        self.grasp_opening_max = self._finite_float(
            "~grasp_opening_max", minimum=0.0, maximum=0.07)
        if (self.grasp_opening_max <= self.grasp_opening_min
                or self.grasp_opening_min <= self.opening_close_command
                or self.opening_open <= self.grasp_opening_max):
            raise ValueError("要求: 闭合目标 < 夹持开口下限 < 上限 < 张开目标")
        self.grasp_effort_min = self._finite_float(
            "~grasp_effort_min", minimum=0.0, strict_min=True)
        self.grasp_check_service = str(rospy.get_param("~grasp_check_service", ""))
        self.gripper_effort = self._finite_float(
            "~gripper_effort", 1.0, 0.5, 2.0)
        self.gripper_tolerance = self._finite_float(
            "~gripper_tolerance", 0.004, 0.0, 0.02, strict_min=True)
        self.gripper_feedback_timeout = self._finite_float(
            "~gripper_feedback_timeout", 4.0, 0.0, 20.0, strict_min=True)
        self.gripper_settle_time = self._finite_float(
            "~gripper_settle_time", 1.5, 0.0, 10.0)

        self.move_to_photo_pose = bool(rospy.get_param("~move_to_photo_pose", False))
        self.photo_joint_target = self._numeric_list(
            "~photo_joint_target",
            rospy.get_param("~photo_joint_target",
                            [1.521, 0.403, -0.521, 0.0, 0.808, 0.0]), 6)
        self.post_grasp_joint_target = self._optional_numeric_list(
            "~post_grasp_joint_target", 6)
        self.enable_place = bool(rospy.get_param("~enable_place", False))
        if self.enable_place:
            self.place_tool_pose = self._numeric_list(
                "~place_tool_pose", rospy.get_param("~place_tool_pose"), 6)
            self.place_approach_height = self._finite_float(
                "~place_approach_height", minimum=0.0, strict_min=True)

        self.bottle_center_offset_xy = self._numeric_list(
            "~bottle_center_offset_xy",
            rospy.get_param("~bottle_center_offset_xy", [0.0, 0.0]), 2)
        # 深度百分位通常落在瓶子朝向相机的一侧表面；不能把该表面点
        # 直接当成瓶子的轴线中心。默认根据检测像素射线、已扫描桌面平面
        # 和瓶高反算瓶子中心，避免用TCP参数掩盖视觉几何误差。
        self.estimate_bottle_center_from_table = bool(
            rospy.get_param("~estimate_bottle_center_from_table", True))
        link6_tool_xyz = self._numeric_list(
            "~link6_tool_xyz",
            rospy.get_param("~link6_tool_xyz"), 3)
        tool_rpy = self._numeric_list(
            "~link6_tool_rpy", rospy.get_param("~link6_tool_rpy", [0.0, 0.0, 0.0]), 3)
        self.T_link6_tool = tf_trans.euler_matrix(*tool_rpy, axes="sxyz")
        self.T_link6_tool[:3, 3] = link6_tool_xyz

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.arm = moveit_commander.MoveGroupCommander("arm")
        self.arm.set_pose_reference_frame("base_link")
        self.arm.set_end_effector_link("link6")
        self.arm.set_planner_id(str(rospy.get_param("~planner_id", "RRTConnect")))
        self.arm.set_planning_time(self._finite_float(
            "~planning_time", 10.0, 2.0, 30.0))
        attempts = int(rospy.get_param("~planning_attempts", 8))
        if not 1 <= attempts <= 30:
            raise ValueError("~planning_attempts必须在1～30内")
        self.arm.set_num_planning_attempts(attempts)
        self.velocity_scale = self._finite_float(
            "~velocity_scale", 0.03, 0.0, 0.10, strict_min=True)
        self.acceleration_scale = self._finite_float(
            "~acceleration_scale", 0.03, 0.0, 0.10, strict_min=True)
        self.arm.set_max_velocity_scaling_factor(self.velocity_scale)
        self.arm.set_max_acceleration_scaling_factor(self.acceleration_scale)
        self.arm.set_goal_position_tolerance(0.003)
        self.arm.set_goal_orientation_tolerance(0.03)
        self.arm.allow_replanning(False)

        self.display_pub = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory,
            queue_size=1, latch=True)
        self.joint_sub = rospy.Subscriber(
            "/joint_states", JointState, self._joint_state_callback,
            queue_size=1, tcp_nodelay=True)
        for name in ("/get_planning_scene", "/apply_planning_scene",
                     "/check_state_validity"):
            rospy.wait_for_service(name, timeout=10.0)
        self.get_scene = rospy.ServiceProxy(
            "/get_planning_scene", GetPlanningScene)
        self.apply_scene = rospy.ServiceProxy(
            "/apply_planning_scene", ApplyPlanningScene)
        self.check_validity = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity)
        self.gripper_service = None
        if self.execute_motion:
            rospy.wait_for_service("/gripper_srv", timeout=5.0)
            self.gripper_service = rospy.ServiceProxy("/gripper_srv", Gripper)

    @staticmethod
    def _validate_transform(matrix, name):
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-4)):
            raise ValueError(name + "不是有效的刚体变换")

    def _finite_float(self, name, default=None, minimum=None, maximum=None,
                      strict_min=False):
        value = rospy.get_param(name) if default is None else rospy.get_param(name, default)
        if isinstance(value, bool):
            raise ValueError(name + "不能是布尔值")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(name + "必须填写实测数字，不能保留MEASURE") from error
        if not math.isfinite(value):
            raise ValueError(name + " 必须是有限数字")
        if minimum is not None:
            invalid = value <= minimum if strict_min else value < minimum
            if invalid:
                raise ValueError("{} 小于允许下限".format(name))
        if maximum is not None and value > maximum:
            raise ValueError("{} 大于允许上限".format(name))
        return value

    @staticmethod
    def _numeric_list(name, value, count):
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise ValueError("{} 必须包含{}个数字".format(name, count))
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError) as error:
            raise ValueError(name + "必须填写实测数字列表") from error
        if any(isinstance(item, bool) for item in value) or not np.isfinite(result).all():
            raise ValueError(name + " 包含非有限数值或布尔值")
        return result

    def _optional_numeric_list(self, name, count):
        value = rospy.get_param(name, [])
        if value == []:
            return None
        return self._numeric_list(name, value, count)

    def _joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in ARM_JOINTS + FINGER_JOINTS):
            return
        if not np.isfinite(list(positions.values())).all():
            return
        stamp = message.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        if stamp <= 0 or not -0.1 <= age <= 1.0:
            return
        efforts = dict(zip(message.name, message.effort))
        with self._joint_lock:
            self._joint_positions = positions
            self._joint_efforts = efforts
            self._joint_message_stamp = stamp
            self._joint_feedback_time = time.monotonic()

    def _fresh_joint_positions(self):
        with self._joint_lock:
            positions = copy.deepcopy(self._joint_positions)
            timestamp = self._joint_feedback_time
            stamp = self._joint_message_stamp
        if positions is None or timestamp is None:
            raise RuntimeError("尚未收到完整 /joint_states 反馈")
        if (time.monotonic() - timestamp > 1.0
                or stamp is None or not -0.1 <= rospy.Time.now().to_sec() - stamp <= 1.0):
            raise RuntimeError("/joint_states 反馈超过1秒，拒绝继续")
        return positions

    def wait_for_joint_feedback(self):
        deadline = time.monotonic() + 5.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                self._fresh_joint_positions()
                return
            except RuntimeError:
                rospy.sleep(0.05)
        raise RuntimeError("5秒内没有完整、新鲜的真实/joint_states反馈")

    def init_camera_and_detector(self):
        global rs
        try:
            import pyrealsense2 as rs
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("当前Python环境缺少RealSense/YOLO依赖: " + str(error)) from error
        if not os.path.isfile(self.model_path):
            raise ValueError("YOLO模型不存在: " + self.model_path)
        rospy.loginfo("正在加载YOLO瓶子检测模型: %s", self.model_path)
        self.model = YOLO(self.model_path)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        config.enable_stream(
            rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as error:
            raise RuntimeError(
                "RealSense启动失败；运行本脚本前停止ROS相机launch以释放设备: {}"
                .format(error))
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.filters = build_filters()
        rospy.loginfo("YOLO与RealSense初始化完成；depth scale=%.6f", self.depth_scale)

    def make_bottle_pose_in_base(self, bottle_xyz_cam):
        """Convert an observed color-optical point to base_link.

        This is retained for diagnostics and for the fallback path.  The
        measured depth point is generally on the visible bottle surface, not
        on the bottle axis.  Grasp planning should therefore prefer
        make_bottle_center_pose_in_base().
        """
        T_cam_point = np.eye(4)
        T_cam_point[:3, 3] = np.asarray(bottle_xyz_cam, dtype=float)
        T_base_point = (
            self.current_ee_pose_mat @ self.T_link6_color_optical @ T_cam_point)
        result = np.eye(4)
        result[:3, 3] = T_base_point[:3, 3]
        result[:3, :3] = build_level_grasp_rotation(
            self.current_ee_pose_mat[:3, :3])
        return result

    def make_bottle_center_pose_in_base(self, pixel, intrinsics):
        """Estimate the bottle axis center from a pixel and the scanned table.

        The YOLO/depth point is the camera-facing surface.  For a standing
        bottle, the center lies on the ray through the detection pixel and at
        bottle_height/2 above the fitted table plane.  Solving that
        intersection removes the systematic ``front-surface`` offset without
        changing the link6-to-tool/TCP transform.
        """
        if not hasattr(self, "table_model"):
            raise RuntimeError("尚未加载桌面模型，无法反算瓶子中心")
        px, py = pixel
        ray_cam = np.asarray(deproject(intrinsics, px, py, 1.0), dtype=float)
        ray_norm = np.linalg.norm(ray_cam)
        if ray_cam.shape != (3,) or not np.isfinite(ray_cam).all() or ray_norm < 1e-9:
            raise RuntimeError("瓶子像素无法生成有效相机射线")
        ray_cam /= ray_norm

        T_base_camera = self.current_ee_pose_mat @ self.T_link6_color_optical
        origin = T_base_camera[:3, 3]
        direction = T_base_camera[:3, :3] @ ray_cam
        normal = np.asarray(self.table_model["plane_normal"], dtype=float)
        offset = float(self.table_model["plane_offset"])
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or not math.isfinite(offset) or abs(normal[2]) < 0.98):
            raise RuntimeError("桌面模型不适合反算瓶子中心")

        # table plane: n dot p + offset = 0.  The bottle center is
        # bottle_height/2 above the table along base Z, hence
        # n dot center + offset = n_z * bottle_height/2.
        denominator = float(np.dot(normal, direction))
        numerator = (normal[2] * self.bottle_height / 2.0
                     - (float(np.dot(normal, origin)) + offset))
        if abs(denominator) < 1e-6:
            raise RuntimeError("瓶子检测射线几乎平行于桌面")
        distance = numerator / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("瓶子中心射线与桌面高度的交点无效")
        center = origin + distance * direction
        if not np.isfinite(center).all():
            raise RuntimeError("反算得到的瓶子中心包含非有限值")

        result = np.eye(4)
        result[:3, 3] = center
        result[:3, :3] = build_level_grasp_rotation(
            self.current_ee_pose_mat[:3, :3])
        return result

    def get_bottle_pose(self, target_label=TARGET_BOTTLE_LABEL, max_frames=60):
        """Detect one bottle and return its observed pose in base_link."""
        self.current_ee_pose_mat = self.current_ee_matrix()
        reference_joints = self._fresh_joint_positions()
        # 丢弃缓冲中的旧画面；SDK时间域不同，不将其毫秒时间戳冒充ROS时间。
        for _ in range(5):
            self.pipeline.wait_for_frames(timeout_ms=5000)
        label_text = target_label if target_label is not None else "任意颜色瓶子"
        rospy.loginfo("正在检测目标瓶子: %s", label_text)
        best_candidate = None
        best_score = -1.0
        last_annotated = None

        for frame_index in range(max_frames):
            if rospy.is_shutdown():
                break
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as error:
                rospy.logwarn("RealSense等待图像超时: %s", error)
                continue
            measured = self._fresh_joint_positions()
            if any(abs(measured[j] - reference_joints[j]) > 0.01 for j in ARM_JOINTS):
                raise RuntimeError("视觉采集期间机械臂发生移动；停止并重新定位")
            aligned = self.align.process(frames)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                continue
            filtered_depth = apply_filters(
                raw_depth, self.filters).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            annotated = color_image.copy()
            intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics
            results = self.model.predict(
                source=color_image, imgsz=IMG_SIZE,
                conf=CONF_THRES, verbose=False)

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                height, width = color_image.shape[:2]
                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = box.tolist()
                    x1, x2 = np.clip([x1, x2], 0, width - 1)
                    y1, y2 = np.clip([y1, y2], 0, height - 1)
                    color_label = classify_bottle_color(
                        color_image, x1, y1, x2, y2)
                    if color_label is None:
                        continue
                    if target_label is not None and color_label != target_label:
                        continue
                    cx = int((x1 + x2) // 2)
                    cy = int((y1 + y2) // 2)
                    distance = get_depth_percentile(
                        filtered_depth, raw_depth, x1, y1, x2, y2,
                        self.depth_scale)
                    if not DEPTH_MIN < distance < DEPTH_MAX:
                        continue
                    xyz_cam = deproject(intrinsics, cx, cy, distance)
                    surface_pose = self.make_bottle_pose_in_base(xyz_cam)
                    if self.estimate_bottle_center_from_table:
                        # 用像素射线+桌面+瓶高求轴线中心，而不是把深度表面点
                        # 的base_link X/Y直接当作瓶子中心。
                        T_base_bottle = self.make_bottle_center_pose_in_base(
                            (cx, cy), intrinsics)
                    else:
                        T_base_bottle = surface_pose
                    draw_detection(
                        annotated, x1, y1, x2, y2, color_label,
                        float(confidence), distance, xyz_cam)
                    center_penalty = (
                        abs(cx - width / 2.0) / width
                        + abs(cy - height / 2.0) / height)
                    score = float(confidence) - 0.10 * center_penalty
                    if score > best_score:
                        best_score = score
                        best_candidate = {
                            "label": color_label,
                            "conf": float(confidence),
                            "center": (cx, cy),
                            "dist_m": float(distance),
                            "xyz_cam": np.asarray(xyz_cam, dtype=float),
                            "surface_base": surface_pose[:3, 3].copy(),
                            "T_base_bottle": T_base_bottle,
                        }

            last_annotated = annotated
            if self.show_detection_window:
                try:
                    cv2.imshow("Bottle Detection For Grasp", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
            if best_candidate is not None and frame_index >= 10:
                candidate = best_candidate
                position = candidate["T_base_bottle"][:3, 3]
                surface = candidate["surface_base"]
                correction = position - surface
                rospy.loginfo(
                    "检测成功: %s conf=%.2f pixel=%s depth=%.4fm "
                    "surface_base=(%.4f, %.4f, %.4f) "
                    "bottle_center_base=(%.4f, %.4f, %.4f) "
                    "correction=(%.4f, %.4f, %.4f)%s",
                    candidate["label"], candidate["conf"], candidate["center"],
                    candidate["dist_m"], surface[0], surface[1], surface[2],
                    position[0], position[1], position[2],
                    correction[0], correction[1], correction[2],
                    " (table-ray center)" if self.estimate_bottle_center_from_table
                    else " (depth surface fallback)")
                return candidate["T_base_bottle"]

        if last_annotated is not None and self.show_detection_window:
            try:
                cv2.imshow("Bottle Detection For Grasp", last_annotated)
                cv2.waitKey(1)
            except cv2.error:
                pass
        rospy.logwarn("超时：未检测到目标瓶子: %s", label_text)
        return None

    def frozen_scene_signature(self):
        rospy.wait_for_service("/piper_static_scan_gate/status", timeout=5.0)
        response = rospy.ServiceProxy(
            "/piper_static_scan_gate/status", Trigger)()
        if not response.success:
            raise RuntimeError("扫描门状态服务返回失败: " + response.message)
        try:
            gate_status = json.loads(response.message)
        except (TypeError, ValueError) as error:
            raise RuntimeError("扫描门返回无效JSON") from error
        if gate_status.get("enabled") is not False:
            raise RuntimeError("扫描门未关闭，拒绝规划/执行")

        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.OCTOMAP |
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        scene = self.get_scene(request).scene
        outer = scene.world.octomap.header.frame_id
        octomap = scene.world.octomap.octomap
        inner = octomap.header.frame_id
        frame = (outer or inner).lstrip("/")
        if (not octomap.data or not np.isfinite(octomap.resolution)
                or octomap.resolution <= 0 or not octomap.id
                or (outer and inner and outer != inner)
                or frame != "dummy_link"):
            raise RuntimeError("环境OctoMap为空、无效或坐标系错误")
        objects = {obj.id: obj for obj in scene.world.collision_objects}
        if "table" not in objects:
            raise RuntimeError("自动table模型未加载")
        table = objects["table"]
        geometry = copy.deepcopy(table)
        geometry.header.seq = 0
        geometry.header.stamp = rospy.Time(0)
        stream = BytesIO()
        geometry.serialize(stream)
        payload = bytes((value & 255 for value in octomap.data))
        origin = scene.world.octomap.origin
        origin_values = (
            origin.position.x, origin.position.y, origin.position.z,
            origin.orientation.x, origin.orientation.y,
            origin.orientation.z, origin.orientation.w)
        signature = (
            frame, octomap.id, bool(octomap.binary),
            float(octomap.resolution), origin_values,
            hashlib.sha256(payload).hexdigest(),
            hashlib.sha256(stream.getvalue()).hexdigest())
        return signature, table

    @staticmethod
    def _pose_matrix(pose):
        matrix = tf_trans.quaternion_matrix([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w])
        matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return matrix

    def require_scan_scene(self):
        signature, table = self.frozen_scene_signature()
        model = json.loads(Path(self.table_model_path).read_text())
        if (model.get("version") != 1
                or model.get("frame_id") != "base_link"
                or model.get("object_id") != "table"):
            raise RuntimeError("桌面模型文件无效")
        lower = np.asarray(model["bounds_min"], dtype=float)
        upper = np.asarray(model["bounds_max"], dtype=float)
        normal = np.asarray(model["plane_normal"], dtype=float)
        offset = float(model["plane_offset"])
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite([lower, upper]).all()
                or np.any(upper <= lower)):
            raise RuntimeError("桌面模型边界无效")
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or not np.isfinite(offset)
                or abs(np.linalg.norm(normal) - 1.0) > 1e-3
                or normal[2] < np.cos(np.deg2rad(10.0))):
            raise RuntimeError("扫描桌面平面参数无效")
        if (len(table.primitives) != 1 or len(table.primitive_poses) != 1
                or table.primitives[0].type != 1
                or not np.allclose(
                    table.primitives[0].dimensions, upper - lower,
                    rtol=0.0, atol=1e-6)):
            raise RuntimeError("场景table尺寸与模型文件不一致")
        actual = self.pose_from_matrix(
            self._pose_matrix(table.pose)
            @ self._pose_matrix(table.primitive_poses[0]))
        actual.header = table.header
        if actual.header.frame_id.lstrip("/") != "base_link":
            actual = self.tf_buffer.transform(
                actual, "base_link", timeout=rospy.Duration(1.0))
        actual_matrix = self._pose_matrix(actual.pose)
        if (not np.allclose(
                actual_matrix[:3, 3], (lower + upper) / 2.0,
                rtol=0.0, atol=1e-4)
                or not np.allclose(
                    actual_matrix[:3, :3], np.eye(3),
                    rtol=0.0, atol=1e-4)):
            raise RuntimeError("场景table位姿与扫描文件不一致")

        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        scene = self.get_scene(request).scene
        acm = scene.allowed_collision_matrix
        names = list(acm.entry_names)
        defaults = dict(zip(
            acm.default_entry_names, acm.default_entry_values))
        for obstacle in ("table", "<octomap>"):
            for link in self.robot.get_link_names():
                if obstacle in names and link in names:
                    allowed = acm.entry_values[
                        names.index(obstacle)].enabled[names.index(link)]
                else:
                    allowed = defaults.get(obstacle, False) or defaults.get(link, False)
                if allowed:
                    raise RuntimeError(
                        "障碍与机器人碰撞被放开: {}/{}".format(obstacle, link))
        validity = GetStateValidityRequest()
        validity.robot_state = scene.robot_state
        validity.group_name = ""
        reply = self.check_validity(validity)
        if not reply.valid:
            pairs = [(item.contact_body_1, item.contact_body_2)
                     for item in reply.contacts]
            raise RuntimeError("当前状态无效/碰撞: " + str(pairs))
        self.table_model = model
        self.scan_signature = signature
        self.acm_before_task = self.get_acm()
        rospy.loginfo("冻结场景、table模型和当前机器人状态检查通过")

    def require_unchanged_scan(self):
        signature, _ = self.frozen_scene_signature()
        if signature != self.scan_signature:
            raise RuntimeError("环境OctoMap/table在任务中改变")

    def current_ee_matrix(self):
        pose = self.arm.get_current_pose("link6")
        if pose.header.frame_id.lstrip("/") != "base_link":
            pose = self.tf_buffer.transform(
                pose, "base_link", timeout=rospy.Duration(1.0))
        return self._pose_matrix(pose.pose)

    @staticmethod
    def pose_from_matrix(matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("目标矩阵无效")
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = matrix[:3, 3]
        quaternion = tf_trans.quaternion_from_matrix(matrix)
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
        return pose

    @staticmethod
    def unpack_plan(result):
        if isinstance(result, tuple):
            if not result[0]:
                raise RuntimeError("MoveIt规划失败")
            result = result[1]
        if result is None or not result.joint_trajectory.points:
            raise RuntimeError("MoveIt返回空轨迹")
        return result

    def validate_robot_state(self, state):
        request = GetStateValidityRequest()
        request.robot_state = state
        request.group_name = ""
        reply = self.check_validity(request)
        if not reply.valid:
            pairs = [(item.contact_body_1, item.contact_body_2)
                     for item in reply.contacts]
            raise RuntimeError("机器人状态无效/碰撞: " + str(pairs))

    def validate_trajectory(self, trajectory):
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        state = copy.deepcopy(self.get_scene(request).scene.robot_state)
        names = list(state.joint_state.name)
        index = {name: position for position, name in enumerate(names)}
        joint_names = list(trajectory.joint_trajectory.joint_names)
        if not joint_names or any(name not in index for name in joint_names):
            raise RuntimeError("轨迹关节为空或不在完整机器人状态中")
        previous = None
        for point in trajectory.joint_trajectory.points:
            goal = np.asarray(point.positions, dtype=float)
            if len(goal) != len(joint_names) or not np.isfinite(goal).all():
                raise RuntimeError("轨迹关节数不匹配或包含非有限值")
            # 返回点之间按每关节最多0.03rad细分，补充全机器人/附着物检查。
            count = 1 if previous is None else max(1, int(np.ceil(np.max(abs(goal - previous)) / 0.03)))
            for alpha in np.linspace(1.0 / count, 1.0, count):
                values = goal if previous is None else previous + (goal - previous) * alpha
                positions = list(state.joint_state.position)
                for name, value in zip(joint_names, values):
                    positions[index[name]] = float(value)
                state.joint_state.position = positions
                self.validate_robot_state(state)
            previous = goal

    def finish_motion(self, trajectory):
        self.require_unchanged_scan()
        self._fresh_joint_positions()
        self.validate_trajectory(trajectory)
        display = DisplayTrajectory()
        display.trajectory_start = self.robot.get_current_state()
        display.trajectory.append(trajectory)
        self.display_pub.publish(display)
        if not self.execute_motion:
            rospy.sleep(1.0)
            raise PlanningOnlyStop("仅规划模式：第一段轨迹已显示，未发送运动命令")
        self.require_unchanged_scan()
        self._fresh_joint_positions()
        if self.grasp_confirmed:
            self.confirm_grasp()
        if not self.arm.execute(trajectory, wait=True):
            raise RuntimeError("MoveIt/FollowJointTrajectory执行失败")
        self.arm.stop()
        final = dict(zip(
            self.arm.get_active_joints(), self.arm.get_current_joint_values()))
        names = trajectory.joint_trajectory.joint_names
        goals = trajectory.joint_trajectory.points[-1].positions
        error = max(abs(final[name] - goal) for name, goal in zip(names, goals))
        if error > 0.05:
            raise RuntimeError("执行后最大关节误差过大: {:.4f}rad".format(error))

    def move_joints(self, values):
        if len(values) != 6:
            raise ValueError("六轴目标必须包含6个数")
        self.arm.set_start_state_to_current_state()
        self.arm.set_joint_value_target(dict(zip(ARM_JOINTS, values)))
        try:
            self.finish_motion(self.unpack_plan(self.arm.plan()))
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def move_pose(self, matrix):
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(self.pose_from_matrix(matrix), "link6")
        try:
            self.finish_motion(self.unpack_plan(self.arm.plan()))
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def move_cartesian(self, matrix):
        self.arm.set_start_state_to_current_state()
        trajectory, fraction = self.arm.compute_cartesian_path(
            [self.pose_from_matrix(matrix).pose], eef_step=0.003,
            avoid_collisions=True)
        if not math.isfinite(fraction) or fraction < 0.999999 or not trajectory.joint_trajectory.points:
            raise RuntimeError(
                "直线路径受阻或不可达，完成比例 {:.6f}".format(fraction))
        limit = self._finite_float(
            "~cartesian_joint_step_limit", 0.15, 0.0, 0.50,
            strict_min=True)
        previous = dict(zip(
            self.arm.get_active_joints(), self.arm.get_current_joint_values()))
        for point in trajectory.joint_trajectory.points:
            current = dict(zip(
                trajectory.joint_trajectory.joint_names, point.positions))
            if any(abs(value - previous[name]) > limit
                   for name, value in current.items()):
                raise RuntimeError("笛卡尔轨迹关节跳变过大")
            previous = current
        trajectory = self.arm.retime_trajectory(
            self.robot.get_current_state(), trajectory,
            velocity_scaling_factor=self.velocity_scale,
            acceleration_scaling_factor=self.acceleration_scale)
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("轨迹时间参数化失败")
        times = [point.time_from_start.to_sec()
                 for point in trajectory.joint_trajectory.points]
        if times[-1] <= 0 or any(second <= first
                                 for first, second in zip(times, times[1:])):
            raise RuntimeError("轨迹时间无效")
        try:
            self.finish_motion(trajectory)
        finally:
            self.arm.stop()
            self.arm.clear_pose_targets()

    def wait_object(self, name, known, attached, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            is_known = name in self.scene.get_known_object_names()
            is_attached = bool(self.scene.get_attached_objects([name]))
            if is_known == known and is_attached == attached:
                return
            rospy.sleep(0.05)
        raise RuntimeError("场景对象未同步: " + name)

    def table_surface_z(self, x, y):
        normal = np.asarray(self.table_model["plane_normal"], dtype=float)
        return -(normal[0] * x + normal[1] * y
                 + float(self.table_model["plane_offset"])) / normal[2]

    def require_support_footprint(self, center_xyz, radius):
        lower = np.asarray(self.table_model["bounds_min"], dtype=float)
        upper = np.asarray(self.table_model["bounds_max"], dtype=float)
        border = radius + self.scan_signature[3] + self.support_edge_margin
        xy = np.asarray(center_xyz[:2], dtype=float)
        if np.any(xy <= lower[:2] + border) or np.any(xy >= upper[:2] - border):
            raise RuntimeError("瓶子接触区域靠近table边界，拒绝抓取")

    def add_bottle(self, center_xyz):
        center_xyz = np.asarray(center_xyz, dtype=float)
        if center_xyz.shape != (3,) or not np.isfinite(center_xyz).all():
            raise ValueError("瓶子中心无效")
        self.require_support_footprint(center_xyz, self.bottle_radius)
        if (center_xyz[2] - self.bottle_height / 2 + self.lift_distance
                <= self.table_model["bounds_max"][2] + self.lift_clearance):
            raise ValueError("lift_distance不足以让瓶底离开table包络；首次运动前修正参数")
        if (self.scene.get_attached_objects([TARGET_BOTTLE_ID])
                or TARGET_BOTTLE_ID in self.scene.get_known_object_names()):
            raise RuntimeError("已有target_bottle，请核对上次真实抓取状态后处理，拒绝覆盖")
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.orientation.w = 1.0
        (pose.pose.position.x, pose.pose.position.y,
         pose.pose.position.z) = center_xyz
        self.scene.add_cylinder(
            TARGET_BOTTLE_ID, pose,
            height=self.bottle_height, radius=self.bottle_radius)
        self.wait_object(TARGET_BOTTLE_ID, known=True, attached=False)
        self.bottle_center_at_grasp = center_xyz.copy()
        self._created_bottle = True
        self.set_contact_pairs(
            [(TARGET_BOTTLE_ID, link) for link in self.robot.get_link_names()], False)

    def get_acm(self):
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        return copy.deepcopy(
            self.get_scene(request).scene.allowed_collision_matrix)

    def apply_acm(self, acm):
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.allowed_collision_matrix = acm
        if not self.apply_scene(scene).success:
            raise RuntimeError("MoveIt拒绝ACM更新")

    def set_contact_pairs(self, pairs, allowed):
        acm = self.get_acm()
        for first, second in pairs:
            for name in (first, second):
                if name not in acm.entry_names:
                    for row in acm.entry_values:
                        row.enabled.append(False)
                    acm.entry_names.append(name)
                    acm.entry_values.append(AllowedCollisionEntry(
                        enabled=[False] * len(acm.entry_names)))
            first_index = acm.entry_names.index(first)
            second_index = acm.entry_names.index(second)
            acm.entry_values[first_index].enabled[second_index] = bool(allowed)
            acm.entry_values[second_index].enabled[first_index] = bool(allowed)
        self.apply_acm(acm)

    def current_gripper_opening(self):
        positions = self._fresh_joint_positions()
        left, right = positions["joint7"], positions["joint8"]
        if not (0 <= left <= 0.035 and -0.035 <= right <= 0
                and abs(left + right) <= 0.002):
            raise RuntimeError("joint7/joint8反馈不符合当前对称夹爪模型")
        return left - right

    def validate_gripper_sweep(self, target_opening):
        start = self.current_gripper_opening()
        sample_count = max(int(abs(target_opening - start) / 0.002) + 2, 2)
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        state = copy.deepcopy(self.get_scene(request).scene.robot_state)
        names = list(state.joint_state.name)
        index = {name: position for position, name in enumerate(names)}
        if any(name not in index for name in FINGER_JOINTS):
            raise RuntimeError("Planning Scene状态缺少joint7/joint8")
        for opening in np.linspace(start, target_opening, sample_count):
            positions = list(state.joint_state.position)
            positions[index["joint7"]] = opening / 2.0
            positions[index["joint8"]] = -opening / 2.0
            state.joint_state.position = positions
            self.validate_robot_state(state)

    def command_gripper(self, opening, wait_for_target=True):
        if not self.execute_motion:
            raise PlanningOnlyStop("仅规划模式不发送夹爪命令")
        if not 0.0 <= opening <= 0.07:
            raise ValueError("夹爪开口越界")
        self.require_unchanged_scan()
        self.validate_gripper_sweep(opening)
        self.require_unchanged_scan()
        result = self.gripper_service(
            gripper_angle=opening,
            gripper_effort=self.gripper_effort,
            gripper_code=1,
            set_zero=0)
        self._gripper_service_sent = True
        if not result.status:
            raise RuntimeError("夹爪驱动拒绝请求")
        if not wait_for_target:
            return
        deadline = time.monotonic() + self.gripper_feedback_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if abs(self.current_gripper_opening() - opening) <= self.gripper_tolerance:
                return
            rospy.sleep(0.05)
        raise RuntimeError("夹爪未在超时内到达目标开口")

    def confirm_grasp(self):
        # 要求一段时间内开口稳定、未闭到底且有实测夹持负载；不把服务成功当抓住。
        deadline = time.monotonic() + self.gripper_feedback_timeout
        stable_since = None
        previous = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            opening = self.current_gripper_opening()
            with self._joint_lock:
                effort = self._joint_efforts.get("joint7", float("nan"))
            valid = (self.grasp_opening_min <= opening <= self.grasp_opening_max
                     and math.isfinite(effort) and abs(effort) >= self.grasp_effort_min)
            if valid and (previous is None or abs(opening - previous) <= 0.001):
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= self.gripper_settle_time:
                    if self.grasp_check_service:
                        rospy.wait_for_service(self.grasp_check_service, timeout=5.0)
                        reply = rospy.ServiceProxy(self.grasp_check_service, Trigger)()
                        if not reply.success:
                            raise RuntimeError("外部夹持检查失败: " + reply.message)
                    rospy.loginfo("开口/负载稳定，夹持检查通过: opening=%.4fm effort=%.3f",
                                  opening, effort)
                    return
            else:
                stable_since = None
            previous = opening
            rospy.sleep(0.05)
        raise RuntimeError("夹持开口/负载未达到实测阈值，保持当前位置；不附着、不提升")

    def build_grasp_targets(self, observed_bottle):
        center_x = (observed_bottle[0, 3]
                    + self.bottle_center_offset_xy[0])
        center_y = (observed_bottle[1, 3]
                    + self.bottle_center_offset_xy[1])
        table_z = self.table_surface_z(center_x, center_y)
        center = np.array([
            center_x, center_y, table_z + self.bottle_height / 2.0])
        self.add_bottle(center)

        T_base_tool_grasp = observed_bottle.copy()
        T_base_tool_grasp[:3, 3] = [
            center_x, center_y, table_z + self.grasp_height]
        approach_axis = T_base_tool_grasp[:3, 2]
        T_base_tool_pre = T_base_tool_grasp.copy()
        T_base_tool_pre[:3, 3] -= approach_axis * self.pregrasp_distance
        inverse_tool = np.linalg.inv(self.T_link6_tool)
        return (
            center,
            T_base_tool_pre @ inverse_tool,
            T_base_tool_grasp @ inverse_tool,
        )

    def plan_grasp_candidate(self, observed_bottle):
        """Plan pregrasp + complete approach before moving, using virtual start state."""
        center, _, _ = self.build_grasp_targets(observed_bottle)
        acm = self.get_acm()
        errors = []
        for yaw_deg in self.approach_yaws:
            tool = observed_bottle.copy()
            tool[:3, :3] = (tf_trans.rotation_matrix(
                np.deg2rad(yaw_deg), [0, 0, 1])[:3, :3] @ observed_bottle[:3, :3])
            tool[:3, 3] = [center[0], center[1],
                          self.table_surface_z(center[0], center[1]) + self.grasp_height]
            grasp = tool @ np.linalg.inv(self.T_link6_tool)
            pre_tool = tool.copy()
            pre_tool[:3, 3] -= tool[:3, 2] * self.pregrasp_distance
            pre = pre_tool @ np.linalg.inv(self.T_link6_tool)
            try:
                self.require_unchanged_scan()
                self.arm.set_start_state_to_current_state()
                self.arm.set_pose_target(self.pose_from_matrix(pre), "link6")
                pre_trajectory = self.unpack_plan(self.arm.plan())
                self.validate_trajectory(pre_trajectory)
                # 在预抓取轨迹的虚拟终点检查最后一段；机器人仍停在原处。
                state = self.robot.get_current_state()
                positions = dict(zip(state.joint_state.name, state.joint_state.position))
                positions.update(zip(pre_trajectory.joint_trajectory.joint_names,
                                     pre_trajectory.joint_trajectory.points[-1].positions))
                state.joint_state.position = [positions[name] for name in state.joint_state.name]
                self.set_contact_pairs([(TARGET_BOTTLE_ID, "link7"),
                                        (TARGET_BOTTLE_ID, "link8")], True)
                self.arm.set_start_state(state)
                approach, fraction = self.arm.compute_cartesian_path(
                    [self.pose_from_matrix(grasp).pose], eef_step=0.003, avoid_collisions=True)
                if not math.isfinite(fraction) or fraction < 0.999999 or not approach.joint_trajectory.points:
                    raise RuntimeError("接近直线不完整: %.6f" % fraction)
                previous = positions
                limit = self._finite_float("~cartesian_joint_step_limit", 0.15, 0.0, 0.5, strict_min=True)
                for point in approach.joint_trajectory.points:
                    current = dict(zip(approach.joint_trajectory.joint_names, point.positions))
                    if any(abs(value - previous[name]) > limit for name, value in current.items()):
                        raise RuntimeError("候选接近段关节跳变")
                    previous = current
                self.validate_trajectory(approach)
                rospy.loginfo("抓取候选通过: 水平偏转%.1f度", yaw_deg)
                return pre_trajectory, grasp
            except RuntimeError as error:
                errors.append("%.1f度: %s" % (yaw_deg, error))
            finally:
                self.apply_acm(acm)
                self.arm.clear_pose_targets()
                self.arm.set_start_state_to_current_state()
        raise RuntimeError("没有有效的预抓取/直线接近候选: " + "; ".join(errors))

    def execute_place(self):
        if not self.enable_place:
            return
        if not self.grasp_confirmed:
            raise RuntimeError("尚未确认抓住瓶子，拒绝放置")
        x, y, z, roll, pitch, yaw = self.place_tool_pose
        tool_place = tf_trans.euler_matrix(roll, pitch, yaw, axes="sxyz")
        tool_place[:3, 3] = [x, y, z]
        place = tool_place @ np.linalg.inv(self.T_link6_tool)
        bottle = place @ self.T_link6_bottle
        if not np.allclose(bottle[:3, 2], [0, 0, 1], atol=0.01):
            raise RuntimeError("当前放置实现只支持瓶身保持直立")
        self.require_support_footprint(bottle[:3, 3], self.bottle_radius)
        expected_bottom = self.table_surface_z(bottle[0, 3], bottle[1, 3])
        actual_bottom = bottle[2, 3] - self.bottle_height / 2
        if abs(actual_bottom - expected_bottom) > 0.005:
            raise RuntimeError("放置目标瓶底不在当前table表面±5mm内，修正place_tool_pose")
        pre = place.copy()
        pre[2, 3] += self.place_approach_height
        if (bottle[2, 3] - self.bottle_height / 2 + self.place_approach_height
                <= self.table_model["bounds_max"][2] + self.lift_clearance):
            raise RuntimeError("预放置高度不足以离开膨胀table")
        rospy.loginfo(">>> MoveIt规划到预放置位")
        self.move_pose(pre)
        self.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
        self.move_cartesian(place)
        self.command_gripper(self.opening_open, wait_for_target=True)
        self.scene.remove_attached_object("link6", name=TARGET_BOTTLE_ID)
        self.wait_object(TARGET_BOTTLE_ID, known=True, attached=False)
        self.grasp_confirmed = False
        # 释放后以实际末端位姿更新世界瓶子；默认无滑动，需要用实物验证。
        bottle = self.current_ee_matrix() @ self.T_link6_bottle
        self.scene.add_cylinder(TARGET_BOTTLE_ID, self.pose_from_matrix(bottle),
                                height=self.bottle_height, radius=self.bottle_radius)
        self.wait_object(TARGET_BOTTLE_ID, known=True, attached=False)
        self.set_contact_pairs([(TARGET_BOTTLE_ID, "link7"),
                                (TARGET_BOTTLE_ID, "link8")], True)
        self.move_cartesian(pre)
        self.apply_acm(self.acm_before_task)
        self.set_contact_pairs([(TARGET_BOTTLE_ID, link) for link in
                                self.robot.get_link_names() + ["table", "<octomap>"]], False)
        rospy.loginfo(">>> 放置并退出完成，世界瓶子仍保留在场景中")

    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始MoveIt视觉抓取任务")
        self.wait_for_joint_feedback()
        self.require_scan_scene()
        if self.scene.get_attached_objects([TARGET_BOTTLE_ID]):
            raise RuntimeError("已有附着瓶子，不能当作空载重启任务")
        self.init_camera_and_detector()

        if self.move_to_photo_pose:
            rospy.logwarn("目标瓶子尚未建模；仅在已确认该路径安全时移动到拍照位")
            self.move_joints(self.photo_joint_target)
        else:
            rospy.loginfo("使用当前安全停稳姿态拍照；未自动移动到旧脚本拍照位")

        observed = self.get_bottle_pose(self.target_label)
        if observed is None:
            raise RuntimeError("没有检测到目标瓶子")
        rospy.loginfo(">>> 张开夹爪")
        if self.execute_motion:
            self.command_gripper(self.opening_open, wait_for_target=True)
        elif abs(self.current_gripper_opening() - self.opening_open) > self.gripper_tolerance:
            raise RuntimeError("仅规划不会张爪；先将实际夹爪置于配置张开量，再预览路径")
        rospy.loginfo(">>> 检查多个预抓取/接近方向")
        pre_trajectory, T_grasp = self.plan_grasp_candidate(observed)
        rospy.loginfo(">>> 执行预抓取轨迹")
        self.finish_motion(pre_trajectory)

        self.set_contact_pairs([
            (TARGET_BOTTLE_ID, "link7"),
            (TARGET_BOTTLE_ID, "link8"),
        ], True)
        rospy.loginfo(">>> 碰撞检查下直线接近瓶子")
        self.move_cartesian(T_grasp)

        rospy.loginfo(">>> 闭合夹爪并检查实测开口")
        self.command_gripper(
            self.opening_close_command, wait_for_target=False)
        self.confirm_grasp()
        base_bottle = np.eye(4)
        base_bottle[:3, 3] = self.bottle_center_at_grasp
        self.T_link6_bottle = np.linalg.inv(self.current_ee_matrix()) @ base_bottle
        self.scene.attach_object(
            TARGET_BOTTLE_ID, link="link6",
            touch_links=["link7", "link8"])
        self.wait_object(TARGET_BOTTLE_ID, known=False, attached=True)
        # 维持显式指尖接触权限；touch_links负责附着物与指尖关系。
        self.grasp_confirmed = True

        start_ee = self.current_ee_matrix()
        lift_target = start_ee.copy()
        lift_target[2, 3] += self.lift_distance
        acm_before_lift = self.get_acm()
        self.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
        if (self.bottle_center_at_grasp[2] - self.bottle_height / 2 + self.lift_distance
                <= self.table_model["bounds_max"][2] + self.lift_clearance):
            raise RuntimeError("lift_distance不足以让瓶底离开膨胀桌面")
        rospy.loginfo(">>> 携带瓶子沿base Z提升")
        self.move_cartesian(lift_target)
        self.confirm_grasp()
        actual_lift = self.current_ee_matrix()[2, 3] - start_ee[2, 3]
        bottle_bottom = (
            self.bottle_center_at_grasp[2]
            - self.bottle_height / 2.0 + actual_lift)
        table_envelope_top = float(self.table_model["bounds_max"][2])
        if bottle_bottom <= table_envelope_top + self.lift_clearance:
            raise RuntimeError("瓶底尚未确认离开table包络")
        self.apply_acm(acm_before_lift)
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        self.validate_robot_state(self.get_scene(request).scene.robot_state)

        if self.post_grasp_joint_target is not None:
            rospy.loginfo(">>> 携带瓶子规划到抓取后关节目标")
            self.move_joints(self.post_grasp_joint_target)
        self.execute_place()
        if self.grasp_confirmed:
            rospy.logwarn(
                "抓取提升完成；瓶子仍作为AttachedCollisionObject附着，未自动放置或回零")

    def run(self):
        try:
            self.execute_vision_grasp()
        except PlanningOnlyStop as error:
            rospy.logwarn("%s", error)
            # 仅回收本次预览临时创建的世界瓶子，实机失败不进入这条路径。
            if self._created_bottle and not self._gripper_service_sent:
                self.scene.remove_world_object(TARGET_BOTTLE_ID)
                self.wait_object(TARGET_BOTTLE_ID, known=False, attached=False)
                self.apply_acm(self.acm_before_task)
                self._created_bottle = False

    def close(self):
        if self.arm is not None:
            self.arm.stop()
            self.arm.clear_pose_targets()
        if self.joint_sub is not None:
            self.joint_sub.unregister()
            self.joint_sub = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def main():
    node = PiperVisionController.__new__(PiperVisionController)
    try:
        node.__init__()
        node.run()
        return 0
    except rospy.ROSInterruptException:
        return 130
    except Exception as error:
        rospy.logerr("MoveIt视觉抓取终止: %s；不自动松爪、detach或回零", error)
        return 1
    finally:
        # __init__失败也会进入这里；只释放已有资源，保留真实夹持对应的场景。
        if hasattr(node, "arm"):
            node.close()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoveIt bottle-grasp entry point (live scan).

The program no longer consumes an offline table model or a hand-driven static
OctoMap session.  On start it clears the previous snapshot, raises the arm to
``scan_start_joint_target``, scans the workspace from three viewpoints, fits the
table, separates the target bottle from every other obstacle, freezes the
snapshot and then grasps and places.

This module only assembles the submodules; the reusable pieces live in the
sibling modules:

- pose_utils / vision_utils / param_utils          (pure helpers)
- cloud_fusion / table_fit / obstacle_model /
  grasp_corridor                                   (pure scan geometry)
- camera / detector / hand_eye                     (SDK / calibration)
- joint_feedback / arm_controller / gripper_controller / scene_manager
- scan_task / grasp_task / place_task              (business orchestration)
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
import tf.transformations as tf_trans
import tf2_ros
try:  # 只用于 PoseStamped 的 TF 转换；缺少 PyKDL 时不影响主流程
    import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped TF conversion
except ImportError:  # pragma: no cover - depends on the interpreter
    print("警告: 未安装 tf2_geometry_msgs/PyKDL；"
          "若末端位姿不在base_link，TF转换会失败", file=sys.stderr)
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetPlanningScene,
    GetStateValidity,
)
from piper_msgs.srv import Gripper

from arm_controller import ArmController
from camera import (
    COLOR_HEIGHT,
    COLOR_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FPS,
    RealSenseCamera,
)
from detector import BottleDetector
from grasp_task import GraspTask
from gripper_controller import GripperController
from hand_eye import HandEyeCalibration
from joint_feedback import ARM_JOINTS, FINGER_JOINTS, JointFeedback  # noqa: F401
from param_utils import finite_float, numeric_list
from place_task import PlaceTask
from pose_utils import (  # noqa: F401
    build_level_grasp_rotation,
    euler_matrix,
    normalize_vector,
    pose_from_matrix,
    pose_matrix,
    quaternion_from_matrix,
    quaternion_matrix,
    validate_transform,
)
from scan_task import ScanTask
from scene_manager import SceneManager
from vision_utils import (  # noqa: F401
    COLOR_DEFS,
    COLOR_RATIO_THRES,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_PERCENTILE,
    SHRINK_RATIO,
    classify_bottle_color,
    deproject,
    draw_detection,
    draw_rejected,
    get_depth_percentile,
)

# ===============================
# 默认参数
# ===============================
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().with_name("bottle.pt"))
DEFAULT_CALIBRATION_PATH = str(
    Path(__file__).resolve().parents[4]
    / "calibration/eye_in_hand_d435i/session_02/result.json")
DEFAULT_ARTIFACT_ROOT = str(
    Path(__file__).resolve().parents[4] / "calibration/bottle_grasp_scan")

# None 表示抓取任意检测到的瓶子；也可以改成 "bottle-y"、"bottle-g"、"bottle-b"
TARGET_BOTTLE_LABEL = None

# 是否显示检测窗口。
SHOW_DETECTION_WINDOW = True


class PiperVisionController:
    """Composition root: parses parameters and wires all task submodules."""

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("piper_bottle_vision_grasp", anonymous=True)

        self.config = SimpleNamespace()
        cfg = self.config
        self.state = SimpleNamespace(
            current_ee_pose_mat=np.eye(4),
            grasp_confirmed=False,
            T_link6_bottle=None,
            T_base_bottle=None,
            bottle_center_at_grasp=None,
            created_bottle=False,
            gripper_service_sent=False,
            scan_result=None,
            card_model=None,
        )
        self.arm_controller = None
        self.joint_feedback = None
        self.camera = None
        self.scan_task = None
        self.grasp_task = None
        self.place_task = None

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

        required = (
            "support_edge_margin", "lift_clearance",
            "bottle_height", "bottle_radius", "grasp_height_from_table",
            "pregrasp_distance", "lift_distance", "lift_table_margin",
            "link6_tool_xyz",
            "grasp_opening_min", "grasp_opening_max", "grasp_effort_min",
            "gripper_opening_open", "gripper_opening_close_command",
            "gripper_effort", "gripper_tolerance", "gripper_feedback_timeout",
            "gripper_settle_time", "planning_time", "velocity_scale",
            "acceleration_scale", "cartesian_joint_step_limit",
            "scan_start_joint_target")
        missing = ["~" + key for key in required if not rospy.has_param("~" + key)]
        if missing:
            raise ValueError("缺少实测任务参数: " + ", ".join(missing))

        cfg.model_path = str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH))
        calibration_path = rospy.get_param(
            "~hand_eye_calibration", DEFAULT_CALIBRATION_PATH)
        cfg.target_label = rospy.get_param(
            "~target_bottle_label", TARGET_BOTTLE_LABEL)
        if cfg.target_label not in (None, "bottle-y", "bottle-g", "bottle-b"):
            raise ValueError("~target_bottle_label必须是bottle-y/g/b或null")
        cfg.show_detection_window = bool(rospy.get_param(
            "~show_detection_window", SHOW_DETECTION_WINDOW))
        cfg.support_edge_margin = finite_float(
            "~support_edge_margin", minimum=0.0)
        cfg.lift_clearance = finite_float(
            "~lift_clearance", minimum=0.0, strict_min=True)
        cfg.bottle_height = finite_float(
            "~bottle_height", minimum=0.0, strict_min=True)
        cfg.bottle_radius = finite_float(
            "~bottle_radius", minimum=0.0, strict_min=True)
        cfg.grasp_height = finite_float(
            "~grasp_height_from_table", minimum=0.0, strict_min=True)
        if cfg.grasp_height >= cfg.bottle_height:
            raise ValueError("~grasp_height_from_table 必须小于瓶高")
        cfg.pregrasp_distance = finite_float(
            "~pregrasp_distance", minimum=0.0, strict_min=True)
        cfg.lift_distance = finite_float(
            "~lift_distance", minimum=0.0, strict_min=True)
        # 抬升时瓶底越过table碰撞盒顶部的最小余量（不是运输余量）。
        cfg.lift_table_margin = finite_float(
            "~lift_table_margin", minimum=0.0, maximum=0.05)

        candidate_yaws = rospy.get_param(
            "~approach_yaw_offsets_deg",
            [0.0, 15.0, -15.0, 30.0, -30.0])
        if not isinstance(candidate_yaws, list) or not candidate_yaws or len(candidate_yaws) > 12:
            raise ValueError("~approach_yaw_offsets_deg需要1～12个方向")
        cfg.approach_yaws = numeric_list(
            "~approach_yaw_offsets_deg", candidate_yaws, len(candidate_yaws))

        cfg.opening_open = finite_float(
            "~gripper_opening_open", minimum=0.0, maximum=0.07)
        cfg.opening_close_command = finite_float(
            "~gripper_opening_close_command", minimum=0.0, maximum=0.07)
        cfg.grasp_opening_min = finite_float(
            "~grasp_opening_min", minimum=0.0, strict_min=True)
        cfg.grasp_opening_max = finite_float(
            "~grasp_opening_max", minimum=0.0, maximum=0.07)
        if (cfg.grasp_opening_max <= cfg.grasp_opening_min
                or cfg.grasp_opening_min <= cfg.opening_close_command
                or cfg.opening_open <= cfg.grasp_opening_max):
            raise ValueError("要求: 闭合目标 < 夹持开口下限 < 上限 < 张开目标")
        cfg.grasp_effort_min = finite_float(
            "~grasp_effort_min", minimum=0.0, strict_min=True)
        cfg.grasp_check_service = str(rospy.get_param("~grasp_check_service", ""))
        cfg.gripper_effort = finite_float(
            "~gripper_effort", minimum=0.5, maximum=2.0)
        cfg.gripper_tolerance = finite_float(
            "~gripper_tolerance", minimum=0.0, maximum=0.02,
            strict_min=True)
        cfg.gripper_feedback_timeout = finite_float(
            "~gripper_feedback_timeout", minimum=0.0, maximum=20.0,
            strict_min=True)
        cfg.gripper_settle_time = finite_float(
            "~gripper_settle_time", minimum=0.0, maximum=10.0)

        # ---------------- 扫描参数 ----------------
        cfg.scan_mode = str(rospy.get_param("~scan_mode", "map"))
        if cfg.scan_mode not in ("map", "semantic", "both"):
            raise ValueError("~scan_mode必须是map/semantic/both")
        cfg.scan_cloud_topic = str(rospy.get_param(
            "~scan_cloud_topic", "/piper/static_scan/points"))
        cfg.scan_camera_frame = str(rospy.get_param(
            "~scan_camera_frame", "camera_color_optical_frame"))
        if not cfg.scan_camera_frame or cfg.scan_camera_frame.startswith("/"):
            raise ValueError("~scan_camera_frame必须是相对TF frame名")
        cfg.scan_start_joint_target = numeric_list(
            "~scan_start_joint_target",
            rospy.get_param("~scan_start_joint_target"), 6)
        offsets = rospy.get_param(
            "~scan_joint1_offsets", [0.0, 0.3927, -0.3927, 0.7854, -0.7854])
        if not isinstance(offsets, list) or not 1 <= len(offsets) <= 9:
            raise ValueError("~scan_joint1_offsets需要1～9个数")
        cfg.scan_joint1_offsets = numeric_list(
            "~scan_joint1_offsets", offsets, len(offsets))
        if any(abs(value) > 0.9 for value in cfg.scan_joint1_offsets):
            raise ValueError("~scan_joint1_offsets单次偏移不能超过0.9rad")
        pitches = rospy.get_param("~scan_joint2_offsets", [0.0])
        if not isinstance(pitches, list) or not 1 <= len(pitches) <= 3:
            raise ValueError("~scan_joint2_offsets需要1～3个数")
        cfg.scan_joint2_offsets = numeric_list(
            "~scan_joint2_offsets", pitches, len(pitches))
        if any(abs(value) > 0.6 for value in cfg.scan_joint2_offsets):
            raise ValueError("~scan_joint2_offsets单次偏移不能超过0.6rad")
        cfg.scan_settle_time = finite_float(
            "~scan_settle_time", 1.0, minimum=0.0, maximum=5.0)
        cfg.scan_frames_per_view = int(rospy.get_param(
            "~scan_frames_per_view", 4))
        if not 1 <= cfg.scan_frames_per_view <= 12:
            raise ValueError("~scan_frames_per_view必须在1～12内")
        cfg.scan_frame_interval = finite_float(
            "~scan_frame_interval", 0.15, minimum=0.0, maximum=1.0)
        cfg.scan_min_frame_votes = int(rospy.get_param(
            "~scan_min_frame_votes", 2))
        if not 1 <= cfg.scan_min_frame_votes <= cfg.scan_frames_per_view:
            raise ValueError("~scan_min_frame_votes必须在1～帧数内")
        cfg.scan_min_views = int(rospy.get_param("~scan_min_views", 1))
        view_count = len(cfg.scan_joint1_offsets) * len(cfg.scan_joint2_offsets)
        if not 1 <= cfg.scan_min_views <= view_count:
            raise ValueError("~scan_min_views不得超过视点数")
        cfg.scan_voxel_m = finite_float(
            "~scan_voxel_m", 0.006, minimum=0.0, maximum=0.05, strict_min=True)
        cfg.scan_cluster_voxel_m = finite_float(
            "~scan_cluster_voxel_m", 0.012, minimum=0.0, maximum=0.1,
            strict_min=True)
        cfg.scan_min_points = int(rospy.get_param("~scan_min_points", 500))
        if cfg.scan_min_points < 50:
            raise ValueError("~scan_min_points至少为50")
        cfg.scan_min_cluster_points = int(rospy.get_param(
            "~scan_min_cluster_points", 40))
        if cfg.scan_min_cluster_points < 10:
            raise ValueError("~scan_min_cluster_points至少为10")
        cfg.scan_stride = int(rospy.get_param("~scan_stride", 2))
        if not 1 <= cfg.scan_stride <= 8:
            raise ValueError("~scan_stride必须在1～8内")
        cfg.scan_min_depth = finite_float(
            "~scan_min_depth", 0.25, minimum=0.05, maximum=1.0)
        cfg.scan_max_range = finite_float(
            "~scan_max_range", 2.0, minimum=0.3, maximum=8.0)
        cfg.scan_roi_min = numeric_list(
            "~scan_roi_min", rospy.get_param(
                "~scan_roi_min", [-0.40, -1.20, -0.40]), 3)
        cfg.scan_roi_max = numeric_list(
            "~scan_roi_max", rospy.get_param(
                "~scan_roi_max", [1.40, 1.20, 1.30]), 3)
        if np.any(np.asarray(cfg.scan_roi_max)
                  <= np.asarray(cfg.scan_roi_min)):
            raise ValueError("~scan_roi_max必须逐轴大于~scan_roi_min")
        if not cfg.scan_roi_min[2] < cfg.scan_roi_max[2]:
            raise ValueError("~scan_roi 高度范围无效")
        cfg.scan_detect_frames = int(rospy.get_param("~scan_detect_frames", 8))
        if not 1 <= cfg.scan_detect_frames <= 60:
            raise ValueError("~scan_detect_frames必须在1～60内")
        cfg.scan_table_clearance = finite_float(
            "~scan_table_clearance", 0.015, minimum=0.0, maximum=0.05)
        cfg.scan_min_obstacle_height = finite_float(
            "~scan_min_obstacle_height", 0.02, minimum=0.0, maximum=0.20)
        cfg.scan_card_carve_margin = finite_float(
            "~scan_card_carve_margin", 0.02, minimum=0.0, maximum=0.10)
        cfg.scan_footprint_margin = finite_float(
            "~scan_footprint_margin", 0.10, minimum=0.0, maximum=0.3)
        cfg.scan_plane_threshold = finite_float(
            "~scan_table_plane_threshold", 0.006, minimum=0.0,
            maximum=0.05, strict_min=True)
        cfg.scan_table_voxel = finite_float(
            "~scan_table_voxel", 0.008, minimum=0.0, maximum=0.05,
            strict_min=True)
        cfg.scan_table_max_tilt = finite_float(
            "~scan_table_max_tilt_deg", 3.0, minimum=0.0, maximum=10.0,
            strict_min=True)
        cfg.scan_table_min_inliers = int(rospy.get_param(
            "~scan_table_min_inliers", 300))
        cfg.scan_table_min_ratio = finite_float(
            "~scan_table_min_ratio", 0.25, minimum=0.0, maximum=1.0,
            strict_min=True)
        cfg.scan_table_iterations = int(rospy.get_param(
            "~scan_table_iterations", 200))
        cfg.scan_table_edge_padding = finite_float(
            "~scan_table_edge_padding", 0.03, minimum=0.0, maximum=0.2)
        cfg.scan_table_support_depth = finite_float(
            "~scan_table_support_depth", 0.02, minimum=0.0, maximum=0.2)
        cfg.scan_table_mask_padding = finite_float(
            "~scan_table_mask_padding", 0.02, minimum=0.0, maximum=0.2)
        # 桌面拟合焦点：用目标瓶表面点把RANSAC限制在局部窗口，避免大ROI里的
        # 地板/邻桌赢掉拟合；footprint只用于碰撞盒范围，不改变平面本身。
        cfg.scan_table_focus_radius = finite_float(
            "~scan_table_focus_radius", 0.30, minimum=0.05, maximum=1.0,
            strict_min=True)
        cfg.scan_table_focus_down = finite_float(
            "~scan_table_focus_down", 0.30, minimum=0.05, maximum=1.0,
            strict_min=True)
        cfg.scan_table_focus_up = finite_float(
            "~scan_table_focus_up", 0.02, minimum=0.0, maximum=0.20)
        cfg.scan_table_footprint_radius = finite_float(
            "~scan_table_footprint_radius", 0.60, minimum=0.10, maximum=2.0,
            strict_min=True)
        cfg.scan_table_surface_epsilon = finite_float(
            "~scan_table_surface_epsilon", 0.002, minimum=0.0, maximum=0.05)
        cfg.scan_table_max_jump = finite_float(
            "~scan_table_max_jump", 0.03, minimum=0.005, maximum=0.20,
            strict_min=True)
        cfg.scan_card_max_top_height = finite_float(
            "~scan_card_max_top_height", 0.30, minimum=0.05, maximum=1.0,
            strict_min=True)
        # ---------------- 静态障碍地图 ----------------
        cfg.scan_map_cloud_voxel = finite_float(
            "~scan_map_cloud_voxel", 0.015, minimum=0.0, maximum=0.10,
            strict_min=True)
        # 底座自身半径13cm + 3cm余量；地图里必须挖掉，否则link1一开始就碰撞。
        cfg.scan_map_base_keepout_radius = finite_float(
            "~scan_map_base_keepout_radius", 0.16, minimum=0.0, maximum=0.50)
        cfg.scan_map_target_carve = finite_float(
            "~scan_map_target_carve", 0.02, minimum=0.0, maximum=0.15)
        cfg.scan_map_target_height_margin = finite_float(
            "~scan_map_target_height_margin", 0.04, minimum=0.0, maximum=0.20)
        cfg.scan_map_support_carve = finite_float(
            "~scan_map_support_carve", 0.015, minimum=0.0, maximum=0.10)
        cfg.scan_map_support_margin = finite_float(
            "~scan_map_support_margin", 0.05, minimum=0.0, maximum=0.30)
        cfg.scan_map_settle_quiet = finite_float(
            "~scan_map_settle_quiet", 1.0, minimum=0.0, maximum=10.0)
        cfg.scan_map_settle_timeout = finite_float(
            "~scan_map_settle_timeout", 20.0, minimum=1.0, maximum=120.0,
            strict_min=True)
        cfg.scan_map_min_bytes = int(rospy.get_param(
            "~scan_map_min_bytes", 1000))
        if cfg.scan_map_min_bytes < 100:
            raise ValueError("~scan_map_min_bytes至少为100")
        cfg.scan_map_require_current_valid = bool(rospy.get_param(
            "~scan_map_require_current_valid", True))
        cfg.scan_card_planarity_m = finite_float(
            "~scan_card_planarity_m", 0.012, minimum=0.0, maximum=0.05,
            strict_min=True)
        cfg.scan_card_min_extent = finite_float(
            "~scan_card_min_extent", 0.10, minimum=0.0, maximum=1.0,
            strict_min=True)
        cfg.scan_card_max_thickness = finite_float(
            "~scan_card_max_thickness", 0.03, minimum=0.0, maximum=0.2,
            strict_min=True)
        cfg.scan_cylinder_radius_tol = finite_float(
            "~scan_cylinder_radius_tol", 0.6, minimum=0.0, maximum=1.0,
            strict_min=True)
        cfg.scan_cylinder_height_tol = finite_float(
            "~scan_cylinder_height_tol", 0.5, minimum=0.0, maximum=1.0,
            strict_min=True)
        cfg.scan_non_target_cylinder = bool(rospy.get_param(
            "~scan_non_target_cylinder", True))
        cfg.scan_obstacle_padding = finite_float(
            "~scan_obstacle_padding", 0.01, minimum=0.0, maximum=0.1)
        # 卡片是薄板，用更小的外扩量，避免碰撞盒明显大于实物。
        cfg.scan_card_padding = finite_float(
            "~scan_card_padding", 0.005, minimum=0.0, maximum=0.05)
        # 卡片附近的薄板视为卡片支架，不加入障碍集（只有侧视点才看得到）。
        cfg.scan_stand_max_distance = finite_float(
            "~scan_stand_max_distance", 0.25, minimum=0.0, maximum=0.6)
        cfg.scan_stand_max_thickness = finite_float(
            "~scan_stand_max_thickness", 0.03, minimum=0.0, maximum=0.10)
        cfg.scan_corridor_radius = finite_float(
            "~scan_corridor_radius", 0.06, minimum=0.0, maximum=0.3)
        cfg.scan_corridor_samples = int(rospy.get_param(
            "~scan_corridor_samples", 24))
        if cfg.scan_corridor_samples < 2:
            raise ValueError("~scan_corridor_samples至少为2")
        cfg.scan_artifact_root = str(rospy.get_param(
            "~scan_artifact_root", DEFAULT_ARTIFACT_ROOT))

        # ---------------- 放置参数 ----------------
        cfg.enable_place = bool(rospy.get_param("~enable_place", True))
        cfg.place_mode = str(rospy.get_param("~place_mode", "card"))
        if cfg.place_mode not in ("card", "fixed"):
            raise ValueError("~place_mode必须是card或fixed")
        cfg.home_joint_target = numeric_list(
            "~home_joint_target", rospy.get_param(
                "~home_joint_target", [0.0] * 6), 6)
        if cfg.enable_place:
            if not rospy.has_param("~place_approach_height"):
                raise ValueError("缺少放置参数: ~place_approach_height")
            cfg.place_approach_height = finite_float(
                "~place_approach_height", minimum=0.0, strict_min=True)
        if cfg.enable_place and cfg.place_mode == "fixed":
            cfg.place_tool_pose = numeric_list(
                "~place_tool_pose", rospy.get_param("~place_tool_pose"), 6)
        if cfg.enable_place and cfg.place_mode == "card":
            for key in ("place_offset_from_card", "place_release_clearance"):
                if not rospy.has_param("~" + key):
                    raise ValueError("缺少card放置参数: ~" + key)
            # 松手点沿卡片正面法向前移的距离；必须大于瓶半径+卡厚+余量。
            cfg.place_offset_from_card = finite_float(
                "~place_offset_from_card", minimum=0.0, maximum=1.0,
                strict_min=True)
            cfg.place_card_offset_xy = numeric_list(
                "~place_card_offset_xy",
                rospy.get_param("~place_card_offset_xy", [0.0, 0.0]), 2)
            cfg.place_card_clearance = finite_float(
                "~place_card_clearance", 0.015, minimum=0.0, maximum=0.1)
            # 松手时瓶底离桌面的最小间隙；松开后瓶子自由下落。
            cfg.place_release_clearance = finite_float(
                "~place_release_clearance", minimum=0.0, maximum=0.05)
            cfg.place_front_mode = str(
                rospy.get_param("~place_front_mode", "card_normal"))
            if cfg.place_front_mode not in (
                    "card_normal", "toward_robot", "fixed_yaw"):
                raise ValueError(
                    "~place_front_mode必须是card_normal/toward_robot/fixed_yaw")
            cfg.place_front_yaw = finite_float(
                "~place_front_yaw_deg", 0.0, minimum=-180.0, maximum=180.0)
            place_yaws = rospy.get_param(
                "~place_yaw_offsets_deg",
                [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0])
            if (not isinstance(place_yaws, list) or not place_yaws
                    or len(place_yaws) > 12):
                raise ValueError("~place_yaw_offsets_deg需要1～12个方向")
            cfg.place_yaws = numeric_list(
                "~place_yaw_offsets_deg", place_yaws, len(place_yaws))

        cfg.bottle_center_offset_xy = numeric_list(
            "~bottle_center_offset_xy",
            rospy.get_param("~bottle_center_offset_xy", [0.0, 0.0]), 2)
        link6_tool_xyz = numeric_list(
            "~link6_tool_xyz",
            rospy.get_param("~link6_tool_xyz"), 3)
        tool_rpy = numeric_list(
            "~link6_tool_rpy",
            rospy.get_param("~link6_tool_rpy", [0.0, 0.0, 0.0]), 3)
        cfg.T_link6_tool = tf_trans.euler_matrix(*tool_rpy, axes="sxyz")
        cfg.T_link6_tool[:3, 3] = link6_tool_xyz

        # true时只规划并校验预抓取/直线接近，跳过抓取执行；扫描阶段仍会运动。
        cfg.dry_run = bool(rospy.get_param("~dry_run", False))
        cfg.planner_id = str(rospy.get_param("~planner_id", "RRTConnect"))
        cfg.planning_time = finite_float(
            "~planning_time", minimum=2.0, maximum=30.0)
        cfg.planning_attempts = int(rospy.get_param("~planning_attempts", 8))
        if not 1 <= cfg.planning_attempts <= 30:
            raise ValueError("~planning_attempts必须在1～30内")
        cfg.velocity_scale = finite_float(
            "~velocity_scale", minimum=0.0, maximum=0.10, strict_min=True)
        cfg.acceleration_scale = finite_float(
            "~acceleration_scale", minimum=0.0, maximum=0.10,
            strict_min=True)
        cfg.cartesian_joint_step_limit = finite_float(
            "~cartesian_joint_step_limit", minimum=0.0, maximum=0.50,
            strict_min=True)

        self._build_components(cfg, calibration_path)

    def _build_components(self, cfg, calibration_path):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.robot = moveit_commander.RobotCommander()
        self.planning_scene = moveit_commander.PlanningSceneInterface(
            synchronous=True)
        move_group = moveit_commander.MoveGroupCommander("arm")
        move_group.set_pose_reference_frame("base_link")
        move_group.set_end_effector_link("link6")
        move_group.set_planner_id(cfg.planner_id)
        move_group.set_planning_time(cfg.planning_time)
        move_group.set_num_planning_attempts(cfg.planning_attempts)
        move_group.set_max_velocity_scaling_factor(cfg.velocity_scale)
        move_group.set_max_acceleration_scaling_factor(cfg.acceleration_scale)
        move_group.set_goal_position_tolerance(0.003)
        move_group.set_goal_orientation_tolerance(0.03)
        move_group.allow_replanning(False)

        self.display_pub = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory,
            queue_size=1, latch=True)
        self.joint_feedback = JointFeedback()
        self.joint_feedback.subscribe()
        for name in ("/get_planning_scene", "/apply_planning_scene",
                     "/check_state_validity", "/compute_ik", "/gripper_srv"):
            rospy.wait_for_service(name, timeout=10.0)
        self.get_scene = rospy.ServiceProxy(
            "/get_planning_scene", GetPlanningScene)
        self.apply_scene = rospy.ServiceProxy(
            "/apply_planning_scene", ApplyPlanningScene)
        self.check_validity = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity)
        gripper_service = rospy.ServiceProxy("/gripper_srv", Gripper)

        self.scene_manager = SceneManager(
            self.get_scene, self.apply_scene, self.check_validity,
            self.robot, self.tf_buffer, self.planning_scene,
            support_edge_margin=cfg.support_edge_margin,
            voxel_size=cfg.scan_voxel_m)
        self.gripper = GripperController(
            self.joint_feedback, self.scene_manager, gripper_service,
            self.state, cfg.gripper_effort,
            cfg.gripper_tolerance, cfg.gripper_feedback_timeout,
            cfg.gripper_settle_time, cfg.grasp_opening_min,
            cfg.grasp_opening_max, cfg.grasp_effort_min,
            cfg.grasp_check_service)
        self.arm_controller = ArmController(
            move_group, self.robot, self.scene_manager, self.joint_feedback,
            self.display_pub, self.state,
            self.gripper.confirm, cfg.velocity_scale, cfg.acceleration_scale,
            cfg.cartesian_joint_step_limit)
        self.hand_eye = HandEyeCalibration(calibration_path)
        self._ensure_camera_tf()
        self.camera = RealSenseCamera(
            COLOR_WIDTH, COLOR_HEIGHT, DEPTH_WIDTH, DEPTH_HEIGHT, FPS)
        self.detector = BottleDetector(cfg.model_path)
        self.scan_task = ScanTask(
            cfg, self.state, self.arm_controller, self.gripper,
            self.scene_manager, self.joint_feedback, self.camera,
            self.hand_eye, self.detector)
        self.grasp_task = GraspTask(
            cfg, self.state, self.arm_controller, self.gripper,
            self.scene_manager, self.joint_feedback, place_task=None)
        self.place_task = PlaceTask(
            cfg, self.state, self.arm_controller, self.gripper,
            self.scene_manager, self.joint_feedback)
        self.grasp_task.place_task = self.place_task

    def _tf_available(self, parent, child):
        try:
            return bool(self.tf_buffer.can_transform(
                parent, child, rospy.Time(0), rospy.Duration(0.05)))
        except Exception:  # tf2 LookupException/ExtrapolationException
            return False

    def _ensure_camera_tf(self):
        """Make the camera optical frame reachable from base_link.

        MoveIt's ``PointCloudOctomapUpdater`` uses the point-cloud *header*
        frame as the sensor frame, so the optical frame must be in TF;
        otherwise the static octomap cannot be built.  The calibration JSON is
        the single source of truth, and any edge that already exists (for
        example because ``camera_extrinsics.launch`` or the RealSense driver is
        running) is left untouched.
        """
        try:
            transforms = self.hand_eye.static_transforms()
        except RuntimeError as error:
            raise RuntimeError(
                "%s；静态障碍地图需要相机TF" % error) from error
        # 给已有TF一点时间出现（例如camera_extrinsics.launch或驱动刚启动），
        # 避免重复发布同一对静态TF。
        missing = list(transforms)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            missing = [(parent, child, matrix)
                       for parent, child, matrix in transforms
                       if not self._tf_available(parent, child)]
            if not missing:
                rospy.loginfo("相机静态TF已存在，沿用现有链路")
                return
            rospy.sleep(0.1)
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster
        broadcaster = StaticTransformBroadcaster()
        messages = []
        for parent, child, matrix in missing:
            matrix = np.asarray(matrix, dtype=float)
            message = TransformStamped()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = parent
            message.child_frame_id = child
            message.transform.translation.x = float(matrix[0, 3])
            message.transform.translation.y = float(matrix[1, 3])
            message.transform.translation.z = float(matrix[2, 3])
            quaternion = quaternion_from_matrix(matrix)
            (message.transform.rotation.x, message.transform.rotation.y,
             message.transform.rotation.z, message.transform.rotation.w) = [
                float(value) for value in quaternion]
            messages.append(message)
        broadcaster.sendTransform(messages)
        # keep a reference: the latched publisher must live as long as the node
        self._static_tf_broadcaster = broadcaster
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if all(self._tf_available(parent, child)
                   for parent, child, _ in transforms):
                rospy.loginfo(
                    "已发布相机静态TF: %s",
                    ", ".join("%s->%s" % (parent, child)
                              for parent, child, _ in missing))
                return
            rospy.sleep(0.1)
        raise RuntimeError(
            "相机静态TF不可用：%s；检查手眼标定文件和camera_extrinsics.launch"
            % ", ".join("%s->%s" % (parent, child)
                        for parent, child, _ in missing))

    def run(self):
        self.camera.start()
        self.detector.load()
        self.scan_task.execute()
        self.grasp_task.execute_vision_grasp()

    def close(self):
        if self.arm_controller is not None:
            self.arm_controller.arm.stop()
            self.arm_controller.arm.clear_pose_targets()
        if self.joint_feedback is not None:
            self.joint_feedback.close()
        if self.camera is not None:
            self.camera.stop()
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
        rospy.logerr(
            "MoveIt视觉抓取终止: %s；不自动松爪、detach或回零", error)
        return 1
    finally:
        # __init__失败也会进入这里；只释放已有资源，保留真实夹持对应的场景。
        if hasattr(node, "arm_controller"):
            node.close()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    sys.exit(main())

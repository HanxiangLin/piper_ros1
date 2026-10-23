#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoveIt bottle-grasp entry point.

This module only assembles the submodules (camera, detector, hand-eye,
joint feedback, scene/ACM, arm, gripper, grasp/place tasks) and runs the
complete task.  Reusable components live in the sibling modules:

- pose_utils / vision_utils / param_utils     (pure helpers)
- camera / detector / hand_eye                (SDK / calibration)
- joint_feedback / arm_controller / gripper_controller / scene_manager
- grasp_task / place_task                     (business orchestration)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
import tf.transformations as tf_trans
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped TF conversion
import tf2_ros
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetPlanningScene,
    GetStateValidity,
)
from piper_msgs.srv import Gripper

from arm_controller import ArmController, PlanningOnlyStop  # noqa: F401
from camera import (  # noqa: F401
    COLOR_HEIGHT,
    COLOR_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FPS,
    RealSenseCamera,
    apply_filters,
    build_filters,
)
from detector import CONF_THRES, IMG_SIZE, BottleDetector  # noqa: F401
from grasp_task import GraspTask
from gripper_controller import GripperController
from hand_eye import HandEyeCalibration
from joint_feedback import (  # noqa: F401
    ARM_JOINTS,
    FINGER_JOINTS,
    JointFeedback,
)
from param_utils import finite_float, numeric_list, optional_numeric_list
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
from scene_manager import TARGET_BOTTLE_ID, SceneManager
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
            bottle_center_at_grasp=None,
            created_bottle=False,
            gripper_service_sent=False,
        )
        self.arm_controller = None
        self.joint_feedback = None
        self.camera = None

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
        cfg.execute_motion = execute
        required = (
            "table_model", "support_edge_margin", "lift_clearance",
            "lift_table_margin",
            "bottle_height", "bottle_radius", "grasp_height_from_table",
            "pregrasp_distance", "lift_distance", "link6_tool_xyz",
            "grasp_opening_min", "grasp_opening_max", "grasp_effort_min",
            "gripper_opening_open", "gripper_opening_close_command",
            "gripper_effort", "gripper_tolerance", "gripper_feedback_timeout",
            "gripper_settle_time", "planning_time", "velocity_scale",
            "acceleration_scale", "cartesian_joint_step_limit")
        missing = ["~" + key for key in required if not rospy.has_param("~" + key)]
        if missing:
            raise ValueError("缺少实测任务参数: " + ", ".join(missing))

        cfg.model_path = str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH))
        cfg.table_model_path = str(rospy.get_param("~table_model"))
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
            "~approach_yaw_offsets_deg", [0.0, 30.0, -30.0])
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

        cfg.move_to_photo_pose = bool(rospy.get_param("~move_to_photo_pose", True))
        cfg.photo_joint_target = numeric_list(
            "~photo_joint_target",
            rospy.get_param("~photo_joint_target",
                            [0.0, 0.35, -0.47, 0.0, 0.65, 0.0]), 6)
        cfg.post_grasp_joint_target = optional_numeric_list(
            "~post_grasp_joint_target", 6)
        cfg.enable_place = bool(rospy.get_param("~enable_place", True))
        cfg.place_mode = str(rospy.get_param("~place_mode", "card"))
        if cfg.place_mode not in ("card", "fixed"):
            raise ValueError("~place_mode必须是card或fixed")
        if cfg.enable_place:
            if not rospy.has_param("~place_approach_height"):
                raise ValueError("缺少放置参数: ~place_approach_height")
            cfg.place_approach_height = finite_float(
                "~place_approach_height", minimum=0.0, strict_min=True)
        if cfg.enable_place and cfg.place_mode == "fixed":
            cfg.place_tool_pose = numeric_list(
                "~place_tool_pose", rospy.get_param("~place_tool_pose"), 6)
        if cfg.enable_place and cfg.place_mode == "card":
            for key in ("place_offset_from_card", "place_card_conf_thres",
                        "place_release_clearance", "place_front_yaw_deg"):
                if not rospy.has_param("~" + key):
                    raise ValueError("缺少card放置参数: ~" + key)
            # 松手点沿“卡片指向机器人”方向水平前移的距离。
            cfg.place_offset_from_card = finite_float(
                "~place_offset_from_card", minimum=0.0, maximum=1.0,
                strict_min=True)
            cfg.place_card_offset_xy = numeric_list(
                "~place_card_offset_xy",
                rospy.get_param("~place_card_offset_xy", [0.0, 0.0]), 2)
            # 立在桌面上的卡片透视角更大，允许比抓取瓶子更低的YOLO置信度。
            cfg.place_card_conf_thres = finite_float(
                "~place_card_conf_thres", minimum=0.05, maximum=0.95)
            # 松手时瓶底离桌面的最小间隙；松开后瓶子自由下落。
            cfg.place_release_clearance = finite_float(
                "~place_release_clearance", minimum=0.0, maximum=0.05)
            cfg.place_card_label = rospy.get_param(
                "~place_card_label", cfg.target_label)
            if cfg.place_card_label not in (
                    None, "bottle-y", "bottle-g", "bottle-b"):
                raise ValueError("~place_card_label必须是bottle-y/g/b或null")
            cfg.place_front_mode = str(
                rospy.get_param("~place_front_mode", "toward_robot"))
            if cfg.place_front_mode not in ("toward_robot", "fixed_yaw"):
                raise ValueError("~place_front_mode必须是toward_robot或fixed_yaw")
            cfg.place_front_yaw = finite_float("~place_front_yaw_deg")
            cfg.place_photo_joint_target = optional_numeric_list(
                "~place_photo_joint_target", 6)

        cfg.bottle_center_offset_xy = numeric_list(
            "~bottle_center_offset_xy",
            rospy.get_param("~bottle_center_offset_xy", [0.0, 0.0]), 2)
        # 深度百分位通常落在瓶子朝向相机的一侧表面；不能把该表面点
        # 直接当成瓶子的轴线中心。默认根据检测像素射线、已扫描桌面平面
        # 和瓶高反算瓶子中心，避免用TCP参数掩盖视觉几何误差。
        cfg.estimate_bottle_center_from_table = bool(
            rospy.get_param("~estimate_bottle_center_from_table", True))
        link6_tool_xyz = numeric_list(
            "~link6_tool_xyz",
            rospy.get_param("~link6_tool_xyz"), 3)
        tool_rpy = numeric_list(
            "~link6_tool_rpy",
            rospy.get_param("~link6_tool_rpy", [0.0, 0.0, 0.0]), 3)
        cfg.T_link6_tool = tf_trans.euler_matrix(*tool_rpy, axes="sxyz")
        cfg.T_link6_tool[:3, 3] = link6_tool_xyz

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
                     "/check_state_validity"):
            rospy.wait_for_service(name, timeout=10.0)
        self.get_scene = rospy.ServiceProxy(
            "/get_planning_scene", GetPlanningScene)
        self.apply_scene = rospy.ServiceProxy(
            "/apply_planning_scene", ApplyPlanningScene)
        self.check_validity = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity)
        gripper_service = None
        if cfg.execute_motion:
            rospy.wait_for_service("/gripper_srv", timeout=5.0)
            gripper_service = rospy.ServiceProxy("/gripper_srv", Gripper)

        self.scene_manager = SceneManager(
            self.get_scene, self.apply_scene, self.check_validity,
            self.robot, self.tf_buffer, cfg.table_model_path,
            self.planning_scene,
            support_edge_margin=cfg.support_edge_margin)
        self.gripper = GripperController(
            self.joint_feedback, self.scene_manager, gripper_service,
            cfg.execute_motion, self.state, cfg.gripper_effort,
            cfg.gripper_tolerance, cfg.gripper_feedback_timeout,
            cfg.gripper_settle_time, cfg.grasp_opening_min,
            cfg.grasp_opening_max, cfg.grasp_effort_min,
            cfg.grasp_check_service)
        self.arm_controller = ArmController(
            move_group, self.robot, self.scene_manager, self.joint_feedback,
            self.display_pub, cfg.execute_motion, self.state,
            self.gripper.confirm, cfg.velocity_scale, cfg.acceleration_scale,
            cfg.cartesian_joint_step_limit)
        self.hand_eye = HandEyeCalibration(calibration_path)
        self.camera = RealSenseCamera(
            COLOR_WIDTH, COLOR_HEIGHT, DEPTH_WIDTH, DEPTH_HEIGHT, FPS)
        self.detector = BottleDetector(cfg.model_path)
        self.grasp_task = GraspTask(
            cfg, self.state, self.arm_controller, self.gripper,
            self.scene_manager, self.joint_feedback, self.detector,
            self.camera, self.hand_eye, place_task=None)
        self.place_task = PlaceTask(
            cfg, self.state, self.arm_controller, self.gripper,
            self.scene_manager, self.joint_feedback,
            self.grasp_task.get_bottle_pose)
        self.grasp_task.place_task = self.place_task

    def run(self):
        try:
            self.grasp_task.execute_vision_grasp()
        except PlanningOnlyStop as error:
            rospy.logwarn("%s", error)
            # 仅回收本次预览临时创建的世界瓶子，实机失败不进入这条路径。
            if self.state.created_bottle and not self.state.gripper_service_sent:
                self.scene_manager.scene.remove_world_object(TARGET_BOTTLE_ID)
                self.scene_manager.wait_object(
                    TARGET_BOTTLE_ID, known=False, attached=False)
                self.scene_manager.apply_acm(
                    self.scene_manager.acm_before_task)
                self.state.created_bottle = False

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

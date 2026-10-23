#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision-guided grasp task orchestration."""

import numpy as np
import rospy
import tf.transformations as tf_trans
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningSceneRequest

from pose_utils import pose_from_matrix
from scene_manager import TARGET_BOTTLE_ID


class GraspTask:
    """Build grasp targets, plan candidates and execute the vision grasp."""

    def __init__(self, config, state, arm, gripper, scene, joint_feedback,
                 detector, camera, hand_eye, place_task):
        self.config = config
        self.state = state
        self.arm = arm
        self.gripper = gripper
        self.scene = scene
        self.joint_feedback = joint_feedback
        self.detector = detector
        self.camera = camera
        self.hand_eye = hand_eye
        self.place_task = place_task

    # ------------------------------------------------------------------
    # Detection helper shared with the place task.
    # ------------------------------------------------------------------
    def get_bottle_pose(self, target_label=None, max_frames=60,
                        center_height=None, description="目标瓶子",
                        use_surface_point=False, conf_thres=None):
        return self.detector.get_pose(
            self.camera, self.arm, self.joint_feedback, self.state,
            self.hand_eye, self.scene.table_model,
            self.config.bottle_height, self.config.bottle_radius,
            self.config.estimate_bottle_center_from_table,
            self.config.show_detection_window,
            target_label=target_label, max_frames=max_frames,
            center_height=center_height, description=description,
            use_surface_point=use_surface_point, conf_thres=conf_thres)

    # ------------------------------------------------------------------
    # Grasp target construction and candidate planning.
    # ------------------------------------------------------------------
    def build_grasp_targets(self, observed_bottle):
        center_x = (observed_bottle[0, 3]
                    + self.config.bottle_center_offset_xy[0])
        center_y = (observed_bottle[1, 3]
                    + self.config.bottle_center_offset_xy[1])
        table_z = self.scene.table_surface_z(center_x, center_y)
        center = np.array([
            center_x, center_y,
            table_z + self.config.bottle_height / 2.0])
        self.scene.add_bottle(
            center, self.config.bottle_height, self.config.bottle_radius,
            self.config.lift_distance, self.config.lift_table_margin,
            self.state)

        T_base_tool_grasp = observed_bottle.copy()
        T_base_tool_grasp[:3, 3] = [
            center_x, center_y, table_z + self.config.grasp_height]
        approach_axis = T_base_tool_grasp[:3, 2]
        T_base_tool_pre = T_base_tool_grasp.copy()
        T_base_tool_pre[:3, 3] -= approach_axis * self.config.pregrasp_distance
        inverse_tool = np.linalg.inv(self.config.T_link6_tool)
        return (
            center,
            T_base_tool_pre @ inverse_tool,
            T_base_tool_grasp @ inverse_tool,
        )

    def plan_grasp_candidate(self, observed_bottle):
        """Plan pregrasp + complete approach before moving, using virtual start state."""
        center, _, _ = self.build_grasp_targets(observed_bottle)
        acm = self.scene.get_acm()
        errors = []
        for yaw_deg in self.config.approach_yaws:
            tool = observed_bottle.copy()
            tool[:3, :3] = (tf_trans.rotation_matrix(
                np.deg2rad(yaw_deg), [0, 0, 1])[:3, :3]
                @ observed_bottle[:3, :3])
            tool[:3, 3] = [
                center[0], center[1],
                self.scene.table_surface_z(center[0], center[1])
                + self.config.grasp_height]
            grasp = tool @ np.linalg.inv(self.config.T_link6_tool)
            pre_tool = tool.copy()
            pre_tool[:3, 3] -= tool[:3, 2] * self.config.pregrasp_distance
            pre = pre_tool @ np.linalg.inv(self.config.T_link6_tool)
            try:
                self.scene.require_unchanged_scan()
                self.arm.arm.set_start_state_to_current_state()
                self.arm.arm.set_pose_target(pose_from_matrix(pre), "link6")
                pre_trajectory = self.arm.unpack_plan(self.arm.arm.plan())
                self.arm.validate_trajectory(pre_trajectory)
                # 在预抓取轨迹的虚拟终点检查最后一段；机器人仍停在原处。
                state = self.arm.robot.get_current_state()
                positions = dict(zip(
                    state.joint_state.name, state.joint_state.position))
                positions.update(zip(
                    pre_trajectory.joint_trajectory.joint_names,
                    pre_trajectory.joint_trajectory.points[-1].positions))
                state.joint_state.position = [
                    positions[name] for name in state.joint_state.name]
                self.scene.set_contact_pairs(
                    [(TARGET_BOTTLE_ID, "link7"),
                     (TARGET_BOTTLE_ID, "link8")], True)
                self.arm.arm.set_start_state(state)
                approach, fraction = self.arm.arm.compute_cartesian_path(
                    [pose_from_matrix(grasp).pose], eef_step=0.003,
                    avoid_collisions=True)
                if (not np.isfinite(fraction) or fraction < 0.999999
                        or not approach.joint_trajectory.points):
                    raise RuntimeError(
                        "接近直线不完整: %.6f" % fraction)
                previous = positions
                limit = self.arm.cartesian_joint_step_limit
                for point in approach.joint_trajectory.points:
                    current = dict(zip(
                        approach.joint_trajectory.joint_names, point.positions))
                    if any(abs(value - previous[name]) > limit
                           for name, value in current.items()):
                        raise RuntimeError("候选接近段关节跳变")
                    previous = current
                self.arm.validate_trajectory(approach)
                rospy.loginfo("抓取候选通过: 水平偏转%.1f度", yaw_deg)
                return pre_trajectory, grasp
            except RuntimeError as error:
                errors.append("%.1f度: %s" % (yaw_deg, error))
            finally:
                self.scene.apply_acm(acm)
                self.arm.arm.clear_pose_targets()
                self.arm.arm.set_start_state_to_current_state()
        raise RuntimeError(
            "没有有效的预抓取/直线接近候选: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # Full vision grasp workflow.
    # ------------------------------------------------------------------
    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始MoveIt视觉抓取任务")
        self.joint_feedback.wait_for_joint_feedback()
        self.scene.require_scan_scene()
        if self.scene.scene.get_attached_objects([TARGET_BOTTLE_ID]):
            raise RuntimeError("已有附着瓶子，不能当作空载重启任务")
        self.camera.start()
        self.detector.load()

        if self.config.move_to_photo_pose:
            rospy.logwarn(
                "目标瓶子尚未建模；仅在已确认该路径安全时移动到拍照位")
            self.arm.move_joints(self.config.photo_joint_target)
        else:
            rospy.loginfo("使用当前安全停稳姿态拍照；未自动移动到旧脚本拍照位")

        observed = self.get_bottle_pose(self.config.target_label)
        if observed is None:
            raise RuntimeError("没有检测到目标瓶子")
        rospy.loginfo(">>> 张开夹爪")
        if self.config.execute_motion:
            self.gripper.command(self.config.opening_open, wait_for_target=True)
        elif abs(self.gripper.current_opening()
                 - self.config.opening_open) > self.config.gripper_tolerance:
            raise RuntimeError(
                "仅规划不会张爪；先将实际夹爪置于配置张开量，再预览路径")
        rospy.loginfo(">>> 检查多个预抓取/接近方向")
        pre_trajectory, T_grasp = self.plan_grasp_candidate(observed)
        rospy.loginfo(">>> 执行预抓取轨迹")
        self.arm.finish_motion(pre_trajectory)

        self.scene.set_contact_pairs([
            (TARGET_BOTTLE_ID, "link7"),
            (TARGET_BOTTLE_ID, "link8"),
        ], True)
        rospy.loginfo(">>> 碰撞检查下直线接近瓶子")
        self.arm.move_cartesian(T_grasp)

        rospy.loginfo(">>> 闭合夹爪并检查实测开口")
        self.gripper.command(
            self.config.opening_close_command, wait_for_target=False)
        self.gripper.confirm()
        base_bottle = np.eye(4)
        base_bottle[:3, 3] = self.state.bottle_center_at_grasp
        self.state.T_link6_bottle = (
            np.linalg.inv(self.arm.current_ee_matrix()) @ base_bottle)
        self.scene.scene.attach_object(
            TARGET_BOTTLE_ID, link="link6",
            touch_links=["link7", "link8"])
        self.scene.wait_object(TARGET_BOTTLE_ID, known=False, attached=True)
        # 维持显式指尖接触权限；touch_links负责附着物与指尖关系。
        self.state.grasp_confirmed = True

        start_ee = self.arm.current_ee_matrix()
        lift_target = start_ee.copy()
        lift_target[2, 3] += self.config.lift_distance
        acm_before_lift = self.scene.get_acm()
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)

        if (self.state.bottle_center_at_grasp[2]
                - self.config.bottle_height / 2
                + self.config.lift_distance
                <= self.scene.table_model["bounds_max"][2]
                + self.config.lift_table_margin):
            raise RuntimeError(
                "lift_distance不足以让瓶底离开table碰撞盒: "
                "瓶底z=%.4f 提升后z=%.4f 盒顶z=%.4f 需要>%.4f"
                % (self.state.bottle_center_at_grasp[2]
                   - self.config.bottle_height / 2,
                   self.state.bottle_center_at_grasp[2]
                   - self.config.bottle_height / 2
                   + self.config.lift_distance,
                   self.scene.table_model["bounds_max"][2],
                   self.scene.table_model["bounds_max"][2]
                   + self.config.lift_table_margin))

        rospy.loginfo(">>> 携带瓶子沿base Z提升")
        self.arm.move_cartesian(lift_target)
        self.gripper.confirm()
        actual_lift = self.arm.current_ee_matrix()[2, 3] - start_ee[2, 3]
        bottle_bottom = (
            self.state.bottle_center_at_grasp[2]
            - self.config.bottle_height / 2.0 + actual_lift)
        table_envelope_top = float(self.scene.table_model["bounds_max"][2])

        if bottle_bottom <= table_envelope_top + self.config.lift_table_margin:
            raise RuntimeError(
                "瓶底尚未离开table碰撞盒: 瓶底z=%.4f 盒顶z=%.4f 实际抬升=%.4f"
                % (bottle_bottom, table_envelope_top, actual_lift))

        self.scene.apply_acm(acm_before_lift)
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        self.scene.validate_robot_state(
            self.scene.get_scene(request).scene.robot_state)

        if self.config.post_grasp_joint_target is not None:
            rospy.loginfo(">>> 携带瓶子规划到抓取后关节目标")
            self.arm.move_joints(self.config.post_grasp_joint_target)
        self.place_task.execute_place()
        if self.state.grasp_confirmed:
            rospy.logwarn(
                "抓取提升完成；瓶子仍作为AttachedCollisionObject附着，"
                "未自动放置或回零")

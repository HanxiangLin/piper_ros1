#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed/card placement task orchestration."""

import math

import numpy as np
import rospy
import tf.transformations as tf_trans

from pose_utils import pose_from_matrix
from scene_manager import TARGET_BOTTLE_ID


class PlaceTask:
    """Build place targets and execute the fixed or vision-guided place."""

    def __init__(self, config, state, arm, gripper, scene, joint_feedback,
                 get_pose):
        self.config = config
        self.state = state
        self.arm = arm
        self.gripper = gripper
        self.scene = scene
        self.joint_feedback = joint_feedback
        self.get_pose = get_pose

    def build_place_targets(self, observed_card):
        """Compute pre-place/place link6 poses in front of the detected card.

        The card stands on the table, so ``get_bottle_pose`` is called with
        ``use_surface_point=True`` and its measured depth point is used
        directly.  Placement is deliberately simple: keep the bottle upright,
        move its center ``place_offset_from_card`` horizontally in front of
        that point (towards the robot by default) at the card's detected
        height, then open the gripper and let the bottle drop onto the table.
        No precise card center or table-height fitting is needed.
        """
        if self.state.T_link6_bottle is None:
            raise RuntimeError("尚未记录link6->瓶子变换，拒绝放置")
        card_x = (float(observed_card[0, 3])
                  + self.config.place_card_offset_xy[0])
        card_y = (float(observed_card[1, 3])
                  + self.config.place_card_offset_xy[1])
        card_z = float(observed_card[2, 3])
        if not np.isfinite([card_x, card_y, card_z]).all():
            raise RuntimeError("卡片位置包含非有限值")

        if self.config.place_front_mode == "toward_robot":
            dx, dy = -card_x, -card_y
        else:
            yaw = math.radians(self.config.place_front_yaw)
            dx, dy = math.cos(yaw), math.sin(yaw)
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            raise RuntimeError(
                "无法确定卡片正前方方向，请检查place_front参数")

        bottle_x = card_x + self.config.place_offset_from_card * dx / norm
        bottle_y = card_y + self.config.place_offset_from_card * dy / norm
        table_z = self.scene.table_surface_z(bottle_x, bottle_y)
        # 松手高度取卡片检测点高度，但瓶底不允许低于桌面+间隙，
        # 避免松手前把瓶子压进桌面。
        release_center_z = max(
            card_z, table_z + self.config.bottle_height / 2.0
            + self.config.place_release_clearance)
        bottle_center = np.array([bottle_x, bottle_y, release_center_z])
        self.scene.require_support_footprint(
            bottle_center, self.config.bottle_radius)

        T_bottle_place = np.eye(4)
        T_bottle_place[:3, 3] = bottle_center
        place = T_bottle_place @ np.linalg.inv(self.state.T_link6_bottle)

        pre = place.copy()
        pre[2, 3] += self.config.place_approach_height
        pre_bottom = (bottle_center[2] - self.config.bottle_height / 2.0
                      + self.config.place_approach_height)
        required_bottom = (
            self.scene.table_model["bounds_max"][2]
            + self.config.lift_clearance)
        if pre_bottom <= required_bottom:
            raise RuntimeError(
                "预放置高度不足以离开膨胀table: 瓶底z=%.4f，需大于%.4f；"
                "请增大place_approach_height至少%.4f"
                % (pre_bottom, required_bottom,
                   required_bottom - pre_bottom + 1e-4))
        rospy.loginfo(
            "放置目标: card=(%.4f, %.4f, %.4f) "
            "bottle_center=(%.4f, %.4f, %.4f) 松手后下落=%.4fm",
            card_x, card_y, card_z,
            bottle_center[0], bottle_center[1], bottle_center[2],
            bottle_center[2] - self.config.bottle_height / 2.0 - table_z)
        return pre, place, bottle_center

    def release_bottle_and_retract(self, pre_place):
        """Open the gripper, detach the bottle, update the world, retract."""
        bottle_actual = (
            self.arm.current_ee_matrix() @ self.state.T_link6_bottle)
        rospy.loginfo(">>> 张开夹爪")
        self.gripper.command(self.config.opening_open, wait_for_target=True)
        self.scene.scene.remove_attached_object(
            "link6", name=TARGET_BOTTLE_ID)
        # detach后对象会回到world（是否known取决于MoveIt版本），
        # 这里只确认它不再是附着对象，随后add_cylinder会重设位姿。
        self.scene.wait_object(TARGET_BOTTLE_ID, known=None, attached=False)
        self.state.grasp_confirmed = False

        # 松开后瓶子自由下落到桌面，世界模型按桌面高度更新，
        # 避免在场景里留下一个悬空的圆柱。
        bottle_actual[2, 3] = (
            self.scene.table_surface_z(
                bottle_actual[0, 3], bottle_actual[1, 3])
            + self.config.bottle_height / 2.0)
        self.scene.scene.add_cylinder(
            TARGET_BOTTLE_ID, pose_from_matrix(bottle_actual),
            height=self.config.bottle_height,
            radius=self.config.bottle_radius)
        self.scene.wait_object(TARGET_BOTTLE_ID, known=True, attached=False)
        self.scene.set_contact_pairs(
            [(TARGET_BOTTLE_ID, link) for link in
             ("link6", "link7", "link8")], True)
        # 松爪会改变负载并让关节回弹；必须等机械臂真正静止再算撤离轨迹。
        rospy.loginfo(">>> 等待机械臂松手后静止")
        self.joint_feedback.wait_for_arm_settled()
        rospy.loginfo(">>> 抬升撤离")
        self.arm.move_cartesian(pre_place)
        self.scene.apply_acm(self.scene.acm_before_task)
        self.scene.set_contact_pairs(
            [(TARGET_BOTTLE_ID, link) for link in
             self.scene.robot.get_link_names() + ["table", "<octomap>"]],
            False)
        rospy.loginfo(">>> 放置完成，世界瓶子仍保留在场景中")

    def execute_fixed_place(self):
        x, y, z, roll, pitch, yaw = self.config.place_tool_pose
        tool_place = tf_trans.euler_matrix(roll, pitch, yaw, axes="sxyz")
        tool_place[:3, 3] = [x, y, z]
        place = tool_place @ np.linalg.inv(self.config.T_link6_tool)
        bottle = place @ self.state.T_link6_bottle
        if not np.allclose(bottle[:3, 2], [0, 0, 1], atol=0.01):
            raise RuntimeError("当前放置实现只支持瓶身保持直立")
        self.scene.require_support_footprint(
            bottle[:3, 3], self.config.bottle_radius)
        expected_bottom = self.scene.table_surface_z(
            bottle[0, 3], bottle[1, 3])
        actual_bottom = bottle[2, 3] - self.config.bottle_height / 2
        if abs(actual_bottom - expected_bottom) > 0.005:
            raise RuntimeError(
                "放置目标瓶底不在当前table表面±5mm内，修正place_tool_pose")
        pre = place.copy()
        pre[2, 3] += self.config.place_approach_height
        if (bottle[2, 3] - self.config.bottle_height / 2
                + self.config.place_approach_height
                <= self.scene.table_model["bounds_max"][2]
                + self.config.lift_clearance):
            raise RuntimeError("预放置高度不足以离开膨胀table")
        rospy.loginfo(">>> MoveIt规划到预放置位")
        self.arm.move_pose(pre)
        acm_before_place = self.scene.get_acm()
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
        try:
            self.arm.move_cartesian(place)
        finally:
            self.scene.apply_acm(acm_before_place)
        self.release_bottle_and_retract(pre)

    def execute_place(self):
        if not self.config.enable_place:
            return
        if not self.state.grasp_confirmed:
            raise RuntimeError("尚未确认抓住瓶子，拒绝放置")
        if not self.scene.scene.get_attached_objects([TARGET_BOTTLE_ID]):
            raise RuntimeError("target_bottle未处于附着状态，拒绝放置")

        if self.config.place_mode == "fixed":
            self.execute_fixed_place()
            return

        if self.config.place_photo_joint_target is not None:
            rospy.loginfo(">>> 携带瓶子移动到放置拍照位")
            self.arm.move_joints(self.config.place_photo_joint_target)
        rospy.loginfo(">>> 检测目标卡片")
        observed_card = self.get_pose(
            self.config.place_card_label,
            use_surface_point=True,
            conf_thres=self.config.place_card_conf_thres,
            description="目标卡片")
        if observed_card is None:
            raise RuntimeError("没有检测到目标卡片")
        pre, place, _ = self.build_place_targets(observed_card)
        rospy.loginfo(">>> MoveIt规划到预放置位")
        self.arm.move_pose(pre)
        acm_before_place = self.scene.get_acm()
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
        try:
            rospy.loginfo(">>> 碰撞检查下直线下降到放置位")
            self.arm.move_cartesian(place)
        finally:
            self.scene.apply_acm(acm_before_place)
        self.release_bottle_and_retract(pre)

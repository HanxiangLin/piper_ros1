#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Place the grasped bottle in front of the card found by the scan.

The card is not re-detected with YOLO: the frozen scan already knows its front
plane (centre, normal, size), so placement works even when the carried bottle
blocks the camera.  The bottle is released ``place_offset_from_card`` in front
of the card face, at the table plane plus ``place_release_clearance`` -- not at
the card's detected centre height, which for a 20 cm card would drop the bottle
from several centimetres.
"""

import math

import numpy as np
import rospy
import tf.transformations as tf_trans

from grasp_corridor import distance_to_box
from joint_feedback import ARM_JOINTS
from obstacle_model import card_pose_and_dims
from pose_utils import pose_from_matrix
from scene_manager import TARGET_BOTTLE_ID


class PlaceTask:
    """Compute the release pose in front of the scanned card and execute it."""

    # 与抓取候选相同的代价结构。腕部项单独加权，专门压"肩不动但腕硬拧"。
    JOINT_WEIGHTS = np.array([1.0, 1.2, 1.2, 1.0, 1.0, 1.0])
    PATH_WEIGHT = 1.0
    APPROACH_WEIGHT = 2.0
    WRIST_WEIGHT = 1.5
    LIMIT_WEIGHT = 20.0
    LIMIT_MARGIN = 0.10
    INCOMPLETE_PENALTY = 100.0

    def __init__(self, config, state, arm, gripper, scene, joint_feedback):
        self.config = config
        self.state = state
        self.arm = arm
        self.gripper = gripper
        self.scene = scene
        self.joint_feedback = joint_feedback

    def _front_direction(self, center, normal):
        """Horizontal direction from the card face towards the free side."""
        if self.config.place_front_mode == "fixed_yaw":
            yaw = math.radians(self.config.place_front_yaw)
            return np.array([math.cos(yaw), math.sin(yaw)])
        if self.config.place_front_mode == "toward_robot":
            offset = -np.asarray(center[:2], dtype=float)
            norm = float(np.linalg.norm(offset))
            if norm < 1e-6:
                raise RuntimeError("无法确定卡片正前方方向")
            return offset / norm
        if self.config.place_front_mode != "card_normal":
            raise ValueError("place_front_mode必须是card_normal/toward_robot/fixed_yaw")
        horizontal = np.asarray(normal[:2], dtype=float)
        norm = float(np.linalg.norm(horizontal))
        if norm < 1e-3:
            raise RuntimeError("卡片法向几乎竖直，无法确定放置方向")
        direction = horizontal / norm
        if float(np.dot(direction, -np.asarray(center[:2], dtype=float))) < 0.0:
            direction = -direction
        return direction

    def build_place_candidates(self, card):
        """Return all yaw candidates for card placement.

        The bottle is a cylinder, so rotating it about its vertical axis does
        not change the placement; that yaw is the free variable used for
        candidate ranking.  ``0°`` is referenced to the base->place radial, the
        same convention as the grasp candidates.
        """
        if self.state.T_link6_bottle is None:
            raise RuntimeError("尚未记录link6->瓶子变换，拒绝放置")
        center = np.asarray(card["center"], dtype=float)
        normal = np.asarray(card["normal"], dtype=float)
        if not np.isfinite(center).all() or not np.isfinite(normal).all():
            raise RuntimeError("卡片模型包含非有限值")

        direction = self._front_direction(center, normal)
        bottle_xy = (center[:2]
                     + self.config.place_offset_from_card * direction)
        table_z = self.scene.table_surface_z(bottle_xy[0], bottle_xy[1])
        bottle_center = np.array([
            bottle_xy[0], bottle_xy[1],
            table_z + self.config.bottle_height / 2.0
            + self.config.place_release_clearance])

        required = (self.config.bottle_radius + card["thickness"] / 2.0
                    + self.config.scan_obstacle_padding
                    + self.config.place_card_clearance)
        if self.config.place_offset_from_card < required:
            raise RuntimeError(
                "place_offset_from_card=%.3f 不足以避开卡片，需至少%.3fm"
                % (self.config.place_offset_from_card, required))
        card_pose, card_dims = card_pose_and_dims(card)
        clearance = distance_to_box(bottle_center, card_pose, card_dims)
        if clearance < self.config.bottle_radius + self.config.place_card_clearance:
            raise RuntimeError(
                "松手点距卡片实测只有%.3fm，瓶半径+余量需%.3fm；"
                "增大place_offset_from_card"
                % (clearance,
                   self.config.bottle_radius + self.config.place_card_clearance))
        self.scene.require_support_footprint(
            bottle_center, self.config.bottle_radius)

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

        # T_link6_bottle 的旋转是抓取姿态的逆；放置时先把瓶子的 yaw 旋转
        # 到目标方位，再乘这个固定变换。这样瓶子仍然竖直，只有夹爪/手腕
        # 绕瓶轴转动。
        R_grasp = np.linalg.inv(self.state.T_link6_bottle[:3, :3])
        grasp_azimuth = math.atan2(R_grasp[1, 2], R_grasp[0, 2])
        place_azimuth = math.atan2(bottle_center[1], bottle_center[0])
        inverse_bottle = np.linalg.inv(self.state.T_link6_bottle)
        candidates = []
        for offset in self.config.place_yaws:
            theta = place_azimuth + math.radians(offset) - grasp_azimuth
            T_bottle_place = np.eye(4)
            T_bottle_place[:3, :3] = tf_trans.rotation_matrix(
                theta, [0, 0, 1])[:3, :3]
            T_bottle_place[:3, 3] = bottle_center
            place = T_bottle_place @ inverse_bottle
            bottle_place = place @ self.state.T_link6_bottle
            if not np.allclose(bottle_place[:3, 2], [0.0, 0.0, 1.0],
                               atol=1e-6):
                raise RuntimeError("放置候选无法保持瓶身直立")
            pre = place.copy()
            pre[2, 3] += self.config.place_approach_height
            candidates.append({
                "label": offset,
                "pre": pre,
                "place": place,
                "bottle_center": bottle_center.copy(),
            })

        rospy.loginfo(
            "放置目标: card=(%.4f, %.4f, %.4f) "
            "bottle_center=(%.4f, %.4f, %.4f) 松手高度=桌面+%.4fm "
            "径向=%.1f° 候选yaw=%s",
            center[0], center[1], center[2],
            bottle_center[0], bottle_center[1], bottle_center[2],
            self.config.place_release_clearance,
            math.degrees(place_azimuth),
            ", ".join("%+.1f" % value for value in self.config.place_yaws))
        return candidates

    def build_place_targets(self, card):
        """Return the 0° radial candidate (legacy single-pose API)."""
        candidates = self.build_place_candidates(card)
        zero = min(candidates, key=lambda item: abs(item["label"]))
        return zero["pre"], zero["place"], zero["bottle_center"]

    def _weighted_joint_distance(self, first, second):
        return float(np.sum(self.JOINT_WEIGHTS * np.abs(first - second)))

    def _trajectory_motion(self, trajectory):
        """Return ``(weighted path length, final joint vector)``."""
        names = list(trajectory.joint_trajectory.joint_names)
        index = {name: number for number, name in enumerate(names)}
        if any(name not in index for name in ARM_JOINTS):
            raise RuntimeError("候选轨迹缺少六轴关节")
        previous = None
        path = 0.0
        for point in trajectory.joint_trajectory.points:
            current = np.array([
                point.positions[index[name]] for name in ARM_JOINTS])
            if previous is not None:
                path += self._weighted_joint_distance(previous, current)
            previous = current
        if previous is None:
            raise RuntimeError("候选轨迹为空")
        return path, previous

    def _candidate_cost(self, pre_trajectory, descent, complete):
        pre_path, q_pre = self._trajectory_motion(pre_trajectory)
        if descent.joint_trajectory.points:
            descent_path, q_end = self._trajectory_motion(descent)
        else:
            descent_path, q_end = 0.0, q_pre
        wrist = abs(float(q_end[3])) + abs(float(q_end[5]))
        margins = self.arm.joint_limit_margins(q_end)
        limit = sum(max(0.0, self.LIMIT_MARGIN - float(margin)) ** 2
                    for margin in margins)
        cost = (self.PATH_WEIGHT * pre_path
                + self.APPROACH_WEIGHT * descent_path
                + self.WRIST_WEIGHT * wrist
                + self.LIMIT_WEIGHT * limit)
        if not complete:
            cost += self.INCOMPLETE_PENALTY
        return float(cost), {
            "path": pre_path,
            "approach": descent_path,
            "wrist": wrist,
            "limit": limit,
            "q_end": q_end,
        }

    def _pre_state(self, pre_trajectory):
        state = self.arm.robot.get_current_state()
        positions = dict(zip(
            state.joint_state.name, state.joint_state.position))
        positions.update(zip(
            pre_trajectory.joint_trajectory.joint_names,
            pre_trajectory.joint_trajectory.points[-1].positions))
        state.joint_state.position = [
            positions[name] for name in state.joint_state.name]
        return state

    def plan_place_candidates(self, candidates):
        """Plan every place yaw and return the lowest-cost valid candidate.

        Hard gates are the same as the grasp: pre-place planning + collision
        validation, a complete straight vertical descent (preferred), joint
        step limit.  The cost favours a short joint-space path, little joint
        change during the descent, a centred wrist and limit margin.
        """
        acm = self.scene.get_acm()
        results = []
        failures = []
        for candidate in candidates:
            pre = candidate["pre"]
            place = candidate["place"]
            try:
                self.arm.arm.set_start_state_to_current_state()
                pre_trajectory = self.arm.plan_pose_nearest(pre)
                state = self._pre_state(pre_trajectory)
                self.scene.set_contact_pairs(
                    [(TARGET_BOTTLE_ID, "table")], True)
                self.arm.arm.set_start_state(state)
                descent, fraction = self.arm.arm.compute_cartesian_path(
                    [pose_from_matrix(place).pose], eef_step=0.003,
                    avoid_collisions=True)
                complete = (
                    np.isfinite(fraction) and fraction >= 0.999999
                    and bool(descent.joint_trajectory.points))
                if descent.joint_trajectory.points:
                    previous = dict(zip(
                        state.joint_state.name,
                        state.joint_state.position))
                    limit = self.arm.cartesian_joint_step_limit
                    for point in descent.joint_trajectory.points:
                        current = dict(zip(
                            descent.joint_trajectory.joint_names,
                            point.positions))
                        if any(abs(value - previous[name]) > limit
                               for name, value in current.items()):
                            raise RuntimeError("候选下降段关节跳变")
                        previous = current
                    self.arm.validate_trajectory(descent)
                cost, details = self._candidate_cost(
                    pre_trajectory, descent, complete)
                results.append((
                    cost, candidate, pre_trajectory, complete, details))
                rospy.loginfo(
                    "放置候选通过: 偏转%+.1f度 %s 代价%.3f "
                    "(路径%.2f 下降%.2f 腕部%.2f 限位%.4f)",
                    candidate["label"],
                    "直线下降完整" if complete else "下降不完整",
                    cost, details["path"], details["approach"],
                    details["wrist"], details["limit"])
            except RuntimeError as error:
                failures.append({
                    "label": candidate["label"],
                    "pre": pre,
                    "error": error,
                })
            finally:
                self.scene.apply_acm(acm)
                self.arm.arm.clear_pose_targets()
                self.arm.arm.set_start_state_to_current_state()
        if not results:
            errors = []
            for failure in failures:
                reason = self.arm.diagnose_pose(
                    failure["pre"], "预放置位%+.1f度" % failure["label"])
                errors.append("%+.1f度: %s%s" % (
                    failure["label"], failure["error"],
                    "" if reason is None else "；%s" % reason))
            raise RuntimeError(
                "没有有效的放置候选: " + "; ".join(errors))
        complete_results = [item for item in results if item[3]]
        pool = complete_results or results
        pool.sort(key=lambda item: item[0])
        cost, candidate, pre_trajectory, complete, details = pool[0]
        if not complete:
            rospy.logwarn(
                "所有放置候选的垂直下降都不完整，执行时将使用OMPL兜底")
        rospy.loginfo(
            "选择放置候选: 水平偏转%+.1f度 代价%.3f "
            "(路径%.2f 下降%.2f 腕部%.2f 限位%.4f)；"
            "有效%d/%d 完整下降%d",
            candidate["label"], cost, details["path"],
            details["approach"], details["wrist"], details["limit"],
            len(results), len(candidates), len(complete_results))
        if len(pool) > 1:
            rospy.loginfo(
                "其余有效放置候选代价: %s",
                ", ".join("%+.1f度=%.3f" % (item[1]["label"], item[0])
                          for item in pool[1:]))
        return candidate, pre_trajectory

    def release_bottle_and_retract(self, pre_place):
        """Open the gripper, detach the bottle, update the world, retract."""
        bottle_actual = (
            self.arm.current_ee_matrix() @ self.state.T_link6_bottle)
        # 松手点离真实桌面只有几毫米，但table碰撞盒取的是整块桌面最高点
        # 加padding，瓶子必然落在盒内。从张爪到撤离都必须允许这一对，
        # 否则 gripper.command() 的指尖扫描校验会把模型重叠当成真碰撞。
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
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
        self.scene.refresh_signature()
        self.scene.set_contact_pairs(
            [(TARGET_BOTTLE_ID, link) for link in
             ("link6", "link7", "link8")], True)
        # 松爪会改变负载并让关节回弹；必须等机械臂真正静止再算撤离轨迹。
        rospy.loginfo(">>> 等待机械臂松手后静止")
        self.joint_feedback.wait_for_arm_settled()
        rospy.loginfo(">>> 抬升撤离")
        self.arm.move_cartesian(pre_place)
        self.scene.apply_acm(self.scene.acm_before_task)
        # 机器人本体必须避开已放置的瓶子；但瓶底就在桌面上，而table是
        # 膨胀包络，所以瓶子-桌面这一对要永久允许，否则放置后的回零/
        # 再规划都会报 ('table','target_bottle') 碰撞。
        self.scene.set_contact_pairs(
            [(TARGET_BOTTLE_ID, link) for link in
             self.scene.robot.get_link_names() + ["<octomap>"]],
            False)
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
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
        try:
            self.arm.move_pose(pre)
        except RuntimeError as error:
            reason = self.arm.diagnose_pose(pre, "预放置位")
            if reason is None:
                raise
            raise RuntimeError("%s；%s" % (error, reason))
        acm_before_place = self.scene.get_acm()
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
        try:
            rospy.loginfo(">>> 碰撞检查下直线下降到放置位")
            self.arm.move_cartesian(place, allow_plan=True)
        except RuntimeError:
            self.scene.apply_acm(acm_before_place)
            raise
        # 下降成功后继续保持瓶子-桌面接触允许：瓶子松手后仍落在膨胀后的
        # table碰撞盒内，张爪/撤离/回零都要依赖这一对。最终ACM由
        # release_bottle_and_retract() 设置。
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

        card = getattr(self.state, "card_model", None)
        if card is None:
            raise RuntimeError("扫描没有识别出卡片，拒绝按卡片放置")
        candidates = self.build_place_candidates(card)
        rospy.loginfo(">>> 检查多个放置方向")
        candidate, pre_trajectory = self.plan_place_candidates(candidates)
        rospy.loginfo(">>> 执行预放置轨迹")
        self.arm.finish_motion(pre_trajectory)
        acm_before_place = self.scene.get_acm()
        self.scene.set_contact_pairs([(TARGET_BOTTLE_ID, "table")], True)
        try:
            rospy.loginfo(">>> 碰撞检查下直线下降到放置位")
            self.arm.move_cartesian(candidate["place"], allow_plan=True)
        except RuntimeError as error:
            # 诊断时仍允许瓶子-桌面接触，避免把table膨胀包络误判成真碰撞；
            # 诊断完再恢复原ACM。
            try:
                reason = self.arm.diagnose_pose(
                    candidate["place"], "放置位")
            finally:
                self.scene.apply_acm(acm_before_place)
            if reason is None:
                raise
            raise RuntimeError("%s；%s" % (error, reason))
        # 成功：保持瓶子-桌面允许，交给 release_bottle_and_retract。
        self.release_bottle_and_retract(candidate["pre"])

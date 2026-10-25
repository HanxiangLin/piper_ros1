#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision-guided grasp orchestration.

The scan has already produced the frozen scene, the target pose and the list of
obstacles, so this module only plans and executes:

1. enumerate approach yaw candidates and reject the ones whose approach corridor
   is already blocked by a non-target obstacle;
2. plan the pre-grasp motion, then verify the straight approach from a virtual
   trajectory end state (the robot has not moved yet);
3. execute, close the gripper, confirm, attach, lift and hand over to the place
   task.
"""

import math

import numpy as np
import rospy
import tf.transformations as tf_trans
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningSceneRequest

from grasp_corridor import filter_by_corridor
from joint_feedback import ARM_JOINTS
from pose_utils import level_grasp_rotation_for_azimuth, pose_from_matrix
from scene_manager import TARGET_BOTTLE_ID


class GraspTask:
    """Build grasp targets, plan candidates and execute the vision grasp."""

    # 候选代价：路径/接近段/腕部/限位余量的相对权重。腕部项故意单独拿出来，
    # 避免选中"肩不动但 joint4/joint6 硬拧"的构型。
    JOINT_WEIGHTS = np.array([1.0, 1.2, 1.2, 1.0, 1.0, 1.0])
    PATH_WEIGHT = 1.0
    APPROACH_WEIGHT = 2.0
    WRIST_WEIGHT = 1.5
    LIMIT_WEIGHT = 20.0
    LIMIT_MARGIN = 0.10

    def __init__(self, config, state, arm, gripper, scene, joint_feedback,
                 place_task):
        self.config = config
        self.state = state
        self.arm = arm
        self.gripper = gripper
        self.scene = scene
        self.joint_feedback = joint_feedback
        self.place_task = place_task

    # ------------------------------------------------------------------
    # Grasp target construction and candidate planning.
    # ------------------------------------------------------------------
    def _grasp_point(self, observed_bottle):
        """Return ``(grasp_pos, bottle_center)`` in base_link.

        Only the observed translation is used: the approach azimuth is derived
        from the base->bottle radial direction, not from the last scan
        viewpoint's end-effector orientation.
        """
        center_x = (observed_bottle[0, 3]
                    + self.config.bottle_center_offset_xy[0])
        center_y = (observed_bottle[1, 3]
                    + self.config.bottle_center_offset_xy[1])
        table_z = self.scene.table_surface_z(center_x, center_y)
        grasp_pos = np.array([
            center_x, center_y, table_z + self.config.grasp_height])
        center = np.array([
            center_x, center_y,
            table_z + self.config.bottle_height / 2.0])
        return grasp_pos, center

    def build_grasp_targets(self, observed_bottle):
        """Return ``(bottle_center, pre, grasp)`` at the radial yaw (0°)."""
        grasp_pos, center = self._grasp_point(observed_bottle)
        azimuth = math.atan2(grasp_pos[1], grasp_pos[0])
        tool = np.eye(4)
        tool[:3, :3] = level_grasp_rotation_for_azimuth(azimuth)
        tool[:3, 3] = grasp_pos
        approach_axis = tool[:3, 2]
        pre_tool = tool.copy()
        pre_tool[:3, 3] -= approach_axis * self.config.pregrasp_distance
        inverse_tool = np.linalg.inv(self.config.T_link6_tool)
        return (
            center,
            pre_tool @ inverse_tool,
            tool @ inverse_tool,
        )

    def _candidates(self, observed_bottle):
        """Return radial-referenced yaw candidates with a free corridor.

        ``approach_yaw_offsets_deg`` is now relative to the base->bottle
        azimuth: 0 degrees means a radial approach; positive values turn the
        approach direction counter-clockwise around base Z.
        """
        grasp_pos, _ = self._grasp_point(observed_bottle)
        azimuth = math.atan2(grasp_pos[1], grasp_pos[0])
        nominal_rotation = level_grasp_rotation_for_azimuth(azimuth)
        inverse_tool = np.linalg.inv(self.config.T_link6_tool)
        candidates = []
        for yaw_deg in self.config.approach_yaws:
            rotation = (tf_trans.rotation_matrix(
                np.deg2rad(yaw_deg), [0, 0, 1])[:3, :3]
                @ nominal_rotation)
            tool = np.eye(4)
            tool[:3, :3] = rotation
            tool[:3, 3] = grasp_pos
            pre_tool = tool.copy()
            pre_tool[:3, 3] -= rotation[:, 2] * self.config.pregrasp_distance
            grasp = tool @ inverse_tool
            pre = pre_tool @ inverse_tool
            candidates.append({
                "label": yaw_deg,
                "start": pre[:3, 3].copy(),
                "end": grasp[:3, 3].copy(),
                "pre": pre,
                "grasp": grasp,
            })

        scan_result = getattr(self.state, "scan_result", None)
        if not scan_result:
            return candidates
        accepted, rejected = filter_by_corridor(
            candidates, scan_result["obstacles"],
            self.config.scan_corridor_radius,
            self.config.scan_corridor_samples)
        for item in rejected:
            rospy.logwarn(
                "接近方向%.1f度被%s挡住: 净距%.3fm < %.3fm",
                item["label"], item["blocking"], item["clearance"],
                self.config.scan_corridor_radius)
        if not accepted:
            raise RuntimeError(
                "没有可行的接近走廊: " + "; ".join(
                    "%.1f度被%s挡住(%.3fm)"
                    % (item["label"], item["blocking"], item["clearance"])
                    for item in rejected))
        return accepted

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

    def _candidate_cost(self, pre_trajectory, approach):
        """Score one valid candidate; lower is better.

        The hard gates (collision, corridor, straight-line completion) are
        already applied by the caller.  This cost only decides *which* valid
        candidate to execute.
        """
        pre_path, q_pre = self._trajectory_motion(pre_trajectory)
        approach_path, _ = self._trajectory_motion(approach)
        wrist = abs(float(q_pre[3])) + abs(float(q_pre[5]))
        margins = self.arm.joint_limit_margins(q_pre)
        limit = sum(max(0.0, self.LIMIT_MARGIN - float(margin)) ** 2
                    for margin in margins)
        cost = (self.PATH_WEIGHT * pre_path
                + self.APPROACH_WEIGHT * approach_path
                + self.WRIST_WEIGHT * wrist
                + self.LIMIT_WEIGHT * limit)
        return float(cost), {
            "path": pre_path,
            "approach": approach_path,
            "wrist": wrist,
            "limit": limit,
            "q_pre": q_pre,
        }

    def plan_grasp_candidate(self, observed_bottle, candidates=None):
        """Plan every candidate and keep the lowest-cost valid one.

        Planning all candidates makes the choice explicit instead of returning
        whichever yaw happened to be first in the list.  The cost prefers a
        short joint-space path, little joint change during the straight
        approach, a centred wrist and healthy joint-limit margins.  Collision
        and straight-line completeness remain hard gates.
        """
        if candidates is None:
            candidates = self._candidates(observed_bottle)
        acm = self.scene.get_acm()
        failures = []
        results = []
        for candidate in candidates:
            pre = candidate["pre"]
            grasp = candidate["grasp"]
            stage = "pregrasp"
            approach = None
            fraction = float("nan")
            try:
                self.arm.arm.set_start_state_to_current_state()
                # 用当前状态作为IK初值、按关节目标规划，避免OMPL随机
                # 采到远端肘/腕分支；内部已校验，失败会回退pose目标。
                pre_trajectory = self.arm.plan_pose_nearest(pre)
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
                stage = "approach"
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
                cost, details = self._candidate_cost(
                    pre_trajectory, approach)
                results.append((cost, candidate, pre_trajectory, details))
                rospy.loginfo(
                    "候选通过: 偏转%+.1f度 代价%.3f "
                    "(路径%.2f 接近%.2f 腕部%.2f 限位%.4f) 净距%.3fm",
                    candidate["label"], cost, details["path"],
                    details["approach"], details["wrist"], details["limit"],
                    candidate.get("clearance", float("inf")))
            except RuntimeError as error:
                # 直线段诊断必须趁虚拟起点和放宽的ACM还在时立即做；
                # 预抓取IK诊断很慢（14组初值），推迟到“所有候选都失败”
                # 再跑，避免某个候选失败就卡住整个候选评估。
                reason = None
                if stage != "pregrasp":
                    reason = self.arm.diagnose_cartesian(grasp, fraction)
                failures.append({
                    "label": candidate["label"],
                    "stage": stage,
                    "pre": pre,
                    "error": error,
                    "reason": reason,
                })
            finally:
                self.scene.apply_acm(acm)
                self.arm.arm.clear_pose_targets()
                self.arm.arm.set_start_state_to_current_state()
        if not results:
            errors = []
            for failure in failures:
                reason = failure["reason"]
                if failure["stage"] == "pregrasp":
                    reason = self.arm.diagnose_pose(
                        failure["pre"],
                        "预抓取位%.1f度" % failure["label"])
                errors.append("%+.1f度: %s%s" % (
                    failure["label"], failure["error"],
                    "" if reason is None else "；%s" % reason))
            raise RuntimeError(
                "没有有效的预抓取/直线接近候选: " + "; ".join(errors))
        if failures:
            rospy.logwarn(
                "未通过的候选: %s",
                "; ".join("%+.1f度: %s" % (item["label"], item["error"])
                          for item in failures))

        results.sort(key=lambda item: item[0])
        cost, candidate, pre_trajectory, details = results[0]
        rospy.loginfo(
            "选择候选: 水平偏转%+.1f度 代价%.3f "
            "(路径%.2f 接近%.2f 腕部%.2f 限位%.4f) 净距%.3fm；"
            "有效候选%d/%d",
            candidate["label"], cost, details["path"],
            details["approach"], details["wrist"], details["limit"],
            candidate.get("clearance", float("inf")),
            len(results), len(candidates))
        if len(results) > 1:
            rospy.loginfo(
                "其余有效候选代价: %s",
                ", ".join("%+.1f度=%.3f" % (item[1]["label"], item[0])
                          for item in results[1:]))
        return pre_trajectory, candidate["grasp"]

    # ------------------------------------------------------------------
    # Full grasp workflow.
    # ------------------------------------------------------------------
    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始MoveIt视觉抓取任务")
        self.joint_feedback.wait_for_joint_feedback()
        observed = getattr(self.state, "T_base_bottle", None)
        if observed is None:
            raise RuntimeError("扫描没有提供目标瓶位姿，拒绝抓取")
        if self.scene.scene.get_attached_objects([TARGET_BOTTLE_ID]):
            raise RuntimeError("已有附着瓶子，不能当作空载重启任务")
        if not self.state.created_bottle:
            raise RuntimeError("目标瓶尚未加入规划场景，拒绝抓取")

        rospy.loginfo(">>> 检查多个预抓取/接近方向")
        candidates = self._candidates(observed)
        pre_trajectory, T_grasp = self.plan_grasp_candidate(observed, candidates)
        if getattr(self.config, "dry_run", False):
            rospy.logwarn(
                ">>> dry_run: 预抓取与直线接近校验通过，"
                "跳过抓取执行（预抓取轨迹未执行）")
            return
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
        # 附着改变了世界列表，这是本程序主动做的变更，重新冻结签名。
        self.scene.refresh_signature()
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

        if self.config.enable_place:
            # 先回零位再放置：抓取后的手腕是"低姿横伸"构型，直接横移到放置位
            # 会让整条手臂挤在桌面高度，直线下降很容易被卡片/桌面截断。
            rospy.loginfo(">>> 抓取完成，先回零位再放置")
            self.arm.move_joints(self.config.home_joint_target)
        self.place_task.execute_place()
        if self.config.enable_place:
            rospy.loginfo(">>> 放置完成，回到零位")
            self.arm.move_joints(self.config.home_joint_target)
        elif self.state.grasp_confirmed:
            rospy.logwarn(
                "抓取提升完成；瓶子仍作为AttachedCollisionObject附着在link6，"
                "未放置也未回零")

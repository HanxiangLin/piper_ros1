#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gripper service wrapper with opening/effort closed-loop confirmation."""

import copy
import math
import time

import numpy as np
import rospy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningSceneRequest
from std_srvs.srv import Trigger

from joint_feedback import FINGER_JOINTS


class GripperController:
    """Commands the gripper and confirms grasp by opening + load feedback."""

    def __init__(self, joint_feedback, scene_manager, gripper_service,
                 state, gripper_effort, gripper_tolerance,
                 gripper_feedback_timeout, gripper_settle_time,
                 grasp_opening_min, grasp_opening_max, grasp_effort_min,
                 grasp_check_service):
        self.joint_feedback = joint_feedback
        self.scene_manager = scene_manager
        self.gripper_service = gripper_service
        self.state = state
        self.gripper_effort = gripper_effort
        self.gripper_tolerance = gripper_tolerance
        self.gripper_feedback_timeout = gripper_feedback_timeout
        self.gripper_settle_time = gripper_settle_time
        self.grasp_opening_min = grasp_opening_min
        self.grasp_opening_max = grasp_opening_max
        self.grasp_effort_min = grasp_effort_min
        self.grasp_check_service = grasp_check_service

    def current_opening(self):
        positions = self.joint_feedback._fresh_joint_positions()
        left, right = positions["joint7"], positions["joint8"]
        if not (0 <= left <= 0.035 and -0.035 <= right <= 0
                and abs(left + right) <= 0.002):
            raise RuntimeError("joint7/joint8反馈不符合当前对称夹爪模型")
        return left - right

    def validate_sweep(self, target_opening):
        start = self.current_opening()
        sample_count = max(int(abs(target_opening - start) / 0.002) + 2, 2)
        request = GetPlanningSceneRequest()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        state = copy.deepcopy(
            self.scene_manager.get_scene(request).scene.robot_state)
        names = list(state.joint_state.name)
        index = {name: position for position, name in enumerate(names)}
        if any(name not in index for name in FINGER_JOINTS):
            raise RuntimeError("Planning Scene状态缺少joint7/joint8")
        for opening in np.linspace(start, target_opening, sample_count):
            positions = list(state.joint_state.position)
            positions[index["joint7"]] = opening / 2.0
            positions[index["joint8"]] = -opening / 2.0
            state.joint_state.position = positions
            self.scene_manager.validate_robot_state(state)

    def command(self, opening, wait_for_target=True, guarded=True):
        if not 0.0 <= opening <= 0.07:
            raise ValueError("夹爪开口越界")
        if guarded:
            self.scene_manager.require_unchanged()
        self.validate_sweep(opening)
        if guarded:
            self.scene_manager.require_unchanged()
        result = self.gripper_service(
            gripper_angle=opening,
            gripper_effort=self.gripper_effort,
            gripper_code=1,
            set_zero=0)
        self.state.gripper_service_sent = True
        if not result.status:
            raise RuntimeError("夹爪驱动拒绝请求")
        if not wait_for_target:
            return
        deadline = time.monotonic() + self.gripper_feedback_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if abs(self.current_opening() - opening) <= self.gripper_tolerance:
                return
            rospy.sleep(0.05)
        raise RuntimeError("夹爪未在超时内到达目标开口")

    def confirm(self):
        """Require a stable opening inside the grasp band with real load."""
        deadline = time.monotonic() + self.gripper_feedback_timeout
        stable_since = None
        previous = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            opening = self.current_opening()
            effort = self.joint_feedback.effort("joint7")
            valid = (self.grasp_opening_min <= opening <= self.grasp_opening_max
                     and math.isfinite(effort)
                     and abs(effort) >= self.grasp_effort_min)
            if valid and (previous is None or abs(opening - previous) <= 0.001):
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= self.gripper_settle_time:
                    if self.grasp_check_service:
                        rospy.wait_for_service(self.grasp_check_service, timeout=5.0)
                        reply = rospy.ServiceProxy(self.grasp_check_service, Trigger)()
                        if not reply.success:
                            raise RuntimeError("外部夹持检查失败: " + reply.message)
                    rospy.loginfo(
                        "开口/负载稳定，夹持检查通过: opening=%.4fm effort=%.3f",
                        opening, effort)
                    return
            else:
                stable_since = None
            previous = opening
            rospy.sleep(0.05)
        raise RuntimeError(
            "夹持开口/负载未达到实测阈值，保持当前位置；不附着、不提升")

    # Backwards-compatible names used by the original monolithic script.
    current_gripper_opening = current_opening
    validate_gripper_sweep = validate_sweep
    command_gripper = command
    confirm_grasp = confirm

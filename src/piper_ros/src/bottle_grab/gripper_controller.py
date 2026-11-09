#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gripper service wrapper with opening/effort and force closed-loop control.

Without a sensor the behaviour is the original one: command an opening, wait
for the fingers, then confirm a stable opening plus joint effort.  With an
RPPS-200 sensor (``force_config`` + ``force_sensor``) the closing is instead
driven by measured normal force: close in small steps, stop at the requested
force, confirm it, and optionally keep trimming the opening while the arm
carries the bottle.
"""

import copy
import math
import threading
import time

import numpy as np
import rospy
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningSceneRequest
from std_srvs.srv import Trigger

from force_control import (
    ForceControlError,
    ForceControlConfig,
    ForceGraspController,
)
from joint_feedback import FINGER_JOINTS


class GripperController:
    """Commands the gripper and confirms grasp by opening + load feedback."""

    def __init__(self, joint_feedback, scene_manager, gripper_service,
                 state, gripper_effort, gripper_tolerance,
                 gripper_feedback_timeout, gripper_settle_time,
                 grasp_opening_min, grasp_opening_max, grasp_effort_min,
                 grasp_check_service, force_sensor=None, force_config=None):
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

        self._service_lock = threading.RLock()
        self._hold_stop = threading.Event()
        self._hold_thread = None
        self._last_force_n = None
        self.force_fault = None
        self.force_sensor = force_sensor
        self.force_controller = None
        if force_sensor is not None and force_config is not None:
            if not isinstance(force_config, ForceControlConfig):
                raise TypeError("force_config 必须是 ForceControlConfig")
            self.force_controller = ForceGraspController(
                force_config, force_sensor,
                read_opening=self.current_opening,
                send_opening=self._send_force_opening,
                sleep=rospy.sleep,
                log=lambda message: rospy.loginfo("力控: %s", message),
                warn=lambda message: rospy.logwarn("力控: %s", message))

    # ------------------------------------------------------------------
    # Basic opening feedback.
    # ------------------------------------------------------------------
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

    @property
    def force_control_enabled(self):
        return self.force_controller is not None

    # ------------------------------------------------------------------
    # Low-level service calls.
    # ------------------------------------------------------------------
    def _send_position(self, opening, wait_for_target=False):
        if not 0.0 <= opening <= 0.07:
            raise ValueError("夹爪开口越界")
        with self._service_lock:
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

    def command(self, opening, wait_for_target=True, guarded=True):
        if not 0.0 <= opening <= 0.07:
            raise ValueError("夹爪开口越界")
        if guarded:
            self.scene_manager.require_unchanged()
        self.validate_sweep(opening)
        if guarded:
            self.scene_manager.require_unchanged()
        self._send_position(opening, wait_for_target=wait_for_target)

    def _send_force_opening(self, opening):
        """Small adjustment used by the force loop; the sweep was validated."""
        self._send_position(opening, wait_for_target=False)

    # ------------------------------------------------------------------
    # Force-controlled grasp.
    # ------------------------------------------------------------------
    def tare_force_sensor(self):
        if not self.force_control_enabled:
            return None
        if self.force_fault:
            raise RuntimeError("力控故障: " + self.force_fault)
        opening = self.current_opening()
        if opening < 0.065:
            raise RuntimeError(
                "力控零点标定要求夹爪空载张开(>=0.065m)，当前只有%.4fm；"
                "请先清空夹爪并手动张开" % opening)
        # 等机械臂/夹爪的残余振动衰减，再采空载基线。
        rospy.sleep(0.2)
        pre_force = self.force_controller.tare()
        rospy.loginfo("力控: 零点标定完成，标定前力%.2fN", pre_force)
        return pre_force

    def grasp_with_force(self):
        if not self.force_control_enabled:
            raise RuntimeError("力控未启用，不能调用grasp_with_force")
        if self.force_fault:
            raise RuntimeError("力控故障: " + self.force_fault)
        self.validate_sweep(self.grasp_opening_min)
        self.scene_manager.require_unchanged()
        try:
            result = self.force_controller.grasp()
        except ForceControlError as error:
            self.force_fault = str(error)
            raise RuntimeError("力控抓取失败: %s" % error) from error
        self._last_force_n = result.force_n
        return result

    def _confirm_force(self):
        config = self.force_controller.config
        deadline = time.monotonic() + self.gripper_feedback_timeout
        stable_since = None
        previous = None
        lower = max(config.min_n, config.target_n - config.tolerance_n)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            opening = self.current_opening()
            try:
                force = self.force_controller.force_now()
            except ForceControlError as error:
                self.force_fault = str(error)
                raise RuntimeError("力传感器故障: %s" % error) from error
            self._last_force_n = force
            if force > config.max_n:
                self.force_fault = (
                    "夹持力%.2fN超过安全上限%.2fN" % (force, config.max_n))
                raise RuntimeError(self.force_fault)
            valid = (self.grasp_opening_min <= opening <= self.grasp_opening_max
                     and lower <= force <= config.max_n)
            if valid:
                if previous is not None:
                    opening_stable = abs(opening - previous[0]) <= 0.001
                    force_stable = abs(force - previous[1]) <= 0.5
                else:
                    opening_stable = force_stable = False
                if opening_stable and force_stable:
                    if stable_since is None:
                        stable_since = time.monotonic()
                    if (time.monotonic() - stable_since
                            >= self.gripper_settle_time):
                        if self.grasp_check_service:
                            rospy.wait_for_service(
                                self.grasp_check_service, timeout=5.0)
                            reply = rospy.ServiceProxy(
                                self.grasp_check_service, Trigger)()
                            if not reply.success:
                                raise RuntimeError(
                                    "外部夹持检查失败: " + reply.message)
                        rospy.loginfo(
                            "力控夹持稳定: force=%.2fN target=%.2fN "
                            "opening=%.4fm", force, config.target_n, opening)
                        return
                else:
                    stable_since = None
            else:
                stable_since = None
            previous = (opening, force)
            rospy.sleep(0.05)
        raise RuntimeError(
            "力控夹持未稳定在目标%.2f±%.2fN（当前%.2fN，开口%.4fm）；"
            "不附着、不提升"
            % (config.target_n, config.tolerance_n,
               self._last_force_n if self._last_force_n is not None
               else float("nan"), self.current_opening()))

    def _confirm_opening_effort(self):
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

    def confirm(self):
        if self.force_control_enabled:
            return self._confirm_force()
        return self._confirm_opening_effort()

    # ------------------------------------------------------------------
    # Force hold during transport.
    # ------------------------------------------------------------------
    def start_force_hold(self):
        if not self.force_control_enabled:
            return
        if self.force_fault:
            raise RuntimeError("力控故障: " + self.force_fault)
        if self._hold_thread is not None and self._hold_thread.is_alive():
            return
        self._hold_stop.clear()
        self._hold_thread = threading.Thread(
            target=self._hold_loop, name="force-hold", daemon=True)
        self._hold_thread.start()
        rospy.loginfo(
            "力控: 保压已启动，目标%.2fN",
            self.force_controller.config.target_n)

    def _hold_loop(self):
        config = self.force_controller.config
        while (not self._hold_stop.is_set()
               and not rospy.is_shutdown()):
            try:
                force = self.force_controller.regulate_once()
                self._last_force_n = force
                rospy.loginfo_throttle(
                    2.0, "力控保压: force=%.2fN target=%.2fN",
                    force, config.target_n)
            except Exception as error:
                self.force_fault = str(error)
                rospy.logerr("力控保压停止: %s", error)
                break
            self._hold_stop.wait(config.hold_period_s)

    def stop_force_hold(self):
        self._hold_stop.set()
        thread = self._hold_thread
        self._hold_thread = None
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=1.0)

    def assert_force_healthy(self):
        if not self.force_control_enabled:
            return None
        if self.force_fault:
            raise RuntimeError("力控故障: " + self.force_fault)
        try:
            force = self.force_controller.check_healthy()
        except ForceControlError as error:
            self.force_fault = str(error)
            raise RuntimeError("力控故障: %s" % error) from error
        self._last_force_n = force
        return force

    def measured_force_n(self):
        return self._last_force_n

    # Backwards-compatible names used by the original monolithic script.
    current_gripper_opening = current_opening
    validate_gripper_sweep = validate_sweep
    command_gripper = command
    confirm_grasp = confirm

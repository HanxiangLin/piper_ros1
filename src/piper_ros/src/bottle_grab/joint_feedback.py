#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real /joint_states feedback freshness and arm-settling checks."""

import copy
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
FINGER_JOINTS = ["joint7", "joint8"]


class JointFeedback:
    """Subscribes to /joint_states and serves fresh, complete joint data."""

    def __init__(self, arm_joints=ARM_JOINTS, finger_joints=FINGER_JOINTS):
        self.arm_joints = list(arm_joints)
        self.finger_joints = list(finger_joints)
        self.all_joints = self.arm_joints + self.finger_joints
        self._lock = threading.Lock()
        self._positions = None
        self._efforts = {}
        self._message_stamp = None
        self._feedback_time = None
        self.subscriber = None

    def subscribe(self, topic="/joint_states"):
        self.subscriber = rospy.Subscriber(
            topic, JointState, self._joint_state_callback,
            queue_size=1, tcp_nodelay=True)

    def close(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None

    def _joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in self.all_joints):
            return
        if not np.isfinite(list(positions.values())).all():
            return
        stamp = message.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        if stamp <= 0 or not -0.1 <= age <= 1.0:
            return
        efforts = dict(zip(message.name, message.effort))
        with self._lock:
            self._positions = positions
            self._efforts = efforts
            self._message_stamp = stamp
            self._feedback_time = time.monotonic()

    def _fresh_joint_positions(self):
        """Return the latest complete, fresh /joint_states positions."""
        with self._lock:
            positions = copy.deepcopy(self._positions)
            timestamp = self._feedback_time
            stamp = self._message_stamp
        if positions is None or timestamp is None:
            raise RuntimeError("尚未收到完整 /joint_states 反馈")
        if (time.monotonic() - timestamp > 1.0
                or stamp is None or not -0.1 <= rospy.Time.now().to_sec() - stamp <= 1.0):
            raise RuntimeError("/joint_states 反馈超过1秒，拒绝继续")
        return positions

    def effort(self, joint_name):
        """Return the latest reported effort for one joint (NaN if unknown)."""
        with self._lock:
            return self._efforts.get(joint_name, float("nan"))

    def wait_for_joint_feedback(self):
        """Block until fresh, complete /joint_states data is available."""
        deadline = time.monotonic() + 5.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                self._fresh_joint_positions()
                return
            except RuntimeError:
                rospy.sleep(0.05)
        raise RuntimeError("5秒内没有完整、新鲜的真实/joint_states反馈")

    def wait_for_arm_settled(self, timeout=2.0, epsilon=0.002, hold=0.25):
        """Wait until every arm joint stops changing.

        Opening or closing the gripper changes the load and makes the arm
        rebound slightly (about 0.013 rad on joint5 in practice).  A motion
        planned before that rebound finishes has a first point that no longer
        matches the real robot state, and MoveIt's trajectory_execution_manager
        rejects it with ``CONTROL_FAILED`` (allowed start tolerance 0.01 rad).
        """
        deadline = time.monotonic() + timeout
        previous = self._fresh_joint_positions()
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            rospy.sleep(0.02)
            current = self._fresh_joint_positions()
            if all(abs(current[name] - previous[name]) <= epsilon
                   for name in self.arm_joints):
                if time.monotonic() - stable_since >= hold:
                    return
            else:
                stable_since = time.monotonic()
            previous = current
        raise RuntimeError("松手后机械臂未静止，拒绝立即撤离")

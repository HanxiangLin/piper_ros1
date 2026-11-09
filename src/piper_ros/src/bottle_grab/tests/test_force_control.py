#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for the pure force-control algorithm."""

import sys
import unittest
from pathlib import Path

BOTTLE_GRAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTTLE_GRAB))

from force_control import (  # noqa: E402
    ForceControlConfig,
    ForceControlError,
    ForceGraspController,
)


class GripperModel:
    """Opening -> normal force model used by the fake sensor."""

    def __init__(self, contact=0.060, stiffness=1000.0, opening=0.070):
        self.contact = contact
        self.stiffness = stiffness
        self.opening = opening

    def force(self):
        return max(0.0, (self.contact - self.opening) * self.stiffness)

    def send(self, opening):
        self.opening = opening


class FakeSensor:
    def __init__(self, model, fail_after=None):
        self.model = model
        self.fail_after = fail_after
        self.calls = 0

    def force_n(self, max_age=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("fake sensor timeout")
        return self.model.force()

    def tare(self):
        return 0.0


def config(**overrides):
    values = dict(
        target_n=8.0, tolerance_n=1.0, max_n=12.0, min_n=4.0,
        contact_n=0.5, coarse_step_m=0.001, fine_step_m=0.0002,
        settle_s=0.0, max_squeeze_m=0.010, min_opening_m=0.035,
        max_opening_m=0.070, max_steps=200, stable_cycles=2)
    values.update(overrides)
    return ForceControlConfig(**values)


def controller(model, cfg=None, sensor=None):
    cfg = cfg or config()
    sensor = sensor or FakeSensor(model)
    return ForceGraspController(
        cfg, sensor, read_opening=lambda: model.opening,
        send_opening=model.send, sleep=lambda seconds: None)


class GraspTest(unittest.TestCase):
    def test_grasp_stops_near_target(self):
        model = GripperModel()
        ctl = controller(model)
        result = ctl.grasp()
        self.assertGreaterEqual(result.force_n, 7.5)
        self.assertLessEqual(result.force_n, 9.0)
        self.assertGreaterEqual(result.opening_m, 0.051)
        self.assertLessEqual(result.opening_m, 0.053)
        self.assertGreater(result.steps, 10)
        self.assertTrue(result.within_tolerance)

    def test_no_contact_reopens_and_raises(self):
        model = GripperModel(contact=0.030)
        ctl = controller(model)
        with self.assertRaisesRegex(ForceControlError, "未检测到接触"):
            ctl.grasp()
        self.assertAlmostEqual(model.opening, 0.070)

    def test_sensor_failure_aborts(self):
        model = GripperModel()
        ctl = controller(model, sensor=FakeSensor(model, fail_after=2))
        with self.assertRaisesRegex(ForceControlError, "力传感器"):
            ctl.grasp()

    def test_overshoot_is_limited(self):
        model = GripperModel(stiffness=10000.0)
        ctl = controller(model, config(max_n=15.0))
        result = ctl.grasp()
        self.assertLessEqual(result.force_n, 15.0)

    def test_regulate_once_direction(self):
        model = GripperModel()
        ctl = controller(model)
        ctl.grasp()
        model.opening = 0.054
        before = model.opening
        ctl.regulate_once()
        self.assertLess(model.opening, before)

        model.opening = 0.050
        before = model.opening
        ctl.regulate_once()
        self.assertGreater(model.opening, before)

    def test_check_healthy(self):
        model = GripperModel()
        ctl = controller(model)
        ctl.grasp()
        self.assertGreaterEqual(ctl.check_healthy(), 7.0)

    def test_tare(self):
        model = GripperModel()
        ctl = controller(model)
        self.assertEqual(ctl.tare(), 0.0)


class ConfigTest(unittest.TestCase):
    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            ForceControlConfig(target_n=0.0)
        with self.assertRaises(ValueError):
            ForceControlConfig(target_n=8.0, max_n=7.0)
        with self.assertRaises(ValueError):
            ForceControlConfig(target_n=8.0, tolerance_n=8.0)
        with self.assertRaises(ValueError):
            ForceControlConfig(target_n=8.0, min_opening_m=0.08)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Force-controlled gripper closing and hold (pure Python, no ROS imports).

The Piper gripper accepts a target opening plus a driver effort.  The effort
alone is not a calibrated force, so this controller closes in small position
steps and stops when the RPPS-200 measures the requested normal force.  If it
overshoots, it opens a little; after the grasp it can keep trimming the opening
while the arm carries the bottle.

Only the callback ``send_opening`` touches hardware; everything else is
deterministic and unit-testable.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional


class ForceControlError(RuntimeError):
    """Force control cannot reach or keep the requested force."""


@dataclass
class ForceControlConfig:
    """Thresholds and step sizes for one grasp."""

    target_n: float = 8.0
    tolerance_n: float = 1.0
    max_n: float = 12.0
    min_n: float = 4.0
    contact_n: float = 0.5
    coarse_step_m: float = 0.0010
    fine_step_m: float = 0.0002
    settle_s: float = 0.10
    max_squeeze_m: float = 0.008
    min_opening_m: float = 0.035
    max_opening_m: float = 0.070
    hold_period_s: float = 0.15
    hold_step_m: float = 0.0001
    hold_deadband_n: float = 0.5
    sensor_timeout_s: float = 0.5
    max_steps: int = 300
    stable_cycles: int = 3
    # 关节/滑块开口到真实指垫内表面间距的固定偏移(m)。
    # URDF 指垫面相对滑块基准面内缩约2mm/侧，实物橡胶垫再增加约0.7mm/侧；
    # 这里只用于日志显示，不参与任何控制或安全判断。
    pad_offset_m: float = 0.0

    def __post_init__(self):
        values = {
            "target_n": self.target_n,
            "tolerance_n": self.tolerance_n,
            "max_n": self.max_n,
            "min_n": self.min_n,
            "contact_n": self.contact_n,
            "coarse_step_m": self.coarse_step_m,
            "fine_step_m": self.fine_step_m,
            "settle_s": self.settle_s,
            "max_squeeze_m": self.max_squeeze_m,
            "min_opening_m": self.min_opening_m,
            "max_opening_m": self.max_opening_m,
            "hold_period_s": self.hold_period_s,
            "hold_step_m": self.hold_step_m,
            "hold_deadband_n": self.hold_deadband_n,
            "sensor_timeout_s": self.sensor_timeout_s,
            "pad_offset_m": self.pad_offset_m,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError("力控参数%s不是有限数字" % name)
        if self.target_n <= 0.0:
            raise ValueError("目标夹持力必须大于0")
        if self.min_n < 0.0 or self.min_n >= self.target_n:
            raise ValueError("力控下限必须满足 0 <= min < target")
        if self.max_n < self.target_n:
            raise ValueError("力控上限必须不小于目标力")
        if not 0.0 < self.tolerance_n < self.target_n:
            raise ValueError("力控容差必须大于0且小于目标力")
        if self.contact_n < 0.0 or self.contact_n >= self.target_n:
            raise ValueError("接触阈值必须满足 0 <= contact < target")
        if self.max_squeeze_m <= 0.0:
            raise ValueError("最大压缩量必须大于0")
        if not 0.0 < self.min_opening_m < self.max_opening_m <= 0.07:
            raise ValueError("夹爪开口范围无效")
        if self.coarse_step_m <= 0.0 or self.fine_step_m <= 0.0:
            raise ValueError("闭合步长必须大于0")
        if self.coarse_step_m < self.fine_step_m:
            raise ValueError("粗闭合步长不能小于精闭合步长")
        if self.hold_step_m <= 0.0:
            raise ValueError("保压步长必须大于0")
        if self.max_steps < 10:
            raise ValueError("力控最大步数至少为10")
        if self.stable_cycles < 1:
            raise ValueError("稳定周期至少为1")
        if self.pad_offset_m < 0.0:
            raise ValueError("指垫偏移必须不小于0")


@dataclass
class ForceGraspResult:
    force_n: float
    opening_m: float
    contact_opening_m: float
    steps: int
    within_tolerance: bool
    pad_opening_m: float = 0.0
    contact_pad_opening_m: float = 0.0


class ForceGraspController:
    """Close, trim and hold the gripper using measured force feedback."""

    def __init__(self, config: ForceControlConfig, sensor,
                 read_opening: Callable[[], float],
                 send_opening: Callable[[float], None],
                 sleep: Callable[[float], None],
                 log: Optional[Callable[[str], None]] = None,
                 warn: Optional[Callable[[str], None]] = None):
        self.config = config
        self.sensor = sensor
        self._read_opening = read_opening
        self._send_opening = send_opening
        self._sleep = sleep
        self._log = log or (lambda message: None)
        self._warn = warn or (lambda message: None)
        self._lower: Optional[float] = None
        self._upper: Optional[float] = None
        self._command_opening: Optional[float] = None
        self._last_force: Optional[float] = None
        self._contact_opening: Optional[float] = None

    # ------------------------------------------------------------------
    # Sensor and opening helpers.
    # ------------------------------------------------------------------
    def force_now(self) -> float:
        try:
            value = self.sensor.force_n(
                max_age=self.config.sensor_timeout_s)
            value = float(value)
        except ForceControlError:
            raise
        except Exception as error:
            raise ForceControlError(
                "力传感器无有效读数: %s" % error) from error
        if not math.isfinite(value):
            raise ForceControlError("力传感器返回非有限值")
        self._last_force = value
        return value

    def _opening(self) -> float:
        try:
            value = float(self._read_opening())
        except Exception as error:
            raise ForceControlError("读取夹爪开口失败: %s" % error) from error
        if not math.isfinite(value):
            raise ForceControlError("夹爪开口反馈不是有限值")
        return value

    def _opening_text(self, opening: float) -> str:
        """Format a slider opening, adding the physical pad gap when known."""
        offset = float(self.config.pad_offset_m)
        if offset > 0.0:
            return "开口%.4fm(指垫%.4fm)" % (
                float(opening), float(opening) + offset)
        return "开口%.4fm" % float(opening)

    def _send(self, opening: float) -> float:
        value = min(self.config.max_opening_m,
                    max(self.config.min_opening_m, float(opening)))
        try:
            self._send_opening(value)
        except ForceControlError:
            raise
        except Exception as error:
            raise ForceControlError("夹爪闭环指令失败: %s" % error) from error
        self._command_opening = value
        return value

    def tare(self) -> float:
        """Tare the sensor; return the force measured before taring."""
        try:
            return float(self.sensor.tare())
        except ForceControlError:
            raise
        except Exception as error:
            raise ForceControlError("力传感器零点标定失败: %s" % error) from error

    # ------------------------------------------------------------------
    # Closing.
    # ------------------------------------------------------------------
    def grasp(self) -> ForceGraspResult:
        config = self.config
        start = self._opening()
        if start < config.min_opening_m - 1e-9:
            raise ForceControlError(
                "夹爪起始%s小于最小开口%.4fm"
                % (self._opening_text(start), config.min_opening_m))
        self._upper = start
        self._lower = config.min_opening_m
        self._command_opening = start
        self._log(
            "力控闭合: 目标%.2fN 容许±%.2fN 上限%.2fN 起始%s"
            % (config.target_n, config.tolerance_n, config.max_n,
               self._opening_text(start)))

        opening = start
        steps = 0
        force = self.force_now()
        contact_opening: Optional[float] = None
        if force >= config.contact_n:
            contact_opening = opening
            self._log("起始位置已有接触: %.2fN" % force)

        while (contact_opening is None and steps < config.max_steps
               and opening > config.min_opening_m + 1e-9):
            opening = max(config.min_opening_m,
                          opening - config.coarse_step_m)
            opening = self._send(opening)
            self._sleep(config.settle_s)
            steps += 1
            force = self.force_now()
            if steps % 5 == 0:
                self._log("粗闭合: %s 力%.2fN"
                          % (self._opening_text(opening), force))
            if force >= config.contact_n or force >= config.max_n:
                contact_opening = opening
                break

        if contact_opening is None:
            self._send(start)
            raise ForceControlError(
                "闭合到最小%s仍未检测到接触（阈值%.2fN），"
                "拒绝继续挤压"
                % (self._opening_text(config.min_opening_m),
                   config.contact_n))

        self._contact_opening = contact_opening
        self._lower = max(config.min_opening_m,
                          contact_opening - config.max_squeeze_m)
        self._log(
            "检测到接触: %s 力%.2fN；允许继续压缩到%s"
            % (self._opening_text(contact_opening), force,
               self._opening_text(self._lower)))

        if force > config.max_n:
            opening, force, steps = self._backoff(opening, force, steps)

        # Fine approach: stop just below the target (not at the lower edge of
        # the tolerance band), then let the trim loop settle on the target.
        fine_stop = config.target_n - 0.25 * config.tolerance_n
        while (force < fine_stop
               and opening > self._lower + 1e-9
               and steps < config.max_steps):
            opening = max(self._lower, opening - config.fine_step_m)
            opening = self._send(opening)
            self._sleep(config.settle_s)
            steps += 1
            force = self.force_now()
            if force > config.max_n:
                opening, force, steps = self._backoff(
                    opening, force, steps)

        stable = 0
        trim_step = config.fine_step_m
        last_sign = 0
        while steps < config.max_steps:
            force = self.force_now()
            error = config.target_n - force
            if abs(error) <= config.tolerance_n:
                stable += 1
                if stable >= config.stable_cycles:
                    break
                self._sleep(config.settle_s)
                steps += 1
                continue
            stable = 0
            sign = 1 if error > 0.0 else -1
            if last_sign and sign != last_sign:
                # 力穿过目标后把步长减半，避免刚性物体在目标两侧来回跳。
                trim_step = max(config.fine_step_m / 8.0, trim_step * 0.5)
            last_sign = sign
            if error > 0.0:
                new_opening = max(self._lower, opening - trim_step)
            else:
                new_opening = min(self._upper, opening + trim_step)
            if abs(new_opening - opening) < 1e-9:
                self._warn(
                    "力控已到行程边界: %s 力%.2fN 目标%.2fN"
                    % (self._opening_text(opening), force, config.target_n))
                break
            opening = self._send(new_opening)
            self._sleep(config.settle_s)
            steps += 1
            force = self.force_now()
            if force > config.max_n:
                opening, force, steps = self._backoff(
                    opening, force, steps)
            if steps % 10 == 0:
                self._log("精调: %s 力%.2fN 步长%.5fm"
                          % (self._opening_text(opening), force, trim_step))

        force = self.force_now()
        self._last_force = force
        if force > config.max_n:
            raise ForceControlError(
                "夹持力%.2fN超过安全上限%.2fN" % (force, config.max_n))
        if force < config.min_n:
            raise ForceControlError(
                "夹持力%.2fN低于下限%.2fN；不抬升" % (force, config.min_n))
        within = abs(force - config.target_n) <= config.tolerance_n
        if not within:
            raise ForceControlError(
                "夹持力%.2fN未进入目标%.2f±%.2fN（%s）"
                % (force, config.target_n, config.tolerance_n,
                   self._opening_text(opening)))
        self._log(
            "力控闭合完成: 力%.2fN %s 接触%s 步数%d"
            % (force, self._opening_text(opening),
               self._opening_text(contact_opening), steps))
        offset = float(config.pad_offset_m)
        return ForceGraspResult(
            force_n=force, opening_m=opening,
            contact_opening_m=contact_opening, steps=steps,
            within_tolerance=within,
            pad_opening_m=float(opening) + offset,
            contact_pad_opening_m=float(contact_opening) + offset)

    def _backoff(self, opening: float, force: float, steps: int):
        config = self.config
        while (force > config.max_n and opening < self._upper - 1e-9
               and steps < config.max_steps):
            opening = min(self._upper,
                          opening + max(config.fine_step_m * 5.0, 0.0005))
            opening = self._send(opening)
            self._sleep(config.settle_s)
            steps += 1
            force = self.force_now()
            self._warn("夹持力超上限，回退到%s 力%.2fN"
                       % (self._opening_text(opening), force))
        if force > config.max_n:
            raise ForceControlError(
                "夹持力%.2fN超过安全上限%.2fN，且已无法继续张开"
                % (force, config.max_n))
        return opening, force, steps

    # ------------------------------------------------------------------
    # Hold during transport.
    # ------------------------------------------------------------------
    def regulate_once(self) -> float:
        """One hold adjustment; return the force measured after the step."""
        config = self.config
        if (self._lower is None or self._upper is None
                or self._command_opening is None):
            raise ForceControlError("尚未完成力控闭合，不能启动保压")
        # 被物体挡住时 current_opening 是物体宽度，不能拿它做保压增量；
        # 必须从“最后一条闭合指令”继续加减。
        opening = self._command_opening
        force = self.force_now()

        if force > config.max_n:
            new_opening = min(self._upper,
                              opening + max(config.hold_step_m * 5.0,
                                            0.0005))
            if new_opening <= opening + 1e-9:
                raise ForceControlError(
                    "夹持力%.2fN超过上限%.2fN且夹爪已全开"
                    % (force, config.max_n))
            self._send(new_opening)
            self._warn("保压: 力%.2fN超上限，张开到%s"
                       % (force, self._opening_text(new_opening)))
            return self.force_now()

        if force < config.min_n:
            new_opening = max(self._lower, opening - config.hold_step_m)
            if new_opening >= opening - 1e-9:
                raise ForceControlError(
                    "夹持力%.2fN低于下限%.2fN且已到最小开口"
                    % (force, config.min_n))
            self._send(new_opening)
            self._warn("保压: 力%.2fN低于下限，闭合到%s"
                       % (force, self._opening_text(new_opening)))
            return self.force_now()

        error = config.target_n - force
        if error > config.hold_deadband_n:
            new_opening = max(self._lower, opening - config.hold_step_m)
        elif error < -config.hold_deadband_n:
            new_opening = min(self._upper, opening + config.hold_step_m)
        else:
            return force

        if abs(new_opening - opening) < 1e-9:
            if force < config.target_n - config.tolerance_n:
                raise ForceControlError(
                    "夹持力%.2fN无法继续增加（%s）"
                    % (force, self._opening_text(opening)))
            return force
        self._send(new_opening)
        return self.force_now()

    def check_healthy(self) -> float:
        config = self.config
        force = self.force_now()
        if (self._command_opening is not None
                and not config.min_opening_m - 1e-9
                <= self._command_opening <= config.max_opening_m + 1e-9):
            raise ForceControlError(
                "保压指令%s超出范围" % self._opening_text(self._command_opening))
        if force < config.min_n:
            raise ForceControlError(
                "夹持力%.2fN低于下限%.2fN" % (force, config.min_n))
        if force > config.max_n:
            raise ForceControlError(
                "夹持力%.2fN超过上限%.2fN" % (force, config.max_n))
        return force

    @property
    def last_force_n(self) -> Optional[float]:
        return self._last_force

    @property
    def contact_opening_m(self) -> Optional[float]:
        return self._contact_opening

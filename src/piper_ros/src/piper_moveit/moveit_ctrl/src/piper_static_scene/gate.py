"""ROS-independent state and timestamp checks for the point-cloud scan gate."""

import math
import threading


class GateState:
    """Track whether fresh point clouds may pass into MoveIt's OctoMap updater."""

    def __init__(self, max_age=0.75, future_tolerance=0.10):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be a finite positive number")
        if not math.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError("future_tolerance must be finite and non-negative")
        self.max_age = float(max_age)
        self.future_tolerance = float(future_tolerance)
        self.lock = threading.Lock()
        self.enabled = False
        self.received = 0
        self.forwarded = 0
        self.session_forwarded = 0
        self.dropped_disabled = 0
        self.dropped_invalid = 0
        self.last_frame = ""
        self.last_stamp = None
        self.last_reason = "尚未收到点云"

    def start(self):
        with self.lock:
            self.enabled = True
            self.session_forwarded = 0
            self.last_reason = "扫描已开启，等待新点云"

    def stop(self):
        with self.lock:
            self.enabled = False
            self.last_reason = "扫描已停止，MoveIt 内部地图保持冻结"

    def accept(self, frame_id, stamp, now):
        """Return whether one cloud may pass. Inputs are seconds and frame text."""
        with self.lock:
            self.received += 1
            self.last_frame = frame_id or ""
            try:
                stamp = float(stamp)
                now = float(now)
            except (TypeError, ValueError):
                stamp = now = float("nan")
            self.last_stamp = stamp if math.isfinite(stamp) else None
            if not self.enabled:
                self.dropped_disabled += 1
                self.last_reason = "扫描关闭"
                return False, self.last_reason
            if not frame_id or frame_id.startswith("/") or any(char.isspace() for char in frame_id):
                self.dropped_invalid += 1
                self.last_reason = "点云 frame_id 无效"
                return False, self.last_reason
            if not math.isfinite(stamp) or not math.isfinite(now) or stamp <= 0:
                self.dropped_invalid += 1
                self.last_reason = "点云时间戳无效"
                return False, self.last_reason
            age = now - stamp
            if age > self.max_age or age < -self.future_tolerance:
                self.dropped_invalid += 1
                self.last_reason = "点云时间戳过旧或超前：{:.3f}s".format(age)
                return False, self.last_reason
            self.forwarded += 1
            self.session_forwarded += 1
            self.last_reason = "正在转发新鲜点云"
            return True, self.last_reason

    def summary(self):
        with self.lock:
            return {
                "enabled": self.enabled,
                "received": self.received,
                "forwarded": self.forwarded,
                "session_forwarded": self.session_forwarded,
                "dropped_disabled": self.dropped_disabled,
                "dropped_invalid": self.dropped_invalid,
                "last_frame": self.last_frame,
                "last_stamp": self.last_stamp,
                "last_reason": self.last_reason,
                "max_age": self.max_age,
            }

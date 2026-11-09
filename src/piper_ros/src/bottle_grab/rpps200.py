#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RPPS-200 / 10x20 tactile sensor: UART frames and ADC->kPa calibration.

The vendor Windows application (``RPPS32x36.dll``) reads the sensor as a
virtual serial port at 2 Mbaud, 8N1.  A frame is 432 bytes::

    row i (i=0..9):  FF FE i  +  20 x uint16 little-endian
    frame          =  10 * 43 bytes + uint16 little-endian checksum
    checksum       =  sum(frame[0:430]) & 0xFFFF

Each ADC is 12-bit (the vendor masks with ``0xFFF``).  The calibration CSV is
the vendor "查表" export: a metadata line, a normal CSV header, then one row
per applied pressure with 200 raw ADC values.  Per-taxel curves are linearly
interpolated and the vendor display offset (15 kPa for 10x20) is removed in
exactly the same way as ``AdcKpaLookup``.

This module deliberately has no ROS dependency so it can be tested and used by
``rpps200_monitor.py`` on a plain Python interpreter.
"""

import bisect
import glob
import math
import os
import statistics
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

ROWS = 10
COLS = 20
POINTS = ROWS * COLS
FRAME_LENGTH = 432
ROW_LENGTH = 43
DATA_LENGTH = 430
HEADER = b"\xff\xfe"
DEFAULT_BAUD = 2000000
DEFAULT_AREA_MM2 = 2.25          # 30 mm x 15 mm sensing area / 200 points
DEFAULT_DISPLAY_OFFSET_KPA = 15.0

# Order copied from the vendor's BaudRateAutoDetector.
BAUD_CANDIDATES = (2000000, 921600, 460800, 230400, 115200)


class SensorError(RuntimeError):
    """The sensor cannot be opened, configured or read."""


class SensorTimeout(SensorError):
    """No fresh frame arrived within the requested timeout."""


@dataclass
class TactileReading:
    """One decoded frame after calibration and baseline subtraction."""

    raw_adc: List[int]
    adc: List[int]
    kpa: List[float]
    force_n: float
    max_kpa: float
    contact_count: int
    centroid_rc: Tuple[float, float]
    stamp: float
    frame_index: int


def _meta(metadata: Dict[str, str], key: str, default: str = "") -> str:
    return metadata.get(key.lower(), default).strip()


def _meta_float(metadata: Dict[str, str], key: str,
                default: float) -> float:
    text = _meta(metadata, key)
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _meta_bool(metadata: Dict[str, str], key: str,
               default: bool = False) -> bool:
    text = _meta(metadata, key).lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _lookup_curve(curve: Tuple[List[int], List[float]],
                  adc: int) -> float:
    """Piecewise linear ADC->kPa, matching the vendor ``CalibrationCurve``."""
    adcs, kpas = curve
    if not adcs:
        return 0.0
    position = bisect.bisect_left(adcs, adc)
    if position < len(adcs) and adcs[position] == adc:
        return kpas[position]
    if position <= 0:
        # Below the first measured point: interpolate from the origin.
        return max(0.0, kpas[0] * float(adc) / float(max(1, adcs[0])))
    if position >= len(adcs):
        return kpas[-1]
    low_adc, high_adc = adcs[position - 1], adcs[position]
    low_kpa, high_kpa = kpas[position - 1], kpas[position]
    if high_adc <= low_adc:
        return max(0.0, (low_kpa + high_kpa) * 0.5)
    ratio = float(adc - low_adc) / float(high_adc - low_adc)
    return max(0.0, low_kpa + (high_kpa - low_kpa) * ratio)


class RPPS200Calibration:
    """Vendor ADC->kPa lookup for a 10x20 RPPS-200 calibration export."""

    def __init__(self, rows: int, cols: int, offset_kpa: float,
                 corrected: bool, source_offset_kpa: float,
                 curves: List[Optional[Tuple[List[int], List[float]]]],
                 global_curve: Optional[Tuple[List[int], List[float]]],
                 path: str, point_count: int, raw_point_count: int):
        self.rows = rows
        self.cols = cols
        self.offset_kpa = offset_kpa
        self.corrected = corrected
        self.source_offset_kpa = source_offset_kpa
        self._curves = curves
        self._global_curve = global_curve
        self.path = path
        self.point_count = point_count
        self.raw_point_count = raw_point_count
        self.valid_channels = sum(1 for curve in curves if curve is not None)
        self.average_adc_mapping = False

    @property
    def point_count_per_channel(self) -> Tuple[int, int]:
        counts = [len(curve[0]) for curve in self._curves if curve]
        if not counts:
            return (0, 0)
        return (min(counts), max(counts))

    @classmethod
    def load(cls, path: str) -> "RPPS200Calibration":
        """Parse a vendor 10x20 ``查表`` CSV (UTF-8 with or without BOM)."""
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise SensorError("RPPS-200 标定文件不存在: " + path)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as stream:
            lines = [line.rstrip("\r\n") for line in stream if line.strip()]
        if len(lines) < 3:
            raise SensorError("RPPS-200 标定文件内容不足: " + path)

        metadata: Dict[str, str] = {}
        parts = [part.strip() for part in lines[0].split(",")]
        for index in range(0, len(parts) - 1, 2):
            metadata[parts[index].lower()] = parts[index + 1]

        rows = int(_meta_float(metadata, "rows", ROWS))
        cols = int(_meta_float(metadata, "cols", COLS))
        if rows <= 0 or cols <= 0:
            raise SensorError("标定文件缺少 Rows/Cols 元数据: " + path)
        points = rows * cols
        if points != POINTS:
            raise SensorError(
                "当前驱动只支持 %dx%d（%d点），标定文件是 %dx%d（%d点）"
                % (ROWS, COLS, POINTS, rows, cols, points))

        offset_kpa = _meta_float(
            metadata, "displaypressureoffsetkpa",
            _meta_float(metadata, "lookuppressureoffsetkpa",
                        DEFAULT_DISPLAY_OFFSET_KPA))
        offset_kpa = min(520.0, max(0.0, offset_kpa))
        corrected = _meta_bool(metadata, "kpacorrected", False)
        source_offset_kpa = offset_kpa
        if corrected:
            for key in ("pressureoffsetkpa", "lookuppressureoffsetkpa",
                        "displaypressureoffsetkpa"):
                if _meta(metadata, key):
                    source_offset_kpa = _meta_float(
                        metadata, key, DEFAULT_DISPLAY_OFFSET_KPA)
                    break

        header = [part.strip() for part in lines[1].split(",")]
        kpa_index = -1
        for index, name in enumerate(header):
            if name.lower() == "kpa":
                kpa_index = index
                break
        if kpa_index < 0:
            for index, name in enumerate(header):
                if "kpa" in name.lower():
                    kpa_index = index
                    break
        if kpa_index < 0:
            raise SensorError("标定文件表头缺少 kPa 列: " + path)
        channel_start = kpa_index + 1
        if len(header) < channel_start + points:
            raise SensorError(
                "标定文件表头只有%d列，不足以容纳%d个测点: %s"
                % (len(header), points, path))

        # channel -> {adc: [sum_kpa, count]}
        channel_maps: List[Dict[int, List[float]]] = [
            {} for _ in range(points)]
        raw_point_count = 0
        for line in lines[2:]:
            values = line.split(",")
            if len(values) <= kpa_index:
                continue
            try:
                file_kpa = float(values[kpa_index].strip())
            except (TypeError, ValueError):
                continue
            if not math.isfinite(file_kpa):
                continue
            if corrected:
                display_kpa = file_kpa + source_offset_kpa - offset_kpa
            else:
                display_kpa = file_kpa - offset_kpa
            display_kpa = max(0.0, display_kpa)
            for channel in range(points):
                column = channel_start + channel
                if column >= len(values):
                    break
                text = values[column].strip()
                if not text:
                    continue
                try:
                    adc = int(float(text))
                except (TypeError, ValueError):
                    continue
                if adc <= 0 or adc > 65535:
                    continue
                entry = channel_maps[channel].get(adc)
                if entry is None:
                    channel_maps[channel][adc] = [display_kpa, 1.0]
                else:
                    entry[0] += display_kpa
                    entry[1] += 1.0
                raw_point_count += 1

        curves: List[Optional[Tuple[List[int], List[float]]]] = []
        global_map: Dict[int, List[float]] = {}
        point_count = 0
        for channel_map in channel_maps:
            if not channel_map:
                curves.append(None)
                continue
            adcs: List[int] = []
            kpas: List[float] = []
            for adc in sorted(channel_map):
                total, count = channel_map[adc]
                adcs.append(adc)
                kpas.append(max(0.0, total / count))
                entry = global_map.setdefault(adc, [0.0, 0.0])
                entry[0] += total / count
                entry[1] += 1.0
                point_count += 1
            curves.append((adcs, kpas))

        global_curve = None
        if global_map:
            adcs = []
            kpas = []
            for adc in sorted(global_map):
                total, count = global_map[adc]
                adcs.append(adc)
                kpas.append(max(0.0, total / count))
            global_curve = (adcs, kpas)
        if point_count == 0:
            raise SensorError("标定文件没有解析到任何 ADC/kPa 数据: " + path)

        return cls(rows, cols, offset_kpa, corrected, source_offset_kpa,
                   curves, global_curve, path, point_count, raw_point_count)

    def kpa(self, channel: int, adc: int) -> float:
        """Return kPa for one channel; ADC must already be tare-corrected."""
        if adc <= 0:
            return 0.0
        curve = None
        if 0 <= channel < len(self._curves):
            curve = self._curves[channel]
        if curve is None:
            curve = self._global_curve
        if curve is None:
            return 0.0
        return _lookup_curve(curve, int(adc))

    @staticmethod
    def force_n(kpa_values: Sequence[float], area_mm2: float,
                scale: float = 1.0) -> float:
        """Total normal force on the pad: sum(p_i) * taxel area."""
        area_m2 = max(0.0, float(area_mm2)) * 1e-6
        total_pa = 0.0
        for value in kpa_values:
            if value > 0.0 and math.isfinite(value):
                total_pa += value * 1000.0
        return total_pa * area_m2 * float(scale)

    @staticmethod
    def contact_stats(kpa_values: Sequence[float], contact_kpa: float
                      ) -> Tuple[float, int, Tuple[float, float]]:
        """Return ``(max_kpa, contact_count, (row, col) centroid)``."""
        maximum = 0.0
        count = 0
        sum_kpa = 0.0
        sum_row = 0.0
        sum_col = 0.0
        for index, value in enumerate(kpa_values):
            if value <= 0.0 or not math.isfinite(value):
                continue
            if value > maximum:
                maximum = value
            if value >= contact_kpa:
                count += 1
            sum_kpa += value
            sum_row += value * (index // COLS)
            sum_col += value * (index % COLS)
        if sum_kpa <= 0.0:
            return (maximum, count, (float("nan"), float("nan")))
        return (maximum, count, (sum_row / sum_kpa, sum_col / sum_kpa))

    def summary(self) -> str:
        low, high = self.point_count_per_channel
        return ("%s: %dx%d, 偏移%.1fkPa, 有效通道%d/%d, "
                "每通道%d~%d点, 总标定点%d"
                % (os.path.basename(self.path), self.rows, self.cols,
                   self.offset_kpa, self.valid_channels, POINTS,
                   low, high, self.point_count))


class RPPS200FrameParser:
    """Incremental 432-byte frame parser with resynchronisation."""

    def __init__(self):
        self.buffer = bytearray()
        self.frames = 0
        self.rejected = 0
        self.checksum_errors = 0

    def feed(self, data: bytes) -> List[List[int]]:
        if data:
            self.buffer.extend(data)
        decoded: List[List[int]] = []
        while True:
            start = self.buffer.find(HEADER)
            if start < 0:
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                break
            if len(self.buffer) - start < FRAME_LENGTH:
                if start:
                    del self.buffer[:start]
                break
            frame = bytes(self.buffer[start:start + FRAME_LENGTH])
            if self.validate(frame):
                decoded.append(self.parse(frame))
                self.frames += 1
                del self.buffer[:start + FRAME_LENGTH]
            else:
                self.rejected += 1
                if not self._headers(frame):
                    self.checksum_errors += 1
                del self.buffer[:start + 1]
        if len(self.buffer) > 16384:
            del self.buffer[:-4096]
        return decoded

    @staticmethod
    def _headers(frame: bytes) -> bool:
        for row in range(ROWS):
            offset = row * ROW_LENGTH
            if (frame[offset] != 0xFF or frame[offset + 1] != 0xFE
                    or frame[offset + 2] != row):
                return False
        return True

    @classmethod
    def validate(cls, frame: bytes) -> bool:
        if len(frame) != FRAME_LENGTH or not cls._headers(frame):
            return False
        expected = int(frame[430]) | (int(frame[431]) << 8)
        actual = sum(frame[:DATA_LENGTH]) & 0xFFFF
        return actual == expected

    @staticmethod
    def parse(frame: bytes) -> List[int]:
        values = [0] * POINTS
        for row in range(ROWS):
            offset = row * ROW_LENGTH + 3
            channels = struct.unpack_from("<20H", frame, offset)
            base = row * COLS
            for col, value in enumerate(channels):
                values[base + col] = int(value) & 0x0FFF
        return values

    def summary(self) -> str:
        return ("帧%d, 拒绝%d, 校验/帧头错误%d, 缓冲%d"
                % (self.frames, self.rejected, self.checksum_errors,
                   len(self.buffer)))


def candidate_serial_ports() -> List[str]:
    """Return likely RPPS-200 serial devices, most specific first."""
    ports: List[str] = []
    if os.name == "posix":
        for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyCH*",
                        "/dev/ttyCH343USB*", "/dev/serial/by-id/*"):
            ports.extend(sorted(glob.glob(pattern)))
    if not ports:
        try:
            from serial.tools import list_ports
        except ImportError:
            list_ports = None
        if list_ports is not None:
            for info in list_ports.comports():
                device = info.device
                if os.name != "posix" or "tty" in device:
                    ports.append(device)
    result: List[str] = []
    seen = set()
    for port in ports:
        real = os.path.realpath(port)
        if real in seen:
            continue
        seen.add(real)
        result.append(port)
    return result


def _probe_port(serial_module, port: str, baud: int,
                probe_seconds: float) -> Optional[object]:
    serial_port = None
    try:
        serial_port = serial_module.Serial(
            port=port, baudrate=baud, bytesize=serial_module.EIGHTBITS,
            parity=serial_module.PARITY_NONE,
            stopbits=serial_module.STOPBITS_ONE, timeout=0.05,
            write_timeout=0.05, xonxoff=False, rtscts=False, dsrdtr=False)
        serial_port.reset_input_buffer()
        try:
            serial_port.dtr = False
            serial_port.rts = False
        except Exception:
            pass
        parser = RPPS200FrameParser()
        deadline = time.monotonic() + probe_seconds
        while time.monotonic() < deadline:
            data = serial_port.read(4096)
            if data and parser.feed(data):
                return serial_port
        return None
    except Exception:
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        return None


def open_rpps200_serial(port: str = "", baud: int = DEFAULT_BAUD,
                        probe_seconds: float = 1.2
                        ) -> Tuple[object, str, int]:
    """Open the first port/baud that produces a valid RPPS-200 frame.

    Returns ``(serial_port, port_name, baud)``.  The caller owns the port.
    """
    try:
        import serial
    except ImportError as error:
        raise SensorError(
            "未安装 pyserial，无法读取 RPPS-200；请先运行 "
            "python3 -m pip install pyserial") from error

    ports = [port] if port else candidate_serial_ports()
    if not ports:
        raise SensorError(
            "没有找到 RPPS-200 串口；请检查 USB 线，运行 lsusb、"
            "dmesg | tail，或用 rpps_port 指定 /dev/ttyACM0 等设备")
    bauds: List[int] = []
    if baud:
        bauds.append(int(baud))
    for value in BAUD_CANDIDATES:
        if value not in bauds:
            bauds.append(value)

    errors: List[str] = []
    for name in ports:
        for rate in bauds:
            opened = _probe_port(serial, name, rate, probe_seconds)
            if opened is not None:
                return (opened, name, rate)
            errors.append("%s@%d" % (name, rate))
    raise SensorError(
        "在 %s 上没有收到有效的 RPPS-200 帧（已尝试 %s）。"
        "检查设备是否被上位机占用、用户是否在 dialout 组，"
        "以及 rpps_baud 是否为 2000000。"
        % (", ".join(ports), ", ".join(errors[-6:])))


class RPPS200Sensor:
    """Background serial reader producing calibrated :class:`TactileReading`."""

    def __init__(self, calibration: RPPS200Calibration, port: str = "",
                 baud: int = DEFAULT_BAUD, area_mm2: float = DEFAULT_AREA_MM2,
                 force_scale: float = 1.0, contact_kpa: float = 1.0,
                 filter_window: int = 3,
                 sensor_timeout: float = 0.5):
        self.calibration = calibration
        self.requested_port = port
        self.requested_baud = int(baud)
        self.area_mm2 = float(area_mm2)
        self.force_scale = float(force_scale)
        self.contact_kpa = float(contact_kpa)
        self.sensor_timeout = float(sensor_timeout)
        self._baseline = [0.0] * POINTS
        self._history: Deque[float] = deque(
            maxlen=max(1, int(filter_window)))
        self._frame_times: Deque[float] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._latest: Optional[TactileReading] = None
        self._frame_index = 0
        self._serial = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.port = ""
        self.baud = 0
        self.last_error = ""
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------
    def start(self, probe_seconds: float = 1.2):
        if self._running:
            return self.port, self.baud
        self._serial, self.port, self.baud = open_rpps200_serial(
            self.requested_port, self.requested_baud, probe_seconds)
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="rpps200-reader", daemon=True)
        self._thread.start()
        return self.port, self.baud

    def stop(self):
        self._stop.set()
        serial_port = self._serial
        self._serial = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=1.0)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running and self.last_error == ""

    def _reader_loop(self):
        parser = RPPS200FrameParser()
        while not self._stop.is_set():
            serial_port = self._serial
            if serial_port is None:
                break
            try:
                data = serial_port.read(4096)
            except Exception as error:
                if not self._stop.is_set():
                    self.last_error = "串口读取失败: %s" % error
                break
            if not data:
                continue
            for adc in parser.feed(data):
                self._publish(adc)
        self._running = False
        if not self._stop.is_set() and not self.last_error:
            self.last_error = "串口读取线程已停止"

    # ------------------------------------------------------------------
    # Decoding.
    # ------------------------------------------------------------------
    def build_reading(self, adc: Sequence[int], frame_index: int = 0,
                      stamp: Optional[float] = None) -> TactileReading:
        baseline = self._baseline
        raw = [0] * POINTS
        effective = [0] * POINTS
        for index in range(POINTS):
            value = int(adc[index]) if index < len(adc) else 0
            raw[index] = value
            corrected = value - int(round(baseline[index]))
            effective[index] = corrected if corrected > 0 else 0
        kpa_values = [self.calibration.kpa(index, value)
                      for index, value in enumerate(effective)]
        force = self.calibration.force_n(
            kpa_values, self.area_mm2, self.force_scale)
        maximum, count, centroid = self.calibration.contact_stats(
            kpa_values, self.contact_kpa)
        return TactileReading(
            raw_adc=raw, adc=effective, kpa=kpa_values, force_n=force,
            max_kpa=maximum, contact_count=count, centroid_rc=centroid,
            stamp=time.monotonic() if stamp is None else stamp,
            frame_index=int(frame_index))

    def _publish(self, adc: Sequence[int]):
        with self._lock:
            frame_index = self._frame_index + 1
            self._frame_index = frame_index
        reading = self.build_reading(adc, frame_index)
        with self._lock:
            self._latest = reading
            self._history.append(reading.force_n)
            self._frame_times.append(reading.stamp)

    # ------------------------------------------------------------------
    # Reads.
    # ------------------------------------------------------------------
    def latest(self, max_age: Optional[float] = None
               ) -> Optional[TactileReading]:
        with self._lock:
            reading = self._latest
        if reading is None:
            return None
        if max_age is not None and time.monotonic() - reading.stamp > max_age:
            return None
        return reading

    def force_n(self, max_age: Optional[float] = None) -> float:
        age = self.sensor_timeout if max_age is None else max_age
        reading = self.latest(age)
        if reading is None:
            if self.last_error:
                raise SensorTimeout(self.last_error)
            raise SensorTimeout("RPPS-200 超过%.2fs没有新帧" % age)
        with self._lock:
            history = list(self._history)
        if not history:
            raise SensorTimeout("RPPS-200 没有有效的力读数")
        return float(statistics.median(history))

    def latest_force_n(self) -> float:
        reading = self.latest()
        return float("nan") if reading is None else reading.force_n

    def fps(self) -> float:
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0.0:
            return 0.0
        return (len(times) - 1) / span

    @property
    def frame_index(self) -> int:
        with self._lock:
            return self._frame_index

    # ------------------------------------------------------------------
    # Tare.
    # ------------------------------------------------------------------
    def set_baseline(self, baseline: Iterable[float]):
        values = [float(value) for value in baseline]
        if len(values) != POINTS:
            raise ValueError("baseline 必须包含%d个值" % POINTS)
        with self._lock:
            self._baseline = values
            self._history.clear()

    def tare(self, frames: int = 30, timeout: float = 3.0) -> float:
        """Capture the unloaded ADC baseline; return the pre-tare force."""
        if not self._running:
            raise SensorError("RPPS-200 未启动，无法零点标定")
        frames = max(3, int(frames))
        samples: List[List[int]] = []
        seen = set()
        deadline = time.monotonic() + max(0.5, float(timeout))
        pre_force = self.latest_force_n()
        while len(samples) < frames and time.monotonic() < deadline:
            reading = self.latest()
            if reading is not None and reading.frame_index not in seen:
                seen.add(reading.frame_index)
                samples.append(list(reading.raw_adc))
            else:
                time.sleep(0.005)
        if len(samples) < frames:
            raise SensorTimeout(
                "零点标定只收到%d/%d帧；检查传感器帧率" %
                (len(samples), frames))
        baseline = [float(statistics.median(column))
                    for column in zip(*samples)]
        last_index = self.frame_index
        self.set_baseline(baseline)
        # set_baseline 清空了力历史；等一帧用新基线解出的数据再返回，
        # 避免紧接着的 force_n() 撞上空历史。
        deadline = time.monotonic() + 0.5
        while self.frame_index <= last_index and time.monotonic() < deadline:
            time.sleep(0.002)
        if self.frame_index <= last_index:
            raise SensorTimeout("零点标定后0.5s内没有新帧")
        return pre_force

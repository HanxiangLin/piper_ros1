#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for the RPPS-200 frame parser and calibration lookup."""

import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

BOTTLE_GRAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOTTLE_GRAB))

from rpps200 import (  # noqa: E402
    FRAME_LENGTH,
    POINTS,
    RPPS200Calibration,
    RPPS200FrameParser,
    RPPS200Sensor,
    SensorError,
)


def calibration_csv(path, rows=(20.0, 40.0, 60.0),
                    adcs=(10, 20, 30)):
    header = ["Timestamp", "kPa"]
    header.extend("r%dc%d" % (row + 1, col + 1)
                  for row in range(10) for col in range(20))
    header.append("AverageADC")
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        stream.write(
            "Rows,10,Cols,20,Data,RawADC,Version,9,KpaCorrected,0,"
            "DisplayPressureOffsetKpa,15.0,"
            "PressureStoredAsMeasured,1\n")
        stream.write(",".join(header) + "\n")
        for kpa, adc in zip(rows, adcs):
            values = [str(adc)] * POINTS
            stream.write(",".join(
                ["2026-11-04 09:57:24.841", "%.1f" % kpa]
                + values + [str(adc)]) + "\n")


def make_frame(values, mask=True):
    if len(values) != POINTS:
        raise ValueError("need 200 values")
    frame = bytearray(FRAME_LENGTH)
    for row in range(10):
        offset = row * 43
        frame[offset:offset + 3] = b"\xff\xfe" + bytes([row])
        for col in range(20):
            value = values[row * 20 + col]
            if mask:
                value &= 0x0FFF
            struct.pack_into("<H", frame, offset + 3 + col * 2, value)
    checksum = sum(frame[:430]) & 0xFFFF
    struct.pack_into("<H", frame, 430, checksum)
    return bytes(frame)


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "cal.csv")
        calibration_csv(self.path)
        self.cal = RPPS200Calibration.load(self.path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_metadata_and_offset(self):
        self.assertEqual((self.cal.rows, self.cal.cols), (10, 20))
        self.assertAlmostEqual(self.cal.offset_kpa, 15.0)
        self.assertEqual(self.cal.valid_channels, POINTS)
        self.assertEqual(self.cal.point_count, POINTS * 3)
        self.assertEqual(self.cal.raw_point_count, POINTS * 3)

    def test_interpolation_and_clamping(self):
        self.assertAlmostEqual(self.cal.kpa(0, 10), 5.0)
        self.assertAlmostEqual(self.cal.kpa(0, 20), 25.0)
        self.assertAlmostEqual(self.cal.kpa(199, 15), 15.0)
        self.assertAlmostEqual(self.cal.kpa(0, 5), 2.5)
        self.assertAlmostEqual(self.cal.kpa(0, 1000), 45.0)
        self.assertEqual(self.cal.kpa(0, 0), 0.0)

    def test_force_and_contact_stats(self):
        values = [5.0] * POINTS
        self.assertAlmostEqual(
            self.cal.force_n(values, 2.25, 1.0), 2.25, places=6)
        maximum, count, centroid = self.cal.contact_stats(values, 1.0)
        self.assertAlmostEqual(maximum, 5.0)
        self.assertEqual(count, POINTS)
        self.assertAlmostEqual(centroid[0], 4.5)
        self.assertAlmostEqual(centroid[1], 9.5)

    def test_missing_file_rejected(self):
        with self.assertRaises(SensorError):
            RPPS200Calibration.load(self.path + ".missing")


class FrameParserTest(unittest.TestCase):
    def setUp(self):
        values = [i for i in range(POINTS)]
        values[0] = 0x1234
        self.values = values
        self.frame = make_frame(values)
        self.parser = RPPS200FrameParser()

    def test_roundtrip_and_mask(self):
        frames = self.parser.feed(self.frame)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0], 0x234)
        self.assertEqual(frames[0][1], 1)
        self.assertEqual(frames[0][199], 199)

    def test_junk_prefix_and_partial_feed(self):
        self.assertEqual(self.parser.feed(b"garbage" + self.frame[:100]), [])
        frames = self.parser.feed(self.frame[100:])
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0], 0x234)

    def test_bad_checksum_rejected_and_resync(self):
        broken = bytearray(self.frame)
        broken[20] ^= 0x01
        self.assertEqual(self.parser.feed(bytes(broken)), [])
        self.assertGreater(self.parser.rejected, 0)
        frames = self.parser.feed(self.frame)
        self.assertEqual(len(frames), 1)

    def test_two_frames(self):
        frames = self.parser.feed(self.frame + self.frame)
        self.assertEqual(len(frames), 2)


class SensorDecodeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        path = os.path.join(self.tempdir.name, "cal.csv")
        calibration_csv(path)
        self.cal = RPPS200Calibration.load(path)
        self.sensor = RPPS200Sensor(self.cal)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_build_reading_force(self):
        reading = self.sensor.build_reading([10] * POINTS, frame_index=7)
        self.assertEqual(reading.frame_index, 7)
        self.assertAlmostEqual(reading.force_n, 2.25, places=6)
        self.assertAlmostEqual(reading.max_kpa, 5.0)
        self.assertEqual(reading.contact_count, POINTS)

    def test_baseline_subtraction(self):
        self.sensor.set_baseline([5] * POINTS)
        reading = self.sensor.build_reading([10] * POINTS)
        self.assertEqual(reading.adc[0], 5)
        self.assertAlmostEqual(reading.force_n, 1.125, places=6)
        self.sensor.set_baseline([10] * POINTS)
        reading = self.sensor.build_reading([10] * POINTS)
        self.assertEqual(reading.adc[0], 0)
        self.assertAlmostEqual(reading.force_n, 0.0)

    def test_tare_requires_running_sensor(self):
        with self.assertRaises(SensorError):
            self.sensor.tare(frames=3)


if __name__ == "__main__":
    unittest.main()

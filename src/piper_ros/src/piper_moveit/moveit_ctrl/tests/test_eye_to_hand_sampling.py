import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from piper_eye_to_hand.sampling import (
    FORMAT, atomic_write_json, choose_stationary_frame, is_duplicate_pose, load_dataset,
)


class SamplingTests(unittest.TestCase):
    def frames(self):
        return [{"stamp": 10 + i * 0.11, "base_T_ee": np.eye(4).tolist(),
                 "camera_T_target": np.eye(4).tolist(), "joint_positions": [0.] * 6,
                 "reprojection_error_px": 0.2 + i * 0.01} for i in range(8)]

    def test_stationary_burst_retains_real_pair(self):
        frames = self.frames()
        chosen, quality = choose_stationary_frame(frames)
        self.assertIs(chosen, frames[0])
        self.assertEqual(quality["frame_count"], 8)

    def test_motion_is_rejected(self):
        frames = self.frames()
        frames[4]["base_T_ee"][0][3] = 0.01
        with self.assertRaises(ValueError):
            choose_stationary_frame(frames)

    def test_target_slip_is_rejected(self):
        frames = self.frames()
        frames[-1]["camera_T_target"][1][3] = 0.01
        with self.assertRaises(ValueError):
            choose_stationary_frame(frames)

    def test_joint_motion_even_if_ee_still(self):
        frames = self.frames()
        frames[-1]["joint_positions"][0] = 0.02
        with self.assertRaises(ValueError):
            choose_stationary_frame(frames)

    def test_repeated_and_short_bursts(self):
        frames = self.frames()
        frames[-1]["stamp"] = frames[-2]["stamp"]
        with self.assertRaises(ValueError):
            choose_stationary_frame(frames)
        with self.assertRaises(ValueError):
            choose_stationary_frame(self.frames()[:3])

    def test_near_duplicate(self):
        frame = self.frames()[0]
        other = copy.deepcopy(frame)
        other["base_T_ee"][0][3] += 0.003
        self.assertTrue(is_duplicate_pose([frame], other))
        other["base_T_ee"][0][3] += 0.02
        self.assertFalse(is_duplicate_pose([frame], other))

    def test_atomic_write_rejects_nan_without_losing_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            data = {"format": FORMAT, "samples": [], "board": {}, "frames": {},
                    "camera_info": {}, "camera_link_T_optical": np.eye(4).tolist()}
            atomic_write_json(path, data)
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"nan": float("nan")})
            self.assertEqual(load_dataset(path), data)


if __name__ == "__main__":
    unittest.main()

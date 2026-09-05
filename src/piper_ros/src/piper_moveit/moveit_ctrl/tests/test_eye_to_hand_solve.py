"""Offline CLI integration: data -> solve -> independent validation -> TF XML."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from piper_eye_to_hand.board import Board
from piper_eye_to_hand.geometry import invert_transform, quaternion_xyzw
from piper_eye_to_hand.sampling import FORMAT
from test_eye_to_hand_geometry import synthetic_samples, transform

spec = importlib.util.spec_from_file_location("eye_to_hand_solve", PACKAGE / "scripts" / "eye_to_hand_solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def dataset(seed, count):
    samples, _, _ = synthetic_samples(seed=seed, count=count)
    for i, sample in enumerate(samples):
        sample.update(id="{}-{}".format(seed, i), stamp=1000. + seed * 100 + i)
    return {
        "format": FORMAT, "board": Board().config,
        "frames": {"base": "base_link", "ee": "link6", "camera_link": "camera_link",
                   "camera_optical": "camera_color_optical_frame"},
        "camera_link_T_optical": transform([-1.2, .1, -.7], [.015, -.005, .004]).tolist(),
        "camera_info": {"K": [900., 0., 640., 0., 900., 360., 0., 0., 1.], "D": [0.] * 5,
                        "width": 1280, "height": 720, "distortion_model": "plumb_bob"},
        "samples": samples,
    }


class SolveCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = dataset(19, 24)
        self.validation = dataset(83, 7)
        self.write("train.json", self.train)
        self.write("validation.json", self.validation)

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data), encoding="utf-8")

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = solver.main(list(args))
        return code, output.getvalue()

    def solve(self):
        code, text = self.run_cli("solve", "--dataset", str(self.root / "train.json"),
                                  "--output", str(self.root / "result.json"))
        self.assertEqual(code, 0, text)
        return json.loads((self.root / "result.json").read_text())

    def validate(self):
        return self.run_cli("validate", "--dataset", str(self.root / "validation.json"),
                            "--calibration", str(self.root / "result.json"),
                            "--output", str(self.root / "report.json"),
                            "--launch-output", str(self.root / "camera.launch"))

    def test_end_to_end_exports_camera_link_with_correct_direction(self):
        result = self.solve()
        self.assertFalse((self.root / "camera.launch").exists())
        self.assertFalse(result["validated"])
        code, message = self.validate()
        self.assertEqual(code, 0, message)
        report = json.loads((self.root / "report.json").read_text())
        self.assertTrue(report["passed"])
        node = ET.parse(self.root / "camera.launch").getroot().find("node")
        args = node.attrib["args"].split()
        self.assertEqual(node.attrib["pkg"], "tf2_ros")
        self.assertEqual(args[-2:], ["base_link", "camera_link"])
        _, x, _ = synthetic_samples()
        expected = x @ invert_transform(self.train["camera_link_T_optical"])
        np.testing.assert_allclose([float(v) for v in args[:3]], expected[:3, 3], atol=1e-10)
        np.testing.assert_allclose([float(v) for v in args[3:7]], quaternion_xyzw(expected), atol=1e-10)

    def test_validation_rejects_training_reuse(self):
        self.solve()
        self.write("validation.json", self.train)
        code, message = self.validate()
        self.assertEqual(code, 1)
        self.assertIn("训练样本", message)
        self.assertFalse((self.root / "camera.launch").exists())

    def test_validation_bad_sample_report_without_launch(self):
        self.solve()
        self.validation["samples"][0]["camera_T_target"][0][3] += .04
        self.write("validation.json", self.validation)
        code, message = self.validate()
        self.assertEqual(code, 2, message)
        self.assertFalse((self.root / "camera.launch").exists())
        report = json.loads((self.root / "report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["translation"]["max"], .04, places=7)

    def test_camera_metadata_change_is_rejected(self):
        self.solve()
        self.validation["camera_info"]["K"][0] += 20
        self.write("validation.json", self.validation)
        self.assertEqual(self.validate()[0], 1)

    def test_existing_output_is_preserved(self):
        self.solve()
        old = (self.root / "result.json").read_bytes()
        code, _ = self.run_cli("solve", "--dataset", str(self.root / "train.json"),
                                "--output", str(self.root / "result.json"))
        self.assertEqual(code, 1)
        self.assertEqual((self.root / "result.json").read_bytes(), old)

    def test_validation_requires_five_new_poses(self):
        self.solve()
        self.validation["samples"] = self.validation["samples"][:4]
        self.write("validation.json", self.validation)
        self.assertEqual(self.validate()[0], 1)

    def test_bad_frame_and_nonfinite_input_rejected(self):
        self.train["frames"]["camera_link"] = "bad frame"
        self.write("train.json", self.train)
        code, _ = self.run_cli("solve", "--dataset", str(self.root / "train.json"),
                                "--output", str(self.root / "result.json"))
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()

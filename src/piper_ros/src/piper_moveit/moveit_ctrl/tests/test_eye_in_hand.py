"""ROS-free eye-in-hand geometry, acquisition metadata and CLI regression tests."""
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
from piper_eye_to_hand.calibration_cli import main as calibration_main, static_launch
from piper_eye_to_hand.collection import Collector, parse_args
from piper_eye_to_hand.geometry import (
    evaluate_eye_in_hand_samples, invert_transform, quaternion_xyzw, solve_eye_in_hand,
)
from piper_eye_to_hand.sampling import check_mode, load_dataset, mode_format
from test_eye_to_hand_geometry import transform


def synthetic_samples(seed=19, count=24, translation_noise=0., rotation_noise=0.,
                      camera_mount=None, target=None, poses=None):
    rng = np.random.RandomState(seed)
    x = (transform([.31, -.24, .51], [.045, -.026, .085])
         if camera_mount is None else camera_mount)
    y = (transform([.11, -.18, .24], [.37, -.1, .7]) if target is None else target)
    if poses is None:
        poses = [transform(rng.normal(0, .35, 3), rng.uniform(-.15, .15, 3))
                 for _ in range(count)]
    samples = []
    for i, base in enumerate(poses):
        camera = invert_transform(x) @ invert_transform(base) @ y
        camera = camera @ transform(rng.normal(0, rotation_noise, 3),
                                    rng.normal(0, translation_noise, 3))
        samples.append({"base_T_ee": base.tolist(), "camera_T_target": camera.tolist(),
                        "id": "{}-{}".format(seed, i), "stamp": 1000. + seed * 100 + i})
    return samples, x, y


def dataset(seed, count, **kwargs):
    samples, _, _ = synthetic_samples(seed, count, **kwargs)
    return {
        "format": mode_format("eye_in_hand"), "mode": "eye_in_hand",
        "board": Board().config,
        "frames": {"base": "base_link", "ee": "link6", "camera_link": "camera_link",
                   "camera_optical": "camera_color_optical_frame"},
        "camera_link_T_optical": transform([-1.2, .1, -.7], [.015, -.005, .004]).tolist(),
        "camera_info": {"K": [900., 0., 640., 0., 900., 360., 0., 0., 1.], "D": [0.] * 5,
                        "width": 1280, "height": 720, "distortion_model": "plumb_bob"},
        "samples": samples,
    }


class EyeInHandGeometryTests(unittest.TestCase):
    def test_exact_mount_direction_multiple_seeds(self):
        for seed in (1, 9, 37):
            samples, x, y = synthetic_samples(seed)
            result = solve_eye_in_hand(samples)
            np.testing.assert_allclose(result["ee_T_camera"], x, atol=1e-9)
            np.testing.assert_allclose(result["base_T_target"], y, atol=1e-9)
            self.assertFalse(np.allclose(result["ee_T_camera"], invert_transform(x)))
            self.assertLess(result["training_residuals"]["translation"]["max"], 1e-9)
            json.dumps(result, allow_nan=False)

    def test_noise_and_new_poses(self):
        samples, x, _ = synthetic_samples(count=36, translation_noise=.0005, rotation_noise=.001)
        original = copy.deepcopy(samples)
        result = solve_eye_in_hand(samples)
        self.assertEqual(samples, original)
        self.assertLess(np.linalg.norm(np.asarray(result["ee_T_camera"])[:3, 3] - x[:3, 3]), .003)
        validation, _, _ = synthetic_samples(seed=83, count=8)
        metrics = evaluate_eye_in_hand_samples(validation, result["ee_T_camera"], result["base_T_target"])
        self.assertLess(metrics["translation"]["max"], .003)
        self.assertLess(metrics["rotation"]["max"], .3)

    def test_moved_stationary_board_is_not_refitted_away(self):
        samples, x, y = synthetic_samples()
        result = solve_eye_in_hand(samples)
        moved_y = y @ transform([0, 0, np.deg2rad(8)], [.02, 0, 0])
        validation, _, _ = synthetic_samples(seed=83, count=7, target=moved_y)
        metrics = evaluate_eye_in_hand_samples(validation, result["ee_T_camera"], result["base_T_target"])
        self.assertAlmostEqual(metrics["translation"]["rms"], .02, places=9)
        self.assertAlmostEqual(metrics["rotation"]["rms"], 8., places=8)

    def test_camera_mount_slip_is_visible(self):
        samples, x, y = synthetic_samples()
        moved_x = x @ transform([0, .1, 0], [.025, 0, 0])
        validation, _, _ = synthetic_samples(seed=83, count=7, camera_mount=moved_x)
        metrics = evaluate_eye_in_hand_samples(validation, x, y)
        self.assertGreater(metrics["rotation"]["max"], 5.)

    def test_degenerate_rotation_rejected(self):
        rng = np.random.RandomState(7)
        for angles in (np.zeros(15), np.linspace(-.8, .8, 15)):
            poses = [transform([0, 0, angle], rng.uniform(-.2, .2, 3)) for angle in angles]
            samples, _, _ = synthetic_samples(poses=poses)
            with self.assertRaisesRegex(ValueError, "non-parallel"):
                solve_eye_in_hand(samples)

    def test_minimum_samples_and_invalid_matrix(self):
        samples, _, _ = synthetic_samples(count=11)
        with self.assertRaisesRegex(ValueError, "At least 12"):
            solve_eye_in_hand(samples)
        samples, _, _ = synthetic_samples()
        samples[0]["base_T_ee"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            solve_eye_in_hand(samples)


class EyeInHandCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train, self.validation = dataset(19, 24), dataset(83, 7)
        self.write("train.json", self.train)
        self.write("validation.json", self.validation)

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data), encoding="utf-8")

    def run_cli(self, *args, mode="eye_in_hand"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = calibration_main(list(args), mode=mode)
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

    def test_end_to_end_correct_tf_parent_and_optical_conversion(self):
        result = self.solve()
        self.assertEqual(result["tf_parent"], "link6")
        self.assertNotIn("base_T_camera", result)
        self.assertFalse(result["validated"])
        self.assertFalse((self.root / "camera.launch").exists())
        code, text = self.validate()
        self.assertEqual(code, 0, text)
        self.assertTrue(json.loads((self.root / "report.json").read_text())["passed"])
        node = ET.parse(self.root / "camera.launch").getroot().find("node")
        args = node.attrib["args"].split()
        self.assertEqual(args[-2:], ["link6", "camera_link"])
        self.assertEqual(node.attrib["name"], "piper_eye_in_hand_camera_tf")
        _, x, _ = synthetic_samples()
        expected = x @ invert_transform(self.train["camera_link_T_optical"])
        np.testing.assert_allclose([float(v) for v in args[:3]], expected[:3, 3], atol=1e-10)
        np.testing.assert_allclose([float(v) for v in args[3:7]], quaternion_xyzw(expected), atol=1e-10)
        np.testing.assert_allclose(result["ee_T_camera_link"], expected, atol=1e-10)

    def test_custom_mount_link_used_in_export(self):
        for data, name in ((self.train, "train.json"), (self.validation, "validation.json")):
            data["frames"]["ee"] = "link5"
            self.write(name, data)
        self.solve()
        self.assertEqual(self.validate()[0], 0)
        args = ET.parse(self.root / "camera.launch").getroot().find("node").attrib["args"].split()
        self.assertEqual(args[-2:], ["link5", "camera_link"])

    def test_failed_validation_writes_report_but_no_launch(self):
        self.solve()
        self.validation["samples"][0]["camera_T_target"][0][3] += .04
        self.write("validation.json", self.validation)
        code, text = self.validate()
        self.assertEqual(code, 2, text)
        self.assertFalse((self.root / "camera.launch").exists())
        self.assertFalse(json.loads((self.root / "report.json").read_text())["passed"])

    def test_training_reuse_rejected(self):
        self.solve()
        self.write("validation.json", self.train)
        self.assertEqual(self.validate()[0], 1)
        self.assertFalse((self.root / "camera.launch").exists())

    def test_wrong_mode_solve_and_resume_rejected_both_directions(self):
        with self.assertRaisesRegex(ValueError, "混用"):
            load_dataset(self.root / "train.json", expected_mode="eye_to_hand")
        self.assertEqual(self.run_cli("solve", "--dataset", str(self.root / "train.json"),
                                     "--output", str(self.root / "wrong.json"), mode="eye_to_hand")[0], 1)
        self.train["format"] = mode_format("eye_to_hand")
        self.train["mode"] = "eye_to_hand"
        self.write("train.json", self.train)
        with self.assertRaisesRegex(ValueError, "混用"):
            load_dataset(self.root / "train.json", expected_mode="eye_in_hand")
        self.assertEqual(self.run_cli("solve", "--dataset", str(self.root / "train.json"),
                                     "--output", str(self.root / "wrong.json"))[0], 1)
        self.assertFalse((self.root / "wrong.json").exists())

    def test_wrong_result_mode_cannot_export_or_validate(self):
        result = self.solve()
        with self.assertRaisesRegex(ValueError, "混用"):
            static_launch(result, mode="eye_to_hand")
        result["format"] = mode_format("eye_to_hand", "calibration")
        result["mode"] = "eye_to_hand"
        self.write("result.json", result)
        self.assertEqual(self.validate()[0], 1)
        self.assertFalse((self.root / "camera.launch").exists())

    def test_changed_mount_frame_rejected(self):
        self.solve()
        self.validation["frames"]["ee"] = "link5"
        self.write("validation.json", self.validation)
        self.assertEqual(self.validate()[0], 1)

    def test_existing_result_not_overwritten(self):
        self.solve()
        original = (self.root / "result.json").read_bytes()
        code, _ = self.run_cli("solve", "--dataset", str(self.root / "train.json"),
                                "--output", str(self.root / "result.json"))
        self.assertEqual(code, 1)
        self.assertEqual(original, (self.root / "result.json").read_bytes())


class EyeInHandCollectionTests(unittest.TestCase):
    def test_mode_specific_arguments_and_saved_metadata(self):
        for mode in ("eye_to_hand", "eye_in_hand"):
            args = parse_args(["--board", "board.json", "--output", "train.json"], mode=mode)
            self.assertEqual(args.preview_topic, "/{}/preview".format(mode))
            self.assertEqual(args.mode, mode)
            collector = Collector.__new__(Collector)  # no ROS imports/node/hardware
            collector.args, collector.board, collector.dataset = args, Board(), None
            candidate = dataset(19, 1)
            collector.check_metadata(candidate)
            check_mode(collector.dataset, mode)
            self.assertEqual(collector.dataset["mode"], mode)

    def test_entrypoints_select_eye_in_hand_without_ros(self):
        for entry in ("collect", "solve"):
            spec = importlib.util.spec_from_file_location(
                "eye_in_hand_" + entry, PACKAGE / "scripts" / ("eye_in_hand_" + entry + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
                module.main(["--help"])
            self.assertEqual(error.exception.code, 0)
            self.assertIn("eye_in_hand", output.getvalue())

    def test_conflicting_mode_field_rejected(self):
        data = dataset(19, 1)
        data["mode"] = "eye_to_hand"
        with self.assertRaisesRegex(ValueError, "混用"):
            check_mode(data, "eye_in_hand")


if __name__ == "__main__":
    unittest.main()

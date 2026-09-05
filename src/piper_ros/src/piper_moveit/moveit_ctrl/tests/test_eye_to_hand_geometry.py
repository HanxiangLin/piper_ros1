"""Deterministic, ROS-free direction, noise and degeneracy checks."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from piper_eye_to_hand.geometry import (  # noqa: E402
    evaluate_samples,
    invert_transform,
    quaternion_xyzw,
    solve_eye_to_hand,
)


def transform(rotation_vector, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float))[0]
    matrix[:3, 3] = translation
    return matrix


def synthetic_samples(seed=19, count=24, translation_noise=0.0, rotation_noise=0.0):
    rng = np.random.RandomState(seed)
    base_T_camera = transform([0.61, -0.35, 1.17], [0.52, -0.31, 0.77])
    ee_T_target = transform([-0.21, 0.34, 0.12], [0.028, -0.014, 0.12])
    samples = []
    for _ in range(count):
        base = transform(rng.normal(0.0, 0.50, 3), rng.uniform(-0.30, 0.30, 3))
        camera = invert_transform(base_T_camera) @ base @ ee_T_target
        if translation_noise or rotation_noise:
            camera = camera @ transform(rng.normal(0.0, rotation_noise, 3),
                                        rng.normal(0.0, translation_noise, 3))
        samples.append({"base_T_ee": base.tolist(), "camera_T_target": camera.tolist()})
    return samples, base_T_camera, ee_T_target


def samples_with_robot_poses(poses):
    _, base_T_camera, ee_T_target = synthetic_samples(count=1)
    return [{"base_T_ee": base.tolist(), "camera_T_target": (
        invert_transform(base_T_camera) @ base @ ee_T_target).tolist()} for base in poses]


class EyeToHandGeometryTests(unittest.TestCase):
    def test_exact_recovers_correct_transform_direction(self):
        for seed in (1, 9, 37):
            samples, expected_x, expected_y = synthetic_samples(seed=seed)
            result = solve_eye_to_hand(samples)
            np.testing.assert_allclose(result["base_T_camera"], expected_x, atol=1.0e-9)
            np.testing.assert_allclose(result["ee_T_target"], expected_y, atol=1.0e-9)
            self.assertFalse(np.allclose(result["base_T_camera"], invert_transform(expected_x)))
            self.assertEqual(result["method"], "PARK")
            self.assertLess(result["training_residuals"]["translation"]["max"], 1.0e-9)
            self.assertLess(result["training_residuals"]["rotation"]["max"], 1.0e-7)
            json.dumps(result, allow_nan=False)

    def test_noisy_solution_and_independent_validation(self):
        samples, expected_x, expected_y = synthetic_samples(
            count=36, translation_noise=0.0005, rotation_noise=0.001)
        original = copy.deepcopy(samples)
        result = solve_eye_to_hand(samples)
        self.assertEqual(samples, original)
        self.assertLess(np.linalg.norm(np.asarray(result["base_T_camera"])[:3, 3]
                                      - expected_x[:3, 3]), 0.003)
        validation, _, _ = synthetic_samples(seed=83, count=10)
        metrics = evaluate_samples(validation, result["base_T_camera"], result["ee_T_target"])
        self.assertLess(metrics["translation"]["max"], 0.003)
        self.assertLess(metrics["rotation"]["max"], 0.3)
        self.assertEqual(metrics["sample_count"], 10)
        self.assertEqual(len(result["training_residuals"]["per_sample"]), len(samples))

    def test_bad_validation_sample_visible_without_refit(self):
        samples, expected_x, expected_y = synthetic_samples(count=12)
        result = solve_eye_to_hand(samples)
        validation, _, _ = synthetic_samples(seed=83, count=5)
        validation[2]["camera_T_target"][0][3] += 0.12
        metrics = evaluate_samples(validation, result["base_T_camera"], result["ee_T_target"])
        self.assertAlmostEqual(metrics["translation"]["max"], 0.12, places=9)
        self.assertAlmostEqual(metrics["translation"]["rms"], 0.12 / math.sqrt(5), places=9)
        self.assertAlmostEqual(metrics["per_sample"][2]["translation_error_m"], 0.12, places=9)
        # A target that moved on the gripper must not be 'fixed' by refitting Y.
        moved_y = expected_y @ transform([0, 0, math.radians(8)], [0.02, 0, 0])
        validation, _, _ = synthetic_samples(seed=71, count=7)
        for sample in validation:
            sample["camera_T_target"] = (invert_transform(expected_x)
                                         @ np.asarray(sample["base_T_ee"]) @ moved_y).tolist()
        moved = evaluate_samples(validation, expected_x, expected_y)
        self.assertAlmostEqual(moved["translation"]["rms"], 0.02, places=10)
        self.assertAlmostEqual(moved["rotation"]["rms"], 8.0, places=9)

    def test_training_outlier_is_not_silently_discarded(self):
        samples, _, _ = synthetic_samples()
        samples[7]["camera_T_target"][1][3] += 0.15
        result = solve_eye_to_hand(samples)
        metrics = result["training_residuals"]
        self.assertEqual(result["sample_count"], 24)
        self.assertEqual(len(metrics["per_sample"]), 24)
        self.assertGreater(metrics["per_sample"][7]["translation_error_m"], 0.10)

    def test_rejects_translation_only_and_single_rotation_axis(self):
        rng = np.random.RandomState(4)
        for angles in (np.zeros(15), np.linspace(-1.0, 1.0, 15)):
            poses = [transform([0, 0, angle], rng.uniform(-0.5, 0.5, 3)) for angle in angles]
            with self.assertRaisesRegex(ValueError, "non-parallel"):
                solve_eye_to_hand(samples_with_robot_poses(poses))

    def test_rejects_tiny_rotation_span_and_nearly_parallel_axes(self):
        rng = np.random.RandomState(7)
        tiny = [transform(rng.normal(0, 0.001, 3), [0, 0, 0]) for _ in range(15)]
        with self.assertRaisesRegex(ValueError, "rotation span"):
            solve_eye_to_hand(samples_with_robot_poses(tiny))
        parallel = [transform([0.0001 * math.sin(a), 0.0, a], [0, 0, 0])
                    for a in np.linspace(-0.8, 0.8, 15)]
        with self.assertRaisesRegex(ValueError, "singular-value ratio"):
            solve_eye_to_hand(samples_with_robot_poses(parallel))

    def test_two_independent_rotation_axes_are_sufficient(self):
        poses = ([transform([angle, 0, 0], [0.1, 0.2, 0.3])
                  for angle in np.linspace(-0.7, 0.7, 8)]
                 + [transform([0, angle, 0], [0.1, 0.2, 0.3])
                    for angle in np.linspace(-0.7, 0.7, 8)])
        result = solve_eye_to_hand(samples_with_robot_poses(poses))
        self.assertGreater(result["excitation"]["rotation_axis_ratio"], 0.1)
        self.assertAlmostEqual(result["excitation"]["translation_span_m"], 0.0)
        self.assertLess(result["training_residuals"]["translation"]["max"], 1.0e-9)

    def test_rejects_missing_samples_keys_and_invalid_se3(self):
        samples, _, _ = synthetic_samples(count=12)
        with self.assertRaisesRegex(ValueError, "12 samples"):
            solve_eye_to_hand(samples[:11])
        for minimum in (True, 2, 3.1):
            with self.assertRaisesRegex(ValueError, "integer >= 3"):
                solve_eye_to_hand(samples, min_samples=minimum)
        broken = copy.deepcopy(samples)
        del broken[0]["camera_T_target"]
        with self.assertRaisesRegex(ValueError, "missing camera_T_target"):
            solve_eye_to_hand(broken)
        bad_matrices = []
        for row, col, value in ((0, 3, float("nan")), (1, 1, float("inf")),
                                (3, 0, 1.0), (0, 0, 2.0), (0, 0, -1.0)):
            matrix = np.eye(4)
            matrix[row, col] = value
            bad_matrices.append(matrix.tolist())
        bad_matrices.extend([np.eye(3).tolist(), [["no"]] * 4])
        for bad in bad_matrices:
            with self.subTest(matrix=bad):
                broken = copy.deepcopy(samples)
                broken[0]["base_T_ee"] = bad
                with self.assertRaises(ValueError):
                    solve_eye_to_hand(broken)

    def test_evaluate_accepts_one_sample_but_rejects_empty_or_invalid_transform(self):
        samples, expected_x, expected_y = synthetic_samples(count=1)
        self.assertEqual(evaluate_samples(samples, expected_x, expected_y)["sample_count"], 1)
        with self.assertRaises(ValueError):
            evaluate_samples([], expected_x, expected_y)
        expected_y[3, 3] = 0
        with self.assertRaisesRegex(ValueError, "bottom row"):
            evaluate_samples(samples, expected_x, expected_y)

    def test_inverse_and_quaternion_helpers(self):
        for rotation_vector in ([0, 0, 0], [0.2, -0.4, 1.2], [math.pi, 0, 0],
                                [0, -math.pi, 0], [0, 0, math.pi]):
            matrix = transform(rotation_vector, [0.5, -0.8, 0.9])
            np.testing.assert_allclose(matrix @ invert_transform(matrix), np.eye(4), atol=1.0e-12)
            quaternion = np.asarray(quaternion_xyzw(matrix))
            np.testing.assert_allclose(quaternion_xyzw(matrix[:3, :3]), quaternion)
            self.assertGreaterEqual(quaternion[3], 0.0)
            self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0)
            if np.linalg.norm(quaternion[:3]) > 1.0e-10:
                axis = quaternion[:3] / np.linalg.norm(quaternion[:3])
                angle = 2.0 * math.atan2(np.linalg.norm(quaternion[:3]), quaternion[3])
                np.testing.assert_allclose(cv2.Rodrigues(axis * angle)[0], matrix[:3, :3], atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()

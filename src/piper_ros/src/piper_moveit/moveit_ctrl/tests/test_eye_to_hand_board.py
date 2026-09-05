"""No-camera, no-ROS tests for board geometry, detection, and printable output."""

import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from piper_eye_to_hand.board import Board, annotate

SPEC = importlib.util.spec_from_file_location("make_board", str(PACKAGE / "scripts" / "eye_to_hand_make_board.py"))
MAKE_BOARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAKE_BOARD)


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.board = Board()
        self.K = np.array([[1050., 0, 640.], [0, 1040., 480.], [0, 0, 1.]])
        self.D = np.zeros(5)
        self.rvec = np.array([0.22, -0.15, 0.04])
        self.tvec = np.array([-0.082, -0.0605, 0.5])

    def synthetic_image(self, hidden_ids=(), swap_first_two=False):
        ppm, margin = 5000., 0.01
        source = self.board.render(ppm, margin)
        side = int(round(self.board.config["marker_length"] * ppm))
        for marker_id in hidden_ids:
            x, y = np.round((self.board.marker_corners(marker_id)[0, :2] + margin) * ppm).astype(int)
            source[y:y + side, x:x + side] = 255
        if swap_first_two:
            for target_id, printed_id in ((0, 1), (1, 0)):
                x, y = np.round((self.board.marker_corners(target_id)[0, :2] + margin) * ppm).astype(int)
                source[y:y + side, x:x + side] = self.board.marker_image(printed_id, side)
        world_corners = np.array([[-margin, -margin, 0],
                                  [self.board.width + margin, -margin, 0],
                                  [self.board.width + margin, self.board.height + margin, 0],
                                  [-margin, self.board.height + margin, 0]], dtype=np.float64)
        destination, _ = cv2.projectPoints(world_corners, self.rvec, self.tvec, self.K, self.D)
        corners = np.array([[0, 0], [source.shape[1], 0],
                            [source.shape[1], source.shape[0]], [0, source.shape[0]]], np.float32)
        homography = cv2.getPerspectiveTransform(corners, destination.reshape(4, 2).astype(np.float32))
        return cv2.warpPerspective(source, homography, (1280, 960), borderValue=255)

    def test_metric_geometry_and_json_roundtrip(self):
        self.assertAlmostEqual(self.board.width, 0.164)
        self.assertAlmostEqual(self.board.height, 0.121)
        np.testing.assert_allclose(self.board.marker_corners(0)[0], [0, 0, 0])
        np.testing.assert_allclose(self.board.marker_corners(11)[2], [0.164, 0.121, 0])
        restored = Board(json.loads(json.dumps(self.board.config)))
        self.assertEqual(restored.config, self.board.config)

    def test_perspective_detection_recovers_camera_T_target(self):
        image = self.synthetic_image()
        result = self.board.detect(image, self.K, self.D)
        self.assertIsNotNone(result)
        self.assertEqual(result["marker_count"], 12)
        self.assertLess(result["reprojection_error_px"], 1.0)
        transform = np.array(result["camera_T_target"])
        np.testing.assert_allclose(transform[:3, 3], self.tvec, atol=0.002)
        expected_rotation, _ = cv2.Rodrigues(self.rvec)
        np.testing.assert_allclose(transform[:3, :3], expected_rotation, atol=0.02)
        self.assertEqual(annotate(image, result).shape, (960, 1280, 3))
        json.dumps(result)

    def test_partial_board_with_four_markers_is_accepted(self):
        result = self.board.detect(self.synthetic_image(hidden_ids=range(4, 12)), self.K, self.D)
        self.assertIsNotNone(result)
        self.assertEqual(result["marker_count"], 4)

    def test_fewer_than_four_markers_are_rejected(self):
        self.assertIsNone(self.board.detect(self.synthetic_image(hidden_ids=range(3, 12)), self.K, self.D))
        self.assertIsNone(self.board.detect(np.full((480, 640), 255, np.uint8), self.K, self.D))

    def test_inconsistent_marker_positions_are_rejected(self):
        self.assertIsNone(self.board.detect(self.synthetic_image(swap_first_two=True), self.K, self.D))

    def test_behind_camera_pose_is_rejected(self):
        with mock.patch("cv2.solvePnP", return_value=(True, np.zeros((3, 1)), np.array([[0.], [0.], [-0.5]]))):
            self.assertIsNone(self.board.detect(self.synthetic_image(), self.K, self.D))

    def test_invalid_configuration_is_rejected(self):
        for config in ({"marker_length": -0.1}, {"separation": 0}, {"markers_x": 4.5},
                       {"first_id": 49}, {"min_markers": 3}, {"markers_x": 1, "markers_y": 3}):
            with self.assertRaises(ValueError):
                Board(config)

    def test_fractional_pixel_dimensions_render_without_clipping(self):
        for millimetres, gap in ((22, 7), (23, 3), (35, 8)):
            board = Board({"marker_length": millimetres / 1000., "separation": gap / 1000.})
            ppm = 150. / 0.0254
            image = board.render(ppm, margin=0)
            self.assertEqual(image.shape, (round(board.height * ppm), round(board.width * ppm)))
            # Last marker's physical outer border must survive pixel rounding.
            self.assertEqual(int(image[-1, -1]), 0)

    def test_pdf_a4_dimensions_marker_cells_and_ruler(self):
        pdf = MAKE_BOARD.make_pdf(self.board).decode("latin1")
        page = re.search(r"/MediaBox \[0 0 ([0-9.]+) ([0-9.]+)\]", pdf)
        self.assertIsNotNone(page)
        ppm = MAKE_BOARD.POINTS_PER_METRE
        self.assertAlmostEqual(float(page.group(1)) / ppm, 0.210, places=8)
        self.assertAlmostEqual(float(page.group(2)) / ppm, 0.297, places=8)
        rectangles = re.findall(r"([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+) re f", pdf)
        cells = self.board.dictionary.markerSize + 2
        for _, _, width, height in rectangles:
            self.assertAlmostEqual(float(width) / ppm * cells, 0.035, places=8)
            self.assertAlmostEqual(float(height) / ppm * cells, 0.035, places=8)
        lines = re.findall(r"([0-9.]+) ([0-9.]+) m ([0-9.]+) ([0-9.]+) l S", pdf)
        horizontal = [(float(x2) - float(x1)) / ppm for x1, y1, x2, y2 in lines if y1 == y2]
        self.assertEqual(len(horizontal), 1)
        self.assertAlmostEqual(horizontal[0], 0.100, places=8)
        self.assertIn("Print at 100%", pdf)
        self.assertTrue(pdf.endswith("%%EOF\n"))

    def test_cli_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(MAKE_BOARD.main(["--output-dir", directory]), 0)
            paths = [Path(directory) / name for name in ("board.pdf", "board.png", "board.json")]
            before = [path.read_bytes() for path in paths]
            self.assertEqual(Board(paths[2]).config, self.board.config)
            image = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(image.shape, (1754, 1240))
            self.assertIsNotNone(self.board.detect(image, self.K, self.D))
            self.assertEqual(MAKE_BOARD.main(["--output-dir", directory]), 1)
            self.assertEqual(before, [path.read_bytes() for path in paths])


if __name__ == "__main__":
    unittest.main()

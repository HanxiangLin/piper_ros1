"""Metric ArUco GridBoard geometry and detection, without a ROS dependency.

The target frame is fixed at the top-left outer corner of the printed board.
Looking at its printed face, X points right, Y points down, and Z points into
the paper. All object coordinates and returned translations are in metres.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "schema_version": 1,
    "board_type": "aruco_grid",
    "dictionary": "DICT_4X4_50",
    "markers_x": 4,
    "markers_y": 3,
    "marker_length": 0.035,
    "separation": 0.008,
    "first_id": 0,
    "min_markers": 4,
    "max_reprojection_error_px": 2.0,
}


class Board:
    """A row-major GridBoard with a stable frame across OpenCV versions.

    ``config`` is a dictionary, a board.json path, or None for the default.
    ``detect`` returns JSON-compatible pose/quality data, or None when the
    board cannot be detected reliably. Invalid input/configuration raises
    ValueError. Detection is supported by OpenCV 4.2 and its newer ArUco API.
    """

    def __init__(self, config=None):
        if isinstance(config, (str, Path)):
            with open(str(config), "r", encoding="utf-8") as stream:
                config = json.load(stream)
        if config is not None and not isinstance(config, dict):
            raise ValueError("Board configuration must be a dictionary or JSON path")
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config or {})
        if self.config["schema_version"] != 1:
            raise ValueError("Unsupported board schema_version")
        if self.config["board_type"] != "aruco_grid":
            raise ValueError("board_type must be aruco_grid")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV with the aruco module is required")
        for name in ("markers_x", "markers_y", "first_id", "min_markers"):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("{} must be an integer".format(name))
        if self.config["markers_x"] < 1 or self.config["markers_y"] < 1:
            raise ValueError("The board must have positive row and column counts")
        self.count = self.config["markers_x"] * self.config["markers_y"]
        if not 4 <= self.config["min_markers"] <= self.count:
            raise ValueError("min_markers must be between 4 and the board marker count")
        for name in ("marker_length", "separation", "max_reprojection_error_px"):
            value = float(self.config[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be finite and positive".format(name))
            self.config[name] = value
        dictionary_name = self.config["dictionary"]
        if not isinstance(dictionary_name, str) or not dictionary_name.startswith("DICT_"):
            raise ValueError("dictionary must be an OpenCV DICT_* name")
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError("Unknown ArUco dictionary: {}".format(dictionary_name))
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.first_id = self.config["first_id"]
        if self.first_id < 0 or self.first_id + self.count > len(self.dictionary.bytesList):
            raise ValueError("Requested marker IDs do not fit the dictionary")
        length = self.config["marker_length"]
        gap = self.config["separation"]
        self.width = self.config["markers_x"] * length + (self.config["markers_x"] - 1) * gap
        self.height = self.config["markers_y"] * length + (self.config["markers_y"] - 1) * gap
        # Derived dimensions are recomputed; a stale JSON field cannot rescale geometry.
        self.config["width_m"] = self.width
        self.config["height_m"] = self.height
        self.config["frame_convention"] = "top_left_x_right_y_down_z_into_paper"
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = (cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                         if hasattr(cv2.aruco, "ArucoDetector") else None)

    def marker_corners(self, marker_id):
        """Return a marker's TL, TR, BR, BL corners in the metric target frame."""
        index = int(marker_id) - self.first_id
        if not 0 <= index < self.count:
            raise ValueError("Marker ID is not part of this board")
        row, column = divmod(index, self.config["markers_x"])
        length = self.config["marker_length"]
        pitch = length + self.config["separation"]
        x, y = column * pitch, row * pitch
        return np.array([[x, y, 0], [x + length, y, 0],
                         [x + length, y + length, 0], [x, y + length, 0]],
                        dtype=np.float64)

    def marker_image(self, marker_id, side_pixels):
        """Generate a marker including its one-cell black border."""
        if not self.first_id <= marker_id < self.first_id + self.count:
            raise ValueError("Marker ID is not part of this board")
        if hasattr(cv2.aruco, "generateImageMarker"):
            return cv2.aruco.generateImageMarker(self.dictionary, marker_id,
                                               int(side_pixels), borderBits=1)
        return cv2.aruco.drawMarker(self.dictionary, marker_id, int(side_pixels),
                                   borderBits=1)

    def render(self, pixels_per_metre=5000.0, margin=0.01):
        """Render a grayscale board preview; the PDF is the print-size authority."""
        if not math.isfinite(pixels_per_metre) or pixels_per_metre <= 0 or margin < 0:
            raise ValueError("Invalid render scale or margin")
        width = int(round((self.width + 2 * margin) * pixels_per_metre))
        height = int(round((self.height + 2 * margin) * pixels_per_metre))
        side = int(round(self.config["marker_length"] * pixels_per_metre))
        if side < self.dictionary.markerSize + 2:
            raise ValueError("Render resolution is too small for the dictionary")
        result = np.full((height, width), 255, dtype=np.uint8)
        for marker_id in range(self.first_id, self.first_id + self.count):
            corner = self.marker_corners(marker_id)[0]
            x = int(round((corner[0] + margin) * pixels_per_metre))
            y = int(round((corner[1] + margin) * pixels_per_metre))
            x_end = int(round((corner[0] + margin + self.config["marker_length"]) * pixels_per_metre))
            y_end = int(round((corner[1] + margin + self.config["marker_length"]) * pixels_per_metre))
            # Round both physical edges, so a fractional pixel pitch cannot
            # overrun the last row/column when rendering without a margin.
            marker = self.marker_image(marker_id, side)
            result[y:y_end, x:x_end] = cv2.resize(marker, (x_end - x, y_end - y),
                                                 interpolation=cv2.INTER_NEAREST)
        return result

    def detect(self, image, K, D=None):
        """Estimate camera_T_target from matching markers and reject poor poses.

        At least min_markers complete, distinct board markers are required.
        Every detected board marker participates in the solve and residual
        checks; inconsistent IDs/corners are rejected rather than silently
        accepted as a different target. An undistorted image must use the
        matching rectified K and zero D, not the original distortion values.
        """
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim not in (2, 3) or not image.size:
            raise ValueError("image must be a nonempty uint8 grayscale/BGR/BGRA image")
        if image.ndim == 3:
            if image.shape[2] not in (3, 4):
                raise ValueError("Expected BGR or BGRA channels")
            conversion = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            gray = cv2.cvtColor(image, conversion)
        else:
            gray = image
        K = np.asarray(K, dtype=np.float64)
        D = np.asarray(D if D is not None else np.zeros(5), dtype=np.float64).reshape(-1)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K must be a finite 3x3 camera matrix with positive focal lengths")
        if D.size not in (0, 4, 5, 8, 12, 14) or not np.all(np.isfinite(D)):
            raise ValueError("D must contain 0, 4, 5, 8, 12, or 14 finite coefficients")
        D = D if D.size else np.zeros(5, dtype=np.float64)
        if self.detector is None:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary,
                                                    parameters=self.parameters)
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        observations = {}
        for marker_id, marker_corners in zip(ids.reshape(-1), corners):
            marker_id = int(marker_id)
            if not self.first_id <= marker_id < self.first_id + self.count:
                continue
            if marker_id in observations:
                return None
            observations[marker_id] = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        if len(observations) < self.config["min_markers"]:
            return None
        marker_ids = sorted(observations)
        object_points = np.concatenate([self.marker_corners(i) for i in marker_ids])
        image_points = np.concatenate([observations[i] for i in marker_ids])
        if not np.all(np.isfinite(image_points)):
            return None
        try:
            success, rotation_vector, translation = cv2.solvePnP(
                object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success or not np.all(np.isfinite(rotation_vector)) or not np.all(np.isfinite(translation)):
                return None
            rotation, _ = cv2.Rodrigues(rotation_vector)
            camera_points = (rotation.dot(object_points.T) + translation.reshape(3, 1)).T
            if np.any(camera_points[:, 2] <= 0):
                return None
            projected, _ = cv2.projectPoints(object_points, rotation_vector, translation, K, D)
        except cv2.error:
            return None
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        limit = self.config["max_reprojection_error_px"]
        if not math.isfinite(rms) or rms > limit or float(np.max(residuals)) > 3 * limit:
            return None
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation.reshape(3)
        return {
            "camera_T_target": transform.tolist(),
            "reprojection_error_px": rms,
            "marker_count": len(marker_ids),
            "marker_ids": marker_ids,
            "image_points": image_points.tolist(),
            "object_points": object_points.tolist(),
        }

    @staticmethod
    def annotate(image, result):
        return annotate(image, result)


def annotate(image, result):
    """Return a BGR image with detected marker boundaries and quality text."""
    output = np.asarray(image).copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif output.shape[2] == 4:
        output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
    if result is None:
        label, color = "Board not accepted", (0, 0, 255)
    else:
        points = np.asarray(result["image_points"], dtype=np.float32).reshape(-1, 1, 4, 2)
        ids = np.asarray(result["marker_ids"], dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(output, list(points), ids)
        label = "{} markers | reprojection RMS {:.2f} px".format(
            result["marker_count"], result["reprojection_error_px"])
        color = (0, 160, 0)
    cv2.putText(output, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return output

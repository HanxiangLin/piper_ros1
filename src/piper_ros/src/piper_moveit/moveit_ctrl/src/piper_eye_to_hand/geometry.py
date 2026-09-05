"""ROS-independent eye-to-hand calibration geometry (metres and radians).

Transform notation ``a_T_b`` maps coordinates from frame b into frame a.
For a fixed external camera and a target rigidly fixed to the end effector::

    base_T_ee[i] @ ee_T_target = base_T_camera @ camera_T_target[i]

The camera frame must be the optical frame used by the target pose estimator.
No sample is silently removed. Excitation thresholds are numerical safeguards,
not claims about calibration accuracy or clearance for hardware operation.
"""

import math

import cv2
import numpy as np


_SE3_TOLERANCE = 1.0e-5
_EXCITATION_THRESHOLDS = {
    "min_max_relative_rotation_deg": 15.0,
    "min_second_axis_rms_deg": 3.0,
    "min_rotation_axis_ratio": 0.1,
}


def _rotation_matrix(value, name="rotation"):
    try:
        rotation = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain numeric values".format(name)) from exc
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("{} must be a finite 3x3 rotation matrix".format(name))
    if not np.allclose(rotation.T @ rotation, np.eye(3),
                       rtol=0.0, atol=_SE3_TOLERANCE):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if not math.isclose(float(np.linalg.det(rotation)), 1.0,
                        rel_tol=0.0, abs_tol=_SE3_TOLERANCE):
        raise ValueError("{} rotation determinant must be +1".format(name))
    return rotation


def _transform(value, name="transform"):
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain numeric values".format(name)) from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("{} must be a finite 4x4 SE(3) matrix".format(name))
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0],
                       rtol=0.0, atol=_SE3_TOLERANCE):
        raise ValueError("{} bottom row must be [0, 0, 0, 1]".format(name))
    _rotation_matrix(matrix[:3, :3], name)
    return matrix


def invert_transform(matrix):
    """Return the inverse of a validated SE(3) transform as an ndarray."""
    matrix = _transform(matrix)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def quaternion_xyzw(matrix_or_rotation):
    """Return a unit ROS-order quaternion [x, y, z, w], choosing w >= 0."""
    try:
        matrix = np.asarray(matrix_or_rotation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("quaternion input must contain numeric values") from exc
    rotation = (_transform(matrix)[:3, :3] if matrix.shape == (4, 4)
                else _rotation_matrix(matrix))
    vector = cv2.Rodrigues(rotation)[0].reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return [0.0, 0.0, 0.0, 1.0]
    xyz = vector * (math.sin(angle / 2.0) / angle)
    quaternion = np.r_[xyz, math.cos(angle / 2.0)]
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion.tolist()


def _samples(samples, minimum):
    if not isinstance(samples, (list, tuple)):
        raise ValueError("samples must be a list or tuple of sample dictionaries")
    if len(samples) < minimum:
        raise ValueError("At least {} samples are required; received {}".format(
            minimum, len(samples)))
    pairs = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("Sample {} must be a dictionary".format(index))
        try:
            base = _transform(sample["base_T_ee"],
                              "sample {} base_T_ee".format(index))
            camera = _transform(sample["camera_T_target"],
                                "sample {} camera_T_target".format(index))
        except KeyError as exc:
            raise ValueError("Sample {} is missing {}".format(index, exc.args[0])) from exc
        pairs.append((base, camera))
    return pairs


def _rotation_error_deg(rotation):
    # atan2 avoids the loss of precision of acos(trace(...)) near identity.
    sine = 0.5 * float(np.linalg.norm([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]))
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def _statistics(values, unit):
    values = np.asarray(values, dtype=np.float64)
    return {
        "unit": unit,
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def evaluate_samples(samples, base_T_camera, ee_T_target):
    """Evaluate B_i Y = X C_i using fixed X and Y, without refitting either.

    Translation residuals are distances in metres between target origins in
    the base frame; rotation residuals are geodesic angles in degrees. This
    function accepts one or more independent validation samples and does not
    apply the calibration motion-excitation test.
    """
    pairs = _samples(samples, minimum=1)
    base_T_camera = _transform(base_T_camera, "base_T_camera")
    ee_T_target = _transform(ee_T_target, "ee_T_target")
    per_sample = []
    for index, (base_T_ee, camera_T_target) in enumerate(pairs):
        robot_target = base_T_ee @ ee_T_target
        camera_target = base_T_camera @ camera_T_target
        translation_error = float(np.linalg.norm(
            robot_target[:3, 3] - camera_target[:3, 3]))
        rotation_error = _rotation_error_deg(
            robot_target[:3, :3].T @ camera_target[:3, :3])
        per_sample.append({
            "index": index,
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error,
        })
    return {
        "sample_count": len(pairs),
        "translation": _statistics(
            [row["translation_error_m"] for row in per_sample], "m"),
        "rotation": _statistics(
            [row["rotation_error_deg"] for row in per_sample], "deg"),
        "per_sample": per_sample,
    }


def _check_excitation(pairs):
    """Check relative robot rotations expressed in the common base frame."""
    rotation_vectors = []
    translation_spans = []
    for i, (base_i, _) in enumerate(pairs):
        for base_j, _ in pairs[i + 1:]:
            # OpenCV input A_i = inv(B_i); relative A_j^-1 A_i = B_j B_i^-1.
            relative_rotation = base_j[:3, :3] @ base_i[:3, :3].T
            rotation_vectors.append(cv2.Rodrigues(relative_rotation)[0].reshape(3))
            translation_spans.append(float(np.linalg.norm(
                base_j[:3, 3] - base_i[:3, 3])))
    vectors = np.asarray(rotation_vectors, dtype=np.float64)
    singular_values = np.linalg.svd(vectors, compute_uv=False) / math.sqrt(len(vectors))
    max_angle = math.degrees(float(np.max(np.linalg.norm(vectors, axis=1))))
    second_axis = math.degrees(float(singular_values[1]))
    axis_ratio = (float(singular_values[1] / singular_values[0])
                  if singular_values[0] > 1.0e-12 else 0.0)
    excitation = {
        "pair_count": len(rotation_vectors),
        "rotation_singular_values_rad": singular_values.tolist(),
        "rotation_axis_ratio": axis_ratio,
        "max_relative_rotation_deg": max_angle,
        "second_axis_rms_deg": second_axis,
        "translation_span_m": max(translation_spans),
        "thresholds": dict(_EXCITATION_THRESHOLDS),
    }
    reasons = []
    if max_angle < _EXCITATION_THRESHOLDS["min_max_relative_rotation_deg"]:
        reasons.append("relative rotation span {:.2f} deg < {:.2f} deg".format(
            max_angle, _EXCITATION_THRESHOLDS["min_max_relative_rotation_deg"]))
    if second_axis < _EXCITATION_THRESHOLDS["min_second_axis_rms_deg"]:
        reasons.append("second rotation-axis RMS {:.2f} deg < {:.2f} deg".format(
            second_axis, _EXCITATION_THRESHOLDS["min_second_axis_rms_deg"]))
    if axis_ratio < _EXCITATION_THRESHOLDS["min_rotation_axis_ratio"]:
        reasons.append("rotation-axis singular-value ratio {:.3f} < {:.3f}".format(
            axis_ratio, _EXCITATION_THRESHOLDS["min_rotation_axis_ratio"]))
    if reasons:
        raise ValueError(
            "Insufficient rotational excitation; use rotations about at least two "
            "non-parallel axes. " + "; ".join(reasons))
    return excitation


def _mean_transform(transforms):
    result = np.eye(4, dtype=np.float64)
    rotations = np.asarray([matrix[:3, :3] for matrix in transforms])
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    result[:3, :3] = u @ correction @ vt
    result[:3, 3] = np.mean([matrix[:3, 3] for matrix in transforms], axis=0)
    return result


def solve_eye_to_hand(samples, min_samples=12):
    """Solve fixed-camera calibration and return a JSON-serializable report.

    Each sample has ``base_T_ee`` and ``camera_T_target`` as 4x4 nested lists.
    Returns X = ``base_T_camera`` and Y = ``ee_T_target``. OpenCV's usual
    eye-in-hand equation is A_i X C_i = constant. Supplying A_i = inv(B_i)
    therefore yields exactly B_i Y = X C_i; the result must NOT be inverted.

    All samples are used with the PARK method. Y is the SE(3) chordal mean of
    inv(B_i) X C_i. Large residuals are reported, not hidden by deleting rows;
    callers must set task-appropriate acceptance limits and validate on new
    samples before publishing a calibrated transform for hardware use.
    """
    if (isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer))
            or min_samples < 3):
        raise ValueError("min_samples must be an integer >= 3")
    pairs = _samples(samples, minimum=int(min_samples))
    excitation = _check_excitation(pairs)
    inverse_bases = [invert_transform(base) for base, _ in pairs]
    try:
        rotation, translation = cv2.calibrateHandEye(
            [base[:3, :3].copy() for base in inverse_bases],
            [base[:3, 3].reshape(3, 1).copy() for base in inverse_bases],
            [camera[:3, :3].copy() for _, camera in pairs],
            [camera[:3, 3].reshape(3, 1).copy() for _, camera in pairs],
            method=cv2.CALIB_HAND_EYE_PARK,
        )
    except cv2.error as exc:
        raise ValueError("OpenCV PARK eye-to-hand solve failed: {}".format(exc)) from exc
    base_T_camera = np.eye(4, dtype=np.float64)
    base_T_camera[:3, :3] = rotation
    base_T_camera[:3, 3] = np.asarray(translation).reshape(3)
    _transform(base_T_camera, "OpenCV PARK base_T_camera result")
    ee_T_target = _mean_transform([
        inverse_base @ base_T_camera @ camera
        for inverse_base, (_, camera) in zip(inverse_bases, pairs)
    ])
    _transform(ee_T_target, "ee_T_target result")
    return {
        "base_T_camera": base_T_camera.tolist(),
        "ee_T_target": ee_T_target.tolist(),
        "method": "PARK",
        "sample_count": len(pairs),
        "training_residuals": evaluate_samples(samples, base_T_camera, ee_T_target),
        "excitation": excitation,
    }

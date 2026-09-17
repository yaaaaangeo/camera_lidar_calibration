"""Does Edge Alignment actually react to a wrong extrinsic?

A metric that reports roughly the same number whether the extrinsic is exactly
right or visibly wrong cannot be trusted to validate anything -- so this file
pushes a known-correct T away from the truth in small, controlled steps (yaw,
then translation) and checks that the reported pixel error moves with it.

Goes through the real `gui.core.verify.project_cloud` (not a hand-rolled
projection) so the test exercises the exact function Step 7 calls, including
its behind-camera / off-image / distortion-limit protections.
"""

from __future__ import annotations

import cv2
import numpy as np

from gui.core import verify
from gui.core.evaluation.edge_alignment import EdgeAlignmentParams, evaluate_edge_alignment
from gui.core.project import Camera
from gui.core.solve import Solution

_WIDTH, _HEIGHT = 640, 480
_FX = _FY = 500.0
_CX, _CY = 320.0, 240.0


def _make_scene():
    """A depth step in 3D that lines up exactly with a drawn vertical image
    edge under the identity transform -- the LiDAR frame is built to coincide
    with the camera frame, so R=I, t=0 is "correct" by construction."""
    image = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    image[:, int(_CX):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Spacing must stay under the default radius_px=3.0 in both axes, or
    # neighbouring grid points straddling the depth step never see each other
    # and no edge is ever flagged.
    u = np.repeat(np.linspace(0, _WIDTH - 1, 250), 200)
    v = np.tile(np.linspace(0, _HEIGHT - 1, 200), 250)
    z = np.where(u < _CX, 5.0, 10.0)
    x = (u - _CX) * z / _FX
    y = (v - _CY) * z / _FY
    cloud = np.column_stack([x, y, z])
    return image, cloud


def _yaw(deg: float) -> np.ndarray:
    """Rotation about the camera's vertical (Y) axis -- a classic mounting-
    angle error, and the one that shifts a vertical edge horizontally."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _mean_error(image, cloud, R, t) -> float:
    sol = Solution(ok=True, R=R, t=t, rmse=0.0, scene_ids=[], n_pairs=0)
    camera = Camera(fx=_FX, fy=_FY, cx=_CX, cy=_CY)
    pr = verify.project_cloud(cloud, sol, camera, _WIDTH, _HEIGHT)
    result = evaluate_edge_alignment(image, pr.uv, pr.depth, EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert result.ok, result.reason
    return result.mean_px


def test_correct_extrinsic_has_low_error():
    image, cloud = _make_scene()
    err = _mean_error(image, cloud, np.eye(3), np.zeros(3))
    assert err < 3.0, f"mean_px={err}"


def _assert_generally_increasing(magnitudes, errors):
    """The spec asks that error "대체로" (generally) rises with the
    perturbation, not that every single step beats the last -- at
    sub-pixel-to-a-few-pixel perturbations, Canny/PCA quantization noise in a
    synthetic scene this coarse is comparable to the shift itself, so strict
    step-by-step monotonicity is not a fair bar. A positive correlation
    between perturbation size and error, plus a clearly larger error at the
    largest perturbation than at zero, is: a metric that does not respond to
    the extrinsic at all would show ~0 or negative correlation here.
    """
    corr = float(np.corrcoef(magnitudes, errors)[0, 1])
    assert corr > 0.5, (magnitudes, errors, corr)
    assert errors[-1] > errors[0], errors


def test_error_increases_with_yaw_perturbation():
    image, cloud = _make_scene()
    degrees = (0.0, 0.1, 0.2, 0.5)
    errors = [_mean_error(image, cloud, _yaw(deg), np.zeros(3)) for deg in degrees]
    _assert_generally_increasing(degrees, errors)


def test_error_increases_with_translation_perturbation():
    image, cloud = _make_scene()
    tx_values = (0.0, 0.005, 0.010, 0.020)
    errors = [_mean_error(image, cloud, np.eye(3), np.array([tx, 0.0, 0.0])) for tx in tx_values]
    _assert_generally_increasing(tx_values, errors)

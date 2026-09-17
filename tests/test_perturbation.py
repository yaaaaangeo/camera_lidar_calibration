"""Unit + end-to-end tests for gui.core.evaluation.perturbation.

Split roughly into two halves: pure arithmetic (composition, mm->m, the
axis-summary logic) that needs no geometry at all, and a smaller set of
geometry-backed tests reusing the depth-step synthetic scene already proven
to behave well under yaw/tx in test_synthetic_extrinsic_sensitivity.py.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

import gui.core.evaluation.perturbation as pert_mod
from gui.core.evaluation.edge_alignment import EdgeAlignmentParams
from gui.core.evaluation.perturbation import (
    AXIS_LABELS,
    DEFAULT_ROTATION_DELTAS_DEG,
    DEFAULT_TRANSLATION_DELTAS_MM,
    QUICK_ROTATION_DELTAS_DEG,
    QUICK_TRANSLATION_DELTAS_MM,
    ROTATION_AXES,
    TRANSLATION_AXES,
    AxisSensitivity,
    FrameMetric,
    PerturbationPoint,
    _pair_frame_metrics,
    _rotation_delta,
    evaluate_perturbation_grid,
    evaluate_single_axis,
    frame_from_current,
    make_delta_range,
    perturb_rotation,
    perturb_translation,
    prepare_frames,
    summarize_axis,
)
from gui.core.project import Camera
from gui.core.solve import Solution

# --------------------------------------------------------------- composition


def test_perturb_rotation_yaw_matches_independent_ry():
    """yaw perturbation must be Ry(delta) @ R -- left-multiplied, about the
    camera's own Y (down) axis, per the module's documented convention."""
    theta = np.radians(12.0)
    c, s = np.cos(theta), np.sin(theta)
    expected_ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    R = np.eye(3)
    perturbed = perturb_rotation(R, "yaw", 12.0)
    assert np.allclose(perturbed, expected_ry)
    # Left-multiplication, not right: with a non-identity R the two differ.
    R2 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about Z
    left = expected_ry @ R2
    right = R2 @ expected_ry
    assert not np.allclose(left, right)
    assert np.allclose(perturb_rotation(R2, "yaw", 12.0), left)


def test_perturb_rotation_pitch_and_roll_axes():
    theta = np.radians(5.0)
    c, s = np.cos(theta), np.sin(theta)
    expected_rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])  # pitch: camera X
    expected_rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # roll: camera Z

    assert np.allclose(perturb_rotation(np.eye(3), "pitch", 5.0), expected_rx)
    assert np.allclose(perturb_rotation(np.eye(3), "roll", 5.0), expected_rz)


def test_perturb_rotation_does_not_mutate_input():
    R = np.eye(3)
    R_copy = R.copy()
    perturb_rotation(R, "yaw", 30.0)
    assert np.array_equal(R, R_copy)


def test_perturb_translation_mm_to_metre_conversion():
    t = np.array([1.0, 2.0, 3.0])
    out = perturb_translation(t, "tx", 10.0)
    assert np.isclose(out[0], 1.0 + 0.010)
    assert out[1] == 2.0 and out[2] == 3.0

    out_tz = perturb_translation(t, "tz", -5.0)
    assert np.isclose(out_tz[2], 3.0 - 0.005)


def test_perturb_translation_does_not_mutate_input():
    t = np.array([1.0, 2.0, 3.0])
    t_copy = t.copy()
    perturb_translation(t, "ty", 20.0)
    assert np.array_equal(t, t_copy)


# ------------------------------------------------------------- summarize_axis


def _point(delta, p95, ok=True, n_valid=10):
    return PerturbationPoint(delta=delta, ok=ok, p95_px=p95, median_px=p95 * 0.8, n_valid_frames=n_valid)


def test_summarize_axis_deltas_and_sign_convention():
    baseline = _point(0.0, 5.0)
    worse = _point(0.1, 6.0)
    better = _point(-0.1, 4.5)
    axis = summarize_axis("yaw", "deg", [worse, baseline, better], baseline, local_min_tolerance_px=0.05)

    assert [p.delta for p in axis.points] == [-0.1, 0.0, 0.1]  # sorted
    assert worse.delta_p95_px > 0    # positive = worse than baseline
    assert better.delta_p95_px < 0   # negative = improvement over baseline
    assert baseline.delta_p95_px == 0.0


def test_summarize_axis_finds_lowest_and_improvement():
    baseline = _point(0.0, 5.0)
    candidate = _point(-0.1, 4.0)
    axis = summarize_axis("yaw", "deg", [baseline, candidate], baseline, local_min_tolerance_px=0.05)
    assert axis.lowest_point is candidate
    assert np.isclose(axis.improvement_p95_px, 1.0)
    assert not axis.is_local_minimum


def test_summarize_axis_local_minimum_within_tolerance():
    baseline = _point(0.0, 5.0)
    tiny_improvement = _point(0.1, 4.98)  # 0.02px better -- noise, not a real improvement
    axis = summarize_axis("tx", "mm", [baseline, tiny_improvement], baseline, local_min_tolerance_px=0.05)
    assert axis.is_local_minimum
    assert axis.lowest_point is tiny_improvement  # still factually the lowest tested point
    assert axis.improvement_p95_px < 0.05


def test_summarize_axis_local_minimum_false_beyond_tolerance():
    baseline = _point(0.0, 5.0)
    real_improvement = _point(0.1, 4.9)  # 0.1px -- beyond the 0.05 tolerance
    axis = summarize_axis("tx", "mm", [baseline, real_improvement], baseline, local_min_tolerance_px=0.05)
    assert not axis.is_local_minimum


def test_summarize_axis_ignores_failed_points_for_lowest():
    baseline = _point(0.0, 5.0)
    failed = _point(0.1, 0.5, ok=False)  # a lower p95 number, but not a real result
    axis = summarize_axis("yaw", "deg", [baseline, failed], baseline, local_min_tolerance_px=0.05)
    assert axis.lowest_point is baseline
    assert axis.is_local_minimum


def test_summarize_axis_flags_coverage_difference():
    baseline = _point(0.0, 5.0, n_valid=10)
    fewer_frames = _point(0.1, 5.5, n_valid=7)
    summarize_axis("yaw", "deg", [baseline, fewer_frames], baseline, local_min_tolerance_px=0.05)
    assert fewer_frames.coverage_warning != ""
    assert baseline.coverage_warning == ""


# -------------------------------------------------------------- prepare_frames


def test_prepare_frames_marks_sync_rejected_and_skips_missing():
    calls = []

    def loader(idx):
        calls.append(idx)
        if idx == 2:
            return None  # missing entirely
        camera_t = idx if idx != 4 else idx + 100_000_000  # frame 4: 100ms off
        image = cv2.cvtColor(np.zeros((60, 80), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        cloud = np.zeros((10, 3), dtype=np.float32)
        return image, cloud, idx, float(idx), camera_t

    frames = prepare_frames(loader, n_total_timeline=5, n_samples=5, max_sync_offset_ms=50.0)
    assert len(calls) == 5
    assert len(frames) == 4  # idx=2 dropped entirely (missing)
    rejected = {f.timeline_index: f.sync_rejected for f in frames}
    assert rejected[4] is True
    assert all(v is False for k, v in rejected.items() if k != 4)
    assert next(f.cache for f in frames if f.timeline_index == 4) is None


def test_frame_from_current_reuses_exact_objects_not_copies():
    """Current-Frame mode's identity-based pairing guarantee: the LoadedFrame
    must hold the very same array objects passed in, not copies re-fetched or
    re-derived some other way."""
    image = cv2.cvtColor(np.zeros((60, 80), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    cloud = np.zeros((10, 3), dtype=np.float32)
    frame = frame_from_current(image, cloud)
    assert frame.image_bgr is image
    assert frame.cloud_xyz is cloud
    assert not frame.sync_rejected
    assert frame.cache is not None


# -------------------------------------------------------- geometry end-to-end
#
# Reuses the depth-step scene already proven (in
# test_synthetic_extrinsic_sensitivity.py) to behave well under yaw/tx.

_WIDTH, _HEIGHT = 640, 480
_FX = _FY = 500.0
_CX, _CY = 320.0, 240.0


def _make_scene():
    image = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    image[:, int(_CX):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    u = np.repeat(np.linspace(0, _WIDTH - 1, 250), 200)
    v = np.tile(np.linspace(0, _HEIGHT - 1, 200), 250)
    z = np.where(u < _CX, 5.0, 10.0)
    x = (u - _CX) * z / _FX
    y = (v - _CY) * z / _FY
    return image, np.column_stack([x, y, z]).astype(np.float32)


def _camera() -> Camera:
    return Camera(fx=_FX, fy=_FY, cx=_CX, cy=_CY)


def _frames_from_single_scene(image, cloud, n=3):
    """A tiny "multi-frame" set: the same scene repeated, standing in for
    prepare_frames' output without needing a fake bag."""
    return prepare_frames(
        lambda idx: (image, cloud, idx, float(idx), idx),  # perfectly synced
        n_total_timeline=n, n_samples=n,
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
    )


def test_evaluate_perturbation_grid_baseline_matches_direct_evaluation():
    from gui.core import verify
    from gui.core.evaluation.edge_alignment import evaluate_edge_alignment

    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    camera = _camera()
    R, t = np.eye(3), np.zeros(3)

    sol = Solution(ok=True, R=R, t=t, rmse=0.0, scene_ids=[], n_pairs=0)
    pr = verify.project_cloud(cloud, sol, camera, _WIDTH, _HEIGHT)
    direct = evaluate_edge_alignment(image, pr.uv, pr.depth, EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    result = evaluate_perturbation_grid(
        frames, R, t, camera, edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
    )
    baseline = result.axes["yaw"].baseline
    assert np.isclose(baseline.mean_px, direct.mean_px)
    assert np.isclose(baseline.p95_px, direct.p95_px)


def test_evaluate_perturbation_grid_never_mutates_baseline_arrays():
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    camera = _camera()
    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()

    evaluate_perturbation_grid(frames, R, t, camera, edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert np.array_equal(R, R_before)
    assert np.array_equal(t, t_before)


def test_evaluate_perturbation_grid_yaw_error_generally_increases():
    """The same "generally increases" check used in
    test_synthetic_extrinsic_sensitivity.py -- here exercised through the full
    grid/pooling/spatial machinery instead of a single direct call."""
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    camera = _camera()

    result = evaluate_perturbation_grid(
        frames, np.eye(3), np.zeros(3), camera,
        rotation_deltas_deg=(0.0, 0.1, 0.2, 0.5),
        translation_deltas_mm=(0.0,),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
    )
    yaw = result.axes["yaw"]
    deltas = [p.delta for p in yaw.points]
    p95s = [p.p95_px for p in yaw.points]
    corr = float(np.corrcoef(deltas, p95s)[0, 1])
    assert corr > 0.5, (deltas, p95s, corr)
    assert yaw.points[-1].p95_px > yaw.baseline.p95_px

    # Spatial breakdown must exist and cover all three horizontal regions.
    assert yaw.baseline.spatial is not None
    for label in ("LEFT", "CENTER", "RIGHT"):
        assert label in yaw.baseline.spatial.horizontal


def test_evaluate_perturbation_grid_reuses_same_frame_set_for_every_trial():
    """prepare_frames-equivalent loading must happen once, not once per
    trial -- checked by counting frame_loader calls, then running a grid with
    several deltas and confirming the loader was not called again."""
    image, cloud = _make_scene()
    calls = []

    def loader(idx):
        calls.append(idx)
        return image, cloud, idx, float(idx), idx

    frames = prepare_frames(loader, n_total_timeline=3, n_samples=3,
                             edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert len(calls) == 3
    calls.clear()

    evaluate_perturbation_grid(
        frames, np.eye(3), np.zeros(3), _camera(),
        rotation_deltas_deg=(0.0, 0.1, 0.2),
        translation_deltas_mm=(0.0, 5.0),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
    )
    assert calls == []  # the loader was never touched again during the grid run


def test_evaluate_perturbation_grid_sync_rejected_frame_excluded_from_every_trial():
    image, cloud = _make_scene()

    def loader(idx):
        camera_t = idx if idx != 1 else idx + 100_000_000  # frame 1: 100ms off
        return image, cloud, idx, float(idx), camera_t

    frames = prepare_frames(loader, n_total_timeline=3, n_samples=3, max_sync_offset_ms=50.0,
                             edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert sum(f.sync_rejected for f in frames) == 1

    # yaw and translation only -- pitch, on this single-plane synthetic scene,
    # has no vertical depth structure to be sensitive to and can legitimately
    # fail to find enough edge points at any nonzero delta; that is a known
    # property of this minimal scene (see test_perturb_rotation_pitch_and_roll_axes
    # for pitch's own composition check), not what this test is about.
    result = evaluate_perturbation_grid(
        frames, np.eye(3), np.zeros(3), _camera(),
        rotation_deltas_deg=(0.0, 0.1),
        translation_deltas_mm=(0.0, 5.0),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
    )
    assert "yaw" in result.axes and "tx" in result.axes
    for axis in result.axes.values():
        for p in axis.points:
            assert p.n_sync_rejected_frames == 1
            assert p.n_valid_frames + p.n_failed_frames == 2  # the 2 usable frames only


def test_evaluate_perturbation_grid_can_be_cancelled_mid_run():
    """Real-thread cancellation, mirroring the Multi-frame Consistency test:
    a slow frame set, cancelled from another thread while the grid is mid-run."""
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    camera = _camera()
    cancel_event = threading.Event()
    progress_calls = []

    def progress(done, total):
        progress_calls.append(done)
        time.sleep(0.01)  # simulate a slow trial so the cancel has time to land mid-run

    result_holder: dict = {}

    def run():
        result_holder["result"] = evaluate_perturbation_grid(
            frames, np.eye(3), np.zeros(3), camera,
            edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
            progress=progress, should_cancel=cancel_event.is_set,
        )

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.05)
    cancel_event.set()
    worker.join(timeout=10.0)

    assert not worker.is_alive()
    result = result_holder["result"]
    assert result.cancelled
    total_trials = 3 * 9 + 3 * 7  # default deltas
    assert len(progress_calls) < total_trials


# --------------------------------------------------- independence: R vs t
#
# A rotation-axis trial must project with the ORIGINAL t, and a translation-
# axis trial must project with the ORIGINAL R -- neither this module's own
# "Independent Extrinsic Parameter Sensitivity" definition, nor a true SE(3)
# composition, perturbs both from a single-axis nudge. Verified by spying on
# _evaluate_trial (the only place R/t actually reach the projector) rather
# than re-deriving the composition math again.


def _spy_evaluate_trial(monkeypatch, calls: list):
    original = pert_mod._evaluate_trial

    def spy(frames, R, t, camera, min_range, max_range, edge_params):
        calls.append((np.array(R, copy=True), np.array(t, copy=True)))
        return original(frames, R, t, camera, min_range, max_range, edge_params)

    monkeypatch.setattr(pert_mod, "_evaluate_trial", spy)


def test_rotation_trial_leaves_translation_untouched(monkeypatch):
    calls: list = []
    _spy_evaluate_trial(monkeypatch, calls)
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    baseline_t = np.array([0.01, 0.02, 0.03])

    pert_mod._run_trial_point(
        "rotation", "yaw", 0.1, frames, np.eye(3), baseline_t, _camera(),
        0.0, 0.0, EdgeAlignmentParams(depth_jump_threshold_m=1.0), 0,
    )
    assert len(calls) == 1
    used_R, used_t = calls[0]
    assert np.array_equal(used_t, baseline_t)          # untouched
    assert not np.allclose(used_R, np.eye(3))          # R WAS perturbed


def test_translation_trial_leaves_rotation_untouched(monkeypatch):
    calls: list = []
    _spy_evaluate_trial(monkeypatch, calls)
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    baseline_R = np.eye(3)

    pert_mod._run_trial_point(
        "translation", "tx", 10.0, frames, baseline_R, np.zeros(3), _camera(),
        0.0, 0.0, EdgeAlignmentParams(depth_jump_threshold_m=1.0), 0,
    )
    assert len(calls) == 1
    used_R, used_t = calls[0]
    assert np.array_equal(used_R, baseline_R)          # untouched
    assert not np.allclose(used_t, np.zeros(3))        # t WAS perturbed


# ------------------------------------------------ camera-axis label mapping


def test_axis_labels_cover_every_axis_and_name_a_camera_axis():
    for axis in (*ROTATION_AXES, *TRANSLATION_AXES):
        assert axis in AXIS_LABELS
    for axis in ROTATION_AXES:
        assert "Cam-" in AXIS_LABELS[axis]
    for axis, cam_axis in (("tx", "Cam-X"), ("ty", "Cam-Y"), ("tz", "Cam-Z")):
        assert cam_axis in AXIS_LABELS[axis]


def test_rotation_axis_labels_match_the_actual_matrix_mapping():
    """A rotation about an axis leaves that axis's own basis vector fixed --
    ties each label's claimed camera axis to what _rotation_delta actually
    does, rather than trusting the two to stay in sync by hand."""
    e_x, e_y, e_z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    assert "Cam-Y" in AXIS_LABELS["yaw"]
    assert np.allclose(_rotation_delta("yaw", 37.0) @ e_y, e_y)
    assert "Cam-X" in AXIS_LABELS["pitch"]
    assert np.allclose(_rotation_delta("pitch", 37.0) @ e_x, e_x)
    assert "Cam-Z" in AXIS_LABELS["roll"]
    assert np.allclose(_rotation_delta("roll", 37.0) @ e_z, e_z)


# ------------------------------------------ pooled vs frame-balanced metrics


def _fm(ok, p95=float("nan"), median=float("nan")) -> FrameMetric:
    return FrameMetric(ok=ok, p95_px=p95, median_px=median)


def test_pooled_and_frame_balanced_metrics_are_computed_independently():
    """median_frame_p95_px must come from per_frame alone, never from
    whatever p95_px (pooled) happens to hold -- set them to deliberately
    inconsistent values and confirm summarize_axis doesn't conflate them."""
    baseline = PerturbationPoint(
        delta=0.0, ok=True, p95_px=999.0, median_px=999.0,  # pooled: deliberately absurd
        per_frame={1: _fm(True, 2.0, 1.5), 2: _fm(True, 4.0, 3.0)},
    )
    axis = summarize_axis("yaw", "deg", [baseline], baseline, local_min_tolerance_px=0.05)
    assert baseline.median_frame_p95_px == 3.0  # median(2.0, 4.0), independent of pooled 999.0
    assert axis.baseline.p95_px == 999.0         # pooled value left alone


def test_frame_balanced_ranking_is_not_dominated_by_one_outlier_frame():
    """3 frames, baseline uniform; a candidate improves 2 of them a lot and
    gets one very bad frame. Point-pooling weighted by edge-point count could
    let that one bad (or point-heavy) frame swing a pooled average, but the
    per-frame MEDIAN treats every frame equally regardless of how many points
    it contributed, so the candidate should still rank as better."""
    baseline = PerturbationPoint(
        delta=0.0, ok=True, p95_px=2.0, median_px=2.0,
        per_frame={1: _fm(True, 2.0), 2: _fm(True, 2.0), 3: _fm(True, 2.0)},
    )
    candidate = PerturbationPoint(
        delta=0.1, ok=True, p95_px=2.0, median_px=2.0,  # pooled left identical on purpose
        per_frame={1: _fm(True, 1.0), 2: _fm(True, 1.0), 3: _fm(True, 100.0)},
    )
    axis = summarize_axis(
        "yaw", "deg", [baseline, candidate], baseline,
        local_min_tolerance_px=0.05, ranking_metric="median_frame_p95",
    )
    assert candidate.median_frame_p95_px == 1.0  # median(1, 1, 100) -- robust to the outlier
    assert axis.lowest_point is candidate
    assert axis.improvement_p95_px > 0  # candidate ranked better under median_frame_p95


# ----------------------------------------------------- paired frame delta


def test_pair_frame_metrics_pairs_same_frame_only():
    baseline = {1: _fm(True, 5.0), 2: _fm(True, 3.0)}
    candidate = {1: _fm(True, 4.0), 2: _fm(True, 10.0)}
    paired = _pair_frame_metrics(baseline, candidate, tolerance=0.05)
    # frame 1: 4-5=-1 (improved), frame 2: 10-3=+7 (worsened) -- never 10-5 or 4-3
    assert paired["n_comparable_frames"] == 2
    assert paired["n_improved_frames"] == 1
    assert paired["n_worsened_frames"] == 1
    assert paired["median_delta_frame_p95_px"] == 3.0  # median(-1, 7)


def test_candidate_frame_failure_is_counted_not_dropped():
    baseline = {1: _fm(True, 5.0), 2: _fm(True, 5.0), 3: _fm(True, 5.0)}
    candidate = {1: _fm(True, 4.0), 2: _fm(False), 3: _fm(True, 4.0)}
    paired = _pair_frame_metrics(baseline, candidate, tolerance=0.05)
    assert paired["n_comparable_frames"] == 2
    assert paired["n_baseline_only_valid"] == 1   # frame 2: baseline ok, candidate failed
    assert paired["n_candidate_only_valid"] == 0
    assert paired["n_both_failed"] == 0
    # A candidate cannot look artificially good by failing its hard frames --
    # the failure is visible in n_baseline_only_valid, not silently absent.
    assert paired["n_comparable_frames"] + paired["n_baseline_only_valid"] == len(baseline)


# --------------------------------------------------- Quick / Full grid size


def test_quick_grid_runs_exactly_13_unique_trials():
    calls = []
    original = pert_mod._evaluate_trial

    def counting_spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    import unittest.mock as mock
    with mock.patch.object(pert_mod, "_evaluate_trial", side_effect=counting_spy):
        evaluate_perturbation_grid(
            frames, np.eye(3), np.zeros(3), _camera(),
            rotation_deltas_deg=QUICK_ROTATION_DELTAS_DEG,
            translation_deltas_mm=QUICK_TRANSLATION_DELTAS_MM,
            edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        )
    # 1 shared baseline + 2 nonzero deltas x 3 rotation axes + 2 nonzero deltas x 3 translation axes
    assert len(calls) == 1 + 2 * 3 + 2 * 3 == 13


def test_full_grid_defaults_are_unchanged():
    """Regression guard: Quick mode is additive: Full must still use exactly
    the same default delta sets as before this feature existed."""
    assert DEFAULT_ROTATION_DELTAS_DEG == (-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5)
    assert DEFAULT_TRANSLATION_DELTAS_MM == (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)


def test_baseline_is_computed_exactly_once_across_full_grid():
    import unittest.mock as mock

    calls = []
    original = pert_mod._evaluate_trial

    def counting_spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    with mock.patch.object(pert_mod, "_evaluate_trial", side_effect=counting_spy):
        evaluate_perturbation_grid(
            frames, np.eye(3), np.zeros(3), _camera(),
            edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        )
    # 9 rotation deltas x 3 axes + 7 translation deltas x 3 axes, each set
    # sharing one delta=0.0 -> (9-1)*3 + (7-1)*3 + 1 shared baseline = 43.
    expected = (len(DEFAULT_ROTATION_DELTAS_DEG) - 1) * 3 + (len(DEFAULT_TRANSLATION_DELTAS_MM) - 1) * 3 + 1
    assert len(calls) == expected == 43


# --------------------------------------------------------------- fine scan


def test_make_delta_range_basic():
    assert make_delta_range(-0.3, 0.1, 0.05) == (-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1)


def test_make_delta_range_forces_zero_even_off_grid():
    values = make_delta_range(0.1, 0.5, 0.1)
    assert 0.0 in values
    assert values[0] == 0.0
    assert np.isclose(values[-1], 0.5)


def test_evaluate_single_axis_fine_scan_matches_grid_at_shared_deltas():
    """Fine Scan must score a delta identically to the Full/Quick grid --
    both go through the same _run_trial_point, so this is really a check that
    evaluate_single_axis wires it the same way, not a second implementation."""
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    camera = _camera()
    params = EdgeAlignmentParams(depth_jump_threshold_m=1.0)

    grid = evaluate_perturbation_grid(
        frames, np.eye(3), np.zeros(3), camera,
        rotation_deltas_deg=(0.0, 0.1), translation_deltas_mm=(0.0,), edge_params=params,
    )
    fine = evaluate_single_axis(
        frames, np.eye(3), np.zeros(3), camera, axis="yaw",
        deltas=make_delta_range(-0.3, 0.1, 0.1), edge_params=params,
    )
    grid_01 = next(p for p in grid.axes["yaw"].points if p.delta == 0.1)
    fine_01 = next(p for p in fine.points if p.delta == 0.1)
    assert np.isclose(grid_01.p95_px, fine_01.p95_px)


def test_evaluate_single_axis_can_be_cancelled():
    image, cloud = _make_scene()
    frames = _frames_from_single_scene(image, cloud, n=1)
    counter = {"n": 0}

    def should_cancel():
        counter["n"] += 1
        return counter["n"] > 2

    axis = evaluate_single_axis(
        frames, np.eye(3), np.zeros(3), _camera(), axis="tx",
        deltas=make_delta_range(-20.0, 20.0, 5.0),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        should_cancel=should_cancel,
    )
    assert len(axis.points) <= 2

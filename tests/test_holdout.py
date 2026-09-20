"""Tests for gui.core.evaluation.holdout (Independent Hold-out Validation).

Split into several groups:
  - CandidateSpec / overlap / independence partition: pure data/arithmetic,
    no geometry.
  - Bag signature comparison and cross-bag time-range note: pure data, no I/O
    (see test_bag_reader.py for the I/O-adjacent path_size_bytes tests).
  - Block bootstrap: pure arithmetic over hand-built (timestamp, delta)
    pairs -- proves block-level, not individual-frame, resampling.
  - evaluate_holdout, geometry-backed: reuses the same depth-step synthetic
    scene proven in test_synthetic_extrinsic_sensitivity.py / test_perturbation.py.
  - evaluate_holdout, pairing surfaced correctly: run_trial_point is
    monkeypatched to return hand-built PerturbationPoints, so baseline-only /
    candidate-only / both-failed bookkeeping, and exposure-status grading in
    diagnostics, can be checked exactly without needing to engineer geometry
    that fails, or improves, for one side only.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

import gui.core.evaluation.holdout as ho_mod
import gui.core.evaluation.perturbation as pert_mod
from gui.core.bag_reader import BagSignature
from gui.core.evaluation.edge_alignment import EdgeAlignmentParams
from gui.core.evaluation.holdout import (
    CandidateSpec,
    OverlapCheck,
    SpatialComparisonRow,
    _compare_bins,
    check_overlap,
    evaluate_holdout,
)
from gui.core.evaluation.perturbation import FrameMetric, LoadedFrame, PerturbationPoint, prepare_frames
from gui.core.evaluation.spatial_analysis import BinStats
from gui.core.project import Camera

# ------------------------------------------------------------- CandidateSpec


def test_candidate_spec_rejects_zero_delta():
    with pytest.raises(ValueError):
        CandidateSpec(axis="yaw", delta=0.0, source="fine")


def test_candidate_spec_rejects_unknown_axis():
    with pytest.raises(ValueError):
        CandidateSpec(axis="bogus", delta=0.1, source="fine")


def test_candidate_spec_rejects_unknown_source():
    with pytest.raises(ValueError):
        CandidateSpec(axis="yaw", delta=0.1, source="grid")


def test_candidate_spec_unit_and_kind():
    rot = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    assert rot.unit == "deg" and rot.kind == "rotation"
    trans = CandidateSpec(axis="tx", delta=10.0, source="quick")
    assert trans.unit == "mm" and trans.kind == "translation"


def test_candidate_spec_is_frozen():
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    with pytest.raises(Exception):
        cand.delta = 0.2  # type: ignore[misc]


# ------------------------------------------------------------------ overlap
#
# Realistic nanosecond-scale timestamps throughout (seconds * 1e9), since the
# default exclusion margin is 2.0 seconds -- tiny fake integers like [1,2,3]
# would all be "within margin" of each other and defeat the point of these
# tests. Exact-equality behaviour (the pre-guard-band rule) is exercised by
# passing exclusion_margin_s=0.0 explicitly.

SEC = 1_000_000_000  # ns per second, for readable test timestamps


def test_check_overlap_exact_duplicate_at_zero_margin():
    result = check_overlap("C:/data/run1.bag", [10 * SEC], "C:/data/run1.bag", [10 * SEC], exclusion_margin_s=0.0)
    assert result.same_source
    assert result.overlapping
    assert result.overlap_timestamps == (10 * SEC,)


def test_check_overlap_different_exact_timestamps_at_zero_margin_is_clean():
    result = check_overlap("C:/data/run1.bag", [10 * SEC], "C:/data/run1.bag", [10 * SEC + 1], exclusion_margin_s=0.0)
    assert result.same_source
    assert not result.overlapping


def test_check_overlap_within_guard_band_is_flagged():
    """Discovery at 10.000s, Hold-out at 10.033s -- different exact
    timestamps, but well inside the default 2.0s guard band: adjacent video
    frames from a moving vehicle can be nearly the same scene."""
    result = check_overlap("C:/data/run1.bag", [10 * SEC], "C:/data/run1.bag", [10 * SEC + int(0.033 * SEC)])
    assert result.same_source
    assert result.overlapping
    assert result.exclusion_margin_s == ho_mod.DEFAULT_EXCLUSION_MARGIN_S


def test_check_overlap_outside_guard_band_is_usable():
    result = check_overlap("C:/data/run1.bag", [10 * SEC], "C:/data/run1.bag", [15 * SEC], exclusion_margin_s=2.0)
    assert result.same_source
    assert not result.overlapping


def test_check_overlap_margin_boundary_is_inclusive():
    result = check_overlap("C:/data/run1.bag", [10 * SEC], "C:/data/run1.bag", [12 * SEC], exclusion_margin_s=2.0)
    assert result.overlapping  # exactly 2.0s away, at the boundary
    result_just_past = check_overlap(
        "C:/data/run1.bag", [10 * SEC], "C:/data/run1.bag", [12 * SEC + 1], exclusion_margin_s=2.0,
    )
    assert not result_just_past.overlapping


def test_check_overlap_finds_correct_match_regardless_of_input_order():
    """searchsorted-based matching must not depend on the caller's list
    order -- discovery/holdout timestamps arrive however sampling produced
    them, not pre-sorted."""
    discovery = [50 * SEC, 10 * SEC, 30 * SEC]
    holdout = [30 * SEC + int(0.5 * SEC), 100 * SEC, 10 * SEC - int(0.5 * SEC)]
    result = check_overlap("C:/data/run1.bag", discovery, "C:/data/run1.bag", holdout, exclusion_margin_s=1.0)
    assert set(result.overlap_timestamps) == {30 * SEC + int(0.5 * SEC), 10 * SEC - int(0.5 * SEC)}


def test_check_overlap_different_bag_is_independent_even_with_shared_timestamps():
    """A separate bag file is treated as independent data purely by file
    identity -- shared LiDAR timestamps between two different recordings do
    not mean the same physical moment was used twice, and the guard band is
    not even applied."""
    result = check_overlap("C:/data/discovery.bag", [10 * SEC], "C:/data/holdout.bag", [10 * SEC])
    assert not result.same_source
    assert not result.overlapping
    assert result.n_overlap == 0


def test_check_overlap_resolves_relative_vs_absolute_paths_to_same_identity(tmp_path):
    bag = tmp_path / "run.bag"
    bag.write_bytes(b"")
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = check_overlap(str(bag), [5 * SEC], "run.bag", [5 * SEC], exclusion_margin_s=0.0)
    finally:
        os.chdir(cwd)
    assert result.same_source
    assert result.overlapping


def test_check_overlap_missing_bag_path_is_never_treated_as_same_source():
    result = check_overlap("", [1 * SEC, 2 * SEC], "", [1 * SEC, 2 * SEC])
    assert not result.same_source
    assert not result.overlapping


def test_check_overlap_large_timestamp_lists_do_not_use_brute_force(monkeypatch):
    """A crude correctness proxy for the O((n+m) log n) searchsorted
    approach: run it on lists large enough (10k x 10k) that an accidental
    O(n*m) implementation would be obviously slow, and require it to finish
    well under a brute-force budget."""
    import time as _time

    rng = np.random.default_rng(0)
    discovery = (rng.uniform(0, 1000, 10_000) * SEC).astype(np.int64).tolist()
    holdout = (rng.uniform(0, 1000, 10_000) * SEC).astype(np.int64).tolist()
    start = _time.perf_counter()
    check_overlap("a.bag", discovery, "a.bag", holdout, exclusion_margin_s=0.01)
    elapsed = _time.perf_counter() - start
    assert elapsed < 2.0, f"took {elapsed:.2f}s -- looks like O(n*m), not O((n+m) log n)"


# ------------------------------------------------------ independence partition


def test_partition_independent_frames_splits_usable_and_excluded():
    candidate = [10 * SEC + int(0.5 * SEC), 20 * SEC, 30 * SEC]
    partition = ho_mod.partition_independent_frames(candidate, "a.bag", [10 * SEC], "a.bag", exclusion_margin_s=2.0)
    assert partition.same_source
    assert partition.requested == 3
    assert partition.n_excluded == 1
    assert partition.n_usable == 2
    assert (10 * SEC + int(0.5 * SEC)) in partition.excluded_timestamps
    assert 20 * SEC in partition.usable_timestamps and 30 * SEC in partition.usable_timestamps


def test_partition_independent_frames_different_bag_excludes_nothing():
    candidate = [10 * SEC, 10 * SEC + 1]
    partition = ho_mod.partition_independent_frames(candidate, "discovery.bag", [10 * SEC], "holdout.bag", 2.0)
    assert not partition.same_source
    assert partition.n_excluded == 0
    assert partition.n_usable == 2


def test_partition_independence_limited_flag():
    few = list(range(3))
    partition_few = ho_mod.IndependencePartition(
        same_source=True, exclusion_margin_s=2.0, requested=3, usable_timestamps=tuple(few),
    )
    assert partition_few.independence_limited

    many = list(range(ho_mod.DEFAULT_MIN_INDEPENDENT_FRAMES + 5))
    partition_many = ho_mod.IndependencePartition(
        same_source=True, exclusion_margin_s=2.0, requested=len(many), usable_timestamps=tuple(many),
    )
    assert not partition_many.independence_limited


# --------------------------------------------------------- spatial/depth rows


def _bin(n_points, p95):
    return BinStats(label="x", mean_px=p95 * 0.8, median_px=p95 * 0.8, p95_px=p95, std_px=0.1,
                     n_points=n_points, n_matched=n_points, n_unmatched=0)


def test_compare_bins_computes_delta_candidate_minus_baseline():
    baseline = {"LEFT": _bin(100, 2.0), "CENTER": _bin(100, 3.0), "RIGHT": _bin(100, 5.0)}
    candidate = {"LEFT": _bin(100, 2.2), "CENTER": _bin(100, 2.4), "RIGHT": _bin(100, 4.0)}
    rows = _compare_bins(baseline, candidate, ("LEFT", "CENTER", "RIGHT"))
    by_label = {r.label: r for r in rows}
    assert np.isclose(by_label["LEFT"].delta_p95_px, 0.2)
    assert np.isclose(by_label["CENTER"].delta_p95_px, -0.6)
    assert np.isclose(by_label["RIGHT"].delta_p95_px, -1.0)
    assert by_label["LEFT"].baseline_n_points == 100


def test_compare_bins_missing_region_on_either_side_is_nan_not_zero():
    baseline = {"LEFT": _bin(100, 2.0)}
    candidate = {"LEFT": _bin(100, 2.2)}
    rows = _compare_bins(baseline, candidate, ("LEFT", "CENTER", "RIGHT"))
    by_label = {r.label: r for r in rows}
    assert np.isnan(by_label["CENTER"].delta_p95_px)
    assert by_label["CENTER"].baseline_n_points == 0
    assert by_label["CENTER"].candidate_n_points == 0


def test_compare_bins_preserves_label_order():
    baseline = {"0-10m": _bin(10, 1.0), "10-20m": _bin(10, 2.0)}
    candidate = {"0-10m": _bin(10, 1.1), "10-20m": _bin(10, 1.9)}
    rows = _compare_bins(baseline, candidate, ("0-10m", "10-20m"))
    assert [r.label for r in rows] == ["0-10m", "10-20m"]


# --------------------------------------------------------- synthetic scene
#
# Same depth-step scene as test_perturbation.py: a vertical intensity edge at
# CX, with points at z=5m to its left and z=10m to its right, so a yaw
# perturbation measurably shifts the projected edge.

_WIDTH, _HEIGHT = 640, 480
_FX = _FY = 500.0
_CX, _CY = _WIDTH / 2, _HEIGHT / 2


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


def _holdout_frames(image, cloud, n=5, **kwargs):
    return prepare_frames(
        lambda idx: (image, cloud, idx, float(idx), idx),
        n_total_timeline=n, n_samples=n,
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        **kwargs,
    )


def test_evaluate_holdout_baseline_and_candidate_score_the_identical_frame_set():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=4)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")

    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert result.ok
    baseline_idx = set(result.baseline_point.per_frame.keys())
    candidate_idx = set(result.candidate_point.per_frame.keys())
    assert baseline_idx == candidate_idx == {f.timeline_index for f in frames}


def test_evaluate_holdout_never_mutates_baseline_arrays():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=2)
    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()
    cand = CandidateSpec(axis="tx", delta=10.0, source="quick")

    evaluate_holdout(frames, R, t, cand, _camera(), edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert np.array_equal(R, R_before)
    assert np.array_equal(t, t_before)


def test_evaluate_holdout_never_searches_beyond_the_locked_candidate(monkeypatch):
    """Exactly two trials -- baseline and the one locked candidate -- no
    matter how many frames are hold-out. A grid or a range would call
    run_trial_point more than twice."""
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=6)
    calls = []
    original = pert_mod.run_trial_point

    def spy(*args, **kwargs):
        calls.append(args[1:3])  # (axis, delta)
        return original(*args, **kwargs)

    monkeypatch.setattr(ho_mod, "run_trial_point", spy)
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")

    evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                      edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert len(calls) == 2
    assert calls[0] == ("yaw", 0.0)     # baseline
    assert calls[1] == ("yaw", -0.1)    # exactly the locked candidate, nothing else


def test_evaluate_holdout_paired_delta_p95_matches_manual_subtraction():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=3)
    cand = CandidateSpec(axis="yaw", delta=0.5, source="full")

    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    baseline, candidate = result.baseline_point, result.candidate_point
    manual = [
        candidate.per_frame[idx].p95_px - baseline.per_frame[idx].p95_px
        for idx in baseline.per_frame
        if baseline.per_frame[idx].ok and candidate.per_frame[idx].ok
    ]
    assert np.isclose(candidate.median_delta_frame_p95_px, float(np.median(manual)))
    assert candidate.n_comparable_frames == len(manual)


def test_evaluate_holdout_sync_rejected_frame_excluded_from_both_sides():
    image, cloud = _make_scene()

    def loader(idx):
        camera_t = idx if idx != 1 else idx + 100_000_000  # frame 1: 100ms off
        return image, cloud, idx, float(idx), camera_t

    frames = prepare_frames(loader, n_total_timeline=4, n_samples=4, max_sync_offset_ms=50.0,
                             edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert sum(f.sync_rejected for f in frames) == 1
    rejected_idx = next(f.timeline_index for f in frames if f.sync_rejected)

    cand = CandidateSpec(axis="tx", delta=10.0, source="quick")
    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert result.n_sync_rejected == 1
    assert result.n_frames_used == 3
    assert rejected_idx not in result.baseline_point.per_frame
    assert rejected_idx not in result.candidate_point.per_frame


def test_evaluate_holdout_spatial_and_depth_use_only_holdout_frames():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=2)
    cand = CandidateSpec(axis="yaw", delta=0.3, source="full")

    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert result.horizontal and result.vertical and result.depth
    assert {r.label for r in result.horizontal} == {"LEFT", "CENTER", "RIGHT"}
    assert {r.label for r in result.vertical} == {"TOP", "CENTER", "BOTTOM"}


def test_evaluate_holdout_no_usable_frames_reports_reason_not_a_crash():
    frame = LoadedFrame(
        timeline_index=0, image_bgr=np.zeros((2, 2, 3), np.uint8), cloud_xyz=np.zeros((0, 3), np.float32),
        lidar_timestamp_ns=0, camera_timestamp_ns=100_000_000, signed_sync_offset_ms=100.0,
        abs_sync_offset_ms=100.0, sync_rejected=True, cache=None,
    )
    cand = CandidateSpec(axis="yaw", delta=0.1, source="fine")
    result = evaluate_holdout([frame], np.eye(3), np.zeros(3), cand, _camera())
    assert not result.ok
    assert not result.cancelled
    assert result.reason


# ----------------------------------------------------------- image-side cache


def test_prepare_frames_cache_is_reused_not_recomputed_by_evaluate_holdout(monkeypatch):
    """compute_edge_orientation_map must run once per frame during
    prepare_frames and never again during evaluate_holdout -- the baseline and
    candidate trials both reuse frame.cache, matching how Perturbation
    Sensitivity's grid reuses it across dozens of trials."""
    image, cloud = _make_scene()
    calls = {"n": 0}
    original = pert_mod.compute_edge_orientation_map

    def spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pert_mod, "compute_edge_orientation_map", spy)

    frames = _holdout_frames(image, cloud, n=4)
    assert calls["n"] == 4
    calls["n"] = 0

    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                      edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    assert calls["n"] == 0


# ----------------------------------------------------------------- cancellation


def test_evaluate_holdout_cancel_before_baseline_runs_nothing():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=2)
    cand = CandidateSpec(axis="yaw", delta=0.1, source="fine")

    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               should_cancel=lambda: True)
    assert result.cancelled
    assert result.axis is None


def test_evaluate_holdout_can_be_cancelled_between_baseline_and_candidate():
    """Real-thread cancellation set after the baseline trial's progress
    callback fires but before the candidate trial starts -- the candidate is
    never scored, and the baseline that already ran is not thrown away."""
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=3)
    cancel_event = threading.Event()
    progress_calls = []

    def progress(done, total):
        progress_calls.append(done)
        if done == 1:
            cancel_event.set()
        time.sleep(0.01)

    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    result_holder = {}

    def run():
        result_holder["result"] = evaluate_holdout(
            frames, np.eye(3), np.zeros(3), cand, _camera(),
            edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
            progress=progress, should_cancel=cancel_event.is_set,
        )

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=10.0)

    assert not worker.is_alive()
    result = result_holder["result"]
    assert result.cancelled
    assert progress_calls == [1]  # candidate's progress(2, 2) never fired
    assert result.axis is not None and result.axis.baseline is not None
    assert result.candidate_point is None  # never computed


# --------------------------------------------------- pairing bookkeeping (mocked)


def _point(delta, per_frame: dict, ok=True) -> PerturbationPoint:
    ok_p95 = [m.p95_px for m in per_frame.values() if m.ok]
    return PerturbationPoint(
        delta=delta, ok=ok, p95_px=float(np.mean(ok_p95)) if ok_p95 else float("nan"),
        median_px=float(np.mean(ok_p95)) if ok_p95 else float("nan"),
        n_valid_frames=sum(1 for m in per_frame.values() if m.ok),
        n_failed_frames=sum(1 for m in per_frame.values() if not m.ok),
        per_frame=per_frame,
    )


def test_evaluate_holdout_surfaces_baseline_only_and_candidate_only_failures(monkeypatch):
    """Frame 2 fails only for the candidate, frame 3 fails only for baseline --
    neither disappears from the aggregate; each is counted in its own bucket,
    never silently dropped or merged with 'both valid'."""
    baseline_per_frame = {
        0: FrameMetric(ok=True, p95_px=2.0, median_px=1.5),
        1: FrameMetric(ok=True, p95_px=3.0, median_px=2.5),
        2: FrameMetric(ok=True, p95_px=4.0, median_px=3.5),   # candidate fails here
        3: FrameMetric(ok=False),                              # baseline itself failed here
    }
    candidate_per_frame = {
        0: FrameMetric(ok=True, p95_px=1.8, median_px=1.4),
        1: FrameMetric(ok=True, p95_px=2.9, median_px=2.4),
        2: FrameMetric(ok=False),
        3: FrameMetric(ok=True, p95_px=5.0, median_px=4.5),
    }
    baseline_point = _point(0.0, baseline_per_frame)
    candidate_point = _point(0.3, candidate_per_frame)

    def fake_run_trial_point(kind, axis, delta, *_a, **_k):
        return baseline_point if delta == 0.0 else candidate_point

    monkeypatch.setattr(ho_mod, "run_trial_point", fake_run_trial_point)
    frames = [
        LoadedFrame(timeline_index=i, image_bgr=np.zeros((2, 2, 3), np.uint8),
                    cloud_xyz=np.zeros((0, 3), np.float32), lidar_timestamp_ns=i, camera_timestamp_ns=i,
                    signed_sync_offset_ms=0.0, abs_sync_offset_ms=0.0, sync_rejected=False, cache=None)
        for i in range(4)
    ]
    cand = CandidateSpec(axis="yaw", delta=0.3, source="full")

    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera())

    cp = result.candidate_point
    assert cp.n_comparable_frames == 2       # frames 0, 1
    assert cp.n_baseline_only_valid == 1     # frame 2: baseline ok, candidate failed
    assert cp.n_candidate_only_valid == 1    # frame 3: candidate ok, baseline failed
    assert cp.n_both_failed == 0


# -------------------------------------------------------- diagnostics wiring


def test_holdout_result_feeds_diagnostics_without_mutating_the_candidate():
    import gui.core.evaluation.diagnostics as dg

    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=5)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    report = dg.build_diagnostic_report(dg.DiagnosticInputs(holdout=result))
    item = next(i for i in report.evidence if i.category == dg.CAT_HOLDOUT)

    assert item.strength != dg.UNAVAILABLE
    assert not item.is_candidate  # never proposed as a new root-cause candidate on its own
    assert cand.axis == "yaw" and cand.delta == 0.2  # untouched by the report build


# ------------------------------------------------------------- bag signature


def _sig(**overrides) -> BagSignature:
    base = dict(file_size_bytes=1_000_000, duration_s=60.0, n_topics=4, total_message_count=5000,
                first_lidar_timestamp_ns=10 * SEC, last_lidar_timestamp_ns=70 * SEC)
    base.update(overrides)
    return BagSignature(**base)


def test_signatures_suspiciously_similar_flags_a_likely_copy():
    a = _sig()
    b = _sig(file_size_bytes=1_000_050)  # a few bytes of filesystem/metadata noise
    assert ho_mod.signatures_suspiciously_similar(a, b)


def test_signatures_suspiciously_similar_rejects_different_message_count():
    assert not ho_mod.signatures_suspiciously_similar(_sig(), _sig(total_message_count=5001))


def test_signatures_suspiciously_similar_rejects_different_topic_count():
    assert not ho_mod.signatures_suspiciously_similar(_sig(), _sig(n_topics=5))


def test_signatures_suspiciously_similar_rejects_different_first_or_last_timestamp():
    assert not ho_mod.signatures_suspiciously_similar(_sig(), _sig(first_lidar_timestamp_ns=11 * SEC))
    assert not ho_mod.signatures_suspiciously_similar(_sig(), _sig(last_lidar_timestamp_ns=71 * SEC))


def test_signatures_suspiciously_similar_rejects_large_size_or_duration_difference():
    assert not ho_mod.signatures_suspiciously_similar(_sig(), _sig(file_size_bytes=2_000_000))
    assert not ho_mod.signatures_suspiciously_similar(_sig(), _sig(duration_s=90.0))


def test_cross_bag_time_ranges_overlap_true_and_false():
    a = _sig(first_lidar_timestamp_ns=0, last_lidar_timestamp_ns=100 * SEC)
    overlapping = _sig(first_lidar_timestamp_ns=50 * SEC, last_lidar_timestamp_ns=150 * SEC)
    disjoint = _sig(first_lidar_timestamp_ns=200 * SEC, last_lidar_timestamp_ns=300 * SEC)
    assert ho_mod.cross_bag_time_ranges_overlap(a, overlapping)
    assert not ho_mod.cross_bag_time_ranges_overlap(a, disjoint)


# --------------------------------------------------------------- bootstrap CI


def test_block_bootstrap_reports_unavailable_below_min_blocks():
    paired = [(1 * SEC, -1.0), (1 * SEC + 100_000_000, -1.2), (2 * SEC, -0.9)]
    result = ho_mod._run_block_bootstrap(paired, block_s=5.0, iterations=200, min_blocks=5, tolerance_px=0.05)
    assert not result.ok
    assert "too few independent temporal blocks" in result.reason
    assert result.n_blocks < 5


def test_block_bootstrap_reports_unavailable_below_two_comparable_frames():
    result = ho_mod._run_block_bootstrap([(1 * SEC, -1.0)], block_s=2.0, iterations=200, min_blocks=5,
                                          tolerance_px=0.05)
    assert not result.ok
    assert "fewer than 2 comparable frames" in result.reason


def test_block_bootstrap_uniform_improvement_gives_a_degenerate_ci_excluding_zero():
    paired = [(i * 3 * SEC, -2.0) for i in range(20)]  # every frame improves by exactly the same amount
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=500, min_blocks=5, tolerance_px=0.05,
                                          rng=np.random.default_rng(0))
    assert result.ok
    assert result.n_blocks >= 5
    assert np.isclose(result.median_delta_ci_low_px, -2.0)
    assert np.isclose(result.median_delta_ci_high_px, -2.0)
    assert result.median_ci_excludes_zero is True


def test_block_bootstrap_one_giant_block_is_reported_unavailable():
    """A block wide enough to swallow every paired frame collapses to a
    single block -- below the default minimum of 5, so no CI is fabricated
    from one block always being drawn."""
    paired = [(i * SEC, float(i)) for i in range(10)]
    result = ho_mod._run_block_bootstrap(paired, block_s=1000.0, iterations=200, min_blocks=5, tolerance_px=0.05)
    assert not result.ok
    assert result.n_blocks == 1


def test_block_bootstrap_resamples_blocks_not_individual_frames():
    """4 "typical" blocks at -1.0px plus 1 "outlier" block at +5.0px (20
    identical-delta frames each) -- the outlier is a small minority of
    individual *frames* (20 / 100 = 20%), but a full 1-in-5 of the *blocks*.

    Resampling individual frames (naive, i.i.d.) draws the outlier's share
    close to 20% almost every time by the law of large numbers over 100
    draws, so its median never leaves -1.0px: a tight, misleadingly
    confident interval. Resampling whole *blocks* (5 draws with replacement)
    can plausibly draw the outlier block 3+ times, occasionally flipping the
    pooled median positive -- correctly widening the interval, because five
    blocks is genuinely not very much independent data. If this module
    resampled individual frames instead of blocks (a plausible first-draft
    bug), the two intervals would look the same.
    """
    block_values = [-1.0, -1.0, -1.0, -1.0, 5.0]
    paired = []
    for i, d in enumerate(block_values):
        base_t = i * 3 * SEC  # 3s apart -- outside a 2.0s block width
        for j in range(20):   # 20 identical-delta frames per block
            paired.append((base_t + j * 50_000_000, d))

    block_result = ho_mod._run_block_bootstrap(
        paired, block_s=2.0, iterations=3000, min_blocks=2, tolerance_px=0.05, rng=np.random.default_rng(0),
    )
    assert block_result.ok
    assert block_result.n_blocks == 5
    block_width = block_result.median_delta_ci_high_px - block_result.median_delta_ci_low_px

    all_deltas = np.array([d for _, d in paired])
    rng = np.random.default_rng(0)
    naive_medians = np.array([
        np.median(rng.choice(all_deltas, size=all_deltas.size, replace=True)) for _ in range(3000)
    ])
    naive_lo, naive_hi = np.percentile(naive_medians, [2.5, 97.5])
    naive_width = naive_hi - naive_lo

    assert naive_width < 0.5, "naive (i.i.d.) resampling should stay tightly at -1.0px here"
    assert block_width > naive_width + 2.0, (block_width, naive_width)


def test_block_bootstrap_improved_ratio_matches_observed_fraction():
    # 8 blocks improve beyond tolerance, 2 do not -- observed ratio 0.8
    paired = []
    for i in range(10):
        d = -1.0 if i < 8 else 0.5
        for j in range(5):
            paired.append((i * 3 * SEC + j * 100_000_000, d))
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=500, min_blocks=5, tolerance_px=0.05,
                                          rng=np.random.default_rng(3))
    assert result.ok
    assert np.isclose(result.improved_ratio, 0.8)
    assert result.improved_ratio_ci_low <= result.improved_ratio <= result.improved_ratio_ci_high


def test_evaluate_holdout_computes_bootstrap_end_to_end_with_correlated_synthetic_frames():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=10)
    cand = CandidateSpec(axis="yaw", delta=0.3, source="full")

    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
                               bootstrap_iterations=200, bootstrap_rng=np.random.default_rng(0))

    assert result.bootstrap is not None
    # Every frame here is the *same* repeated scene -- one block, so the
    # default 5-block minimum is (correctly) not met and no CI is fabricated.
    assert not result.bootstrap.ok
    assert "too few independent temporal blocks" in result.bootstrap.reason


# -------------------------------------------------------------- provenance


def test_evaluate_holdout_records_provenance_when_supplied():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=5)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    discovery = ho_mod.DiscoveryProvenance(bag="disc.bag", timestamps=(1 * SEC, 2 * SEC), exclusion_margin_s=2.0)

    result = evaluate_holdout(
        frames, np.eye(3), np.zeros(3), cand, _camera(),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        discovery=discovery, holdout_bag="holdout.bag",
        requested_timestamps=(10 * SEC, 20 * SEC, 30 * SEC, 40 * SEC, 50 * SEC),
        excluded_near_discovery=(15 * SEC,),
        signature_warning="Hold-out bag appears identical to the Discovery bag.",
        cross_bag_time_overlap_note="note",
    )

    assert result.discovery is discovery
    assert result.provenance is not None
    assert result.provenance.bag == "holdout.bag"
    assert result.provenance.excluded_near_discovery == (15 * SEC,)
    assert result.provenance.requested_timestamps == (10 * SEC, 20 * SEC, 30 * SEC, 40 * SEC, 50 * SEC)
    assert result.signature_warning
    assert result.cross_bag_time_overlap_note == "note"
    assert len(result.baseline_fingerprint) == 12


def test_evaluate_holdout_baseline_fingerprint_is_deterministic_and_sensitive_to_t():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=2)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    kwargs = dict(edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))

    r1 = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(), **kwargs)
    r2 = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(), **kwargs)
    r3 = evaluate_holdout(frames, np.eye(3), np.array([0.01, 0.0, 0.0]), cand, _camera(), **kwargs)

    assert r1.baseline_fingerprint == r2.baseline_fingerprint
    assert r1.baseline_fingerprint != r3.baseline_fingerprint


def test_evaluate_holdout_without_provenance_kwargs_leaves_it_none():
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=2)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))
    assert result.discovery is None
    assert result.provenance is None
    assert result.signature_warning == ""


# --------------------------------------------------- exposure / validation status


def _uniform_improvement_holdout_result(monkeypatch, n=20, delta_px=-2.0, source="fine"):
    """A clean, deterministic 'candidate clearly and uniformly better than
    baseline' result via a monkeypatched run_trial_point -- the geometry-
    backed synthetic scene's baseline (R=eye) is already near-optimal (see
    test_synthetic_extrinsic_sensitivity.py), so most non-zero deltas make it
    *worse*, not better, and cannot exercise a Strong-rated path directly."""
    baseline_per_frame = {i: FrameMetric(ok=True, p95_px=5.0, median_px=4.0) for i in range(n)}
    candidate_per_frame = {i: FrameMetric(ok=True, p95_px=5.0 + delta_px, median_px=4.0 + delta_px) for i in range(n)}
    baseline_point = _point(0.0, baseline_per_frame)
    candidate_point = _point(-0.1, candidate_per_frame)

    def fake_run_trial_point(kind, axis, delta, *_a, **_k):
        return baseline_point if delta == 0.0 else candidate_point

    monkeypatch.setattr(ho_mod, "run_trial_point", fake_run_trial_point)
    frames = [
        LoadedFrame(timeline_index=i, image_bgr=np.zeros((2, 2, 3), np.uint8), cloud_xyz=np.zeros((0, 3), np.float32),
                    lidar_timestamp_ns=i * 3 * SEC, camera_timestamp_ns=i * 3 * SEC,
                    signed_sync_offset_ms=0.0, abs_sync_offset_ms=0.0, sync_rejected=False, cache=None)
        for i in range(n)
    ]
    cand = CandidateSpec(axis="yaw", delta=-0.1, source=source)
    return evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                             bootstrap_iterations=300, bootstrap_block_s=2.0,
                             bootstrap_rng=np.random.default_rng(1))


def test_assess_holdout_independent_status_can_reach_strong(monkeypatch):
    import gui.core.evaluation.diagnostics as dg

    result = _uniform_improvement_holdout_result(monkeypatch)
    assert result.bootstrap.ok and result.bootstrap.median_ci_excludes_zero is True

    item = dg.assess_holdout(result, ho_mod.STATUS_INDEPENDENT)
    assert item.title == "Independent hold-out validation"
    assert item.strength == dg.STRONG


def test_assess_holdout_independent_repeated_status_is_not_capped(monkeypatch):
    import gui.core.evaluation.diagnostics as dg

    result = _uniform_improvement_holdout_result(monkeypatch)
    item = dg.assess_holdout(result, ho_mod.STATUS_INDEPENDENT_REPEATED)
    assert item.title == "Independent hold-out validation"
    assert item.strength == dg.STRONG


@pytest.mark.parametrize("status", ho_mod.EXPOSED_STATUSES)
def test_assess_holdout_exposed_statuses_rename_and_cap_strength(monkeypatch, status):
    import gui.core.evaluation.diagnostics as dg

    result = _uniform_improvement_holdout_result(monkeypatch)

    item = dg.assess_holdout(result, status)
    assert item.title == "Exploratory validation"
    assert dg._STRENGTH_RANK[item.strength] <= dg._STRENGTH_RANK[dg.WEAK]
    assert any(ho_mod.STATUS_LABELS[status] in o for o in item.observations)


def test_assess_holdout_strong_requires_ci_excluding_zero(monkeypatch):
    """A point estimate that would otherwise qualify as Strong is capped at
    Moderate when the bootstrap interval for the median still includes zero
    -- see assess_holdout's docstring.

    5 blocks of 20 frames each, spaced 3s apart (well outside the 2.0s block
    width, so each stays its own block): 4 "good" blocks improve by -1.0px
    (80% of frames, comfortably past every Strong threshold on imp_ratio /
    worse_ratio / coverage / paired median), 1 "bad" block worsens by
    +5.0px. The point estimate looks Strong; but block-level resampling can
    plausibly draw the bad block 3+ times out of 5, which is enough by
    itself to flip the pooled median positive -- so the 95% interval spans
    zero, and Strong must be capped to Moderate.
    """
    import gui.core.evaluation.diagnostics as dg

    block_deltas_px = [-1.0, -1.0, -1.0, -1.0, 5.0]
    n_per_block = 20
    baseline_per_frame: dict = {}
    candidate_per_frame: dict = {}
    frames: list = []
    idx = 0
    for b, d in enumerate(block_deltas_px):
        base_t = b * 3 * SEC
        for j in range(n_per_block):
            baseline_per_frame[idx] = FrameMetric(ok=True, p95_px=5.0, median_px=4.0)
            candidate_per_frame[idx] = FrameMetric(ok=True, p95_px=5.0 + d, median_px=4.0 + d)
            frames.append(LoadedFrame(
                timeline_index=idx, image_bgr=np.zeros((2, 2, 3), np.uint8), cloud_xyz=np.zeros((0, 3), np.float32),
                lidar_timestamp_ns=base_t + j * 50_000_000, camera_timestamp_ns=base_t + j * 50_000_000,
                signed_sync_offset_ms=0.0, abs_sync_offset_ms=0.0, sync_rejected=False, cache=None,
            ))
            idx += 1

    baseline_point = _point(0.0, baseline_per_frame)
    candidate_point = _point(-0.1, candidate_per_frame)

    def fake_run_trial_point(kind, axis, delta, *_a, **_k):
        return baseline_point if delta == 0.0 else candidate_point

    monkeypatch.setattr(ho_mod, "run_trial_point", fake_run_trial_point)
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    result = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                               bootstrap_iterations=1000, bootstrap_block_s=2.0,
                               bootstrap_rng=np.random.default_rng(0))

    assert result.candidate_point.improved_frame_ratio == pytest.approx(0.8)
    assert result.bootstrap.ok and result.bootstrap.n_blocks == 5
    assert result.bootstrap.median_ci_excludes_zero is False, "test setup must produce a CI spanning zero"

    item = dg.assess_holdout(result, ho_mod.STATUS_INDEPENDENT)
    assert item.strength == dg.MODERATE
    assert any("block-bootstrap" in c for c in item.caveats)


# --------------------------- candidate identity / provenance (physical vs session)


def test_candidate_identity_equal_for_identical_inputs():
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    a = ho_mod.candidate_identity(np.eye(3), np.zeros(3), cand)
    b = ho_mod.candidate_identity(np.eye(3), np.zeros(3), cand)
    assert a == b
    assert hash(a) == hash(b)


def test_candidate_identity_differs_when_delta_differs():
    a = ho_mod.candidate_identity(np.eye(3), np.zeros(3), CandidateSpec(axis="yaw", delta=-0.1, source="fine"))
    b = ho_mod.candidate_identity(np.eye(3), np.zeros(3), CandidateSpec(axis="yaw", delta=-0.2, source="fine"))
    assert a != b
    assert a.candidate_fingerprint != b.candidate_fingerprint


def test_candidate_identity_differs_when_baseline_t_differs():
    """Same axis/delta/source, different baseline T -> a different candidate
    identity: the actual candidate R/t (and hence candidate_fingerprint) also
    differ, and baseline_fingerprint alone already distinguishes them."""
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    a = ho_mod.candidate_identity(np.eye(3), np.zeros(3), cand)
    b = ho_mod.candidate_identity(np.eye(3), np.array([0.05, 0.0, 0.0]), cand)
    assert a != b
    assert a.baseline_fingerprint != b.baseline_fingerprint
    assert a.candidate_fingerprint != b.candidate_fingerprint
    # axis/delta alone would have looked identical -- proving the
    # requirement that "Yaw -0.1deg" is not enough to establish sameness.
    assert (a.axis, a.delta) == (b.axis, b.delta)


def test_candidate_identity_equal_regardless_of_discovery_revision():
    """discovery_revision lives on CandidateProvenance now, never on
    CandidateIdentity -- the exact requirement this split exists for: the
    same physical candidate found under two different discovery revisions,
    including across an app restart where the persistent counter has moved
    on, must still be recognised as the same candidate."""
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    a = ho_mod.candidate_identity(np.eye(3), np.zeros(3), cand)
    b = ho_mod.candidate_identity(np.eye(3), np.zeros(3), cand)
    prov_a = ho_mod.candidate_provenance(cand, discovery_revision=7)
    prov_b = ho_mod.candidate_provenance(cand, discovery_revision=0)
    assert a == b
    assert prov_a != prov_b  # provenance itself still differs, just not identity


def test_candidate_identity_equal_regardless_of_source_when_transform_matches():
    """Quick and Fine landing on the exact same delta produce the exact same
    transform -- physically the same candidate, even though `source` (kept
    on CandidateProvenance, not CandidateIdentity) differs."""
    baseline_R, baseline_t = np.eye(3), np.zeros(3)
    quick = ho_mod.candidate_identity(baseline_R, baseline_t, CandidateSpec(axis="yaw", delta=-0.1, source="quick"))
    fine = ho_mod.candidate_identity(baseline_R, baseline_t, CandidateSpec(axis="yaw", delta=-0.1, source="fine"))
    assert quick == fine


def test_candidate_identity_never_mutates_baseline_arrays():
    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()
    ho_mod.candidate_identity(R, t, CandidateSpec(axis="tx", delta=10.0, source="quick"))
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)


def test_candidate_provenance_carries_source_and_revision():
    cand = CandidateSpec(axis="yaw", delta=-0.1, source="fine")
    prov = ho_mod.candidate_provenance(cand, discovery_revision=3)
    assert prov.source == "fine"
    assert prov.discovery_revision == 3


def _sig2(**overrides) -> BagSignature:
    base = dict(file_size_bytes=2_000_000, duration_s=120.0, n_topics=3, total_message_count=8000,
                first_lidar_timestamp_ns=0, last_lidar_timestamp_ns=100 * SEC)
    base.update(overrides)
    return BagSignature(**base)


def _cid(axis="yaw", delta=-0.1, source="fine", discovery_revision=0):
    """(CandidateIdentity, CandidateProvenance) -- callers pass this to
    determine_and_record_exposure/record_exposure as `*_cid(...)`, matching
    those functions' (candidate_id, candidate_prov) parameter order."""
    spec = CandidateSpec(axis=axis, delta=delta, source=source)
    identity = ho_mod.candidate_identity(np.eye(3), np.zeros(3), spec)
    provenance = ho_mod.candidate_provenance(spec, discovery_revision)
    return identity, provenance


# ------------------------------------------------------ slice / source identity


def test_slice_identity_same_path_and_timestamps_is_the_same_slice():
    a = ho_mod.slice_identity("path:run.bag", [3 * SEC, 1 * SEC, 2 * SEC])
    b = ho_mod.slice_identity("path:run.bag", [1 * SEC, 2 * SEC, 3 * SEC])  # order differs
    assert a == b  # timestamps are sorted before hashing -- order must not matter


def test_slice_identity_differs_for_different_timestamps_or_source():
    base = ho_mod.slice_identity("path:run.bag", [1 * SEC, 2 * SEC])
    different_ts = ho_mod.slice_identity("path:run.bag", [1 * SEC, 3 * SEC])
    different_source = ho_mod.slice_identity("path:other.bag", [1 * SEC, 2 * SEC])
    assert base != different_ts
    assert base != different_source


def test_resolve_source_identity_no_signature_falls_back_to_path():
    registry = {}
    key_a = ho_mod.resolve_source_identity(registry, "C:/data/run.bag", None)
    key_b = ho_mod.resolve_source_identity(registry, "C:/data/run.bag", None)
    key_other = ho_mod.resolve_source_identity(registry, "C:/data/other.bag", None)
    assert key_a == key_b
    assert key_a != key_other
    assert key_a.startswith("path:")


def test_resolve_source_identity_copied_bag_same_signature_folds_together():
    """holdout.bag and holdout_copy.bag: different paths, byte-identical
    content -- must resolve to the SAME source once a signature is known."""
    registry = {}
    sig = _sig2()
    key_a = ho_mod.resolve_source_identity(registry, "C:/data/holdout.bag", sig)
    ho_mod.record_exposure(registry, key_a, sig, ho_mod.slice_identity(key_a, [1 * SEC]), [1 * SEC], *_cid(), 2.0, None)
    key_b = ho_mod.resolve_source_identity(registry, "C:/data/holdout_copy.bag", sig)
    assert key_a == key_b
    assert len(registry) == 1


def test_resolve_source_identity_different_signature_is_a_different_source():
    registry = {}
    sig_a = _sig2()
    key_a = ho_mod.resolve_source_identity(registry, "a.bag", sig_a)
    ho_mod.record_exposure(registry, key_a, sig_a, ho_mod.slice_identity(key_a, [1 * SEC]), [1 * SEC], *_cid(), 2.0, None)
    sig_b = _sig2(total_message_count=9999)  # clearly a different recording
    key_b = ho_mod.resolve_source_identity(registry, "b.bag", sig_b)
    assert key_a != key_b


# ----------------------------------------------------------- exposure registry


def test_exposure_first_reveal_is_independent():
    registry = {}
    status = ho_mod.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC, 2 * SEC], *_cid(),
                                                    reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_INDEPENDENT
    assert len(registry) == 1


def test_exposure_copied_bag_same_signature_folds_into_one_source():
    """bag_A.bag and bag_A_copy.bag with an identical BagSignature: SourceIdentity
    must match, so peeking at A and then evaluating a different candidate on
    the copy cannot regain Independent status."""
    registry = {}
    sig = _sig2()
    ts = [10 * SEC, 20 * SEC, 30 * SEC, 40 * SEC, 50 * SEC]
    s1 = ho_mod.determine_and_record_exposure(registry, "C:/data/bag_A.bag", sig, ts, *_cid(delta=-0.1),
                                                reveal_margin_s=2.0)
    assert s1 == ho_mod.STATUS_INDEPENDENT
    assert len(registry) == 1

    s2 = ho_mod.determine_and_record_exposure(registry, "C:/data/bag_A_copy.bag", sig, ts,
                                                *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert s2 == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED
    assert len(registry) == 1  # still one source, not two


def test_exposure_same_candidate_rerun_is_independent_repeated():
    registry = {}
    ts = [1 * SEC, 2 * SEC, 3 * SEC]
    cand = _cid()
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, ts, *cand, reveal_margin_s=2.0)
    status = ho_mod.determine_and_record_exposure(registry, "a.bag", None, ts, *_cid(), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_INDEPENDENT_REPEATED


def test_exposure_90_percent_overlap_different_candidate_is_exploratory():
    """A 50-frame slice, then a 48/50-overlapping resample with a different
    candidate -- must still be recognised as the same exposed hold-out, not
    an unseen dataset."""
    registry = {}
    original = [i * SEC for i in range(50)]
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, original, *_cid(delta=-0.1), reveal_margin_s=2.0)

    resampled = original[1:-1]  # 48 of 50 timestamps still present, exact matches
    assert len(resampled) / len(original) >= 0.9
    status = ho_mod.determine_and_record_exposure(registry, "a.bag", None, resampled,
                                                    *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED


def test_exposure_near_timestamp_within_reveal_margin_counts_as_overlap():
    """10.000s was revealed; 10.033s is a different exact timestamp but well
    inside a 2.0s reveal margin -- must not be read as an unseen frame."""
    registry = {}
    revealed = [10 * SEC, 11 * SEC, 12 * SEC, 13 * SEC, 14 * SEC]
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, revealed, *_cid(delta=-0.1), reveal_margin_s=2.0)

    near = [t + int(0.033 * SEC) for t in revealed]  # every timestamp shifted by 33ms
    status = ho_mod.determine_and_record_exposure(registry, "a.bag", None, near,
                                                    *_cid(axis="ty", delta=5.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED


def test_exposure_resampling_with_different_frame_count_cannot_regain_independence():
    """10 timestamps, then the same span re-sampled to 9 (one dropped) --
    different slice_identity hash, but the same hold-out data in substance;
    a different candidate on it must still be Exploratory."""
    registry = {}
    run1 = list(range(1 * SEC, 11 * SEC, SEC))         # 10 timestamps
    run2 = list(range(1 * SEC, 10 * SEC, SEC))         # 9 timestamps, subset of run1
    assert ho_mod.slice_identity("k", run1) != ho_mod.slice_identity("k", run2)  # hashes differ, as expected

    ho_mod.determine_and_record_exposure(registry, "a.bag", None, run1, *_cid(delta=-0.1), reveal_margin_s=2.0)
    status = ho_mod.determine_and_record_exposure(registry, "a.bag", None, run2,
                                                    *_cid(axis="tz", delta=-5.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED


def test_exposure_truly_unseen_time_range_on_same_source_is_new_slice():
    registry = {}
    revealed = [t for t in range(100 * SEC, 121 * SEC, SEC)]   # 100-120s
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, revealed, *_cid(delta=-0.1), reveal_margin_s=2.0)

    unseen = [t for t in range(500 * SEC, 521 * SEC, SEC)]     # 500-520s -- far away, no overlap
    status = ho_mod.determine_and_record_exposure(registry, "a.bag", None, unseen,
                                                    *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_INDEPENDENT_NEW_SLICE
    assert status not in ho_mod.EXPOSED_STATUSES


def test_exposure_completely_different_source_is_independent():
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", _sig2(), [1 * SEC], *_cid(delta=-0.1), reveal_margin_s=2.0)
    status = ho_mod.determine_and_record_exposure(
        registry, "b.bag", _sig2(total_message_count=1), [1 * SEC],
        *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0,
    )
    assert status == ho_mod.STATUS_INDEPENDENT


def test_exposure_record_exposure_never_overwrites_first_reveal():
    registry = {}
    cand_a = _cid(delta=-0.1)
    cand_b = _cid(axis="tx", delta=10.0, source="quick")
    sid_a = ho_mod.resolve_source_identity(registry, "a.bag", None)
    slice_a = ho_mod.slice_identity(sid_a, [1 * SEC])
    ho_mod.record_exposure(registry, sid_a, None, slice_a, [1 * SEC], *cand_a, 2.0, None)
    # matched=None again -- if this incorrectly overwrote, the source's
    # first slice's candidate would change to cand_b.
    ho_mod.record_exposure(registry, sid_a, None, slice_a, [1 * SEC], *cand_b, 2.0,
                            matched=registry[sid_a].revealed_slices[0])
    assert len(registry[sid_a].revealed_slices) == 1
    assert registry[sid_a].revealed_slices[0].candidate == cand_a[0]
    assert registry[sid_a].revealed_slices[0].candidate_provenance == cand_a[1]


# ------------------------------------------------- exposure registry: persistence


def test_serialize_deserialize_exposure_registry_round_trips():
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", _sig2(), [1 * SEC, 2 * SEC, 3 * SEC],
                                          *_cid(delta=-0.1), reveal_margin_s=2.0)
    ho_mod.determine_and_record_exposure(registry, "b.bag", None, [10 * SEC], *_cid(axis="tx", delta=5.0),
                                          reveal_margin_s=1.5)

    data = ho_mod.serialize_exposure_registry(registry, discovery_revision=4)
    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(data)

    assert warnings == []
    assert restored == registry  # dataclass equality, field by field
    assert discovery_revision == 4


def test_serialize_exposure_registry_uses_only_plain_json_safe_types():
    """No Python repr/pickle anywhere -- every value must be a plain
    str/int/float/bool/list/dict/None, exactly as requirement 6 asks."""
    import json

    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", _sig2(), [1 * SEC], *_cid(delta=-0.1), reveal_margin_s=2.0)
    data = ho_mod.serialize_exposure_registry(registry, discovery_revision=2)
    json.dumps(data)  # raises TypeError on anything not JSON-safe


def test_deserialize_exposure_registry_missing_or_empty_is_empty_registry():
    assert ho_mod.deserialize_exposure_registry(None) == ({}, 0, [])
    assert ho_mod.deserialize_exposure_registry({}) == ({}, 0, [])


def test_deserialize_exposure_registry_unknown_version_warns_and_is_empty():
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC], *_cid(delta=-0.1), reveal_margin_s=2.0)
    data = ho_mod.serialize_exposure_registry(registry, discovery_revision=5)
    data["version"] = 999

    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(data)
    assert restored == {}
    assert discovery_revision == 0
    assert len(warnings) == 1 and "999" in warnings[0]


def test_deserialize_exposure_registry_skips_one_malformed_source_not_all():
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC], *_cid(delta=-0.1), reveal_margin_s=2.0)
    ho_mod.determine_and_record_exposure(registry, "b.bag", None, [2 * SEC], *_cid(axis="tx", delta=5.0),
                                          reveal_margin_s=2.0)
    data = ho_mod.serialize_exposure_registry(registry)
    # corrupt source "a.bag"'s entry -- drop a required field
    a_key = ho_mod.resolve_source_identity({}, "a.bag", None)
    del data["holdout_exposure"][a_key]["revealed_slices"][0]["candidate_identity"]["axis"]

    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(data)
    assert len(warnings) == 1
    assert a_key not in restored
    b_key = ho_mod.resolve_source_identity({}, "b.bag", None)
    assert b_key in restored  # the other source still loaded


# --------------------------------------- schema migration (v1/v2 -> v3)
#
# See VALIDATION_HISTORY_SCHEMA_VERSION's module-level comment for the full
# migration policy this section exercises: a v1/v2 project's old per-slice
# "discovery_generation" was a session-local number with no relationship to
# the new persistent discovery_revision counter, so it must be DISCARDED on
# migration (reset to 0), never reinterpreted as a starting revision value.


def _downgrade_to_v2(data: dict) -> dict:
    """Reshape a current (v3) serialize_exposure_registry() dict to look
    like what v2 actually wrote on disk: no top-level "discovery_revision"
    (that key did not exist yet), and each slice's candidate_provenance
    keyed by the old "discovery_generation" name instead of the new
    "discovery_revision" one."""
    out = {"version": 2, "holdout_exposure": {}}
    for source_id, source in data["holdout_exposure"].items():
        slices = []
        for s in source["revealed_slices"]:
            prov = s["candidate_provenance"]
            slices.append({
                **{k: v for k, v in s.items() if k not in ("candidate_provenance",)},
                "candidate_provenance": {"source": prov["source"], "discovery_generation": prov["discovery_revision"]},
            })
        out["holdout_exposure"][source_id] = {"signature": source["signature"], "revealed_slices": slices}
    return out


def _downgrade_to_v1(data: dict) -> dict:
    """Reshape a current (v3) dict to look like v1: candidate_identity and
    candidate_provenance merged into one "candidate" dict, old
    "discovery_generation" key name, no top-level "discovery_revision"."""
    out = {"version": 1, "holdout_exposure": {}}
    for source_id, source in data["holdout_exposure"].items():
        slices = []
        for s in source["revealed_slices"]:
            ident, prov = s["candidate_identity"], s["candidate_provenance"]
            merged = {**ident, "source": prov["source"], "discovery_generation": prov["discovery_revision"]}
            slices.append({
                "slice_identity": s["slice_identity"], "timestamps_ns": s["timestamps_ns"],
                "candidate": merged, "reveal_margin_s": s["reveal_margin_s"],
            })
        out["holdout_exposure"][source_id] = {"signature": source["signature"], "revealed_slices": slices}
    return out


def test_migrate_v2_discards_old_generation_never_reinterprets_as_revision():
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC],
                                          *_cid(delta=-0.1, discovery_revision=99), reveal_margin_s=2.0)
    v2_data = _downgrade_to_v2(ho_mod.serialize_exposure_registry(registry, discovery_revision=99))
    assert "discovery_revision" not in v2_data  # confirms this really looks like v2, not v3

    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(v2_data)
    assert warnings == []
    # The project-level counter starts fresh, not "recovered" from anywhere
    # in the old data (v2 never persisted one at all).
    assert discovery_revision == 0
    source_id = ho_mod.resolve_source_identity({}, "a.bag", None)
    slice_ = restored[source_id].revealed_slices[0]
    # The old per-slice generation (99) is discarded, not carried over --
    # migrated provenance starts at 0, same scale as a brand-new project's.
    assert slice_.candidate_provenance.discovery_revision == 0
    assert slice_.candidate_provenance.source == "fine"
    # Identity itself (the physical transform) is unaffected by migration.
    assert slice_.candidate.axis == "yaw" and slice_.candidate.delta == -0.1


def test_migrate_v1_discards_old_generation_never_reinterprets_as_revision():
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC],
                                          *_cid(delta=-0.1, discovery_revision=42), reveal_margin_s=2.0)
    v1_data = _downgrade_to_v1(ho_mod.serialize_exposure_registry(registry, discovery_revision=42))
    assert "discovery_revision" not in v1_data

    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(v1_data)
    assert warnings == []
    assert discovery_revision == 0
    source_id = ho_mod.resolve_source_identity({}, "a.bag", None)
    slice_ = restored[source_id].revealed_slices[0]
    assert slice_.candidate_provenance.discovery_revision == 0
    assert slice_.candidate.baseline_fingerprint  # identity fields still present, non-empty


def test_migrated_v2_history_then_a_real_discovery_run_still_catches_a_candidate_switch():
    """End-to-end migration correctness: a v2 project's old history
    (generation discarded, revision reset to 0) is loaded, then a *real*
    post-migration Discovery success bumps the fresh counter to 1 -- which
    must already be enough to flag a *different* candidate on the same
    slice as Exploratory (1 > 0), exactly the conservative-by-design
    behaviour the migration policy comment promises: migrated history can
    never look falsely "fresh" forever, only until the very next Discovery
    run."""
    registry = {}
    ho_mod.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC, 2 * SEC],
                                          *_cid(axis="yaw", delta=-0.1, discovery_revision=99), reveal_margin_s=2.0)
    v2_data = _downgrade_to_v2(ho_mod.serialize_exposure_registry(registry, discovery_revision=99))
    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(v2_data)
    assert warnings == []
    assert discovery_revision == 0

    # "a real Discovery run after migration" -- the fresh, persistent
    # counter advances from its migrated starting point (0) to 1.
    post_migration_revision = discovery_revision + 1
    status = ho_mod.determine_and_record_exposure(
        restored, "a.bag", None, [1 * SEC, 2 * SEC],
        *_cid(axis="tx", delta=10.0, source="quick", discovery_revision=post_migration_revision),
        reveal_margin_s=2.0,
    )
    assert status == ho_mod.STATUS_EXPOSED_DISCOVERY_UPDATED


def _simulate_restart(registry: dict, discovery_revision: int = 0) -> dict:
    """Round-trip through the exact (de)serialization the project file uses
    -- standing in for quitting and relaunching the app between the two
    calls a test makes around this. Returns only the restored registry (most
    callers only care about that); pass `discovery_revision` through when a
    test also needs to simulate the project's persisted revision counter
    surviving the same restart (see `_simulate_restart_full`)."""
    restored, _, warnings = ho_mod.deserialize_exposure_registry(
        ho_mod.serialize_exposure_registry(registry, discovery_revision)
    )
    assert warnings == []
    return restored


def _simulate_restart_full(registry: dict, discovery_revision: int = 0) -> "tuple[dict, int]":
    """Like `_simulate_restart`, but also returns the restored
    `discovery_revision` -- used by tests that need to carry the persistent
    revision counter across a simulated restart themselves (this module has
    no Project/step_verify.py dependency, so tests exercise the counter
    purely through serialize/deserialize, the same way step_verify.py's
    `_sync_holdout_history_from_project` does)."""
    restored, revision, warnings = ho_mod.deserialize_exposure_registry(
        ho_mod.serialize_exposure_registry(registry, discovery_revision)
    )
    assert warnings == []
    return restored, revision


def test_exposure_survives_restart_then_different_candidate_is_exploratory():
    session1 = {}
    ho_mod.determine_and_record_exposure(session1, "a.bag", None, [10 * SEC, 20 * SEC, 30 * SEC],
                                          *_cid(delta=-0.1), reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)  # "app restarted, project reopened"
    status = ho_mod.determine_and_record_exposure(session2, "a.bag", None, [10 * SEC, 20 * SEC, 30 * SEC],
                                                    *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED


def test_exposure_survives_restart_same_candidate_stays_independent_repeated():
    session1 = {}
    cand = _cid(delta=-0.1)
    ho_mod.determine_and_record_exposure(session1, "a.bag", None, [10 * SEC, 20 * SEC], *cand, reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)
    status = ho_mod.determine_and_record_exposure(session2, "a.bag", None, [10 * SEC, 20 * SEC], *_cid(delta=-0.1),
                                                    reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_INDEPENDENT_REPEATED


def test_exposure_survives_restart_same_candidate_same_revision_stays_independent_repeated():
    """`discovery_revision` is project-PERSISTENT (see CandidateProvenance's
    docstring) -- unlike the old session-local `discovery_generation`, it
    does not reset across a restart, so re-evaluating the exact same
    baseline+candidate at the SAME persisted revision (7 in both sessions,
    e.g. no Discovery run happened between the reveal and the restart) must
    still be `Independent — repeated evaluation`."""
    session1 = {}
    ho_mod.determine_and_record_exposure(session1, "a.bag", None, [10 * SEC, 20 * SEC],
                                          *_cid(axis="yaw", delta=-0.1, discovery_revision=7), reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)
    status = ho_mod.determine_and_record_exposure(
        session2, "a.bag", None, [10 * SEC, 20 * SEC],
        *_cid(axis="yaw", delta=-0.1, discovery_revision=7), reveal_margin_s=2.0,
    )
    assert status == ho_mod.STATUS_INDEPENDENT_REPEATED


def test_exposure_revision_advance_after_restart_exposes_even_identical_candidate():
    """The precise bug this phase fixes: candidate B revealed at persisted
    revision 7 in session 1; app restarts (revision correctly restored as 7,
    never reset to 0 -- see test above); Fine Scan runs once more, advancing
    the persisted revision to 8; the *exact same* candidate transform (same
    axis/delta/baseline, same fingerprint) is re-evaluated against the same
    Hold-out slice. Because the revision genuinely advanced since the
    reveal, this must be `Exploratory -- discovery updated`, even though
    `candidate_id` is numerically identical to what was first shown --
    tuning happened after the reveal, so it still counts as exposure. A
    session-local generation counter that reset to 0 on restart would have
    missed this entirely (1 > 7 is False); a persistent one that continues
    to 8 catches it (8 > 7 is True)."""
    session1 = {}
    ho_mod.determine_and_record_exposure(session1, "a.bag", None, [10 * SEC, 20 * SEC],
                                          *_cid(axis="yaw", delta=-0.1, discovery_revision=7), reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)
    status = ho_mod.determine_and_record_exposure(
        session2, "a.bag", None, [10 * SEC, 20 * SEC],
        *_cid(axis="yaw", delta=-0.1, discovery_revision=8), reveal_margin_s=2.0,
    )
    assert status == ho_mod.STATUS_EXPOSED_DISCOVERY_UPDATED


def test_exposure_survives_restart_copied_bag_still_folds_into_same_source():
    session1 = {}
    sig = _sig2()
    ho_mod.determine_and_record_exposure(session1, "C:/data/bag_A.bag", sig, [10 * SEC, 20 * SEC],
                                          *_cid(delta=-0.1), reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)
    status = ho_mod.determine_and_record_exposure(session2, "C:/data/bag_A_copy.bag", sig, [10 * SEC, 20 * SEC],
                                                    *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED
    assert len(session2) == 1  # still one source after the copy folds in


def test_exposure_survives_restart_partial_overlap_still_detected():
    session1 = {}
    original = [i * SEC for i in range(50)]
    ho_mod.determine_and_record_exposure(session1, "a.bag", None, original, *_cid(delta=-0.1), reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)
    resampled = original[1:-1]  # 48/50 overlap
    status = ho_mod.determine_and_record_exposure(session2, "a.bag", None, resampled,
                                                    *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED


def test_exposure_survives_restart_truly_unseen_slice_is_independent_new_slice():
    session1 = {}
    revealed = [t for t in range(100 * SEC, 121 * SEC, SEC)]
    ho_mod.determine_and_record_exposure(session1, "a.bag", None, revealed, *_cid(delta=-0.1), reveal_margin_s=2.0)

    session2 = _simulate_restart(session1)
    unseen = [t for t in range(500 * SEC, 521 * SEC, SEC)]
    status = ho_mod.determine_and_record_exposure(session2, "a.bag", None, unseen,
                                                    *_cid(axis="tx", delta=10.0, source="quick"), reveal_margin_s=2.0)
    assert status == ho_mod.STATUS_INDEPENDENT_NEW_SLICE


def test_serialization_never_touches_rt_or_calibration_state(monkeypatch):
    """End-to-end: evaluate_holdout -> record exposure -> serialize ->
    (simulated restart) -> deserialize -> classify again -- none of it may
    touch baseline R/t, and nothing on this path is a Solution/project/YAML
    write (evaluate_holdout and the (de)serialize functions take no such
    object)."""
    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=5)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")

    result = evaluate_holdout(frames, R, t, cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
                               holdout_bag="a.bag", holdout_signature=_sig2(), discovery_revision=0)
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)

    registry = {}
    ho_mod.determine_and_record_exposure(registry, result.provenance.bag, result.signature,
                                          result.provenance.usable_timestamps, result.candidate_identity,
                                          result.candidate_provenance, reveal_margin_s=2.0)
    data = ho_mod.serialize_exposure_registry(registry)
    restored, discovery_revision, warnings = ho_mod.deserialize_exposure_registry(data)
    assert warnings == []
    assert discovery_revision == 0
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)

    status_after_restart = ho_mod.determine_and_record_exposure(
        restored, result.provenance.bag, result.signature, result.provenance.usable_timestamps,
        result.candidate_identity, result.candidate_provenance, reveal_margin_s=2.0,
    )
    assert status_after_restart == ho_mod.STATUS_INDEPENDENT_REPEATED
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)


def test_evaluate_holdout_never_records_exposure_itself():
    """A failed or cancelled Hold-out run must never be treated as exposure
    -- structurally guaranteed here: evaluate_holdout's signature has no
    registry parameter at all, so it is physically incapable of recording
    anything regardless of ok/cancelled. Recording is the caller's decision,
    made only once a result is actually shown (see step_verify.py's
    _on_holdout_ready)."""
    import inspect
    params = inspect.signature(evaluate_holdout).parameters
    assert "registry" not in params
    assert not any("exposure" in name for name in params)


def test_evaluate_holdout_never_advances_discovery_revision_itself():
    """The persistent discovery_revision counter (see CandidateProvenance's
    docstring) only ever advances from a *successful Perturbation Grid or
    Fine Scan* result (see step_verify.py's _note_discovery_success, called
    only from _on_pert_ready/_on_fine_ready) -- never from a Hold-out run.
    Structurally guaranteed here the same way as the test above:
    evaluate_holdout only ever *reads* discovery_revision (an int it copies
    onto the returned candidate_provenance unchanged, see the parameter's
    docstring) -- it holds no counter of its own and returns no incremented
    value, so it is physically incapable of bumping anything regardless of
    how many times, or how successfully, it runs."""
    import inspect
    params = inspect.signature(evaluate_holdout).parameters
    assert "discovery_revision" in params
    assert params["discovery_revision"].default == 0  # a passthrough value, not a counter this function owns


def test_failed_and_cancelled_holdout_results_carry_no_provenance():
    """The concrete mechanism the UI guard in _on_holdout_ready relies on: a
    failed (no usable frames) or cancelled result's `provenance` is always
    None, regardless of whether `candidate_identity` happened to be computed
    -- so `result.provenance is not None and ...` alone is enough to skip
    exposure recording for either case, exactly reproducing the UI's guard
    condition here at the core level."""
    rejected_frame = LoadedFrame(
        timeline_index=0, image_bgr=np.zeros((2, 2, 3), np.uint8), cloud_xyz=np.zeros((0, 3), np.float32),
        lidar_timestamp_ns=0, camera_timestamp_ns=100_000_000, signed_sync_offset_ms=100.0,
        abs_sync_offset_ms=100.0, sync_rejected=True, cache=None,
    )
    cand = CandidateSpec(axis="yaw", delta=0.1, source="fine")

    failed = evaluate_holdout([rejected_frame], np.eye(3), np.zeros(3), cand, _camera(), holdout_bag="a.bag")
    assert not failed.ok
    assert failed.provenance is None

    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=3)
    cancelled = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(), holdout_bag="a.bag",
                                  should_cancel=lambda: True)
    assert cancelled.cancelled
    assert cancelled.provenance is None

    for result in (failed, cancelled):
        would_record = (
            result.ok and not result.cancelled and result.provenance is not None
            and result.provenance.bag and result.candidate_identity is not None
        )
        assert not would_record


# ------------------------------------------------------- bootstrap: identity


def test_bootstrap_result_records_non_overlapping_block_method():
    paired = [(i * 3 * SEC, -1.0) for i in range(10)]
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=100, min_blocks=5, tolerance_px=0.05,
                                          rng=np.random.default_rng(0))
    assert result.method == "non_overlapping_block" == ho_mod.DEFAULT_BOOTSTRAP_METHOD


def test_bootstrap_rejects_unknown_method():
    with pytest.raises(ValueError):
        ho_mod._run_block_bootstrap([(0, -1.0), (SEC, -1.0)], block_s=2.0, iterations=10, min_blocks=1,
                                     tolerance_px=0.05, method="moving_block")


def test_bootstrap_reports_effective_block_structure():
    # 3 blocks of 5 frames each, 3s apart (each block isolated at 2.0s width)
    paired = []
    for b in range(3):
        for j in range(5):
            paired.append((b * 3 * SEC + j * 100_000_000, -1.0))
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=100, min_blocks=2, tolerance_px=0.05,
                                          rng=np.random.default_rng(0))
    assert result.n_comparable_frames == 15
    assert result.n_blocks == 3
    assert result.frames_per_block_median == 5
    assert result.frames_per_block_min == 5 and result.frames_per_block_max == 5


def test_bootstrap_50_frames_but_few_blocks_still_unavailable():
    """Many frames packed into very few temporal blocks must not be treated
    as enough independent data -- effective block count gates CI
    availability, not raw frame count."""
    paired = [(i * 10_000_000, -1.0) for i in range(50)]  # 50 frames spanning only 0.5s total
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=100, min_blocks=5, tolerance_px=0.05,
                                          rng=np.random.default_rng(0))
    assert result.n_comparable_frames == 50
    assert result.n_blocks < 5
    assert not result.ok
    assert "too few independent temporal blocks" in result.reason


def test_bootstrap_sparse_sampling_flagged_when_most_blocks_are_single_frame():
    # 8 blocks, each 3s apart (wider than the 2.0s block), one frame each.
    paired = [(i * 3 * SEC, -1.0 if i % 2 == 0 else 0.5) for i in range(8)]
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=200, min_blocks=5, tolerance_px=0.05,
                                          rng=np.random.default_rng(0))
    assert result.n_blocks == 8
    assert result.frames_per_block_max == 1
    assert result.sparse_sampling is True


def test_bootstrap_dense_sampling_not_flagged_sparse():
    paired = []
    for b in range(6):
        for j in range(4):  # 4 frames per block -- not single-frame
            paired.append((b * 3 * SEC + j * 200_000_000, -1.0))
    result = ho_mod._run_block_bootstrap(paired, block_s=2.0, iterations=200, min_blocks=5, tolerance_px=0.05,
                                          rng=np.random.default_rng(0))
    assert result.sparse_sampling is False


def test_exclusion_margin_and_bootstrap_block_duration_are_independent():
    """Changing the Discovery Exclusion Margin must never change which
    bootstrap block a frame falls into (or vice versa) -- they are separate
    parameters to separate functions with no shared state."""
    image, cloud = _make_scene()
    frames_a = _holdout_frames(image, cloud, n=6)
    frames_b = _holdout_frames(image, cloud, n=6)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")
    kwargs = dict(edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
                  bootstrap_rng=np.random.default_rng(0))

    # exclusion_margin_s is not even a parameter of evaluate_holdout (it only
    # affects which timestamps partition_independent_frames hands it as
    # `frames` upstream) -- varying bootstrap_block_s alone must not require,
    # or implicitly touch, any margin-related state.
    r1 = evaluate_holdout(frames_a, np.eye(3), np.zeros(3), cand, _camera(), bootstrap_block_s=1.0, **kwargs)
    r2 = evaluate_holdout(frames_b, np.eye(3), np.zeros(3), cand, _camera(), bootstrap_block_s=5.0, **kwargs)
    assert r1.bootstrap.block_s == 1.0
    assert r2.bootstrap.block_s == 5.0
    # Same frames, same candidate -- only the block duration differs, so the
    # underlying paired deltas (and hence the point estimate) are identical.
    assert r1.candidate_point.median_delta_frame_p95_px == r2.candidate_point.median_delta_frame_p95_px


# --------------------------------------------------- diagnostic status wiring


def test_assess_holdout_shows_all_status_labels():
    import gui.core.evaluation.diagnostics as dg

    for status in (ho_mod.STATUS_INDEPENDENT, ho_mod.STATUS_INDEPENDENT_NEW_SLICE,
                   ho_mod.STATUS_INDEPENDENT_REPEATED,
                   ho_mod.STATUS_EXPOSED_CANDIDATE_CHANGED, ho_mod.STATUS_EXPOSED_DISCOVERY_UPDATED):
        item = dg.assess_holdout(None, status)
        # even with result=None the function returns early, but exercising
        # every status value here guards against a KeyError in STATUS_LABELS
        assert item.title in ("Independent hold-out validation", "Exploratory validation")
        # only the two EXPOSED_STATUSES rename/cap the category
        assert (item.title == "Exploratory validation") == (status in ho_mod.EXPOSED_STATUSES)


def test_assess_holdout_new_slice_status_is_independent_not_capped(monkeypatch):
    """STATUS_INDEPENDENT_NEW_SLICE is independent evidence, not exploratory
    -- Diagnostic Evidence must show the same status label the UI does, and
    must not cap its strength the way an EXPOSED_STATUSES value would."""
    import gui.core.evaluation.diagnostics as dg

    result = _uniform_improvement_holdout_result(monkeypatch)
    item = dg.assess_holdout(result, ho_mod.STATUS_INDEPENDENT_NEW_SLICE)
    assert item.title == "Independent hold-out validation"
    assert item.strength == dg.STRONG
    assert dict(item.metrics)["Validation status"] == ho_mod.STATUS_LABELS[ho_mod.STATUS_INDEPENDENT_NEW_SLICE]


def test_assess_holdout_metrics_include_validation_status_label(monkeypatch):
    import gui.core.evaluation.diagnostics as dg

    result = _uniform_improvement_holdout_result(monkeypatch)
    item = dg.assess_holdout(result, ho_mod.STATUS_INDEPENDENT_REPEATED)
    labels = dict(item.metrics)
    assert labels.get("Validation status") == ho_mod.STATUS_LABELS[ho_mod.STATUS_INDEPENDENT_REPEATED]


def test_assess_holdout_reports_effective_temporal_blocks_metric(monkeypatch):
    import gui.core.evaluation.diagnostics as dg

    result = _uniform_improvement_holdout_result(monkeypatch)
    item = dg.assess_holdout(result, ho_mod.STATUS_INDEPENDENT)
    labels = dict(item.metrics)
    assert "Effective Temporal Blocks" in labels
    assert "Bootstrap method" in labels and labels["Bootstrap method"] == ho_mod.DEFAULT_BOOTSTRAP_METHOD


def test_full_flow_never_mutates_rt_or_extrinsic_state(monkeypatch):
    """End-to-end: evaluate_holdout -> exposure classification/recording ->
    diagnostics, exactly the sequence step_verify.py's _on_holdout_ready runs
    -- none of it may touch the baseline R/t the caller passed in, and
    nothing resembling a Solution/project/YAML write exists anywhere on this
    path (evaluate_holdout takes no such object at all)."""
    import gui.core.evaluation.diagnostics as dg

    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()
    image, cloud = _make_scene()
    frames = _holdout_frames(image, cloud, n=5)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="full")

    result = evaluate_holdout(frames, R, t, cand, _camera(),
                               edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
                               holdout_bag="holdout.bag", holdout_signature=_sig2(), discovery_revision=0)
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)

    registry = {}
    status = ho_mod.determine_and_record_exposure(
        registry, result.provenance.bag, result.signature, result.provenance.usable_timestamps,
        result.candidate_identity, result.candidate_provenance, reveal_margin_s=2.0,
    )
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)

    report = dg.build_diagnostic_report(dg.DiagnosticInputs(holdout=result, holdout_status=status))
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)
    assert any(i.category == dg.CAT_HOLDOUT for i in report.evidence)

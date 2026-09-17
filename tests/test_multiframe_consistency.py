"""Unit + end-to-end tests for gui.core.evaluation.multiframe_consistency."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from gui.core.evaluation.edge_alignment import EdgeAlignmentParams
from gui.core.evaluation.multiframe_consistency import (
    compute_robust_stats,
    evaluate_multiframe_consistency,
    flag_outliers_hampel,
    sample_frame_indices,
)
from gui.core.project import Camera
from gui.core.solve import Solution


# ------------------------------------------------------------------ sampling


def test_sample_frame_indices_covers_full_range_evenly():
    idx = sample_frame_indices(10_000, 100)
    assert len(idx) == 100
    assert idx[0] == 0
    assert idx[-1] == 9_999
    assert idx == sorted(set(idx))


def test_sample_frame_indices_all_when_fewer_frames_than_requested():
    assert sample_frame_indices(5, 100) == [0, 1, 2, 3, 4]


def test_sample_frame_indices_empty_timeline():
    assert sample_frame_indices(0, 100) == []


# ------------------------------------------------------------- robust stats


def test_compute_robust_stats_known_values():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_robust_stats(values)
    assert stats["median"] == 3.0
    assert stats["q1"] == 2.0
    assert stats["q3"] == 4.0
    assert stats["iqr"] == 2.0


def test_flag_outliers_hampel_detects_single_spike():
    values = np.array([2.0, 2.1, 1.9, 2.05, 1.95, 20.0])
    robust = compute_robust_stats(values)
    is_outlier, z = flag_outliers_hampel(values, robust, k=3.0)
    assert is_outlier[-1]
    assert not is_outlier[:-1].any()


def test_flag_outliers_hampel_degenerate_mad_falls_back():
    values = np.array([2.0, 2.0, 2.0, 2.0, 50.0])
    robust = compute_robust_stats(values)  # MAD is 0 here
    is_outlier, _ = flag_outliers_hampel(values, robust, k=3.0)
    assert is_outlier[-1]


# --------------------------------------------------------- end-to-end mixture
#
# A fixed extrinsic evaluated over a synthetic "timeline" of step-edge scenes:
# most frames have the depth step aligned with the image edge (low, near-
# identical error every time); a few are deliberately built with the step
# shifted far from the image edge (should read as clear outliers); a few more
# have no data at all (should count as failed, not silently vanish).

# Kept small (and the grid below correspondingly sparse-but-radius-safe) since
# this scene is projected and evaluated ~40 times in one test.
_WIDTH, _HEIGHT = 320, 240
_FX = _FY = 250.0
_CX, _CY = 160.0, 120.0


def _camera() -> Camera:
    return Camera(fx=_FX, fy=_FY, cx=_CX, cy=_CY)


def _identity_solution() -> Solution:
    return Solution(ok=True, R=np.eye(3), t=np.zeros(3), rmse=0.0, scene_ids=[], n_pairs=0)


def _frame(step_col: float):
    image = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    image[:, int(_CX):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Spacing must stay under the default radius_px=3.0 in both axes, or
    # neighbouring grid points straddling the depth step never see each other.
    u = np.repeat(np.linspace(0, _WIDTH - 1, 140), 110)
    v = np.tile(np.linspace(0, _HEIGHT - 1, 110), 140)
    z = np.where(u < step_col, 5.0, 10.0)
    x = (u - _CX) * z / _FX
    y = (v - _CY) * z / _FY
    cloud = np.column_stack([x, y, z]).astype(np.float32)
    return image, cloud


def _kind_of(idx: int) -> str:
    if idx % 10 == 0:
        return "missing"
    if idx % 7 == 0:
        return "misaligned"
    return "aligned"


def _frame_loader(idx: int):
    kind = _kind_of(idx)
    if kind == "missing":
        return None
    step_col = _CX if kind == "aligned" else _CX + 150.0
    image, cloud = _frame(step_col)
    # A benign, constant 1ms sync offset (camera_timestamp_ns = lidar + 1ms)
    # -- this loader is for the outlier/missing tests below, which run with
    # max_sync_offset_ms=None (no rejection), so the exact value here is not
    # otherwise exercised.
    return image, cloud, idx, float(idx), idx + 1_000_000


def test_evaluate_multiframe_consistency_separates_outliers_and_missing():
    result = evaluate_multiframe_consistency(
        frame_loader=_frame_loader,
        n_total_timeline=40,
        n_samples=40,
        sol=_identity_solution(),
        camera=_camera(),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        hampel_k=3.0,
        top_k_worst=10,
    )

    missing = [i for i in range(40) if _kind_of(i) == "missing"]
    misaligned = [i for i in range(40) if _kind_of(i) == "misaligned"]

    assert result.n_total == 40
    assert result.n_failed == len(missing)
    assert result.n_valid == 40 - len(missing)
    assert np.isclose(result.failure_ratio, len(missing) / 40)

    outlier_indices = {f.timeline_index for f in result.frame_results if f.is_outlier}
    assert set(misaligned).issubset(outlier_indices)

    worst_indices = {f.timeline_index for f in result.worst_frames}
    assert set(misaligned).issubset(worst_indices)

    aligned_means = [f.mean_px for f in result.frame_results if _kind_of(f.timeline_index) == "aligned" and f.ok]
    misaligned_means = [f.mean_px for f in result.frame_results if _kind_of(f.timeline_index) == "misaligned" and f.ok]
    assert min(misaligned_means) > max(aligned_means)


def test_evaluate_multiframe_consistency_respects_cancel():
    calls = []

    def loader(idx):
        calls.append(idx)
        return _frame_loader(idx)

    result = evaluate_multiframe_consistency(
        frame_loader=loader,
        n_total_timeline=40,
        n_samples=40,
        sol=_identity_solution(),
        camera=_camera(),
        should_cancel=lambda: len(calls) >= 5,
    )
    assert len(calls) == 5
    assert result.n_total == 5


def test_evaluate_multiframe_consistency_can_be_interrupted_mid_run():
    """The test above only proves "if should_cancel() is already True, the
    loop stops" -- which a should_cancel that is a queued, delayed operation
    could also satisfy, just late. This test proves the stronger property the
    GUI actually depends on: a real OS thread mid-loop notices a
    threading.Event.set() from another thread and stops promptly, without
    processing every requested frame.

    Mirrors production exactly at the synchronization level: the GUI's Cancel
    button calls _Worker.cancel_multiframe() (a plain, un-queued method call
    across threads -- not a Qt signal/slot -- so it is not subject to the
    worker's Qt event loop at all) which does nothing but
    `threading.Event.set()`, and the worker thread's loop polls that same
    Event's `.is_set` once per frame. Using a real threading.Thread here
    instead of QThread exercises that identical Event-based handoff without
    needing PySide6.
    """
    cancel_event = threading.Event()
    n_requested = 200

    def slow_loader(idx: int):
        time.sleep(0.01)  # simulate slow bag I/O so cancellation mid-loop is meaningful
        return _frame_loader(idx)

    result_holder: dict = {}

    def run():
        result_holder["result"] = evaluate_multiframe_consistency(
            frame_loader=slow_loader,
            n_total_timeline=n_requested,
            n_samples=n_requested,
            sol=_identity_solution(),
            camera=_camera(),
            edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
            should_cancel=cancel_event.is_set,
        )

    worker_thread = threading.Thread(target=run)
    t0 = time.perf_counter()
    worker_thread.start()
    time.sleep(0.08)          # let a handful of frames process (~8 at 10ms each)
    cancel_event.set()        # cancel WHILE the worker is still looping
    worker_thread.join(timeout=10.0)
    elapsed = time.perf_counter() - t0

    assert not worker_thread.is_alive(), "worker did not stop after the cancel Event was set"
    result = result_holder["result"]
    assert result.n_total < n_requested, "ran to completion despite mid-run cancellation"
    # A run that ignored cancellation would take >= 200 * 10ms = 2s; stopping
    # promptly after the ~80ms mark leaves generous headroom either way.
    assert elapsed < 1.0, f"took {elapsed:.2f}s -- cancellation did not take effect promptly"


# ----------------------------------------------------------- sync offset
#
# On a moving vehicle a loose camera/LiDAR pairing produces exactly the same
# symptom as a rotation error: a pixel-space shift. These tests hold the
# geometry fixed (every frame is the perfectly aligned scene) and vary only
# the sync offset the loader reports, to make sure a badly-synced frame is
# kept out of the geometric statistics rather than silently counted as
# evidence about the extrinsic.


def _sync_kind_of(idx: int) -> str:
    if idx % 10 == 0:
        return "missing"
    if idx % 13 == 0:
        return "bad_sync"
    return "aligned"


def _sync_frame_loader(idx: int):
    kind = _sync_kind_of(idx)
    if kind == "missing":
        return None
    image, cloud = _frame(_CX)  # always the perfectly aligned scene
    # idx doubles as a (tiny, unrealistic-but-fine-for-arithmetic) LiDAR
    # timestamp in ns; camera_timestamp_ns is offset from it by the desired
    # signed amount so evaluate_multiframe_consistency's own derivation --
    # (camera - lidar) / 1e6 -- reproduces exactly 80.0ms / 3.0ms.
    offset_ns = 80_000_000 if kind == "bad_sync" else 3_000_000
    return image, cloud, idx, float(idx), idx + offset_ns


def test_signed_sync_offset_sign_convention():
    """Nails down the definition itself: signed = (camera - lidar) / 1e6.
    Positive means the camera image is the *later* of the two; negative means
    the camera image came first."""
    lidar_t_ns = 10_000_000_000

    def loader_camera_later(_idx):
        image, cloud = _frame(_CX)
        return image, cloud, lidar_t_ns, 0.0, lidar_t_ns + 7_000_000  # +7ms

    def loader_camera_earlier(_idx):
        image, cloud = _frame(_CX)
        return image, cloud, lidar_t_ns, 0.0, lidar_t_ns - 7_000_000  # -7ms

    for loader, expected_sign in ((loader_camera_later, +1), (loader_camera_earlier, -1)):
        result = evaluate_multiframe_consistency(
            frame_loader=loader,
            n_total_timeline=5,
            n_samples=5,
            sol=_identity_solution(),
            camera=_camera(),
            edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        )
        signed = [f.signed_sync_offset_ms for f in result.frame_results]
        assert all(np.isclose(v, expected_sign * 7.0) for v in signed), signed
        assert all(np.isclose(f.abs_sync_offset_ms, 7.0) for f in result.frame_results)


def test_evaluate_multiframe_consistency_rejects_bad_sync_separately_from_geometric_failure():
    result = evaluate_multiframe_consistency(
        frame_loader=_sync_frame_loader,
        n_total_timeline=40,
        n_samples=40,
        sol=_identity_solution(),
        camera=_camera(),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        max_sync_offset_ms=50.0,
    )

    missing = [i for i in range(40) if _sync_kind_of(i) == "missing"]
    bad_sync = [i for i in range(40) if _sync_kind_of(i) == "bad_sync"]
    aligned = [i for i in range(40) if _sync_kind_of(i) == "aligned"]

    assert result.n_total == 40
    assert result.n_sync_rejected == len(bad_sync)
    assert result.n_failed == len(missing)
    assert result.n_valid == len(aligned)
    assert np.isclose(result.sync_rejected_ratio, len(bad_sync) / 40)

    # A sync-rejected frame never ran the geometric metric at all, so it must
    # not appear as an ordinary failure or as a candidate for "worst frame".
    rejected_indices = {f.timeline_index for f in result.frame_results if f.sync_rejected}
    assert rejected_indices == set(bad_sync)
    for f in result.frame_results:
        if f.sync_rejected:
            assert not f.ok
            assert np.isnan(f.mean_px)
    worst_indices = {f.timeline_index for f in result.worst_frames}
    assert worst_indices.isdisjoint(rejected_indices)

    # The offset itself is still reported for every frame a camera image was
    # found for (aligned + bad_sync = 36), median/max computed exactly.
    all_offsets = np.array(
        [3.0] * len(aligned) + [80.0] * len(bad_sync)
    )
    assert result.sync_offset_median_ms == float(np.median(all_offsets))
    assert result.sync_offset_max_ms == 80.0
    assert result.sync_offset_p95_ms == float(np.percentile(all_offsets, 95))


def test_evaluate_multiframe_consistency_reports_sync_offsets_without_rejecting_by_default():
    result = evaluate_multiframe_consistency(
        frame_loader=_sync_frame_loader,
        n_total_timeline=40,
        n_samples=40,
        sol=_identity_solution(),
        camera=_camera(),
        edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        # max_sync_offset_ms left at its default (None): nothing is rejected
        # on sync alone, but the offsets must still be collected and reported.
    )
    missing = [i for i in range(40) if _sync_kind_of(i) == "missing"]
    assert result.n_sync_rejected == 0
    assert result.n_valid == 40 - len(missing)
    assert result.sync_offset_max_ms == 80.0
    assert np.isfinite(result.sync_offset_median_ms)

"""Is the current T_cam_lidar sitting at a local minimum of Edge Alignment error,
or would a tiny nudge in some axis do better?

Strictly a read-only diagnostic. Nothing here ever writes back to a `Solution`
or a project file, offers an "apply this instead" action, or decides that the
extrinsic is wrong -- it evaluates `edge_alignment.evaluate_edge_alignment` at
small, temporary copies of T near the one already computed, and reports how
the error moved. Whether that means anything is for the person reading the
numbers to judge.

Coordinate convention (worked out from `gui.core.solve.Solution.transform`,
`(R @ xyz.T).T + t`, which maps a LiDAR-frame point into the camera frame; the
camera frame is OpenCV-standard, confirmed by `solve.is_upright`'s docstring:
X right, Y down, Z forward). A rotation perturbation expressed "in the camera
frame" is applied *after* R has already mapped the point there, i.e. on the
left: `R_perturbed = R_delta @ R_original`, with `t` left untouched; a
translation perturbation is `t_perturbed = t_original + delta_m * e_axis`,
with `R` left untouched.

This is deliberately called "Independent Extrinsic Parameter Sensitivity", not
"SE(3) left composition" -- a true SE(3) left composition `T' = ΔT @ T` would
also carry translation through the rotation delta (`t' = R_delta @ t`, since a
rotation about the current origin moves where `t` points), which is not what
happens here. Each of the six parameters (roll, pitch, yaw, tx, ty, tz) is
wiggled *on its own*, holding every other parameter exactly at baseline -- the
question this module answers is "how sensitive is the error to *this one*
parameter", not "what nearby rigid transform fits better", and conflating the
two would misdescribe what the numbers mean.

Axis names follow how a camera's own attitude is normally described, but note
this is NOT the common robotics roll=X/pitch=Y/yaw=Z convention (that convention
describes a body frame with Z up; a camera's own Y axis points down, so its
"yaw" -- turning to look left/right -- rotates about Y, not Z). Every axis name
in this module's public API and any UI built on it should carry its camera
axis explicitly for exactly this reason -- see `AXIS_LABELS`:
  - yaw   / pan  = rotation about camera Y (down)    -> Ry  = "Yaw / Pan (Cam-Y)"
  - pitch / tilt = rotation about camera X (right)   -> Rx  = "Pitch / Tilt (Cam-X)"
  - roll         = rotation about camera Z (forward) -> Rz  = "Roll (Cam-Z)"
  - tx = camera X (right), ty = camera Y (down), tz = camera Z (forward)

There is no pre-existing "roll/pitch/yaw of the extrinsic" convention anywhere
else in this codebase to conflict with -- `gui.core.project.FilterBox.rotation`
defines yaw/pitch/roll too, but about the *LiDAR's* own Z/Y/X axes, for an
unrelated purpose (orienting a crop box in LiDAR space).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from gui.core.evaluation.edge_alignment import EdgeAlignmentParams, EdgeAlignmentResult, evaluate_edge_alignment
from gui.core.evaluation.edge_correspondence import EdgeOrientationMap, compute_edge_orientation_map
from gui.core.evaluation.multiframe_consistency import sample_frame_indices
from gui.core.evaluation.spatial_analysis import SpatialAnalysisResult, analyze_spatial

DEFAULT_ROTATION_DELTAS_DEG = (-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5)
DEFAULT_TRANSLATION_DELTAS_MM = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)
# "Quick" search: 3 values per axis x 6 axes = 18, minus 5 duplicate delta=0
# baselines (shared across axes) = 13 trials total -- for a fast first look
# before committing to the much heavier Full grid (9x3 + 7x3 - 5 = 43 trials).
QUICK_ROTATION_DELTAS_DEG = (-0.2, 0.0, 0.2)
QUICK_TRANSLATION_DELTAS_MM = (-10.0, 0.0, 10.0)
ROTATION_AXES = ("roll", "pitch", "yaw")
TRANSLATION_AXES = ("tx", "ty", "tz")
DEFAULT_LOCAL_MIN_TOLERANCE_PX = 0.05

# Display labels that always carry the camera axis a name maps to -- this
# module's yaw/pitch/roll is NOT the common robotics roll=X/pitch=Y/yaw=Z
# convention (see module docstring), so any UI showing a bare axis name risks
# being misread against that more familiar convention.
AXIS_LABELS = {
    "roll": "Roll (Cam-Z)",
    "pitch": "Pitch / Tilt (Cam-X)",
    "yaw": "Yaw / Pan (Cam-Y)",
    "tx": "Tx (Cam-X / Right)",
    "ty": "Ty (Cam-Y / Down)",
    "tz": "Tz (Cam-Z / Forward)",
}


def _rotation_delta(axis: str, delta_deg: float) -> np.ndarray:
    theta = np.radians(delta_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == "yaw":     # camera Y (down)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "pitch":   # camera X (right)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "roll":    # camera Z (forward / optical axis)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"unknown rotation axis: {axis!r}")


def perturb_rotation(R: np.ndarray, axis: str, delta_deg: float) -> np.ndarray:
    """R_delta(axis, delta_deg) @ R -- a camera-frame rotation applied on top
    of R, leaving the input untouched (matrix multiplication always allocates
    a new array; R itself is never written to)."""
    return _rotation_delta(axis, delta_deg) @ R


def perturb_translation(t: np.ndarray, axis: str, delta_mm: float) -> np.ndarray:
    """t + delta, along the camera axis named by `axis` -- a fresh array; the
    input `t` is never modified in place."""
    i = {"tx": 0, "ty": 1, "tz": 2}[axis]
    out = np.array(t, dtype=np.float64, copy=True)
    out[i] += delta_mm / 1000.0
    return out


def make_delta_range(min_value: float, max_value: float, step: float) -> tuple:
    """Inclusive `[min_value, max_value]` stepped by `step`, for Fine Scan's
    user-chosen range on one axis -- degrees or millimetres, same either way.
    `0.0` is always included even if not exactly on the grid, since every axis
    summary needs a baseline point to compare against.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if max_value < min_value:
        raise ValueError("max_value must be >= min_value")
    n = int(round((max_value - min_value) / step)) + 1
    # Rounded to kill float-accumulation noise (e.g. 3 * 0.1 != 0.30000000004)
    # that would otherwise make "0.0" fail to dedupe against a step landing on
    # it, or produce two near-identical deltas a user asked for as one.
    values = {round(min_value + i * step, 9) for i in range(n)}
    values.add(0.0)
    return tuple(sorted(values))


@dataclass
class LoadedFrame:
    """One pre-loaded, pairing-resolved frame plus its image-only cache --
    everything a perturbation trial needs except the extrinsic itself, so it
    can be projected and scored against many candidate T's without touching
    the bag or recomputing Canny/Sobel again."""

    timeline_index: int
    image_bgr: np.ndarray
    cloud_xyz: np.ndarray
    lidar_timestamp_ns: int
    camera_timestamp_ns: int
    signed_sync_offset_ms: float
    abs_sync_offset_ms: float
    sync_rejected: bool = False
    cache: Optional[EdgeOrientationMap] = None  # None exactly when sync_rejected


def prepare_frames(
    frame_loader: Callable[[int], Optional[tuple]],
    n_total_timeline: int,
    n_samples: int,
    max_sync_offset_ms: Optional[float] = None,
    edge_params: EdgeAlignmentParams = EdgeAlignmentParams(),
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list:
    """Sample, pair, and cache the fixed frame set every perturbation trial
    re-scores -- the same sampling (`sample_frame_indices`) and nearest-pairing
    policy `multiframe_consistency.evaluate_multiframe_consistency` uses, just
    loaded once up front instead of loaded-scored-discarded per frame, because
    here the *same* frames are re-evaluated once per perturbation (dozens of
    times), not once each.

    `frame_loader(idx)` has the identical contract as
    `evaluate_multiframe_consistency`'s: returns `(image_bgr, cloud_xyz,
    lidar_timestamp_ns, offset_s, camera_timestamp_ns)` or `None`.

    A frame missing entirely (loader returns None) is dropped -- there is
    nothing to hold onto. A frame whose sync offset exceeds
    `max_sync_offset_ms` is kept (with `sync_rejected=True`, `cache=None`) so
    every perturbation trial excludes it identically; dropping it here instead
    would still exclude it everywhere, but keeping the record lets a caller
    report *why* the frame count is what it is.
    """
    indices = sample_frame_indices(n_total_timeline, n_samples)
    total = len(indices)
    frames: list = []
    for done, idx in enumerate(indices, start=1):
        if should_cancel is not None and should_cancel():
            break
        loaded = frame_loader(idx)
        if loaded is not None:
            image_bgr, cloud_xyz, lidar_t_ns, _offset_s, camera_t_ns = loaded
            signed_ms = (camera_t_ns - lidar_t_ns) / 1e6
            abs_ms = abs(signed_ms)
            rejected = max_sync_offset_ms is not None and abs_ms > max_sync_offset_ms
            cache = None if rejected else compute_edge_orientation_map(
                image_bgr, edge_params.canny_low, edge_params.canny_high,
            )
            frames.append(LoadedFrame(
                timeline_index=idx, image_bgr=image_bgr, cloud_xyz=cloud_xyz,
                lidar_timestamp_ns=lidar_t_ns, camera_timestamp_ns=camera_t_ns,
                signed_sync_offset_ms=signed_ms, abs_sync_offset_ms=abs_ms,
                sync_rejected=rejected, cache=cache,
            ))
        if progress is not None:
            progress(done, total)
    return frames


def frame_from_current(image_bgr: np.ndarray, cloud_xyz: np.ndarray,
                        edge_params: EdgeAlignmentParams = EdgeAlignmentParams()) -> LoadedFrame:
    """Current-Frame mode's frame set is a single `LoadedFrame` built directly
    from whatever is already displayed -- no bag, no pairing, no sync
    question, and (critically) the exact same `image_bgr`/`cloud_xyz` objects
    the screen was drawn from, for the same identity-based pairing guarantee
    `_eval_current_frame` already relies on. `evaluate_perturbation_grid`
    itself cannot tell this apart from a one-frame Multi-frame run."""
    return LoadedFrame(
        timeline_index=-1, image_bgr=image_bgr, cloud_xyz=cloud_xyz,
        lidar_timestamp_ns=0, camera_timestamp_ns=0,
        signed_sync_offset_ms=0.0, abs_sync_offset_ms=0.0, sync_rejected=False,
        cache=compute_edge_orientation_map(image_bgr, edge_params.canny_low, edge_params.canny_high),
    )


@dataclass
class FrameMetric:
    """One trial's result for one specific frame -- kept per-frame (not just
    pooled into the trial's overall numbers) so a candidate can be compared
    against baseline frame-by-frame instead of only through a pooled-point
    statistic that a single edge-rich frame could dominate."""

    ok: bool
    p95_px: float = float("nan")
    median_px: float = float("nan")


@dataclass
class PerturbationPoint:
    delta: float   # degrees for a rotation axis, millimetres for a translation axis
    ok: bool
    reason: str = ""
    mean_px: float = float("nan")
    median_px: float = float("nan")
    p95_px: float = float("nan")
    max_px: float = float("nan")
    match_rate: float = float("nan")
    n_edge_points: int = 0
    n_valid_frames: int = 0
    n_failed_frames: int = 0
    n_sync_rejected_frames: int = 0
    delta_median_px: float = float("nan")  # vs this axis's own delta=0 point (pooled); negative = improvement
    delta_p95_px: float = float("nan")     # pooled -- kept as a diagnostic even where frame-balanced ranking is used
    coverage_warning: str = ""
    spatial: Optional[SpatialAnalysisResult] = None

    # Frame-balanced fields -- this trial's own per-frame results (keyed by
    # LoadedFrame.timeline_index), and the two standalone (non-paired)
    # descriptive stats derived from them.
    per_frame: dict = field(default_factory=dict)   # timeline_index -> FrameMetric
    median_frame_p95_px: float = float("nan")
    median_frame_median_px: float = float("nan")

    # Paired vs baseline -- computed only over frames where BOTH this point
    # and baseline succeeded (see summarize_axis); never mixes one frame's
    # candidate result against a different frame's baseline result.
    n_comparable_frames: int = 0
    n_baseline_only_valid: int = 0
    n_candidate_only_valid: int = 0
    n_both_failed: int = 0
    n_improved_frames: int = 0
    n_worsened_frames: int = 0
    n_unchanged_frames: int = 0
    improved_frame_ratio: float = float("nan")
    median_delta_frame_p95_px: float = float("nan")
    p95_delta_frame_p95_px: float = float("nan")


RANKING_METRIC_NAMES = {
    "pooled_p95": "Pooled P95",
    "median_frame_p95": "Median of per-frame P95",
}


@dataclass
class AxisSensitivity:
    axis: str
    unit: str  # "deg" | "mm"
    points: list = field(default_factory=list)          # PerturbationPoint, sorted by delta
    baseline: Optional[PerturbationPoint] = None          # the delta=0 point
    # "pooled_p95" (point-pooled P95 -- what Current-Frame mode always uses,
    # since with one frame it is numerically identical to the frame-balanced
    # version anyway) or "median_frame_p95" (Multi-frame mode's default: the
    # median of each trial's own per-frame P95 values, so one edge-rich frame
    # cannot dominate which perturbation looks "best"). Only decides which
    # value lowest_point/ranking_*/is_local_minimum are judged by -- both
    # underlying numbers are always present on every PerturbationPoint.
    ranking_metric: str = "pooled_p95"
    lowest_point: Optional[PerturbationPoint] = None      # best by `ranking_metric` (may be baseline)
    # Legacy name, kept for existing callers: despite "p95" it holds the
    # improvement in whichever `ranking_metric` was used -- under
    # "median_frame_p95" that is NOT a pooled-P95 improvement. New code should
    # read `ranking_improvement_px` (same value) alongside `ranking_metric_name`.
    improvement_p95_px: float = float("nan")
    is_local_minimum: bool = True
    # The ranking metric's value at baseline and at `lowest_point`, and
    # baseline - lowest (positive = the lowest tested point is lower).
    ranking_baseline_px: float = float("nan")
    ranking_best_px: float = float("nan")
    ranking_improvement_px: float = float("nan")

    @property
    def ranking_metric_name(self) -> str:
        """Human-readable name of `ranking_metric`, derived rather than stored
        so it can never disagree with the metric actually used."""
        return RANKING_METRIC_NAMES.get(self.ranking_metric, self.ranking_metric)


@dataclass
class PerturbationResult:
    mode: str  # "current_frame" | "multi_frame"
    axes: dict = field(default_factory=dict)  # axis name -> AxisSensitivity
    n_frames_used: int = 0
    n_sync_rejected: int = 0
    cancelled: bool = False
    reason: str = ""


def _pool_for_spatial(results: list) -> "tuple[EdgeAlignmentResult, int, int]":
    """Concatenate every `ok=True` result's per-point arrays into one
    EdgeAlignmentResult-shaped object, so `spatial_analysis.analyze_spatial`
    can be reused exactly as-is instead of learning a multi-frame variant."""
    ok_results = [r for r in results if r.ok]
    if not ok_results:
        return EdgeAlignmentResult(ok=False, reason="no valid frame"), 0, len(results)

    pixels = np.concatenate([r.edge_pixels for r in ok_results])
    depths = np.concatenate([r.edge_depths for r in ok_results])
    errors = np.concatenate([r.edge_errors_px for r in ok_results])
    matched = np.concatenate([r.edge_matched for r in ok_results])
    n_edge_points = int(errors.shape[0])
    n_matched = int(matched.sum())

    pooled = EdgeAlignmentResult(
        ok=True,
        mean_px=float(np.mean(errors)),
        median_px=float(np.median(errors)),
        p95_px=float(np.percentile(errors, 95)),
        max_px=float(np.max(errors)),
        n_projected=sum(r.n_projected for r in ok_results),
        n_edge_points=n_edge_points,
        n_matched=n_matched,
        n_unmatched=n_edge_points - n_matched,
        match_rate=n_matched / n_edge_points if n_edge_points else float("nan"),
        edge_pixels=pixels, edge_depths=depths, edge_errors_px=errors, edge_matched=matched,
    )
    return pooled, len(ok_results), len(results) - len(ok_results)


def _evaluate_trial(
    frames: list, R: np.ndarray, t: np.ndarray, camera,
    min_range: float, max_range: float, edge_params: EdgeAlignmentParams,
) -> "tuple[EdgeAlignmentResult, int, int, Optional[SpatialAnalysisResult], dict]":
    """Project + score every usable (non sync-rejected) frame against one
    candidate (R, t): pool the results for Spatial analysis (unchanged from
    before), and *also* keep each frame's own P95/median individually (keyed
    by `LoadedFrame.timeline_index`) -- pooling alone would let a single
    edge-rich frame dominate the pooled statistic, which is fine for Spatial
    (it is meant to describe the whole sampled scene) but wrong for deciding
    which candidate is "better" in Multi-frame mode, where every frame should
    count once regardless of how many edge points it happened to contribute.

    LiDAR projection is redone here every time -- it depends on T and cannot
    be cached; the image-side Canny/orientation work each frame already
    carries in `frame.cache` is reused unchanged.
    """
    from gui.core import verify
    from gui.core.solve import Solution

    sol = Solution(ok=True, R=R, t=t, rmse=0.0, scene_ids=[], n_pairs=0)
    results = []
    per_frame: dict = {}
    image_shape = None
    for frame in frames:
        if frame.sync_rejected:
            continue
        h, w = frame.image_bgr.shape[:2]
        image_shape = image_shape or (w, h)
        pr = verify.project_cloud(frame.cloud_xyz, sol, camera, w, h, min_range=min_range, max_range=max_range)
        if pr.n_visible == 0:
            result = EdgeAlignmentResult(ok=False, reason="화면에 투영된 점이 없습니다.")
        else:
            result = evaluate_edge_alignment(
                frame.image_bgr, pr.uv, pr.depth, edge_params, cached_orientation_map=frame.cache,
            )
        results.append(result)
        per_frame[frame.timeline_index] = FrameMetric(
            ok=result.ok, p95_px=result.p95_px, median_px=result.median_px,
        )

    pooled, n_valid, n_failed = _pool_for_spatial(results)
    spatial = analyze_spatial(pooled, *image_shape) if image_shape and pooled.ok else None
    return pooled, n_valid, n_failed, spatial, per_frame


def _pair_frame_metrics(baseline_per_frame: dict, candidate_per_frame: dict, tolerance: float) -> dict:
    """Compare a candidate's per-frame results against baseline's, frame by
    frame -- never a candidate's frame N against baseline's frame M. Only
    frames where BOTH succeeded ("comparable") contribute to the paired delta
    statistics; a frame either side failed on is counted, never silently
    dropped (see the module's Failed-frame-coverage requirement).
    """
    deltas = []
    n_comparable = n_baseline_only = n_candidate_only = n_both_failed = 0
    n_improved = n_worsened = n_unchanged = 0
    for idx, b in baseline_per_frame.items():
        c = candidate_per_frame.get(idx)
        c_ok = c is not None and c.ok
        if b.ok and c_ok:
            n_comparable += 1
            d = c.p95_px - b.p95_px
            deltas.append(d)
            if d < -tolerance:
                n_improved += 1
            elif d > tolerance:
                n_worsened += 1
            else:
                n_unchanged += 1
        elif b.ok and not c_ok:
            n_baseline_only += 1
        elif c_ok and not b.ok:
            n_candidate_only += 1
        else:
            n_both_failed += 1

    deltas_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n_comparable_frames": n_comparable,
        "n_baseline_only_valid": n_baseline_only,
        "n_candidate_only_valid": n_candidate_only,
        "n_both_failed": n_both_failed,
        "n_improved_frames": n_improved,
        "n_worsened_frames": n_worsened,
        "n_unchanged_frames": n_unchanged,
        "improved_frame_ratio": (n_improved / n_comparable) if n_comparable else float("nan"),
        "median_delta_frame_p95_px": float(np.median(deltas_arr)) if deltas_arr.size else float("nan"),
        "p95_delta_frame_p95_px": float(np.percentile(deltas_arr, 95)) if deltas_arr.size else float("nan"),
    }


def frame_coverage_breakdown(point: PerturbationPoint) -> list:
    """(label, count) for a candidate point's paired coverage vs baseline,
    named by which side actually failed: "Candidate failed" is a frame
    baseline scored but this candidate could not (`n_baseline_only_valid`),
    never to be confused with "Candidate-only valid" (baseline failed, the
    candidate scored)."""
    return [
        ("Both valid", point.n_comparable_frames),
        ("Candidate failed", point.n_baseline_only_valid),
        ("Candidate-only valid", point.n_candidate_only_valid),
        ("Both failed", point.n_both_failed),
    ]


def _rank_value(point: PerturbationPoint, ranking_metric: str) -> float:
    return point.median_frame_p95_px if ranking_metric == "median_frame_p95" else point.p95_px


def summarize_axis(
    axis: str, unit: str, points: list, baseline_point: PerturbationPoint,
    local_min_tolerance_px: float = DEFAULT_LOCAL_MIN_TOLERANCE_PX,
    ranking_metric: str = "pooled_p95",
) -> AxisSensitivity:
    """Pure arithmetic over an already-evaluated set of points for one axis --
    factored out from `evaluate_perturbation_grid` so it is testable without
    any geometry, projection, or image processing at all.

    Fills in, in place: each point's standalone `median_frame_p95_px` /
    `median_frame_median_px` (median across that trial's OWN valid frames,
    unpaired); each non-baseline point's pooled `delta_median_px` /
    `delta_p95_px` (negative = improvement, kept as a diagnostic regardless of
    `ranking_metric`); and the paired frame-balanced comparison against
    baseline (`_pair_frame_metrics`) plus `coverage_warning` when a
    candidate's *comparable* frame count differs from baseline's own valid
    count (a real accuracy claim would need the same frames counted).

    `ranking_metric` ("pooled_p95" or "median_frame_p95") decides only which
    number `lowest_point`/`ranking_*`/`is_local_minimum` are judged
    by -- Multi-frame mode uses "median_frame_p95" so one edge-rich frame
    cannot single-handedly make a perturbation look best; Current-Frame mode
    (a single frame) uses "pooled_p95", which is numerically identical to the
    frame-balanced version there anyway.
    """
    points = sorted(points, key=lambda p: p.delta)
    for p in points:
        ok_frame_p95 = [m.p95_px for m in p.per_frame.values() if m.ok]
        ok_frame_median = [m.median_px for m in p.per_frame.values() if m.ok]
        p.median_frame_p95_px = float(np.median(ok_frame_p95)) if ok_frame_p95 else float("nan")
        p.median_frame_median_px = float(np.median(ok_frame_median)) if ok_frame_median else float("nan")

    baseline_point.delta_median_px = 0.0
    baseline_point.delta_p95_px = 0.0
    for p in points:
        if p is baseline_point:
            continue
        paired = _pair_frame_metrics(baseline_point.per_frame, p.per_frame, local_min_tolerance_px)
        for key, value in paired.items():
            setattr(p, key, value)
        if not p.ok:
            continue
        p.delta_median_px = p.median_px - baseline_point.median_px
        p.delta_p95_px = p.p95_px - baseline_point.p95_px
        if p.n_valid_frames != baseline_point.n_valid_frames:
            p.coverage_warning = "Comparable frame coverage differs from baseline."

    ok_points = [p for p in points if p.ok]
    lowest = min(ok_points, key=lambda p: _rank_value(p, ranking_metric)) if ok_points else baseline_point
    baseline_value = _rank_value(baseline_point, ranking_metric)
    best_value = _rank_value(lowest, ranking_metric)
    improvement = baseline_value - best_value
    return AxisSensitivity(
        axis=axis, unit=unit, points=points, baseline=baseline_point,
        ranking_metric=ranking_metric, lowest_point=lowest, improvement_p95_px=improvement,
        is_local_minimum=bool(improvement <= local_min_tolerance_px),
        ranking_baseline_px=baseline_value, ranking_best_px=best_value, ranking_improvement_px=improvement,
    )


def _run_trial_point(
    kind: str, axis: str, delta: float, usable_frames: list,
    baseline_R: np.ndarray, baseline_t: np.ndarray, camera,
    min_range: float, max_range: float, edge_params: EdgeAlignmentParams,
    n_sync_rejected: int,
) -> PerturbationPoint:
    """Perturb one parameter, evaluate every usable frame against the result,
    and package it as a PerturbationPoint -- the one piece of per-trial work
    shared by the Quick/Full grid (`evaluate_perturbation_grid`) and Fine Scan
    (`evaluate_single_axis`), so the two cannot silently drift apart."""
    if delta == 0.0:
        R, t = baseline_R, baseline_t
    elif kind == "rotation":
        R, t = perturb_rotation(baseline_R, axis, delta), baseline_t
    else:
        R, t = baseline_R, perturb_translation(baseline_t, axis, delta)

    pooled, n_valid, n_failed, spatial, per_frame = _evaluate_trial(
        usable_frames, R, t, camera, min_range, max_range, edge_params,
    )
    return PerturbationPoint(
        delta=delta, ok=pooled.ok, reason=pooled.reason,
        mean_px=pooled.mean_px, median_px=pooled.median_px,
        p95_px=pooled.p95_px, max_px=pooled.max_px, match_rate=pooled.match_rate,
        n_edge_points=pooled.n_edge_points if pooled.ok else 0,
        n_valid_frames=n_valid, n_failed_frames=n_failed,
        n_sync_rejected_frames=n_sync_rejected, spatial=spatial, per_frame=per_frame,
    )


def evaluate_perturbation_grid(
    frames: list,
    baseline_R: np.ndarray,
    baseline_t: np.ndarray,
    camera,
    min_range: float = 0.0,
    max_range: float = 0.0,
    mode: str = "multi_frame",
    rotation_deltas_deg: tuple = DEFAULT_ROTATION_DELTAS_DEG,
    translation_deltas_mm: tuple = DEFAULT_TRANSLATION_DELTAS_MM,
    edge_params: EdgeAlignmentParams = EdgeAlignmentParams(),
    local_min_tolerance_px: float = DEFAULT_LOCAL_MIN_TOLERANCE_PX,
    progress: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> PerturbationResult:
    """Score `baseline_R`/`baseline_t`, then every requested rotation/translation
    perturbation, against the same fixed `frames` -- never re-selecting or
    re-pairing frames per trial (see `prepare_frames`'s docstring for why that
    matters for a fair comparison), and never mutating `baseline_R`/`baseline_t`
    themselves (`perturb_rotation`/`perturb_translation` always return new
    arrays).

    delta=0.0 is evaluated exactly once for the whole run and shared as every
    axis's baseline point -- evaluating the unperturbed T six times (once per
    axis) would be pure waste.
    """
    n_sync_rejected = sum(1 for f in frames if f.sync_rejected)
    usable = [f for f in frames if not f.sync_rejected]
    if not usable:
        return PerturbationResult(mode=mode, n_frames_used=0, n_sync_rejected=n_sync_rejected,
                                   reason="사용 가능한 frame이 없습니다 (모두 sync 제외되었거나 비어 있음).")

    trials: list = []
    for axis in ROTATION_AXES:
        for delta in rotation_deltas_deg:
            trials.append(("rotation", axis, float(delta)))
    for axis in TRANSLATION_AXES:
        for delta in translation_deltas_mm:
            trials.append(("translation", axis, float(delta)))

    total = len(trials)
    cache: dict = {}          # (kind, axis, delta) -> PerturbationPoint, dedups every delta=0
    baseline_point: Optional[PerturbationPoint] = None
    cancelled = False

    for done, (kind, axis, delta) in enumerate(trials, start=1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break

        if delta == 0.0 and baseline_point is not None:
            point = baseline_point
        else:
            point = _run_trial_point(
                kind, axis, delta, usable, baseline_R, baseline_t, camera,
                min_range, max_range, edge_params, n_sync_rejected,
            )
            if delta == 0.0:
                baseline_point = point  # delta_median_px/delta_p95_px set to 0.0 by summarize_axis below

        cache[(kind, axis, delta)] = point
        if progress is not None:
            progress(done, total)

    if baseline_point is None:
        # Cancelled before delta=0 (the first trial of the first axis) finished.
        return PerturbationResult(mode=mode, n_frames_used=len(usable), n_sync_rejected=n_sync_rejected,
                                   cancelled=True, reason="baseline 평가 전에 취소되었습니다.")

    # Multi-frame mode ranks by the median of each trial's OWN per-frame P95
    # values, not the pooled P95, so one edge-rich frame cannot single-
    # handedly decide which perturbation looks best (see AxisSensitivity's
    # docstring). Current-Frame mode has exactly one frame, where the two are
    # numerically identical, so "pooled_p95" there changes nothing.
    ranking_metric = "median_frame_p95" if mode == "multi_frame" else "pooled_p95"

    axes: dict = {}
    for kind, axis_names, deltas, unit in (
        ("rotation", ROTATION_AXES, rotation_deltas_deg, "deg"),
        ("translation", TRANSLATION_AXES, translation_deltas_mm, "mm"),
    ):
        for axis in axis_names:
            points = [cache[(kind, axis, float(d))] for d in deltas if (kind, axis, float(d)) in cache]
            if not points:
                continue
            axes[axis] = summarize_axis(axis, unit, points, baseline_point, local_min_tolerance_px, ranking_metric)

    return PerturbationResult(
        mode=mode, axes=axes, n_frames_used=len(usable), n_sync_rejected=n_sync_rejected, cancelled=cancelled,
    )


def evaluate_single_axis(
    frames: list,
    baseline_R: np.ndarray,
    baseline_t: np.ndarray,
    camera,
    axis: str,
    deltas: tuple,
    min_range: float = 0.0,
    max_range: float = 0.0,
    mode: str = "multi_frame",
    edge_params: EdgeAlignmentParams = EdgeAlignmentParams(),
    local_min_tolerance_px: float = DEFAULT_LOCAL_MIN_TOLERANCE_PX,
    progress: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> AxisSensitivity:
    """Fine Scan: re-evaluate a single axis over a caller-chosen delta range
    (typically from `make_delta_range`), instead of the Quick/Full grid's
    fixed set across all six axes -- for zooming into one axis after a first
    look, e.g. "Yaw -0.30..+0.10 step 0.05". `0.0` is always included even if
    the caller's range does not land on it exactly, since a baseline point is
    required.

    Shares `_run_trial_point`/`_evaluate_trial`/`summarize_axis` with
    `evaluate_perturbation_grid`, so the two can never silently diverge in how
    a trial is scored -- only which deltas are tried differs.
    """
    kind = "rotation" if axis in ROTATION_AXES else "translation"
    unit = "deg" if kind == "rotation" else "mm"
    deltas = tuple(sorted({float(d) for d in deltas} | {0.0}))

    n_sync_rejected = sum(1 for f in frames if f.sync_rejected)
    usable = [f for f in frames if not f.sync_rejected]
    if not usable:
        return AxisSensitivity(axis=axis, unit=unit)

    points: list = []
    baseline_point: Optional[PerturbationPoint] = None
    total = len(deltas)
    for done, delta in enumerate(deltas, start=1):
        if should_cancel is not None and should_cancel():
            break
        point = _run_trial_point(
            kind, axis, delta, usable, baseline_R, baseline_t, camera,
            min_range, max_range, edge_params, n_sync_rejected,
        )
        if delta == 0.0:
            baseline_point = point
        points.append(point)
        if progress is not None:
            progress(done, total)

    if baseline_point is None:
        return AxisSensitivity(axis=axis, unit=unit, points=points)

    ranking_metric = "median_frame_p95" if mode == "multi_frame" else "pooled_p95"
    return summarize_axis(axis, unit, points, baseline_point, local_min_tolerance_px, ranking_metric)

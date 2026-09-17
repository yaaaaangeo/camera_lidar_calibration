"""Does a fixed extrinsic hold up across a whole recording, or just one frame?

Everything else in this package judges one frame. A frame picked by hand is
also a frame the person happened to look at, which is exactly the kind of
sample that misses an extrinsic that is fine dead ahead and wrong at grazing
angles, or fine in daylight frames and wrong wherever a shadow confuses Canny.
This module runs `edge_alignment.evaluate_edge_alignment` independently over
many frames spread across the recording and reports the spread, not just
another mean.

Deliberately knows nothing about rosbags or `gui.core.bag_reader`: it takes a
`frame_loader(timeline_index) -> (image_bgr, cloud_xyz, lidar_timestamp_ns,
offset_s, camera_timestamp_ns) | None` callback and processes one frame at a
time, so a caller can back it with whatever it already uses to read frames,
and a frame's image/cloud is never held onto once its error is computed.
`gui.core.verify` is imported here (and nowhere else in this package) because
projecting each frame's cloud is unavoidably per-frame work that only this
module does.

The sync offset -- the gap between the LiDAR sweep this frame is centred on
and whichever camera image the loader actually paired it with -- is computed
here, once, as `(camera_timestamp_ns - lidar_timestamp_ns) / 1e6`, from the
two raw timestamps the loader hands back; the loader's only job is finding
that camera image (nearest timestamp, first-after, ...), not doing the
arithmetic. On a moving vehicle, a camera/LiDAR pairing that is off by tens of
milliseconds looks exactly like a rotation error in the pixel statistics, so a
frame whose pairing is worse than `max_sync_offset_ms` (judged on the absolute
offset) is kept out of the geometric statistics entirely (`sync_rejected`, its
own bucket) rather than silently counted as evidence about the extrinsic. The
signed offset is kept too -- a magnitude alone cannot distinguish "the camera
is consistently late" from "consistently early", which a future temporal-
offset/drift analysis would need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from gui.core.evaluation.edge_alignment import EdgeAlignmentParams, evaluate_edge_alignment

DEFAULT_HAMPEL_K = 3.0
_MAD_NORMAL_CONSTANT = 1.4826  # scales raw MAD to be std-comparable under a normal distribution


def sample_frame_indices(n_total: int, n_samples: int) -> list[int]:
    """Up to `n_samples` indices into `range(n_total)`, spread as evenly as the
    timeline allows.

    `np.linspace(0, n_total - 1, n_samples)` naturally covers a 10,000-frame
    timeline requested at 100 samples without walking every frame; when
    `n_samples >= n_total` every index is returned once, and duplicates that
    rounding can produce at the low end are collapsed rather than evaluated
    twice.
    """
    if n_total <= 0 or n_samples <= 0:
        return []
    if n_samples >= n_total:
        return list(range(n_total))
    raw = np.linspace(0, n_total - 1, n_samples)
    return sorted(set(int(round(v)) for v in raw))


def compute_robust_stats(values: np.ndarray) -> dict:
    """Median, scaled MAD, and IQR (with Q1/Q3) -- the shared basis for the
    Hampel outlier flag and for the always-reported spread numbers. NaN fields
    on empty input rather than raising."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"median": float("nan"), "mad": float("nan"), "q1": float("nan"), "q3": float("nan"), "iqr": float("nan")}
    median = float(np.median(values))
    mad_raw = float(np.median(np.abs(values - median)))
    q1, q3 = np.percentile(values, [25, 75])
    return {"median": median, "mad": _MAD_NORMAL_CONSTANT * mad_raw, "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def flag_outliers_hampel(values: np.ndarray, robust: dict, k: float = DEFAULT_HAMPEL_K) -> tuple:
    """Hampel/X84: a value is an outlier when its robust z-score --
    (value - median) / MAD -- exceeds `k`. Unlike a plain 5x-median rule, this
    accounts for how spread out the *non*-outlier values already are, so it
    does not call a clean-but-noisy dataset's ordinary spread "outliers" nor
    miss a genuine one in a very tight dataset.

    Falls back to an absolute epsilon distance from the median when MAD is
    ~0 (every frame agrees almost exactly) -- a z-score divides by MAD, and
    dividing by ~0 would flag ordinary rounding noise as an infinite-sigma
    outlier.

    Returns (is_outlier bool array, robust z-score array).
    """
    median = robust["median"]
    mad = robust["mad"]
    if mad > 1e-6:
        z = (values - median) / mad
    else:
        z = (values - median) / max(0.05, abs(median) * 0.05, 1e-9)
    return np.abs(z) > k, z


@dataclass
class FrameEvalResult:
    timeline_index: int   # position in the caller's full timeline -- what the "이동" button seeks to
    lidar_timestamp_ns: int
    offset_s: float
    ok: bool
    reason: str = ""
    mean_px: float = float("nan")
    median_px: float = float("nan")
    p95_px: float = float("nan")
    n_edge_points: int = 0
    match_rate: float = float("nan")
    is_outlier: bool = False
    robust_z: float = float("nan")
    # 0 when no camera frame was found to pair against at all (a missing-data
    # failure, not a sync problem) -- the two sync fields below stay NaN then.
    # A jump back to this exact frame (e.g. a Worst-Frame "이동" click) must
    # pin the reload to this timestamp, not re-derive a pairing from
    # lidar_timestamp_ns alone -- see gui.core.bag_reader.pick_camera_for_frame.
    camera_timestamp_ns: int = 0
    # camera - lidar: positive means the camera image is the later of the
    # two. Kept alongside the absolute value because the *sign* is what a
    # future temporal-drift/offset analysis would need -- a magnitude alone
    # cannot tell "the camera is consistently late" from "consistently early".
    signed_sync_offset_ms: float = float("nan")
    abs_sync_offset_ms: float = float("nan")
    sync_rejected: bool = False   # a pairing was found, but past max_sync_offset_ms -- excluded from geometric stats


@dataclass
class MultiFrameConsistencyResult:
    frame_results: list = field(default_factory=list)   # list[FrameEvalResult], one per sampled frame
    n_total: int = 0
    n_valid: int = 0
    n_failed: int = 0
    n_outlier: int = 0
    n_sync_rejected: int = 0
    valid_ratio: float = float("nan")
    failure_ratio: float = float("nan")
    outlier_ratio: float = float("nan")
    sync_rejected_ratio: float = float("nan")
    mean_px: float = float("nan")
    median_px: float = float("nan")
    std_px: float = float("nan")
    p95_px: float = float("nan")
    max_px: float = float("nan")
    mad_px: float = float("nan")
    iqr_px: float = float("nan")
    # Over every frame a camera image was actually found for (valid, failed-
    # for-other-reasons, and sync_rejected alike) -- a separate question from
    # the geometric error above, and reported even when nothing was rejected.
    # These three are absolute-value statistics (what the reject threshold is
    # judged against); sync_offset_signed_median_ms keeps the sign, e.g. for a
    # future temporal-drift analysis that cares which direction the camera is
    # consistently off by, not just by how much.
    sync_offset_median_ms: float = float("nan")
    sync_offset_p95_ms: float = float("nan")
    sync_offset_max_ms: float = float("nan")
    sync_offset_signed_median_ms: float = float("nan")
    worst_frames: list = field(default_factory=list)    # list[FrameEvalResult], valid only, mean_px desc
    reason: str = ""   # set only when the whole run could not produce a result


def _empty_result(n_total: int, reason: str, frame_results: Optional[list] = None) -> MultiFrameConsistencyResult:
    return MultiFrameConsistencyResult(frame_results=frame_results or [], n_total=n_total, reason=reason)


def evaluate_multiframe_consistency(
    frame_loader: Callable[[int], Optional[tuple]],
    n_total_timeline: int,
    n_samples: int,
    sol,
    camera,
    min_range: float = 0.0,
    max_range: float = 0.0,
    edge_params: EdgeAlignmentParams = EdgeAlignmentParams(),
    hampel_k: float = DEFAULT_HAMPEL_K,
    top_k_worst: int = 10,
    max_sync_offset_ms: Optional[float] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> MultiFrameConsistencyResult:
    """Sample `n_samples` frames evenly from a `n_total_timeline`-long
    timeline, run edge alignment on each independently against the fixed
    `sol`/`camera`, and aggregate.

    `frame_loader(timeline_index)` returns `(image_bgr, cloud_xyz,
    lidar_timestamp_ns, offset_s, camera_timestamp_ns)` or `None` when that
    moment has nothing to evaluate at all (no image and no LiDAR data in
    range) -- such a frame is recorded as failed, not silently skipped, so
    `failure_ratio` reflects it. `signed_sync_offset_ms` /
    `abs_sync_offset_ms` are derived from the two timestamps here (one
    definition, in one place), as `(camera_timestamp_ns -
    lidar_timestamp_ns) / 1e6` -- positive means the camera image is the
    later of the two.

    `max_sync_offset_ms`, when given, rejects a frame *before* running the
    geometric metric on it if the loader's own camera/LiDAR pairing (by
    absolute offset) is looser than that -- on a moving vehicle a bad sync
    looks exactly like a rotation error in the pixel statistics, so it must
    not be mixed into `mean_px`/`worst_frames`. `None` (default) never rejects
    on sync alone; the offsets are still collected and reported either way.

    `should_cancel`, checked once per frame, lets a caller stop a long run
    early (e.g. the user changed their mind about evaluating 2000 frames); the
    frames processed so far are still aggregated and returned.
    """
    from gui.core import verify

    indices = sample_frame_indices(n_total_timeline, n_samples)
    total = len(indices)
    if total == 0:
        return _empty_result(n_total_timeline, "평가할 frame이 없습니다.")

    frame_results: list = []
    for done, idx in enumerate(indices, start=1):
        if should_cancel is not None and should_cancel():
            break

        loaded = frame_loader(idx)
        base = FrameEvalResult(timeline_index=idx, lidar_timestamp_ns=0, offset_s=float("nan"), ok=False)
        if loaded is None:
            base.reason = "이미지 또는 포인트가 없습니다."
            frame_results.append(base)
        else:
            image_bgr, cloud_xyz, lidar_t_ns, offset_s, camera_t_ns = loaded
            base.lidar_timestamp_ns = lidar_t_ns
            base.offset_s = offset_s
            base.camera_timestamp_ns = camera_t_ns
            base.signed_sync_offset_ms = (camera_t_ns - lidar_t_ns) / 1e6
            base.abs_sync_offset_ms = abs(base.signed_sync_offset_ms)

            if max_sync_offset_ms is not None and base.abs_sync_offset_ms > max_sync_offset_ms:
                base.sync_rejected = True
                base.reason = (
                    f"camera/LiDAR sync 차이 {base.abs_sync_offset_ms:.1f}ms > 허용 {max_sync_offset_ms:.0f}ms"
                )
                frame_results.append(base)
            else:
                h, w = image_bgr.shape[:2]
                pr = verify.project_cloud(
                    cloud_xyz, sol, camera, w, h, min_range=min_range, max_range=max_range,
                )
                if pr.n_visible == 0:
                    base.reason = "화면에 투영된 점이 없습니다."
                    frame_results.append(base)
                else:
                    result = evaluate_edge_alignment(image_bgr, pr.uv, pr.depth, edge_params)
                    if not result.ok:
                        base.reason = result.reason
                        frame_results.append(base)
                    else:
                        base.ok = True
                        base.mean_px = result.mean_px
                        base.median_px = result.median_px
                        base.p95_px = result.p95_px
                        base.n_edge_points = result.n_edge_points
                        base.match_rate = result.match_rate
                        frame_results.append(base)

        if progress is not None:
            progress(done, total)

    sync_rejected = [f for f in frame_results if f.sync_rejected]
    others = [f for f in frame_results if not f.sync_rejected]
    valid = [f for f in others if f.ok]
    failed = [f for f in others if not f.ok]
    n_total = len(frame_results)
    valid_ratio = len(valid) / n_total if n_total else float("nan")
    failure_ratio = len(failed) / n_total if n_total else float("nan")
    sync_rejected_ratio = len(sync_rejected) / n_total if n_total else float("nan")

    abs_sync_values = np.array([f.abs_sync_offset_ms for f in frame_results if np.isfinite(f.abs_sync_offset_ms)])
    signed_sync_values = np.array([f.signed_sync_offset_ms for f in frame_results if np.isfinite(f.signed_sync_offset_ms)])
    sync_median_ms = float(np.median(abs_sync_values)) if abs_sync_values.size else float("nan")
    sync_p95_ms = float(np.percentile(abs_sync_values, 95)) if abs_sync_values.size else float("nan")
    sync_max_ms = float(np.max(abs_sync_values)) if abs_sync_values.size else float("nan")
    sync_signed_median_ms = float(np.median(signed_sync_values)) if signed_sync_values.size else float("nan")

    if len(valid) < 2:
        reason = f"유효한 frame이 {len(valid)}개뿐입니다 (STD 계산에는 최소 2개 필요)."
        result = _empty_result(n_total, reason, frame_results=frame_results)
        result.n_failed = len(failed)
        result.n_sync_rejected = len(sync_rejected)
        result.valid_ratio = valid_ratio
        result.failure_ratio = failure_ratio
        result.sync_rejected_ratio = sync_rejected_ratio
        result.sync_offset_median_ms = sync_median_ms
        result.sync_offset_p95_ms = sync_p95_ms
        result.sync_offset_max_ms = sync_max_ms
        result.sync_offset_signed_median_ms = sync_signed_median_ms
        return result

    frame_means = np.array([f.mean_px for f in valid])
    robust = compute_robust_stats(frame_means)
    is_outlier, z_scores = flag_outliers_hampel(frame_means, robust, k=hampel_k)
    for f, outlier, z in zip(valid, is_outlier, z_scores):
        f.is_outlier = bool(outlier)
        f.robust_z = float(z)

    outliers = [f for f in valid if f.is_outlier]
    worst = sorted(valid, key=lambda f: f.mean_px, reverse=True)[:top_k_worst]

    return MultiFrameConsistencyResult(
        frame_results=frame_results,
        n_total=n_total,
        n_valid=len(valid),
        n_failed=len(failed),
        n_outlier=len(outliers),
        n_sync_rejected=len(sync_rejected),
        valid_ratio=valid_ratio,
        failure_ratio=failure_ratio,
        outlier_ratio=len(outliers) / len(valid) if valid else float("nan"),
        sync_rejected_ratio=sync_rejected_ratio,
        mean_px=float(np.mean(frame_means)),
        median_px=robust["median"],
        std_px=float(np.std(frame_means, ddof=1)),
        p95_px=float(np.percentile(frame_means, 95)),
        max_px=float(np.max(frame_means)),
        mad_px=robust["mad"],
        iqr_px=robust["iqr"],
        sync_offset_median_ms=sync_median_ms,
        sync_offset_p95_ms=sync_p95_ms,
        sync_offset_max_ms=sync_max_ms,
        sync_offset_signed_median_ms=sync_signed_median_ms,
        worst_frames=worst,
    )

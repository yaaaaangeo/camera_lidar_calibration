"""Camera-board motion analysis for temporal calibration triage.

This module is deliberately independent of Qt and rosbag readers.  A caller
hands it detected board poses ordered by time; it computes frame-to-frame
translation/rotation deltas, classifies stable/marginal/moving spans, and can
export the frame-level result to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv

import numpy as np

STABLE = "STABLE"
MARGINAL = "MARGINAL"
MOVING = "MOVING"
INVALID = "INVALID"


@dataclass(frozen=True)
class BoardPoseSample:
    timestamp_ns: int
    frame_index: int
    R: np.ndarray
    t: np.ndarray
    quality: float = 1.0
    reproj_rms: float = float("nan")
    reproj_median: float = float("nan")
    reproj_p95: float = float("nan")
    reproj_max: float = float("nan")


@dataclass
class BoardMotionFrame:
    timestamp_ns: int
    frame_index: int
    translation_delta_mm: float = float("nan")
    rotation_delta_deg: float = float("nan")
    translation_velocity_mm_s: float = float("nan")
    rotation_velocity_deg_s: float = float("nan")
    motion_score: float = float("nan")
    velocity_score: float = float("nan")
    status: str = "UNKNOWN"
    valid_pose: bool = True
    invalid_reason: str = ""
    reproj_rms: float = float("nan")
    reproj_median: float = float("nan")
    reproj_p95: float = float("nan")
    reproj_max: float = float("nan")


@dataclass
class StableSegment:
    start_frame_index: int
    end_frame_index: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_timestamp_ns - self.start_timestamp_ns) / 1e9)


@dataclass
class BoardMotionResult:
    frames: list[BoardMotionFrame] = field(default_factory=list)
    stable_segments: list[StableSegment] = field(default_factory=list)
    total_detected: int = 0
    stable_frames: int = 0
    marginal_frames: int = 0
    moving_frames: int = 0
    invalid_frames: int = 0
    valid_pose_ratio: float = float("nan")
    invalid_pose_ratio: float = float("nan")
    stable_ratio: float = float("nan")
    translation_median_mm: float = float("nan")
    translation_p10_mm: float = float("nan")
    translation_p25_mm: float = float("nan")
    translation_p75_mm: float = float("nan")
    translation_p90_mm: float = float("nan")
    translation_p95_mm: float = float("nan")
    translation_max_mm: float = float("nan")
    translation_velocity_median_mm_s: float = float("nan")
    translation_velocity_p95_mm_s: float = float("nan")
    translation_velocity_max_mm_s: float = float("nan")
    rotation_median_deg: float = float("nan")
    rotation_p10_deg: float = float("nan")
    rotation_p25_deg: float = float("nan")
    rotation_p75_deg: float = float("nan")
    rotation_p90_deg: float = float("nan")
    rotation_p95_deg: float = float("nan")
    rotation_max_deg: float = float("nan")
    rotation_velocity_median_deg_s: float = float("nan")
    rotation_velocity_p95_deg_s: float = float("nan")
    rotation_velocity_max_deg_s: float = float("nan")
    reproj_rms_median_px: float = float("nan")
    reproj_rms_p95_px: float = float("nan")
    reproj_median_px_median: float = float("nan")
    reproj_p95_px_median: float = float("nan")
    reproj_max_px_p95: float = float("nan")
    reason: str = ""

    @property
    def stable_timestamps_ns(self) -> list[int]:
        return [f.timestamp_ns for f in self.frames if f.status == STABLE]


def rotation_angle_deg(R_prev: np.ndarray, R_next: np.ndarray) -> float:
    """Angle of R_prev.T @ R_next in degrees."""
    dR = np.asarray(R_prev, float).T @ np.asarray(R_next, float)
    cos_angle = (float(np.trace(dR)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def classify_motion(
    translation_delta_mm: float,
    rotation_delta_deg: float,
    translation_threshold_mm: float,
    rotation_threshold_deg: float,
    marginal_ratio: float = 0.75,
) -> str:
    """Stable below thresholds, moving clearly above, marginal near boundary."""
    t_norm = translation_delta_mm / max(translation_threshold_mm, 1e-9)
    r_norm = rotation_delta_deg / max(rotation_threshold_deg, 1e-9)
    score = max(t_norm, r_norm)
    if score < marginal_ratio:
        return STABLE
    if score <= 1.0:
        return MARGINAL
    return MOVING


def analyze_board_motion(
    samples: list[BoardPoseSample],
    translation_threshold_mm: float = 5.0,
    rotation_threshold_deg: float = 0.1,
    marginal_ratio: float = 0.75,
    min_stable_segment_frames: int = 2,
    attempted_count: int | None = None,
    invalid_frames: list[BoardMotionFrame] | None = None,
) -> BoardMotionResult:
    invalid_frames = list(invalid_frames or [])
    total_attempted = int(attempted_count if attempted_count is not None else len(samples) + len(invalid_frames))
    if len(samples) < 2:
        total = max(total_attempted, len(samples) + len(invalid_frames))
        return BoardMotionResult(
            frames=invalid_frames,
            total_detected=len(samples),
            invalid_frames=max(0, total - len(samples)),
            valid_pose_ratio=(len(samples) / total) if total else float("nan"),
            invalid_pose_ratio=(max(0, total - len(samples)) / total) if total else float("nan"),
            reason="Need at least two valid board poses.",
        )

    ordered = sorted(samples, key=lambda s: (s.timestamp_ns, s.frame_index))
    invalid_sorted = sorted(invalid_frames, key=lambda f: (f.timestamp_ns, f.frame_index))
    frames: list[BoardMotionFrame] = [
        BoardMotionFrame(
            timestamp_ns=ordered[0].timestamp_ns,
            frame_index=ordered[0].frame_index,
            reproj_rms=ordered[0].reproj_rms,
            reproj_median=ordered[0].reproj_median,
            reproj_p95=ordered[0].reproj_p95,
            reproj_max=ordered[0].reproj_max,
        )
    ]
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        if _has_invalid_between(prev, cur, invalid_sorted):
            frames.append(
                BoardMotionFrame(
                    timestamp_ns=cur.timestamp_ns,
                    frame_index=cur.frame_index,
                    reproj_rms=cur.reproj_rms,
                    reproj_median=cur.reproj_median,
                    reproj_p95=cur.reproj_p95,
                    reproj_max=cur.reproj_max,
                )
            )
            continue
        trans_mm = float(np.linalg.norm(np.asarray(cur.t, float) - np.asarray(prev.t, float)) * 1000.0)
        rot_deg = rotation_angle_deg(prev.R, cur.R)
        dt_s = max((cur.timestamp_ns - prev.timestamp_ns) / 1e9, 1e-9)
        trans_vel = trans_mm / dt_s
        rot_vel = rot_deg / dt_s
        t_norm = trans_mm / max(translation_threshold_mm, 1e-9)
        r_norm = rot_deg / max(rotation_threshold_deg, 1e-9)
        score = float(max(t_norm, r_norm))
        frames.append(
            BoardMotionFrame(
                timestamp_ns=cur.timestamp_ns,
                frame_index=cur.frame_index,
                translation_delta_mm=trans_mm,
                rotation_delta_deg=rot_deg,
                translation_velocity_mm_s=trans_vel,
                rotation_velocity_deg_s=rot_vel,
                motion_score=score,
                velocity_score=float(max(t_norm / dt_s, r_norm / dt_s)),
                status=classify_motion(
                    trans_mm, rot_deg, translation_threshold_mm, rotation_threshold_deg, marginal_ratio
                ),
                reproj_rms=cur.reproj_rms,
                reproj_median=cur.reproj_median,
                reproj_p95=cur.reproj_p95,
                reproj_max=cur.reproj_max,
            )
        )

    stable_segments = _stable_segments(frames, min_stable_segment_frames)
    combined_frames = sorted(frames + invalid_frames, key=lambda f: (f.timestamp_ns, f.frame_index))
    trans = np.array([f.translation_delta_mm for f in frames if np.isfinite(f.translation_delta_mm)])
    rot = np.array([f.rotation_delta_deg for f in frames if np.isfinite(f.rotation_delta_deg)])
    trans_vel = np.array([f.translation_velocity_mm_s for f in frames if np.isfinite(f.translation_velocity_mm_s)])
    rot_vel = np.array([f.rotation_velocity_deg_s for f in frames if np.isfinite(f.rotation_velocity_deg_s)])
    reproj_rms = np.array([s.reproj_rms for s in ordered if np.isfinite(s.reproj_rms)])
    reproj_med = np.array([s.reproj_median for s in ordered if np.isfinite(s.reproj_median)])
    reproj_p95 = np.array([s.reproj_p95 for s in ordered if np.isfinite(s.reproj_p95)])
    reproj_max = np.array([s.reproj_max for s in ordered if np.isfinite(s.reproj_max)])
    stable = sum(f.status == STABLE for f in frames)
    marginal = sum(f.status == MARGINAL for f in frames)
    moving = sum(f.status == MOVING for f in frames)
    invalid = max(0, total_attempted - len(frames))
    total = max(total_attempted, len(frames) + len(invalid_frames))

    return BoardMotionResult(
        frames=combined_frames,
        stable_segments=stable_segments,
        total_detected=len(frames),
        stable_frames=stable,
        marginal_frames=marginal,
        moving_frames=moving,
        invalid_frames=invalid,
        valid_pose_ratio=len(frames) / total if total else float("nan"),
        invalid_pose_ratio=invalid / total if total else float("nan"),
        stable_ratio=stable / len(frames) if frames else float("nan"),
        translation_median_mm=_stat(trans, "median"),
        translation_p10_mm=_stat(trans, "p10"),
        translation_p25_mm=_stat(trans, "p25"),
        translation_p75_mm=_stat(trans, "p75"),
        translation_p90_mm=_stat(trans, "p90"),
        translation_p95_mm=_stat(trans, "p95"),
        translation_max_mm=_stat(trans, "max"),
        translation_velocity_median_mm_s=_stat(trans_vel, "median"),
        translation_velocity_p95_mm_s=_stat(trans_vel, "p95"),
        translation_velocity_max_mm_s=_stat(trans_vel, "max"),
        rotation_median_deg=_stat(rot, "median"),
        rotation_p10_deg=_stat(rot, "p10"),
        rotation_p25_deg=_stat(rot, "p25"),
        rotation_p75_deg=_stat(rot, "p75"),
        rotation_p90_deg=_stat(rot, "p90"),
        rotation_p95_deg=_stat(rot, "p95"),
        rotation_max_deg=_stat(rot, "max"),
        rotation_velocity_median_deg_s=_stat(rot_vel, "median"),
        rotation_velocity_p95_deg_s=_stat(rot_vel, "p95"),
        rotation_velocity_max_deg_s=_stat(rot_vel, "max"),
        reproj_rms_median_px=_stat(reproj_rms, "median"),
        reproj_rms_p95_px=_stat(reproj_rms, "p95"),
        reproj_median_px_median=_stat(reproj_med, "median"),
        reproj_p95_px_median=_stat(reproj_p95, "median"),
        reproj_max_px_p95=_stat(reproj_max, "p95"),
    )


def _stat(values: np.ndarray, kind: str) -> float:
    if values.size == 0:
        return float("nan")
    if kind == "median":
        return float(np.median(values))
    if kind == "p10":
        return float(np.percentile(values, 10))
    if kind == "p25":
        return float(np.percentile(values, 25))
    if kind == "p75":
        return float(np.percentile(values, 75))
    if kind == "p90":
        return float(np.percentile(values, 90))
    if kind == "p95":
        return float(np.percentile(values, 95))
    if kind == "max":
        return float(np.max(values))
    raise ValueError(kind)


def low_motion_frame_indices(result: BoardMotionResult, fraction: float) -> set[int]:
    """Pick the lowest-motion frames, preferring contiguous runs when tied.

    The score still comes from the existing absolute-threshold analysis. This
    helper only ranks already-computed frames for diagnostics; it does not
    change STABLE/MARGINAL/MOVING classification.
    """
    frames = [f for f in result.frames if np.isfinite(f.motion_score)]
    if not frames or fraction <= 0:
        return set()
    n = max(1, int(np.ceil(len(frames) * min(float(fraction), 1.0))))
    order = sorted(frames, key=lambda f: (f.motion_score, f.frame_index))
    cutoff = order[n - 1].motion_score
    selected = {f.frame_index for f in order if f.motion_score < cutoff}
    boundary = [f for f in frames if f.motion_score == cutoff]
    need = n - len(selected)
    if need > 0:
        selected.update(_prefer_contiguous(boundary, selected, need))
    return selected


def low_velocity_frame_indices(result: BoardMotionResult, fraction: float) -> set[int]:
    frames = [f for f in result.frames if np.isfinite(f.velocity_score)]
    if not frames or fraction <= 0:
        return set()
    n = max(1, int(np.ceil(len(frames) * min(float(fraction), 1.0))))
    order = sorted(frames, key=lambda f: (f.velocity_score, f.frame_index))
    cutoff = order[n - 1].velocity_score
    selected = {f.frame_index for f in order if f.velocity_score < cutoff}
    boundary = [f for f in frames if f.velocity_score == cutoff]
    need = n - len(selected)
    if need > 0:
        selected.update(_prefer_contiguous(boundary, selected, need))
    return selected


def _has_invalid_between(prev: BoardPoseSample, cur: BoardPoseSample, invalid: list[BoardMotionFrame]) -> bool:
    lo_t, hi_t = sorted((prev.timestamp_ns, cur.timestamp_ns))
    lo_i, hi_i = sorted((prev.frame_index, cur.frame_index))
    for f in invalid:
        if lo_t < f.timestamp_ns < hi_t:
            return True
        if lo_i < f.frame_index < hi_i:
            return True
    return False


def _prefer_contiguous(frames: list[BoardMotionFrame], selected: set[int], need: int) -> set[int]:
    if need <= 0:
        return set()
    remaining = {f.frame_index: f for f in frames}
    out: set[int] = set()
    while remaining and len(out) < need:
        best_idx = min(
            remaining,
            key=lambda idx: (
                -int((idx - 1) in selected or (idx + 1) in selected or (idx - 1) in out or (idx + 1) in out),
                idx,
            ),
        )
        out.add(best_idx)
        remaining.pop(best_idx, None)
    return out


def _stable_segments(frames: list[BoardMotionFrame], min_frames: int) -> list[StableSegment]:
    out: list[StableSegment] = []
    start = None
    for i, frame in enumerate(frames + [BoardMotionFrame(0, 0, status=MOVING)]):
        if i < len(frames) and frame.status == STABLE:
            if start is None:
                start = i
            continue
        if start is not None:
            end = i - 1
            if end - start + 1 >= min_frames:
                first, last = frames[start], frames[end]
                out.append(
                    StableSegment(
                        start_frame_index=first.frame_index,
                        end_frame_index=last.frame_index,
                        start_timestamp_ns=first.timestamp_ns,
                        end_timestamp_ns=last.timestamp_ns,
                        n_frames=end - start + 1,
                    )
                )
            start = None
    return out


def export_stable_frames_csv(result: BoardMotionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "timestamp_ns",
            "frame_index",
            "translation_delta_mm",
            "rotation_delta_deg",
            "motion_score",
            "status",
        ])
        for f in result.frames:
            if f.status != STABLE:
                continue
            writer.writerow([
                f.timestamp_ns,
                f.frame_index,
                _csv_float(f.translation_delta_mm),
                _csv_float(f.rotation_delta_deg),
                _csv_float(f.motion_score),
                f.status,
            ])
    return path


def _csv_float(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.9g}"

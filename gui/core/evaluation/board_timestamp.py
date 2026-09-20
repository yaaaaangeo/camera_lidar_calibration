"""Board-specific LiDAR acquisition timestamp diagnostics.

The LiDAR detector already isolates the calibration board plane inside the ROI.
This module consumes those detector plane indices plus the PointCloud2
per-point timestamp field and reports where, within a sweep, the board was
actually observed. It does not pair differently, correct timestamps, or modify
any extrinsic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import bisect

import numpy as np

from gui.core.evaluation.timestamp_diagnostic import TimestampStats, stats_ms


@dataclass
class BoardTimestampFrame:
    frame_index: int
    lidar_record_ns: int
    camera_header_ns: int
    sweep_start_ns: int
    sweep_mid_ns: int
    sweep_end_ns: int
    board_min_ns: int
    board_median_ns: int
    board_mean_ns: int
    board_max_ns: int
    board_span_ms: float
    board_point_count: int
    board_median_offset_from_start_ms: float
    camera_to_sweep_start_ms: float
    camera_to_sweep_mid_ms: float
    camera_to_sweep_end_ms: float
    camera_to_board_median_ms: float
    status: str = "UNKNOWN"
    current_camera_index: int = -1
    current_camera_record_ns: int = 0
    previous_camera_header_ns: int | None = None
    previous_offset_ms: float = float("nan")
    next_camera_header_ns: int | None = None
    next_offset_ms: float = float("nan")
    nearest_camera_index: int = -1
    nearest_camera_header_ns: int = 0
    nearest_offset_ms: float = float("nan")
    camera_index_shift: int = 0
    plane_detected: bool = False
    full_target_detected: bool = False
    board_timestamp_valid: bool = False
    board_pose_valid: bool = False
    lidar_plane_point_count: int = 0
    lidar_circle_candidate_count: int = 0
    lidar_circle_count: int = 0
    lidar_radius_mean_m: float = float("nan")
    lidar_radius_min_m: float = float("nan")
    lidar_radius_max_m: float = float("nan")
    lidar_detection_reason: str = ""


@dataclass
class BoardTimestampResult:
    frames: list[BoardTimestampFrame] = field(default_factory=list)
    failed_frames: int = 0
    reason: str = ""
    processed_frames: int = 0
    plane_detection_success: int = 0
    enough_plane_points_success: int = 0
    circle_candidate_success: int = 0
    four_or_more_circles_success: int = 0
    geometry_validation_success: int = 0
    full_target_detection_success: int = 0
    board_timestamp_valid_count: int = 0
    board_pose_valid_count: int = 0
    lidar_failure_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def all_stats(self) -> dict:
        return summarize(self.frames)

    @property
    def stable_stats(self) -> dict:
        return summarize([f for f in self.frames if f.status == "STABLE"])


def summarize(frames: list[BoardTimestampFrame]) -> dict:
    if not frames:
        empty = TimestampStats()
        return {
            "n": 0,
            "camera_to_sweep_start": empty,
            "camera_to_sweep_mid": empty,
            "camera_to_sweep_end": empty,
            "camera_to_board_median": empty,
            "nearest_to_board": empty,
            "board_offset_from_start": empty,
            "board_span": empty,
            "board_point_count_mean": float("nan"),
            "board_point_count_median": float("nan"),
            "same_camera_frame_ratio": float("nan"),
            "previous_camera_frame_ratio": float("nan"),
            "next_camera_frame_ratio": float("nan"),
            "shift_histogram": {k: 0 for k in (-2, -1, 0, 1, 2)},
            "plane_detection_success": 0,
            "full_target_detection_success": 0,
            "board_timestamp_valid": 0,
            "board_pose_valid": 0,
        }
    counts = np.asarray([f.board_point_count for f in frames], dtype=float)
    shifts = [f.camera_index_shift for f in frames]
    n = len(frames)
    return {
        "n": n,
        "camera_to_sweep_start": stats_ms([f.camera_to_sweep_start_ms for f in frames]),
        "camera_to_sweep_mid": stats_ms([f.camera_to_sweep_mid_ms for f in frames]),
        "camera_to_sweep_end": stats_ms([f.camera_to_sweep_end_ms for f in frames]),
        "camera_to_board_median": stats_ms([f.camera_to_board_median_ms for f in frames]),
        "nearest_to_board": stats_ms([f.nearest_offset_ms for f in frames]),
        "board_offset_from_start": stats_ms([f.board_median_offset_from_start_ms for f in frames]),
        "board_span": stats_ms([f.board_span_ms for f in frames]),
        "board_point_count_mean": float(np.mean(counts)),
        "board_point_count_median": float(np.median(counts)),
        "same_camera_frame_ratio": shifts.count(0) / n,
        "previous_camera_frame_ratio": shifts.count(-1) / n,
        "next_camera_frame_ratio": shifts.count(1) / n,
        "shift_histogram": {k: shifts.count(k) for k in (-2, -1, 0, 1, 2)},
        "plane_detection_success": sum(f.plane_detected for f in frames),
        "full_target_detection_success": sum(f.full_target_detected for f in frames),
        "board_timestamp_valid": sum(f.board_timestamp_valid for f in frames),
        "board_pose_valid": sum(f.board_pose_valid for f in frames),
    }


def frame_from_timestamps(
    frame_index: int,
    lidar_record_ns: int,
    camera_header_ns: int,
    sweep_start_ns: int,
    sweep_end_ns: int,
    board_point_timestamps_s,
    status: str = "UNKNOWN",
    current_camera_index: int = -1,
    current_camera_record_ns: int = 0,
    camera_headers: list[tuple[int, int, int]] | None = None,
    plane_detected: bool = True,
    full_target_detected: bool = False,
    board_pose_valid: bool = False,
    lidar_plane_point_count: int = 0,
    lidar_circle_candidate_count: int = 0,
    lidar_circle_count: int = 0,
    lidar_radii: list[float] | None = None,
    lidar_detection_reason: str = "",
) -> BoardTimestampFrame:
    ts = np.asarray(board_point_timestamps_s, dtype=float)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        raise ValueError("no valid board point timestamps")
    board_ns = np.rint(ts * 1e9).astype(np.int64)
    board_min = int(np.min(board_ns))
    board_max = int(np.max(board_ns))
    board_median = int(np.median(board_ns))
    board_mean = int(np.mean(board_ns))
    sweep_mid = int((sweep_start_ns + sweep_end_ns) // 2)
    pairing = camera_pairing_for_board_time(
        board_median, current_camera_index, camera_headers or []
    )
    radii = np.asarray(lidar_radii or [], dtype=float)
    radii = radii[np.isfinite(radii)]
    return BoardTimestampFrame(
        frame_index=frame_index,
        lidar_record_ns=lidar_record_ns,
        camera_header_ns=camera_header_ns,
        sweep_start_ns=sweep_start_ns,
        sweep_mid_ns=sweep_mid,
        sweep_end_ns=sweep_end_ns,
        board_min_ns=board_min,
        board_median_ns=board_median,
        board_mean_ns=board_mean,
        board_max_ns=board_max,
        board_span_ms=(board_max - board_min) / 1e6,
        board_point_count=int(ts.size),
        board_median_offset_from_start_ms=(board_median - sweep_start_ns) / 1e6,
        camera_to_sweep_start_ms=(camera_header_ns - sweep_start_ns) / 1e6,
        camera_to_sweep_mid_ms=(camera_header_ns - sweep_mid) / 1e6,
        camera_to_sweep_end_ms=(camera_header_ns - sweep_end_ns) / 1e6,
        camera_to_board_median_ms=(camera_header_ns - board_median) / 1e6,
        status=status,
        current_camera_index=current_camera_index,
        current_camera_record_ns=current_camera_record_ns,
        previous_camera_header_ns=pairing["previous_camera_header_ns"],
        previous_offset_ms=pairing["previous_offset_ms"],
        next_camera_header_ns=pairing["next_camera_header_ns"],
        next_offset_ms=pairing["next_offset_ms"],
        nearest_camera_index=pairing["nearest_camera_index"],
        nearest_camera_header_ns=pairing["nearest_camera_header_ns"],
        nearest_offset_ms=pairing["nearest_offset_ms"],
        camera_index_shift=pairing["camera_index_shift"],
        plane_detected=plane_detected,
        full_target_detected=full_target_detected,
        board_timestamp_valid=True,
        board_pose_valid=board_pose_valid,
        lidar_plane_point_count=int(lidar_plane_point_count),
        lidar_circle_candidate_count=int(lidar_circle_candidate_count),
        lidar_circle_count=int(lidar_circle_count),
        lidar_radius_mean_m=float(np.mean(radii)) if radii.size else float("nan"),
        lidar_radius_min_m=float(np.min(radii)) if radii.size else float("nan"),
        lidar_radius_max_m=float(np.max(radii)) if radii.size else float("nan"),
        lidar_detection_reason=lidar_detection_reason,
    )


def camera_pairing_for_board_time(
    board_ns: int,
    current_camera_index: int,
    camera_headers: list[tuple[int, int, int]],
) -> dict:
    """Find previous/next/header-nearest camera frames for a board timestamp.

    `camera_headers` entries are `(camera_index, record_ns, header_ns)`. They
    may arrive in record order; the nearest search is done in header-time order.
    """
    nan = float("nan")
    out = {
        "previous_camera_header_ns": None,
        "previous_offset_ms": nan,
        "next_camera_header_ns": None,
        "next_offset_ms": nan,
        "nearest_camera_index": -1,
        "nearest_camera_header_ns": 0,
        "nearest_offset_ms": nan,
        "camera_index_shift": 0,
    }
    if not camera_headers:
        return out
    by_index = {int(i): int(h) for i, _, h in camera_headers}
    prev_h = by_index.get(int(current_camera_index) - 1)
    next_h = by_index.get(int(current_camera_index) + 1)
    if prev_h is not None:
        out["previous_camera_header_ns"] = prev_h
        out["previous_offset_ms"] = (prev_h - board_ns) / 1e6
    if next_h is not None:
        out["next_camera_header_ns"] = next_h
        out["next_offset_ms"] = (next_h - board_ns) / 1e6

    ordered = sorted((int(h), int(i)) for i, _, h in camera_headers)
    headers = [h for h, _ in ordered]
    pos = bisect.bisect_left(headers, int(board_ns))
    candidates = []
    if pos > 0:
        candidates.append(ordered[pos - 1])
    if pos < len(ordered):
        candidates.append(ordered[pos])
    nearest_h, nearest_i = min(candidates, key=lambda item: (abs(item[0] - board_ns), item[1]))
    out["nearest_camera_index"] = nearest_i
    out["nearest_camera_header_ns"] = nearest_h
    out["nearest_offset_ms"] = (nearest_h - board_ns) / 1e6
    out["camera_index_shift"] = nearest_i - int(current_camera_index)
    return out


def export_frames_csv(result: BoardTimestampResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame_index",
            "status",
            "lidar_record_ns",
            "sweep_start_ns",
            "sweep_mid_ns",
            "sweep_end_ns",
            "board_min_ns",
            "board_median_ns",
            "board_mean_ns",
            "board_max_ns",
            "board_median_offset_from_start_ms",
            "board_span_ms",
            "board_point_count",
            "camera_header_ns",
            "current_camera_index",
            "current_camera_record_ns",
            "camera_to_sweep_start_ms",
            "camera_to_sweep_mid_ms",
            "camera_to_sweep_end_ms",
            "camera_to_board_median_ms",
            "previous_camera_header_ns",
            "previous_offset_ms",
            "next_camera_header_ns",
            "next_offset_ms",
            "nearest_camera_index",
            "nearest_camera_header_ns",
            "nearest_offset_ms",
            "camera_index_shift",
            "plane_detected",
            "full_target_detected",
            "board_timestamp_valid",
            "board_pose_valid",
            "lidar_plane_point_count",
            "lidar_circle_candidate_count",
            "lidar_circle_count",
            "lidar_radius_mean_m",
            "lidar_radius_min_m",
            "lidar_radius_max_m",
            "lidar_detection_reason",
        ])
        for f in result.frames:
            writer.writerow([
                f.frame_index,
                f.status,
                f.lidar_record_ns,
                f.sweep_start_ns,
                f.sweep_mid_ns,
                f.sweep_end_ns,
                f.board_min_ns,
                f.board_median_ns,
                f.board_mean_ns,
                f.board_max_ns,
                f.board_median_offset_from_start_ms,
                f.board_span_ms,
                f.board_point_count,
                f.camera_header_ns,
                f.current_camera_index,
                f.current_camera_record_ns,
                f.camera_to_sweep_start_ms,
                f.camera_to_sweep_mid_ms,
                f.camera_to_sweep_end_ms,
                f.camera_to_board_median_ms,
                f.previous_camera_header_ns if f.previous_camera_header_ns is not None else "",
                f.previous_offset_ms,
                f.next_camera_header_ns if f.next_camera_header_ns is not None else "",
                f.next_offset_ms,
                f.nearest_camera_index,
                f.nearest_camera_header_ns,
                f.nearest_offset_ms,
                f.camera_index_shift,
                int(f.plane_detected),
                int(f.full_target_detected),
                int(f.board_timestamp_valid),
                int(f.board_pose_valid),
                f.lidar_plane_point_count,
                f.lidar_circle_candidate_count,
                f.lidar_circle_count,
                f.lidar_radius_mean_m,
                f.lidar_radius_min_m,
                f.lidar_radius_max_m,
                f.lidar_detection_reason,
            ])
    return path

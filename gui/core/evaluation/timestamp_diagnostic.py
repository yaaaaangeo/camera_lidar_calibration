"""Timestamp-source comparison for Camera/LiDAR evaluation.

This is diagnostic-only infrastructure: it lets callers compare pairing based
on record time, camera header time, and LiDAR sweep start/mid/end without
changing the production pairing path or applying any temporal correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TimestampMode(str, Enum):
    RECORD_TIME = "record_time"
    SENSOR_HEADER = "sensor_header"
    LIDAR_SWEEP_START = "lidar_sweep_start"
    LIDAR_SWEEP_MID = "lidar_sweep_mid"
    LIDAR_SWEEP_END = "lidar_sweep_end"


class PairingMode(str, Enum):
    RECORD_TO_RECORD = "record_to_record"
    HEADER_TO_SWEEP_START = "header_to_sweep_start"
    HEADER_TO_SWEEP_MID = "header_to_sweep_mid"
    HEADER_TO_SWEEP_END = "header_to_sweep_end"


PAIRING_LABELS = {
    PairingMode.RECORD_TO_RECORD: "Record -> Record",
    PairingMode.HEADER_TO_SWEEP_START: "Header -> Sweep Start",
    PairingMode.HEADER_TO_SWEEP_MID: "Header -> Sweep Mid",
    PairingMode.HEADER_TO_SWEEP_END: "Header -> Sweep End",
}


@dataclass(frozen=True)
class TimestampStats:
    n: int = 0
    mean_ms: float = float("nan")
    median_ms: float = float("nan")
    std_ms: float = float("nan")
    p95_abs_ms: float = float("nan")
    min_ms: float = float("nan")
    max_ms: float = float("nan")


@dataclass(frozen=True)
class CameraStamp:
    record_ns: int
    header_ns: int


@dataclass(frozen=True)
class LidarStamp:
    record_ns: int
    header_ns: int
    point_min_ns: Optional[int] = None
    point_max_ns: Optional[int] = None
    header_start_delta_ms: float = float("nan")
    warning: str = ""

    @property
    def point_mid_ns(self) -> Optional[int]:
        if self.point_min_ns is None or self.point_max_ns is None:
            return None
        return int((self.point_min_ns + self.point_max_ns) // 2)


@dataclass
class TimestampPairingSummary:
    mode: PairingMode
    n_pairs: int = 0
    sync_stats: TimestampStats = TimestampStats()
    warning: str = ""


@dataclass
class TimestampDiagnosticFrame:
    timeline_index: int
    lidar_record_ns: int
    lidar_reference_ns: int
    camera_record_ns: int
    camera_reference_ns: int
    signed_sync_offset_ms: float


@dataclass
class TimestampDiagnosticMetrics:
    mode: PairingMode
    n_total: int = 0
    n_valid: int = 0
    n_failed: int = 0
    reason: str = ""
    mean_px: float = float("nan")
    median_px: float = float("nan")
    p95_px: float = float("nan")
    matched_mean_px: float = float("nan")
    matched_median_px: float = float("nan")
    matched_std_px: float = float("nan")
    matched_p95_px: float = float("nan")
    matched_p99_px: float = float("nan")
    matched_max_px: float = float("nan")
    match_rate_mean: float = float("nan")
    match_rate_median: float = float("nan")
    within_2px: float = float("nan")
    within_3px: float = float("nan")
    within_5px: float = float("nan")
    within_10px: float = float("nan")
    within_15px: float = float("nan")
    median_delta_u_px: float = float("nan")
    median_delta_v_px: float = float("nan")
    sync_stats: TimestampStats = TimestampStats()
    frames: list[TimestampDiagnosticFrame] = field(default_factory=list)
    horizontal: dict = field(default_factory=dict)
    depth: dict = field(default_factory=dict)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def header_stamp_ns(msg) -> Optional[int]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return stamp_to_ns(stamp)


def stats_ms(values) -> TimestampStats:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return TimestampStats()
    return TimestampStats(
        n=int(arr.size),
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        std_ms=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        p95_abs_ms=float(np.percentile(np.abs(arr), 95)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
    )


def point_time_bounds_ns(point_timestamps) -> tuple[Optional[int], Optional[int]]:
    if point_timestamps is None:
        return None, None
    arr = np.asarray(point_timestamps, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    # Current supported field is epoch seconds (float64 on the tested Ouster
    # bag). Reject tiny relative values explicitly instead of guessing a unit.
    if np.nanmedian(np.abs(arr)) < 1_000_000.0:
        return None, None
    return int(float(np.min(arr)) * 1e9), int(float(np.max(arr)) * 1e9)


def make_lidar_stamp(record_ns: int, msg, point_timestamps=None,
                     start_tolerance_ms: float = 1.0) -> LidarStamp:
    header_ns = header_stamp_ns(msg)
    if header_ns is None:
        raise ValueError("LiDAR message has no header.stamp")
    pmin, pmax = point_time_bounds_ns(point_timestamps)
    warning = ""
    delta = float("nan")
    if pmin is not None:
        delta = (header_ns - pmin) / 1e6
        if abs(delta) > start_tolerance_ms:
            warning = f"LiDAR header differs from point timestamp min by {delta:.3f} ms"
    return LidarStamp(record_ns=record_ns, header_ns=header_ns, point_min_ns=pmin,
                      point_max_ns=pmax, header_start_delta_ms=delta, warning=warning)


def camera_reference_ns(stamp: CameraStamp, mode: TimestampMode) -> int:
    if mode == TimestampMode.RECORD_TIME:
        return stamp.record_ns
    if mode == TimestampMode.SENSOR_HEADER:
        return stamp.header_ns
    raise ValueError(f"unsupported camera timestamp mode: {mode}")


def lidar_reference_ns(stamp: LidarStamp, mode: TimestampMode) -> int:
    if mode == TimestampMode.RECORD_TIME:
        return stamp.record_ns
    if mode in (TimestampMode.SENSOR_HEADER, TimestampMode.LIDAR_SWEEP_START):
        return stamp.point_min_ns if stamp.point_min_ns is not None else stamp.header_ns
    if mode == TimestampMode.LIDAR_SWEEP_MID:
        if stamp.point_mid_ns is None:
            raise ValueError("LiDAR point timestamp is required for sweep mid")
        return stamp.point_mid_ns
    if mode == TimestampMode.LIDAR_SWEEP_END:
        if stamp.point_max_ns is None:
            raise ValueError("LiDAR point timestamp is required for sweep end")
        return stamp.point_max_ns
    raise ValueError(f"unsupported lidar timestamp mode: {mode}")


def modes_to_sources(mode: PairingMode) -> tuple[TimestampMode, TimestampMode]:
    if mode == PairingMode.RECORD_TO_RECORD:
        return TimestampMode.RECORD_TIME, TimestampMode.RECORD_TIME
    if mode == PairingMode.HEADER_TO_SWEEP_START:
        return TimestampMode.SENSOR_HEADER, TimestampMode.LIDAR_SWEEP_START
    if mode == PairingMode.HEADER_TO_SWEEP_MID:
        return TimestampMode.SENSOR_HEADER, TimestampMode.LIDAR_SWEEP_MID
    if mode == PairingMode.HEADER_TO_SWEEP_END:
        return TimestampMode.SENSOR_HEADER, TimestampMode.LIDAR_SWEEP_END
    raise ValueError(mode)


def nearest_index(sorted_values: np.ndarray, value: int) -> int:
    if len(sorted_values) == 0:
        raise ValueError("empty timestamp array")
    pos = int(np.searchsorted(sorted_values, value))
    candidates = []
    if pos < len(sorted_values):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    return min(candidates, key=lambda i: abs(int(sorted_values[i]) - value))


def pair_by_mode(
    camera_stamps: list[CameraStamp],
    lidar_stamps: list[LidarStamp],
    mode: PairingMode,
    lidar_indices: list[int],
) -> tuple[list[tuple[int, int, int, int]], TimestampPairingSummary]:
    """Return (lidar_index, camera_index, lidar_ref_ns, camera_ref_ns) tuples."""
    cam_mode, lid_mode = modes_to_sources(mode)
    cam_refs = np.asarray([camera_reference_ns(c, cam_mode) for c in camera_stamps], dtype=np.int64)
    pairs = []
    offsets = []
    warnings = []
    for li in lidar_indices:
        try:
            lid_ref = lidar_reference_ns(lidar_stamps[li], lid_mode)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        ci = nearest_index(cam_refs, lid_ref)
        cam_ref = int(cam_refs[ci])
        pairs.append((li, ci, int(lid_ref), cam_ref))
        offsets.append((cam_ref - lid_ref) / 1e6)
        if lidar_stamps[li].warning:
            warnings.append(lidar_stamps[li].warning)
    warning = "; ".join(sorted(set(warnings))[:3])
    return pairs, TimestampPairingSummary(
        mode=mode, n_pairs=len(pairs), sync_stats=stats_ms(offsets), warning=warning,
    )

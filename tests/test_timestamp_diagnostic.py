from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gui.core.evaluation.timestamp_diagnostic import (
    CameraStamp,
    LidarStamp,
    PairingMode,
    TimestampMode,
    camera_reference_ns,
    lidar_reference_ns,
    make_lidar_stamp,
    pair_by_mode,
    point_time_bounds_ns,
)


@dataclass
class _Stamp:
    sec: int
    nanosec: int


@dataclass
class _Header:
    stamp: _Stamp


@dataclass
class _Msg:
    header: _Header


def _msg(ns: int) -> _Msg:
    return _Msg(_Header(_Stamp(ns // 1_000_000_000, ns % 1_000_000_000)))


def test_camera_record_and_header_selection():
    stamp = CameraStamp(record_ns=150, header_ns=100)
    assert camera_reference_ns(stamp, TimestampMode.RECORD_TIME) == 150
    assert camera_reference_ns(stamp, TimestampMode.SENSOR_HEADER) == 100


def test_lidar_sweep_start_mid_end_from_point_timestamps():
    pts = np.array([1_789_352_583.0, 1_789_352_583.04, 1_789_352_583.10], dtype=np.float64)
    pmin, pmax = point_time_bounds_ns(pts)
    base = 1_789_352_583_000_000_000
    lidar = LidarStamp(record_ns=base + 200_000_000, header_ns=base,
                       point_min_ns=pmin, point_max_ns=pmax)

    assert lidar_reference_ns(lidar, TimestampMode.LIDAR_SWEEP_START) == base
    assert lidar_reference_ns(lidar, TimestampMode.LIDAR_SWEEP_MID) == base + 50_000_000
    assert lidar_reference_ns(lidar, TimestampMode.LIDAR_SWEEP_END) == base + 100_000_000


def test_lidar_header_start_validation_warning():
    base_s = 1_789_352_583.0
    base_ns = 1_789_352_583_000_000_000
    lidar = make_lidar_stamp(
        record_ns=base_ns + 200_000_000,
        msg=_msg(base_ns + 5_000_000),
        point_timestamps=np.array([base_s, base_s + 0.1], dtype=np.float64),
        start_tolerance_ms=1.0,
    )
    assert lidar.header_start_delta_ms == 5.0
    assert "differs" in lidar.warning


def test_positive_negative_sync_offset_sign_convention():
    cams = [CameraStamp(record_ns=50, header_ns=50), CameraStamp(record_ns=400, header_ns=400)]
    lids = [LidarStamp(record_ns=200, header_ns=200)]

    pairs, summary = pair_by_mode(cams, lids, PairingMode.RECORD_TO_RECORD, [0])
    assert pairs[0][3] - pairs[0][2] == -150
    assert summary.sync_stats.median_ms == -0.00015

    cams = [CameraStamp(record_ns=300, header_ns=300)]
    pairs, summary = pair_by_mode(cams, lids, PairingMode.RECORD_TO_RECORD, [0])
    assert pairs[0][3] - pairs[0][2] == 100
    assert summary.sync_stats.median_ms == 0.0001


def test_nearest_pairing_uses_selected_timestamp_sources():
    cams = [
        CameraStamp(record_ns=1_000, header_ns=10_000),
        CameraStamp(record_ns=2_000, header_ns=20_000),
    ]
    lids = [
        LidarStamp(record_ns=1_900, header_ns=9_900, point_min_ns=9_900,
                   point_max_ns=19_900),
    ]

    record_pairs, _ = pair_by_mode(cams, lids, PairingMode.RECORD_TO_RECORD, [0])
    header_pairs, _ = pair_by_mode(cams, lids, PairingMode.HEADER_TO_SWEEP_START, [0])

    assert record_pairs[0][1] == 1  # nearest camera record to LiDAR record 1900
    assert header_pairs[0][1] == 0  # nearest camera header to LiDAR sweep start 9900


def test_missing_point_timestamp_fails_sweep_mid_safely():
    cams = [CameraStamp(record_ns=0, header_ns=0)]
    lids = [LidarStamp(record_ns=0, header_ns=0)]

    pairs, summary = pair_by_mode(cams, lids, PairingMode.HEADER_TO_SWEEP_MID, [0])

    assert pairs == []
    assert "point timestamp is required" in summary.warning

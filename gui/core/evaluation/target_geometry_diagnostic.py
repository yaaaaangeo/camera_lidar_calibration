"""Target geometry diagnostics for LiDAR board detection.

This module is intentionally diagnostic-only.  It compares configured target
geometry, previous FAST-Calib candidate geometry, and scaled variants without
writing project configuration, changing extrinsics, or applying corrections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from gui.core.detect_lidar import LidarDetection
from gui.core.project import Target


PREVIOUS_FAST_CALIB_TARGET = Target(
    marker_size=0.1188,
    delta_width_qr_center=0.3267,
    delta_height_qr_center=0.2079,
    delta_width_circles=0.2970,
    delta_height_circles=0.2376,
    circle_radius=0.07128,
)


PARAMETER_MEANINGS = {
    "marker_size": ("m", "ArUco/QR marker side length", "length, not radius"),
    "delta_width_qr_center": ("m", "horizontal marker center-to-center spacing", "center-to-center"),
    "delta_height_qr_center": ("m", "vertical marker center-to-center spacing", "center-to-center"),
    "delta_width_circles": ("m", "horizontal hole center-to-center spacing", "center-to-center"),
    "delta_height_circles": ("m", "vertical hole center-to-center spacing", "center-to-center"),
    "circle_radius": ("m", "hole radius", "radius, not diameter"),
}


TARGET_USAGE = {
    "marker_size": [
        "gui.core.detect_camera.board_model",
        "gui.core.detect_lidar.find_holes_by_occupancy",
        "gui.core.detect_lidar.find_holes_experimental",
        "gui.core.synth",
    ],
    "delta_width_qr_center": [
        "gui.core.detect_camera.board_model",
        "gui.core.detect_lidar.find_holes_by_occupancy",
        "gui.core.detect_lidar.find_holes_experimental",
        "gui.core.synth",
    ],
    "delta_height_qr_center": [
        "gui.core.detect_camera.board_model",
        "gui.core.detect_lidar.find_holes_by_occupancy",
        "gui.core.detect_lidar.find_holes_experimental",
        "gui.core.synth",
    ],
    "delta_width_circles": [
        "gui.core.detect_camera.board_model",
        "gui.core.detect_lidar._select_rectangle",
        "gui.core.roi_auto.guess_box",
        "gui.core.synth",
    ],
    "delta_height_circles": [
        "gui.core.detect_camera.board_model",
        "gui.core.detect_lidar._select_rectangle",
        "gui.core.roi_auto.guess_box",
        "gui.core.synth",
    ],
    "circle_radius": [
        "gui.core.detect_lidar.detect radius limits",
        "gui.core.detect_lidar ring/occupancy thresholds",
        "gui.core.roi_auto.guess_box",
        "gui.core.synth",
    ],
}


@dataclass
class ParameterTraceRow:
    parameter: str
    config_value: float
    runtime_value: float
    unit: str
    meaning: str
    convention: str
    used_in: str


@dataclass
class ParameterComparisonRow:
    parameter: str
    previous: float
    current: float
    ratio: float
    difference: float
    convention_same: bool


@dataclass
class GeometryDetectionSummary:
    label: str
    scale: float = 1.0
    attempted: int = 0
    plane_detected: int = 0
    boundary_success: int = 0
    circle_ge1: int = 0
    circle_ge2: int = 0
    circle_ge3: int = 0
    circle_ge4: int = 0
    geometry_passed: int = 0
    full_target: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)


def parameter_trace(target: Target) -> list[ParameterTraceRow]:
    data = asdict(target)
    rows = []
    for key, value in data.items():
        unit, meaning, convention = PARAMETER_MEANINGS[key]
        rows.append(
            ParameterTraceRow(
                parameter=key,
                config_value=float(value),
                runtime_value=float(getattr(target, key)),
                unit=unit,
                meaning=meaning,
                convention=convention,
                used_in=", ".join(TARGET_USAGE[key]),
            )
        )
    return rows


def compare_to_previous(current: Target, previous: Target = PREVIOUS_FAST_CALIB_TARGET) -> list[ParameterComparisonRow]:
    rows = []
    for key, cur in asdict(current).items():
        prev = float(getattr(previous, key))
        cur = float(cur)
        rows.append(
            ParameterComparisonRow(
                parameter=key,
                previous=prev,
                current=cur,
                ratio=cur / prev if prev else float("nan"),
                difference=cur - prev,
                convention_same=True,
            )
        )
    return rows


def ratio_stats(rows: list[ParameterComparisonRow]) -> dict:
    ratios = np.asarray([r.ratio for r in rows], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
    }


def summarize_detections(label: str, detections: list[LidarDetection], scale: float = 1.0) -> GeometryDetectionSummary:
    summary = GeometryDetectionSummary(label=label, scale=scale, attempted=len(detections))
    for det in detections:
        if len(det.plane) > 0:
            summary.plane_detected += 1
        if len(det.edges) >= 3:
            summary.boundary_success += 1
        if det.n_circles >= 1:
            summary.circle_ge1 += 1
        if det.n_circles >= 2:
            summary.circle_ge2 += 1
        if det.n_circles >= 3:
            summary.circle_ge3 += 1
        if det.n_circles >= 4:
            summary.circle_ge4 += 1
        if det.ok:
            summary.geometry_passed += 1
            summary.full_target += 1
        else:
            reason = det.reason or "not full target"
            summary.failure_reasons[reason] = summary.failure_reasons.get(reason, 0) + 1
    return summary


def plane_extent_xy(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return {"n": 0, "width_m": float("nan"), "height_m": float("nan"), "area_m2": float("nan")}
    extent = np.ptp(pts[:, :2], axis=0)
    width, height = float(np.max(extent)), float(np.min(extent))
    return {"n": int(len(pts)), "width_m": width, "height_m": height, "area_m2": width * height}


def extent_distribution(extents: list[dict]) -> dict:
    out = {"n": len(extents)}
    for key in ("width_m", "height_m", "area_m2"):
        vals = np.asarray([e[key] for e in extents], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[key] = {"p25": float("nan"), "median": float("nan"), "p75": float("nan"), "p95": float("nan")}
        else:
            out[key] = {
                "p25": float(np.percentile(vals, 25)),
                "median": float(np.median(vals)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
            }
    return out


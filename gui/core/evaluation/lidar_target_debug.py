"""Debug snapshot export for LiDAR target detection.

The functions here are diagnostic-only: they draw what the current detector
already did, without changing ROI, plane selection, target geometry, extrinsic,
or calibration output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import cv2
import numpy as np

from gui.core.project import Target


@dataclass
class DebugSnapshot:
    frame_index: int
    category: str
    reason: str
    out_dir: Path
    metrics: dict = field(default_factory=dict)


def plane_local_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(3), np.eye(3), np.empty((0, 2))
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh[:3].copy()
    if axes.shape[0] < 3:
        axes = np.eye(3)
    local = centered @ axes[:2].T
    return center, axes, local


def project_local(points: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    if points is None:
        return np.empty((0, 2))
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.empty((0, 2))
    return (pts - center) @ axes[:2].T


def extent_metrics(local_xy: np.ndarray) -> dict:
    xy = np.asarray(local_xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return {"n": 0, "width_m": float("nan"), "height_m": float("nan"), "area_m2": float("nan")}
    ext = np.ptp(xy, axis=0)
    width = float(np.max(ext))
    height = float(np.min(ext))
    return {"n": int(len(xy)), "width_m": width, "height_m": height, "area_m2": width * height}


def board_likeness(local_xy: np.ndarray, boundary_xy: np.ndarray, det, target: Target) -> dict:
    ext = extent_metrics(local_xy)
    expected_w = target.delta_width_circles + 2 * target.circle_radius
    expected_h = target.delta_height_circles + 2 * target.circle_radius
    expected_area = max(expected_w * expected_h, 1e-9)
    measured_area = ext["area_m2"]
    size_similarity = float(np.exp(-abs(np.log(max(measured_area, 1e-9) / expected_area))))
    expected_aspect = max(expected_w, expected_h) / max(min(expected_w, expected_h), 1e-9)
    measured_aspect = ext["width_m"] / max(ext["height_m"], 1e-9)
    aspect_similarity = float(np.exp(-abs(np.log(max(measured_aspect, 1e-9) / expected_aspect))))
    boundary_score = float(min(len(boundary_xy) / max(len(local_xy), 1), 1.0))
    return {
        **ext,
        "expected_width_m": expected_w,
        "expected_height_m": expected_h,
        "size_similarity": size_similarity,
        "aspect_similarity": aspect_similarity,
        "boundary_score": boundary_score,
        "hole_candidates": int(getattr(det, "n_circles", 0)),
        "detector_ok": bool(getattr(det, "ok", False)),
        "detector_reason": getattr(det, "reason", ""),
    }


def draw_plane_local_xy(
    path: str | Path,
    plane_xy: np.ndarray,
    boundary_xy: np.ndarray,
    circle_xy: np.ndarray,
    radii: list[float],
    current: Target,
    previous: Target | None = None,
    title: str = "",
    current_label: str = "CURRENT",
    previous_label: str = "PREVIOUS",
    size: int = 1200,
) -> None:
    canvas = np.full((size, size, 3), 255, np.uint8)
    all_xy = [plane_xy]
    if len(boundary_xy):
        all_xy.append(boundary_xy)
    if len(circle_xy):
        all_xy.append(circle_xy)
    lim_pts = np.vstack([a for a in all_xy if len(a)]) if any(len(a) for a in all_xy) else np.zeros((1, 2))
    min_xy = np.min(lim_pts, axis=0)
    max_xy = np.max(lim_pts, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    pad = max(float(np.max(span)) * 0.08, 0.2)
    min_xy -= pad
    max_xy += pad
    scale = (size - 120) / max(float(np.max(max_xy - min_xy)), 1e-6)

    def px(xy):
        arr = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.empty_like(arr)
        out[:, 0] = 60 + (arr[:, 0] - min_xy[0]) * scale
        out[:, 1] = size - 60 - (arr[:, 1] - min_xy[1]) * scale
        return np.rint(out).astype(np.int32)

    _draw_points(canvas, px(plane_xy), (185, 185, 185), radius=1)
    _draw_expected(canvas, px, current, (80, 80, 255), current_label)
    if previous is not None:
        _draw_expected(canvas, px, previous, (255, 120, 60), previous_label)
    _draw_points(canvas, px(boundary_xy), (0, 180, 255), radius=3)
    centers_px = px(circle_xy)
    for i, c in enumerate(centers_px):
        r_px = int(round((radii[i] if i < len(radii) else current.circle_radius) * scale))
        cv2.circle(canvas, tuple(c), max(2, r_px), (0, 160, 0), 2)
        cv2.circle(canvas, tuple(c), 4, (0, 120, 0), -1)
    cv2.circle(canvas, tuple(px([[0, 0]])[0]), 5, (0, 0, 0), -1)
    cv2.putText(canvas, title[:130], (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def draw_cloud_xy(path: str | Path, points: np.ndarray, title: str = "", size: int = 1200) -> None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    canvas = np.full((size, size, 3), 255, np.uint8)
    if len(pts):
        xy = pts[:, :2]
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-3)
        pad = max(float(np.max(span)) * 0.05, 0.2)
        min_xy -= pad
        max_xy += pad
        scale = (size - 120) / max(float(np.max(max_xy - min_xy)), 1e-6)
        pix = np.empty((len(xy), 2), dtype=np.int32)
        pix[:, 0] = np.rint(60 + (xy[:, 0] - min_xy[0]) * scale).astype(np.int32)
        pix[:, 1] = np.rint(size - 60 - (xy[:, 1] - min_xy[1]) * scale).astype(np.int32)
        z = pts[:, 2]
        z_span = max(float(np.ptp(z)), 1e-9)
        shade = np.rint(40 + 180 * (z - float(np.min(z))) / z_span).astype(np.uint8)
        for p, s in zip(pix, shade):
            if 0 <= p[0] < size and 0 <= p[1] < size:
                cv2.circle(canvas, tuple(p), 1, (int(s), int(s), int(s)), -1)
    cv2.putText(canvas, title[:130], (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def draw_projection(
    path: str | Path,
    image_bgr: np.ndarray,
    sol,
    camera,
    roi_points: np.ndarray,
    plane_points: np.ndarray,
    boundary_points: np.ndarray,
    max_points: int = 20000,
) -> None:
    out = image_bgr.copy()
    _draw_projected(out, roi_points, sol, camera, (120, 120, 120), max_points=max_points)
    _draw_projected(out, plane_points, sol, camera, (0, 255, 255), max_points=max_points)
    _draw_projected(out, boundary_points, sol, camera, (0, 0, 255), max_points=max_points)
    cv2.imwrite(str(path), out)


def export_snapshot(
    out_dir: str | Path,
    frame_index: int,
    category: str,
    image_bgr: np.ndarray,
    sol,
    camera,
    target_current: Target,
    target_previous: Target,
    xyz: np.ndarray,
    roi_points: np.ndarray,
    det,
) -> DebugSnapshot:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plane = np.asarray(getattr(det, "plane", np.empty((0, 3))), dtype=float)
    edges = np.asarray(getattr(det, "edges", np.empty((0, 3))), dtype=float)
    raw_centers = getattr(det, "centers", None)
    centers = np.empty((0, 3), dtype=float) if raw_centers is None else np.asarray(raw_centers, dtype=float).reshape(-1, 3)
    center, axes, plane_xy = plane_local_frame(plane)
    edge_xy = project_local(edges, center, axes)
    circle_xy = project_local(centers, center, axes)
    normal = axes[2] if len(plane) else np.array([float("nan"), float("nan"), float("nan")])
    plane_d = float(-np.dot(normal, center)) if len(plane) else float("nan")
    metrics = {
        "frame_index": int(frame_index),
        "category": category,
        "roi_point_count": int(len(roi_points)),
        "plane_point_count": int(len(plane)),
        "boundary_point_count": int(len(edges)),
        "circle_count": int(getattr(det, "n_circles", 0)),
        "radii_m": [float(r) for r in getattr(det, "radii", [])],
        "reason": getattr(det, "reason", ""),
        "roi_bbox": _bbox(roi_points),
        "plane_bbox": _bbox(plane),
        "plane_centroid": [float(v) for v in center],
        "plane_normal": [float(v) for v in normal],
        "plane_equation": {"normal": [float(v) for v in normal], "d": plane_d},
        "plane_rms": float(getattr(det, "plane_rms", float("nan"))),
        "spacing": float(getattr(det, "spacing", float("nan"))),
        "plane_likeness_current": board_likeness(plane_xy, edge_xy, det, target_current),
        "plane_likeness_previous": board_likeness(plane_xy, edge_xy, det, target_previous),
    }
    draw_cloud_xy(out / "original_cloud.png", xyz, title=f"frame {frame_index} | original cloud")
    draw_cloud_xy(out / "roi_cloud.png", roi_points, title=f"frame {frame_index} | ROI")
    draw_cloud_xy(out / "plane_cloud.png", plane, title=f"frame {frame_index} | RANSAC plane")
    draw_plane_local_xy(
        out / "plane_local_xy.png",
        plane_xy,
        edge_xy,
        circle_xy,
        list(getattr(det, "radii", [])),
        target_current,
        target_previous,
        title=f"frame {frame_index} | {category} | {getattr(det, 'reason', '')}",
    )
    draw_plane_local_xy(
        out / "boundary.png",
        plane_xy,
        edge_xy,
        np.empty((0, 2)),
        [],
        target_current,
        target_previous,
        title=f"frame {frame_index} | boundary candidates",
    )
    draw_plane_local_xy(
        out / "circles.png",
        plane_xy,
        edge_xy,
        circle_xy,
        list(getattr(det, "radii", [])),
        target_current,
        target_previous,
        title=f"frame {frame_index} | circle candidates",
    )
    draw_plane_local_xy(
        out / "current_geometry_overlay.png",
        plane_xy,
        edge_xy,
        circle_xy,
        list(getattr(det, "radii", [])),
        target_current,
        None,
        title=f"frame {frame_index} | current geometry overlay",
        current_label="CURRENT",
    )
    draw_plane_local_xy(
        out / "previous_geometry_overlay.png",
        plane_xy,
        edge_xy,
        circle_xy,
        list(getattr(det, "radii", [])),
        target_previous,
        None,
        title=f"frame {frame_index} | previous geometry overlay",
        current_label="PREVIOUS",
    )
    draw_projection(out / "camera_projection.png", image_bgr, sol, camera, roi_points, plane, edges)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return DebugSnapshot(frame_index=int(frame_index), category=category, reason=getattr(det, "reason", ""), out_dir=out, metrics=metrics)


def _bbox(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return {"n": 0, "min": [float("nan")] * 3, "max": [float("nan")] * 3, "extent": [float("nan")] * 3}
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    return {
        "n": int(len(pts)),
        "min": [float(v) for v in mn],
        "max": [float(v) for v in mx],
        "extent": [float(v) for v in (mx - mn)],
    }


def _draw_points(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int = 1) -> None:
    for p in pts:
        if 0 <= p[0] < canvas.shape[1] and 0 <= p[1] < canvas.shape[0]:
            cv2.circle(canvas, tuple(p), radius, color, -1)


def _draw_expected(canvas: np.ndarray, px, target: Target, color: tuple[int, int, int], label: str) -> None:
    centers = np.array([
        [-target.delta_width_circles / 2, -target.delta_height_circles / 2],
        [target.delta_width_circles / 2, -target.delta_height_circles / 2],
        [target.delta_width_circles / 2, target.delta_height_circles / 2],
        [-target.delta_width_circles / 2, target.delta_height_circles / 2],
    ])
    cp = px(centers)
    cv2.polylines(canvas, [cp], True, color, 2, cv2.LINE_AA)
    r_px = int(round(target.circle_radius * np.linalg.norm(px([[1, 0]])[0] - px([[0, 0]])[0])))
    for p in cp:
        cv2.circle(canvas, tuple(p), max(2, r_px), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, tuple(cp[0] + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_projected(out, pts, sol, camera, color, max_points=20000):
    if pts is None or len(pts) == 0 or sol is None or not getattr(sol, "ok", False):
        return
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    cam_pts = sol.transform(pts)
    keep = cam_pts[:, 2] > 1e-3
    if not keep.any():
        return
    import cv2 as _cv2
    uv, _ = _cv2.projectPoints(cam_pts[keep], np.zeros(3), np.zeros(3), camera.matrix(), camera.dist())
    uv = uv.reshape(-1, 2)
    h, w = out.shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for u, v in uv[inside].astype(int):
        cv2.circle(out, (int(u), int(v)), 1, color, -1)

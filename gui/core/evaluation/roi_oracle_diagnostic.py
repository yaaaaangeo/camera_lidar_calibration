"""AUTO ROI vs manual/oracle ROI diagnostics.

The production ROI placement and LiDAR detector are not changed here.  This
module only compares ROI choices by feeding each ROI into the same diagnostic
multi-plane pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

from gui.core import roi_auto
from gui.core.detect_lidar import apply_region, region_mask
from gui.core.evaluation.plane_rerank_diagnostic import (
    BoardSize,
    FrameRerankResult,
    PlaneCandidate,
    RerankConfig,
    extract_plane_candidates,
)
from gui.core.project import FilterBox, Target


@dataclass
class RoiProjection:
    camera_bbox: tuple[float, float, float, float] | None = None
    roi_bbox: tuple[float, float, float, float] | None = None
    roi_center_px: tuple[float, float] | None = None
    centroid_distance_px: float = float("nan")
    overlap_ratio: float = float("nan")
    iou: float = float("nan")


@dataclass
class RoiRunMetrics:
    label: str
    roi_point_count: int = 0
    plane_candidate_count: int = 0
    best_plane_index: int | None = None
    best_plane_width_m: float = float("nan")
    best_plane_height_m: float = float("nan")
    best_board_likeness_score: float = float("nan")
    best_boundary_count: int = 0
    best_circle_count: int = 0
    circle_ge4: bool = False
    full_target_success: bool = False
    board_like: bool = False
    projection: RoiProjection = field(default_factory=RoiProjection)


@dataclass
class RoiComparisonFrame:
    frame_index: int
    auto: RoiRunMetrics | None = None
    manual: RoiRunMetrics | None = None
    roi_case: str = "UNKNOWN"
    auto_result: FrameRerankResult | None = None
    manual_result: FrameRerankResult | None = None


@dataclass
class RoiComparisonSummary:
    n_frames: int
    auto_board_like: int
    manual_board_like: int
    auto_boundary_success: int
    manual_boundary_success: int
    auto_circle_ge1: int
    manual_circle_ge1: int
    auto_circle_ge4: int
    manual_circle_ge4: int
    auto_full_target: int
    manual_full_target: int
    case_counts: dict[str, int]


def compare_auto_manual_roi(
    frame_index: int,
    xyz: np.ndarray,
    auto_roi,
    manual_roi,
    target: Target,
    params,
    ring: np.ndarray | None = None,
    image_bgr: np.ndarray | None = None,
    camera=None,
    sol=None,
    camera_bbox: tuple[float, float, float, float] | None = None,
    config: RerankConfig | None = None,
) -> RoiComparisonFrame:
    config = config or RerankConfig(board_size=BoardSize(0.74, 0.64))
    auto_metrics = auto_result = None
    manual_metrics = manual_result = None
    if auto_roi is not None:
        auto_result = _run_roi(frame_index, xyz, auto_roi, target, params, ring, config)
        auto_metrics = metrics_from_result("AUTO", xyz, auto_roi, auto_result, image_bgr, camera, sol, camera_bbox)
    if manual_roi is not None:
        manual_result = _run_roi(frame_index, xyz, manual_roi, target, params, ring, config)
        manual_metrics = metrics_from_result("MANUAL", xyz, manual_roi, manual_result, image_bgr, camera, sol, camera_bbox)
    return RoiComparisonFrame(
        frame_index=frame_index,
        auto=auto_metrics,
        manual=manual_metrics,
        roi_case=classify_roi_case(auto_metrics, manual_metrics),
        auto_result=auto_result,
        manual_result=manual_result,
    )


def margin_sweep(
    frame_index: int,
    xyz: np.ndarray,
    sol,
    cam_det,
    target: Target,
    params,
    ring: np.ndarray | None,
    margins: list[float],
    config: RerankConfig | None = None,
) -> list[RoiRunMetrics]:
    out = []
    config = config or RerankConfig(board_size=BoardSize(0.74, 0.64))
    for margin in margins:
        placement = roi_auto.place(xyz, sol, cam_det, target, scene_id=str(frame_index), margin=margin)
        if not placement.ok or placement.box is None:
            out.append(RoiRunMetrics(label=f"margin_{margin:g}", projection=RoiProjection()))
            continue
        result = _run_roi(frame_index, xyz, placement.box, target, params, ring, config)
        out.append(metrics_from_result(f"margin_{margin:g}", xyz, placement.box, result))
    return out


def board_sized_crop_roi(candidate: PlaneCandidate, target: Target, scale: float = 1.25) -> FilterBox:
    """Diagnostic crop around a candidate plane's own centroid and orientation."""
    pts = np.asarray(candidate.points, dtype=float)
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    axes = vh[:3]
    # box_from_frame's Euler extraction lives in detect_lidar; importing here
    # avoids creating another ROI representation.
    from gui.core.detect_lidar import box_from_frame

    width = (target.delta_width_circles + 2 * target.circle_radius) * scale
    height = (target.delta_height_circles + 2 * target.circle_radius) * scale
    half = np.array([max(width, height) / 2, min(width, height) / 2, 0.08])
    return box_from_frame(center, axes.T, half)


def plane_crop_diagnostic(
    frame_index: int,
    xyz: np.ndarray,
    candidate: PlaneCandidate,
    target: Target,
    params,
    ring: np.ndarray | None,
    config: RerankConfig | None = None,
) -> tuple[RoiRunMetrics, FrameRerankResult]:
    crop = board_sized_crop_roi(candidate, target)
    result = _run_roi(frame_index, xyz, crop, target, params, ring, config or RerankConfig(board_size=BoardSize(0.74, 0.64)))
    return metrics_from_result("BOARD_SIZED_CROP", xyz, crop, result), result


def classify_roi_case(auto: RoiRunMetrics | None, manual: RoiRunMetrics | None) -> str:
    if manual is None:
        return "NO_MANUAL_ROI"
    auto_good = bool(auto and auto.board_like and (auto.best_circle_count > 0 or auto.full_target_success))
    manual_improves = manual.board_like and (
        manual.full_target_success
        or manual.best_circle_count > max(auto.best_circle_count if auto else 0, 0)
        or (manual.best_boundary_count > max(auto.best_boundary_count if auto else 0, 0) and manual.best_circle_count > 0)
    )
    if manual_improves and not auto_good:
        return "ROI-A_AUTO_ROI_MAIN_CAUSE"
    if manual.board_like and not manual.full_target_success:
        return "ROI-B_POST_ROI_CROP_BOUNDARY_CIRCLE"
    if not manual.board_like:
        return "ROI-C_NO_BOARD_PLANE_IN_MANUAL_ROI"
    return "ROI-OK"


def metrics_from_result(
    label: str,
    xyz: np.ndarray,
    roi,
    result: FrameRerankResult,
    image_bgr: np.ndarray | None = None,
    camera=None,
    sol=None,
    camera_bbox: tuple[float, float, float, float] | None = None,
) -> RoiRunMetrics:
    best = result.best
    projection = RoiProjection(camera_bbox=camera_bbox)
    if image_bgr is not None and camera is not None and sol is not None:
        projection = projection_metrics(image_bgr.shape[:2], camera, sol, roi, camera_bbox)
    return RoiRunMetrics(
        label=label,
        roi_point_count=int(region_mask(np.asarray(xyz), roi).sum()),
        plane_candidate_count=len(result.candidates),
        best_plane_index=None if best is None else best.index,
        best_plane_width_m=float("nan") if best is None else best.scores["width_m"],
        best_plane_height_m=float("nan") if best is None else best.scores["height_m"],
        best_board_likeness_score=float("nan") if best is None else best.total_score,
        best_boundary_count=0 if best is None else len(best.det.edges),
        best_circle_count=0 if best is None else best.det.n_circles,
        circle_ge4=bool(best and best.det.n_circles >= 4),
        full_target_success=bool(best and best.det.ok),
        board_like=bool(best and best.board_like),
        projection=projection,
    )


def projection_metrics(
    image_shape_hw: tuple[int, int],
    camera,
    sol,
    roi,
    camera_bbox: tuple[float, float, float, float] | None,
) -> RoiProjection:
    corners = np.asarray(roi.corners() if hasattr(roi, "corners") else roi.as_box().corners(), dtype=float)
    uv = _project_points(corners, camera, sol)
    finite = np.isfinite(uv).all(axis=1)
    if not finite.any():
        return RoiProjection(camera_bbox=camera_bbox)
    h, w = image_shape_hw
    inside = uv[finite]
    x0, y0 = np.min(inside, axis=0)
    x1, y1 = np.max(inside, axis=0)
    roi_bbox = (float(np.clip(x0, 0, w - 1)), float(np.clip(y0, 0, h - 1)),
                float(np.clip(x1, 0, w - 1)), float(np.clip(y1, 0, h - 1)))
    center = tuple(float(v) for v in np.mean(inside, axis=0))
    if camera_bbox is None:
        return RoiProjection(camera_bbox=None, roi_bbox=roi_bbox, roi_center_px=center)
    cb_c = ((camera_bbox[0] + camera_bbox[2]) / 2, (camera_bbox[1] + camera_bbox[3]) / 2)
    dist = float(math.hypot(center[0] - cb_c[0], center[1] - cb_c[1]))
    inter = _intersection_area(camera_bbox, roi_bbox)
    cam_area = _area(camera_bbox)
    union = cam_area + _area(roi_bbox) - inter
    return RoiProjection(
        camera_bbox=tuple(float(v) for v in camera_bbox),
        roi_bbox=roi_bbox,
        roi_center_px=center,
        centroid_distance_px=dist,
        overlap_ratio=float(inter / cam_area) if cam_area > 0 else float("nan"),
        iou=float(inter / union) if union > 0 else float("nan"),
    )


def draw_roi_projection_overlay(
    path: str | Path,
    image_bgr: np.ndarray,
    camera,
    sol,
    camera_bbox: tuple[float, float, float, float] | None,
    auto_roi=None,
    manual_roi=None,
) -> None:
    out = image_bgr.copy()
    if camera_bbox is not None:
        _draw_bbox(out, camera_bbox, (0, 255, 0), "camera board")
    if auto_roi is not None:
        _draw_roi_bbox(out, camera, sol, auto_roi, (0, 200, 255), "AUTO ROI")
    if manual_roi is not None:
        _draw_roi_bbox(out, camera, sol, manual_roi, (255, 80, 0), "MANUAL ROI")
    cv2.imwrite(str(path), out)


def load_manual_roi_csv(path: str | Path) -> dict[int, FilterBox]:
    rois: dict[int, FilterBox] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("frame_index"):
                continue
            vals = {k: float(row.get(k, 0.0) or 0.0) for k in (
                "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "yaw", "pitch", "roll"
            )}
            box = FilterBox(**vals)
            if box.is_set:
                rois[int(row["frame_index"])] = box
    return rois


def export_manual_roi_template(path: str | Path, frames: list[int], auto_rois: dict[int, FilterBox] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_index", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "yaw", "pitch", "roll", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            box = (auto_rois or {}).get(frame)
            row = {"frame_index": frame, "note": "fill_or_adjust_from_step5_manual_roi"}
            for key in fields[1:-1]:
                row[key] = "" if box is None else f"{getattr(box, key):.6f}"
            writer.writerow(row)
    return path


def summarize_comparisons(frames: list[RoiComparisonFrame]) -> RoiComparisonSummary:
    def metric(which: str, pred):
        vals = [getattr(f, which) for f in frames if getattr(f, which) is not None]
        return sum(1 for v in vals if pred(v))

    cases: dict[str, int] = {}
    for f in frames:
        cases[f.roi_case] = cases.get(f.roi_case, 0) + 1
    return RoiComparisonSummary(
        n_frames=len(frames),
        auto_board_like=metric("auto", lambda m: m.board_like),
        manual_board_like=metric("manual", lambda m: m.board_like),
        auto_boundary_success=metric("auto", lambda m: m.best_boundary_count >= 3),
        manual_boundary_success=metric("manual", lambda m: m.best_boundary_count >= 3),
        auto_circle_ge1=metric("auto", lambda m: m.best_circle_count >= 1),
        manual_circle_ge1=metric("manual", lambda m: m.best_circle_count >= 1),
        auto_circle_ge4=metric("auto", lambda m: m.best_circle_count >= 4),
        manual_circle_ge4=metric("manual", lambda m: m.best_circle_count >= 4),
        auto_full_target=metric("auto", lambda m: m.full_target_success),
        manual_full_target=metric("manual", lambda m: m.full_target_success),
        case_counts=cases,
    )


def export_comparisons(frames: list[RoiComparisonFrame], out_dir: str | Path) -> RoiComparisonSummary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_comparisons(frames)
    payload = {"summary": asdict(summary), "frames": [_frame_to_dict(f) for f in frames]}
    (out / "auto_vs_manual_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "auto_vs_manual_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "frame", "auto_board_like", "manual_board_like", "auto_circles", "manual_circles",
            "auto_full_target", "manual_full_target", "roi_case",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in frames:
            writer.writerow({
                "frame": row.frame_index,
                "auto_board_like": "" if row.auto is None else int(row.auto.board_like),
                "manual_board_like": "" if row.manual is None else int(row.manual.board_like),
                "auto_circles": "" if row.auto is None else row.auto.best_circle_count,
                "manual_circles": "" if row.manual is None else row.manual.best_circle_count,
                "auto_full_target": "" if row.auto is None else int(row.auto.full_target_success),
                "manual_full_target": "" if row.manual is None else int(row.manual.full_target_success),
                "roi_case": row.roi_case,
            })
    return summary


def _run_roi(frame_index, xyz, roi, target, params, ring, config):
    current, candidates = extract_plane_candidates(xyz, roi, target, params, ring=ring, config=config)
    return FrameRerankResult(frame_index=frame_index, current=current, candidates=candidates, config=config)


def _project_points(points, camera, sol):
    cam_pts = sol.transform(np.asarray(points, dtype=float))
    keep = cam_pts[:, 2] > 1e-6
    uv = np.full((len(cam_pts), 2), np.nan, dtype=float)
    if keep.any():
        pts, _ = cv2.projectPoints(cam_pts[keep], np.zeros(3), np.zeros(3), camera.matrix(), camera.dist())
        uv[keep] = pts.reshape(-1, 2)
    return uv


def _draw_roi_bbox(out, camera, sol, roi, color, label):
    uv = _project_points(np.asarray(roi.corners(), dtype=float), camera, sol)
    finite = np.isfinite(uv).all(axis=1)
    if not finite.any():
        return
    x0, y0 = np.min(uv[finite], axis=0)
    x1, y1 = np.max(uv[finite], axis=0)
    _draw_bbox(out, (x0, y0, x1, y1), color, label)


def _draw_bbox(out, bbox, color, label):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
    cv2.putText(out, label, (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _area(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _intersection_area(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return _area((x0, y0, x1, y1))


def _frame_to_dict(frame: RoiComparisonFrame) -> dict:
    return {
        "frame_index": frame.frame_index,
        "roi_case": frame.roi_case,
        "auto": None if frame.auto is None else asdict(frame.auto),
        "manual": None if frame.manual is None else asdict(frame.manual),
    }


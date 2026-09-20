"""Diagnostic-only multi-plane extraction and board-likeness re-ranking.

This module deliberately does not replace the production LiDAR detector.  It
uses the same ROI and the same downstream boundary/circle code to ask a narrower
question: if more than one RANSAC plane is available in the ROI, is the board a
second or third candidate rather than the current first plane?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

from gui.core import detect_lidar as dl
from gui.core.evaluation.lidar_target_debug import draw_plane_local_xy, plane_local_frame, project_local
from gui.core.project import Target


@dataclass(frozen=True)
class BoardSize:
    width_m: float | None = None
    height_m: float | None = None

    @property
    def known(self) -> bool:
        return self.width_m is not None and self.height_m is not None and self.width_m > 0 and self.height_m > 0


@dataclass
class RerankConfig:
    max_plane_candidates: int = 5
    min_remaining_points: int = 50
    min_plane_points: int = 50
    hard_max_rms_m: float = 0.08
    board_size: BoardSize = field(default_factory=BoardSize)
    board_like_threshold: float = 0.55


@dataclass
class PlaneCandidate:
    index: int
    points: np.ndarray
    source_indices: np.ndarray
    normal: np.ndarray
    d: float
    rms_m: float
    remaining_point_count: int
    total_roi_points: int
    det: dl.LidarDetection
    scores: dict
    total_score: float
    hard_reject_reason: str = ""

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def circle_count(self) -> int:
        return int(self.det.n_circles)

    @property
    def boundary_count(self) -> int:
        return int(len(self.det.edges))

    @property
    def board_like(self) -> bool:
        if self.hard_reject_reason:
            return False
        if self.total_score < 0.55:
            return False
        size = self.scores.get("size_similarity")
        aspect = self.scores.get("aspect_similarity")
        if np.isfinite(size) and size < 0.40:
            return False
        if np.isfinite(aspect) and aspect < 0.40:
            return False
        return True


@dataclass
class FrameRerankResult:
    frame_index: int
    current: dl.LidarDetection
    candidates: list[PlaneCandidate]
    config: RerankConfig = field(default_factory=lambda: RerankConfig(board_size=BoardSize(0.74, 0.64)))

    @property
    def best(self) -> PlaneCandidate | None:
        valid = [c for c in self.candidates if not c.hard_reject_reason]
        if not valid:
            return None
        return max(valid, key=lambda c: c.total_score)

    @property
    def current_scores(self) -> dict | None:
        if len(self.current.plane) == 0:
            return None
        return score_candidate(
            self.current.plane,
            self.current,
            np.array([np.nan, np.nan, np.nan]),
            float("nan"),
            getattr(self.current, "plane_rms", float("nan")),
            max(len(self.current.filtered), 1),
            self.config,
        )

    @property
    def current_board_like(self) -> bool:
        scores = self.current_scores
        if not scores:
            return False
        size = scores.get("size_similarity")
        aspect = scores.get("aspect_similarity")
        if scores["total_score"] < self.config.board_like_threshold:
            return False
        if np.isfinite(size) and size < 0.40:
            return False
        if np.isfinite(aspect) and aspect < 0.40:
            return False
        return True


@dataclass
class RerankSummary:
    n_frames: int = 0
    current_plane_selected: int = 0
    reranked_plane_selected: int = 0
    current_board_like: int = 0
    other_candidate_board_like: int = 0
    no_board_like_candidate: int = 0
    current_boundary_success: int = 0
    reranked_boundary_success: int = 0
    current_circle_ge1: int = 0
    current_circle_ge2: int = 0
    current_circle_ge3: int = 0
    current_circle_ge4: int = 0
    reranked_circle_ge1: int = 0
    reranked_circle_ge2: int = 0
    reranked_circle_ge3: int = 0
    reranked_circle_ge4: int = 0
    current_full_target: int = 0
    reranked_full_target: int = 0
    avg_candidate_count: float = 0.0
    current_extent: dict = field(default_factory=dict)
    reranked_extent: dict = field(default_factory=dict)


class _AllRegion:
    def mask(self, xyz):
        return np.ones(len(xyz), dtype=bool)


def current_plane_selection_report() -> dict:
    return {
        "ransac_runs_per_detection": 1,
        "plane_candidates_kept": 1,
        "selection_basis": "fit_plane() returns the RANSAC hypothesis with the largest inlier count on the ROI cloud.",
        "largest_plane_selected": True,
        "first_valid_plane_selected": False,
        "point_count_used_for_selection": True,
        "extent_size_aspect_used_for_selection": False,
        "boundary_or_circle_used_for_selection": False,
        "why_large_planes_win": (
            "ROI points are scored only by plane inlier count before boundary/circle detection. "
            "A wall, floor, or long strip with more inliers than the board becomes the only selected plane."
        ),
    }


def extract_plane_candidates(
    xyz: np.ndarray,
    roi,
    target: Target,
    params: dl.DetectParams | None = None,
    ring: np.ndarray | None = None,
    config: RerankConfig | None = None,
) -> tuple[dl.LidarDetection, list[PlaneCandidate]]:
    """Run production detection once, then diagnostic repeated RANSAC on the ROI."""
    params = params or dl.DetectParams()
    config = config or RerankConfig()
    current = dl.detect(xyz, roi, target, params, ring=ring)

    xyz = np.asarray(xyz, np.float32)
    keep = dl.region_mask(xyz, roi)
    keep_indices = np.flatnonzero(keep)
    walks_rings = params.method == "ring" or (params.method == "original" and ring is not None)
    if walks_rings:
        filtered = xyz[keep]
        filtered_indices = keep_indices
        ring_filtered = None if ring is None else np.asarray(ring)[keep]
    else:
        filtered = dl.voxel_down(xyz[keep], params.voxel)
        filtered_indices = np.arange(len(filtered), dtype=np.int64)
        ring_filtered = None

    remaining = filtered.astype(np.float64)
    remaining_indices = filtered_indices.copy()
    remaining_ring = None if ring_filtered is None else np.asarray(ring_filtered).copy()
    total_roi_points = int(len(filtered))
    candidates: list[PlaneCandidate] = []
    rng = np.random.default_rng(0)

    for i in range(config.max_plane_candidates):
        if len(remaining) < config.min_remaining_points:
            break
        normal, d, inliers = dl.fit_plane(remaining, params.plane_threshold, rng=rng)
        if normal is None or int(inliers.sum()) < config.min_plane_points:
            break
        plane = remaining[inliers].astype(np.float32)
        source_indices = remaining_indices[inliers]
        candidate_ring = None if remaining_ring is None else remaining_ring[inliers]
        residual = plane.astype(np.float64) @ normal + d
        rms = float(np.sqrt(np.mean(residual**2))) if len(residual) else float("nan")
        cand_det = dl.detect(plane, _AllRegion(), target, params, ring=candidate_ring)
        scores = score_candidate(plane, cand_det, normal, d, rms, total_roi_points, config)
        hard = hard_reject_reason(plane, rms, config)
        candidates.append(
            PlaneCandidate(
                index=i + 1,
                points=plane,
                source_indices=np.asarray(source_indices, dtype=np.int64),
                normal=np.asarray(normal, dtype=float),
                d=float(d),
                rms_m=rms,
                remaining_point_count=int(len(remaining)),
                total_roi_points=total_roi_points,
                det=cand_det,
                scores=scores,
                total_score=float(scores["total_score"]),
                hard_reject_reason=hard,
            )
        )
        remaining = remaining[~inliers]
        remaining_indices = remaining_indices[~inliers]
        if remaining_ring is not None:
            remaining_ring = remaining_ring[~inliers]
    return current, candidates


def score_candidate(
    plane: np.ndarray,
    det: dl.LidarDetection,
    normal: np.ndarray,
    d: float,
    rms_m: float,
    total_roi_points: int,
    config: RerankConfig,
) -> dict:
    center, axes, local_xy = plane_local_frame(plane)
    ext = _extent(local_xy)
    boundary_xy = project_local(det.edges, center, axes)
    size_score, aspect_score = _size_aspect_scores(ext["width_m"], ext["height_m"], config.board_size)
    rect_score = _rectangularity(local_xy)
    plane_score = float(math.exp(-((rms_m if np.isfinite(rms_m) else 1.0) / 0.03) ** 2))
    density = float(len(plane) / max(ext["area_m2"], 1e-9)) if np.isfinite(ext["area_m2"]) else 0.0
    density_score = float(1.0 - math.exp(-density / 250.0))
    boundary_ratio = len(boundary_xy) / max(len(plane), 1)
    boundary_score = float(min(boundary_ratio / 0.08, 1.0))
    circle_score = float(min(det.n_circles / 4.0, 1.0))
    point_ratio = float(len(plane) / max(total_roi_points, 1))

    parts = {
        "size_similarity": size_score,
        "aspect_similarity": aspect_score,
        "rectangularity": rect_score,
        "plane_quality": plane_score,
        "point_density": density_score,
        "boundary_support": boundary_score,
        "circle_support": circle_score,
    }
    weights = {
        "size_similarity": 0.25,
        "aspect_similarity": 0.15,
        "rectangularity": 0.15,
        "plane_quality": 0.10,
        "point_density": 0.10,
        "boundary_support": 0.10,
        "circle_support": 0.15,
    }
    total = _weighted_score(parts, weights)
    return {
        **ext,
        "centroid": [float(v) for v in center],
        "normal": [float(v) for v in normal],
        "plane_d": float(d),
        "plane_rms_m": float(rms_m),
        "point_count": int(len(plane)),
        "point_ratio": point_ratio,
        "density_points_per_m2": density,
        "boundary_count": int(len(det.edges)),
        "boundary_ratio": float(boundary_ratio),
        "circle_count": int(det.n_circles),
        "circle_candidate_count": int(det.n_clusters),
        "radii_m": [float(r) for r in det.radii],
        "size_similarity": size_score,
        "aspect_similarity": aspect_score,
        "rectangularity": rect_score,
        "plane_quality": plane_score,
        "point_density": density_score,
        "boundary_support": boundary_score,
        "circle_support": circle_score,
        "total_score": total,
    }


def hard_reject_reason(plane: np.ndarray, rms_m: float, config: RerankConfig) -> str:
    if len(plane) < config.min_plane_points:
        return f"too_few_points:{len(plane)}"
    if not np.isfinite(rms_m):
        return "nonfinite_rms"
    if rms_m > config.hard_max_rms_m:
        return f"rms_too_high:{rms_m:.4f}"
    return ""


def summarize_results(results: list[FrameRerankResult]) -> RerankSummary:
    summary = RerankSummary(n_frames=len(results))
    cur_extents, best_extents, candidate_counts = [], [], []
    for r in results:
        best = r.best
        candidate_counts.append(len(r.candidates))
        if len(r.current.plane):
            summary.current_plane_selected += 1
            cur_score = r.current_scores
            cur_extents.append(cur_score)
            if r.current_board_like:
                summary.current_board_like += 1
        if len(r.current.edges) >= 3:
            summary.current_boundary_success += 1
        summary.current_circle_ge1 += int(r.current.n_circles >= 1)
        summary.current_circle_ge2 += int(r.current.n_circles >= 2)
        summary.current_circle_ge3 += int(r.current.n_circles >= 3)
        summary.current_circle_ge4 += int(r.current.n_circles >= 4)
        summary.current_full_target += int(r.current.ok)

        if best is not None:
            summary.reranked_plane_selected += 1
            best_extents.append(best.scores)
            if len(best.det.edges) >= 3:
                summary.reranked_boundary_success += 1
            summary.reranked_circle_ge1 += int(best.det.n_circles >= 1)
            summary.reranked_circle_ge2 += int(best.det.n_circles >= 2)
            summary.reranked_circle_ge3 += int(best.det.n_circles >= 3)
            summary.reranked_circle_ge4 += int(best.det.n_circles >= 4)
            summary.reranked_full_target += int(best.det.ok)

        current_is_board = r.current_board_like
        other_board = any(c.board_like for c in r.candidates[1:])
        if other_board:
            summary.other_candidate_board_like += 1
        if not current_is_board and not other_board:
            summary.no_board_like_candidate += 1
    summary.avg_candidate_count = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    summary.current_extent = _extent_distribution(cur_extents)
    summary.reranked_extent = _extent_distribution(best_extents)
    return summary


def export_frame_result(result: FrameRerankResult, out_dir: str | Path, target: Target) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = result.best
    payload = {
        "frame_index": result.frame_index,
        "current_selected": _det_summary(result.current),
        "best_candidate_index": None if best is None else best.index,
        "best_candidate_score": None if best is None else best.total_score,
        "candidates": [candidate_to_dict(c) for c in result.candidates],
    }
    (out / "frame_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for c in result.candidates:
        c_json = out / f"candidate_{c.index:02d}.json"
        c_json.write_text(json.dumps(candidate_to_dict(c), indent=2, sort_keys=True), encoding="utf-8")
        center, axes, plane_xy = plane_local_frame(c.points)
        edge_xy = project_local(c.det.edges, center, axes)
        centers = np.empty((0, 3), dtype=float) if c.det.centers is None else np.asarray(c.det.centers, dtype=float)
        circle_xy = project_local(centers, center, axes)
        draw_plane_local_xy(
            out / f"candidate_{c.index:02d}_xy.png",
            plane_xy,
            edge_xy,
            circle_xy,
            list(c.det.radii),
            target,
            None,
            title=f"frame {result.frame_index} candidate {c.index} score={c.total_score:.3f}",
        )
    if result.candidates:
        _copy_candidate_png(out, result.candidates[0].index, "current_selected.png")
    if best is not None:
        _copy_candidate_png(out, best.index, "reranked_selected.png")


def export_summary(results: list[FrameRerankResult], out_dir: str | Path) -> RerankSummary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    (out / "summary.json").write_text(
        json.dumps({"plane_selection": current_plane_selection_report(), "summary": asdict(summary)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_index", "candidate_count", "current_points", "current_circles",
            "best_index", "best_score", "best_width_m", "best_height_m", "best_circles",
            "best_boundary_count", "case",
        ])
        writer.writeheader()
        for r in results:
            best = r.best
            current_like = r.current_board_like
            other_like = any(c.board_like for c in r.candidates[1:])
            case = "A_current_board_like" if current_like else ("B_other_candidate_board_like" if other_like else "C_no_board_like_candidate")
            writer.writerow({
                "frame_index": r.frame_index,
                "candidate_count": len(r.candidates),
                "current_points": len(r.current.plane),
                "current_circles": r.current.n_circles,
                "best_index": "" if best is None else best.index,
                "best_score": "" if best is None else f"{best.total_score:.6f}",
                "best_width_m": "" if best is None else f"{best.scores['width_m']:.6f}",
                "best_height_m": "" if best is None else f"{best.scores['height_m']:.6f}",
                "best_circles": "" if best is None else best.det.n_circles,
                "best_boundary_count": "" if best is None else len(best.det.edges),
                "case": case,
            })
    return summary


def candidate_to_dict(c: PlaneCandidate) -> dict:
    return {
        "index": c.index,
        "hard_reject_reason": c.hard_reject_reason,
        "board_like": c.board_like,
        "total_score": c.total_score,
        "plane_point_count": c.n_points,
        "remaining_point_count": c.remaining_point_count,
        "total_roi_points": c.total_roi_points,
        "plane_equation": {"normal": [float(v) for v in c.normal], "d": float(c.d)},
        "scores": c.scores,
        "detector": _det_summary(c.det),
    }


def _det_summary(det: dl.LidarDetection) -> dict:
    return {
        "ok": bool(det.ok),
        "reason": det.reason,
        "plane_point_count": int(len(det.plane)),
        "boundary_point_count": int(len(det.edges)),
        "circle_count": int(det.n_circles),
        "circle_candidate_count": int(det.n_clusters),
        "radii_m": [float(r) for r in det.radii],
        "plane_rms": float(getattr(det, "plane_rms", float("nan"))),
    }


def _extent(local_xy: np.ndarray) -> dict:
    xy = np.asarray(local_xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return {"width_m": float("nan"), "height_m": float("nan"), "area_m2": float("nan"), "aspect_ratio": float("nan")}
    ext = np.ptp(xy, axis=0)
    width = float(np.max(ext))
    height = float(np.min(ext))
    return {
        "width_m": width,
        "height_m": height,
        "area_m2": width * height,
        "aspect_ratio": width / max(height, 1e-9),
    }


def _size_aspect_scores(width: float, height: float, board_size: BoardSize) -> tuple[float, float]:
    if not board_size.known or not np.isfinite(width) or not np.isfinite(height):
        return float("nan"), float("nan")
    measured = np.array([max(width, height), min(width, height)], dtype=float)
    expected_a = np.array([max(board_size.width_m, board_size.height_m), min(board_size.width_m, board_size.height_m)])
    expected_b = expected_a[::-1]
    size_a = _pair_size_score(measured, expected_a)
    size_b = _pair_size_score(measured, expected_b)
    expected = expected_a if size_a >= size_b else expected_b
    size = max(size_a, size_b)
    aspect = math.exp(-abs(math.log((measured[0] / max(measured[1], 1e-9)) / (expected[0] / max(expected[1], 1e-9)))))
    return float(size), float(aspect)


def _pair_size_score(measured: np.ndarray, expected: np.ndarray) -> float:
    err = np.max(np.abs(np.log(np.maximum(measured, 1e-9) / np.maximum(expected, 1e-9))))
    return float(math.exp(-err))


def _rectangularity(local_xy: np.ndarray) -> float:
    xy = np.asarray(local_xy, dtype=np.float32).reshape(-1, 2)
    if len(xy) < 3:
        return 0.0
    hull = cv2.convexHull(xy)
    hull_area = float(abs(cv2.contourArea(hull)))
    (_, _), (w, h), _ = cv2.minAreaRect(xy)
    obb_area = max(float(w * h), 1e-9)
    return float(np.clip(hull_area / obb_area, 0.0, 1.0))


def _weighted_score(parts: dict, weights: dict) -> float:
    num = 0.0
    den = 0.0
    for key, weight in weights.items():
        value = parts.get(key, float("nan"))
        if np.isfinite(value):
            num += float(value) * weight
            den += weight
    return float(num / den) if den else 0.0


def _extent_distribution(rows: list[dict]) -> dict:
    out = {}
    for key in ("width_m", "height_m", "area_m2", "total_score"):
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[key] = {
            "median": float(np.median(vals)) if vals.size else float("nan"),
            "p95": float(np.percentile(vals, 95)) if vals.size else float("nan"),
        }
    return out


def _copy_candidate_png(out: Path, index: int, name: str) -> None:
    src = out / f"candidate_{index:02d}_xy.png"
    dst = out / name
    if src.exists():
        dst.write_bytes(src.read_bytes())

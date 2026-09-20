"""4-hole correspondence symmetry diagnostics.

This module is diagnostic-only.  It never changes the production solver,
extrinsic files, temporal correction, deskew, or any project YAML.  Its job is
to expose when a symmetric rectangular target lets several physical
correspondence assignments fit equally well in 3D.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gui.core.evaluation.calibration_correspondence import (
    leave_one_out_pair_errors,
    residual_stats_mm,
    rotation_geodesic_deg,
)
from gui.core.project import RESULT_DIR
from gui.core.solve import solve_rigid, sort_centers


SYMMETRY_CANDIDATES = {
    "NORMAL": (0, 1, 2, 3),
    "REVERSED": (3, 2, 1, 0),
    "MIRROR": (1, 0, 3, 2),
    "HALF_TURN": (2, 3, 0, 1),
}


ORDERING_NOTES = {
    "feature_id": (
        "ordered_0..ordered_3 are solver ordering labels, not physical hole IDs. "
        "They must not be documented as top-left/top-right/etc unless an "
        "independent asymmetric feature or marker ID proves that mapping."
    ),
    "sort_centers": (
        "sort_centers() centers the four points, estimates a board plane with "
        "SVD, takes the first two in-plane singular vectors as local axes, flips "
        "one axis using the target centroid/view direction, sorts by bearing, "
        "then rotates the cycle so a long rectangle edge starts the sequence."
    ),
    "axis_sign": (
        "PCA/SVD axes have sign ambiguity. The current rule reduces it with a "
        "view-direction check, but a symmetric rectangle still has half-turn and "
        "mirror-equivalent assignments that 4-point RMSE cannot distinguish."
    ),
}


@dataclass
class PlausibilityThresholds:
    max_translation_from_seed_m: float = 1.0
    max_rotation_from_seed_deg: float = 10.0
    max_translation_norm_m: float = 10.0
    seed_gate_enabled: bool = True
    min_match_rate: float = 0.05
    min_projection_margin_px: float = 2.0
    min_projection_p95_margin_px: float = 3.0
    min_match_rate_ratio: float = 0.75
    max_cross_scene_rotation_p95_deg: float = 10.0
    max_cross_scene_translation_p95_mm: float = 500.0


@dataclass
class ProjectionProxyParams:
    max_projected_points: int = 50_000
    max_lidar_edge_points: int = 20_000
    canny_low: int = 50
    canny_high: int = 150
    depth_jump_threshold_m: float = 0.3
    grid_px: int = 3
    thresholds_px: tuple[float, ...] = (5.0, 10.0, 15.0)


@dataclass
class CandidateResult:
    name: str
    permutation: list[int]
    mapping: list[str]
    R: list[list[float]]
    t: list[float]
    det_R: float
    orthogonality_error_fro: float
    fit_stats_mm: dict[str, float]
    residual_norms_mm: list[float]
    loo_errors_mm: list[float]
    loo_stats_mm: dict[str, float]
    delta_rotation_seed_deg: float | None = None
    delta_translation_seed_xyz_mm: list[float] | None = None
    delta_translation_seed_norm_mm: float | None = None
    plausibility_warnings: list[str] = field(default_factory=list)
    projection_frame407: dict[str, Any] | None = None
    projection_summary: dict[str, Any] | None = None
    cross_scene_summary: dict[str, Any] | None = None
    status: str = "UNRANKED"


@dataclass
class CandidateDecision:
    status: str
    selection_source: str
    preferred_candidate_name: str | None
    confidence: str
    reason: str
    candidate_names: list[str]
    thresholds: dict[str, Any]


def pca_plane_xy(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    rel = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    return rel @ vt[:2].T


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)


def candidate_permutations(include_all: bool = True) -> list[tuple[str, tuple[int, ...]]]:
    named = list(SYMMETRY_CANDIDATES.items())
    if not include_all:
        return named
    seen = {perm for _, perm in named}
    for perm in itertools.permutations(range(4)):
        if perm not in seen:
            named.append((f"PERM_{''.join(map(str, perm))}", perm))
            seen.add(perm)
    return named


def fit_candidate(
    name: str,
    permutation: tuple[int, ...],
    lidar_points: np.ndarray,
    camera_points: np.ndarray,
    seed_R: np.ndarray | None = None,
    seed_t: np.ndarray | None = None,
    thresholds: PlausibilityThresholds = PlausibilityThresholds(),
) -> CandidateResult:
    L = np.asarray(lidar_points, np.float64).reshape(4, 3)
    C = np.asarray(camera_points, np.float64).reshape(4, 3)[list(permutation)]
    R, t = solve_rigid(L, C)
    pred = (R @ L.T).T + t
    residual_norms = np.linalg.norm(pred - C, axis=1) * 1000.0
    loo_errors, _ = leave_one_out_pair_errors(L, C)
    det_R = float(np.linalg.det(R))
    orth = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))

    dR = dtn = None
    dt_xyz = None
    warnings: list[str] = []
    if seed_R is not None and seed_t is not None:
        dR = rotation_geodesic_deg(seed_R, R)
        dt = (t - np.asarray(seed_t, np.float64).reshape(3)) * 1000.0
        dt_xyz = [float(x) for x in dt]
        dtn = float(np.linalg.norm(dt))
        if thresholds.seed_gate_enabled and dR > thresholds.max_rotation_from_seed_deg:
            warnings.append("ROTATION_FAR_FROM_SEED")
        if thresholds.seed_gate_enabled and dtn > thresholds.max_translation_from_seed_m * 1000.0:
            warnings.append("TRANSLATION_FAR_FROM_SEED")
    if np.linalg.norm(t) > thresholds.max_translation_norm_m:
        warnings.append("TRANSLATION_NORM_IMPLAUSIBLE")
    if abs(det_R - 1.0) > 1e-6 or orth > 1e-6:
        warnings.append("ROTATION_NUMERIC_WARNING")

    return CandidateResult(
        name=name,
        permutation=list(permutation),
        mapping=[f"lidar_ordered_{i}->camera_ordered_{j}" for i, j in enumerate(permutation)],
        R=R.tolist(),
        t=[float(x) for x in t],
        det_R=det_R,
        orthogonality_error_fro=orth,
        fit_stats_mm=residual_stats_mm(residual_norms),
        residual_norms_mm=[float(x) for x in residual_norms],
        loo_errors_mm=[float(x) for x in loo_errors],
        loo_stats_mm=residual_stats_mm(np.asarray(loo_errors, np.float64)),
        delta_rotation_seed_deg=dR,
        delta_translation_seed_xyz_mm=dt_xyz,
        delta_translation_seed_norm_mm=dtn,
        plausibility_warnings=warnings,
    )


def analyze_symmetry_candidates(
    lidar_points: np.ndarray,
    camera_points: np.ndarray,
    seed_R: np.ndarray | None = None,
    seed_t: np.ndarray | None = None,
    include_all_permutations: bool = True,
    thresholds: PlausibilityThresholds = PlausibilityThresholds(),
) -> list[CandidateResult]:
    candidates = [
        fit_candidate(name, perm, lidar_points, camera_points, seed_R, seed_t, thresholds)
        for name, perm in candidate_permutations(include_all_permutations)
    ]
    candidates.sort(key=lambda c: (c.fit_stats_mm["rmse"], c.loo_stats_mm["max"], c.name))
    _assign_status(candidates)
    return candidates


def fit_candidate_set(
    name: str,
    permutation: tuple[int, ...],
    scenes: list[tuple[str, np.ndarray, np.ndarray]],
    seed_R: np.ndarray | None = None,
    seed_t: np.ndarray | None = None,
    thresholds: PlausibilityThresholds = PlausibilityThresholds(),
) -> CandidateResult:
    """Fit one symmetry candidate across scenes.

    Camera points are kept in the detector's physical canonical order.  Only the
    LiDAR side is geometrically canonicalized, because the current LiDAR target
    detector has no physical hole IDs.
    """
    scene_blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sid, lidar_pts, camera_pts in scenes:
        L = sort_centers(np.asarray(lidar_pts, np.float64), "lidar")
        C = np.asarray(camera_pts, np.float64).reshape(4, 3)[list(permutation)]
        scene_blocks.append((sid, L, C))
    if not scene_blocks:
        raise ValueError("no usable scenes")

    L_all = np.vstack([b[1] for b in scene_blocks])
    C_all = np.vstack([b[2] for b in scene_blocks])
    out = _fit_arrays(name, permutation, L_all, C_all, seed_R, seed_t, thresholds)
    out.cross_scene_summary = _cross_scene_consistency(scene_blocks, np.asarray(out.R), np.asarray(out.t))
    return out


def analyze_calibration_symmetry(
    scenes: list[tuple[str, np.ndarray, np.ndarray]],
    seed_R: np.ndarray | None = None,
    seed_t: np.ndarray | None = None,
    include_all_permutations: bool = False,
    thresholds: PlausibilityThresholds = PlausibilityThresholds(),
    projection_metrics_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[CandidateResult], CandidateDecision]:
    """Production-workflow symmetry gate, without applying the selected transform."""
    candidates = [
        fit_candidate_set(name, perm, scenes, seed_R, seed_t, thresholds)
        for name, perm in candidate_permutations(include_all_permutations)
    ]
    projection_metrics_by_candidate = projection_metrics_by_candidate or {}
    for c in candidates:
        if c.name in projection_metrics_by_candidate:
            c.projection_summary = dict(projection_metrics_by_candidate[c.name])
    candidates.sort(key=lambda c: (c.fit_stats_mm["rmse"], c.loo_stats_mm["max"], c.name))
    _assign_status(candidates)
    decision = decide_candidate(candidates, thresholds, physical_ids_from_both_sensors=False)
    return candidates, decision


def _fit_arrays(
    name: str,
    permutation: tuple[int, ...],
    lidar: np.ndarray,
    camera: np.ndarray,
    seed_R: np.ndarray | None,
    seed_t: np.ndarray | None,
    thresholds: PlausibilityThresholds,
) -> CandidateResult:
    L = np.asarray(lidar, np.float64).reshape(-1, 3)
    C = np.asarray(camera, np.float64).reshape(-1, 3)
    R, t = solve_rigid(L, C)
    pred = (R @ L.T).T + t
    residual_norms = np.linalg.norm(pred - C, axis=1) * 1000.0
    loo_errors, _ = leave_one_out_pair_errors(L, C) if len(L) >= 4 else ([], [])
    det_R = float(np.linalg.det(R))
    orth = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))

    dR = dtn = None
    dt_xyz = None
    warnings: list[str] = []
    if seed_R is not None and seed_t is not None:
        dR = rotation_geodesic_deg(seed_R, R)
        dt = (t - np.asarray(seed_t, np.float64).reshape(3)) * 1000.0
        dt_xyz = [float(x) for x in dt]
        dtn = float(np.linalg.norm(dt))
        if thresholds.seed_gate_enabled and dR > thresholds.max_rotation_from_seed_deg:
            warnings.append("ROTATION_FAR_FROM_SEED")
        if thresholds.seed_gate_enabled and dtn > thresholds.max_translation_from_seed_m * 1000.0:
            warnings.append("TRANSLATION_FAR_FROM_SEED")
    if np.linalg.norm(t) > thresholds.max_translation_norm_m:
        warnings.append("TRANSLATION_NORM_IMPLAUSIBLE")
    if abs(det_R - 1.0) > 1e-6 or orth > 1e-6:
        warnings.append("ROTATION_NUMERIC_WARNING")

    return CandidateResult(
        name=name,
        permutation=list(permutation),
        mapping=[f"lidar_ordered_{i}->camera_physical_{j}" for i, j in enumerate(permutation)],
        R=R.tolist(),
        t=[float(x) for x in t],
        det_R=det_R,
        orthogonality_error_fro=orth,
        fit_stats_mm=residual_stats_mm(residual_norms),
        residual_norms_mm=[float(x) for x in residual_norms],
        loo_errors_mm=[float(x) for x in loo_errors],
        loo_stats_mm=residual_stats_mm(np.asarray(loo_errors, np.float64)),
        delta_rotation_seed_deg=dR,
        delta_translation_seed_xyz_mm=dt_xyz,
        delta_translation_seed_norm_mm=dtn,
        plausibility_warnings=warnings,
    )


def _cross_scene_consistency(
    scene_blocks: list[tuple[str, np.ndarray, np.ndarray]],
    joint_R: np.ndarray,
    joint_t: np.ndarray,
) -> dict[str, Any]:
    d_rots: list[float] = []
    d_ts: list[float] = []
    rows: list[dict[str, Any]] = []
    for sid, L, C in scene_blocks:
        if len(L) < 3:
            continue
        R, t = solve_rigid(L, C)
        dR = rotation_geodesic_deg(joint_R, R)
        dt = float(np.linalg.norm(t - joint_t) * 1000.0)
        d_rots.append(dR)
        d_ts.append(dt)
        rows.append({"scene": sid, "delta_rotation_deg": dR, "delta_translation_mm": dt})
    return {
        "scene_count": len(rows),
        "rotation_deg": _distribution(d_rots),
        "translation_mm": _distribution(d_ts),
        "per_scene": rows,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, np.float64)
    if len(a) == 0:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _assign_status(candidates: list[CandidateResult]) -> None:
    if not candidates:
        return
    best = candidates[0]
    best_rmse = best.fit_stats_mm["rmse"]
    near = [
        c for c in candidates
        if abs(c.fit_stats_mm["rmse"] - best_rmse) <= max(0.5, best_rmse * 0.05)
    ]
    for c in candidates:
        if c.plausibility_warnings:
            c.status = "PHYSICALLY_IMPLAUSIBLE"
        elif len(near) > 1 and c in near:
            c.status = "AMBIGUOUS"
        elif c is best:
            c.status = "CLEAR"
        else:
            c.status = "WORSE_FIT"


def physically_viable(c: CandidateResult) -> bool:
    hard = {"TRANSLATION_NORM_IMPLAUSIBLE", "ROTATION_NUMERIC_WARNING"}
    return not (set(c.plausibility_warnings) & hard)


def projection_summary_score(summary: dict[str, Any] | None) -> tuple[bool, float, float, float]:
    if not summary or not summary.get("ok", True):
        return False, float("inf"), float("inf"), 0.0
    median = summary.get("matched_median_px", summary.get("median_px", float("inf")))
    p95 = summary.get("matched_p95_px", summary.get("p95_px", float("inf")))
    rate = summary.get("match_rate", 0.0)
    try:
        return True, float(median), float(p95), float(rate)
    except (TypeError, ValueError):
        return False, float("inf"), float("inf"), 0.0


def cross_scene_ok(c: CandidateResult, thresholds: PlausibilityThresholds) -> bool:
    summary = c.cross_scene_summary
    if not summary or int(summary.get("scene_count", 0)) < 2:
        return False
    rot = (summary.get("rotation_deg") or {}).get("p95", float("inf"))
    trans = (summary.get("translation_mm") or {}).get("p95", float("inf"))
    return (
        np.isfinite(rot)
        and np.isfinite(trans)
        and float(rot) <= thresholds.max_cross_scene_rotation_p95_deg
        and float(trans) <= thresholds.max_cross_scene_translation_p95_mm
    )


def decide_candidate(
    candidates: list[CandidateResult],
    thresholds: PlausibilityThresholds = PlausibilityThresholds(),
    physical_ids_from_both_sensors: bool = False,
) -> CandidateDecision:
    """Conservative, diagnostic-only candidate policy.

    This does not mutate the production solve.  A candidate is preferred only
    when non-RMSE evidence is clearly decisive.  Otherwise the result remains
    CALIBRATION_AMBIGUOUS so the UI can warn rather than silently applying a
    symmetry choice.
    """
    viable = [c for c in candidates if physically_viable(c)]
    threshold_dict = {
        "max_translation_from_seed_m": thresholds.max_translation_from_seed_m,
        "max_rotation_from_seed_deg": thresholds.max_rotation_from_seed_deg,
        "max_translation_norm_m": thresholds.max_translation_norm_m,
        "seed_gate_enabled": thresholds.seed_gate_enabled,
        "min_match_rate": thresholds.min_match_rate,
        "min_projection_margin_px": thresholds.min_projection_margin_px,
        "min_projection_p95_margin_px": thresholds.min_projection_p95_margin_px,
        "min_match_rate_ratio": thresholds.min_match_rate_ratio,
        "max_cross_scene_rotation_p95_deg": thresholds.max_cross_scene_rotation_p95_deg,
        "max_cross_scene_translation_p95_mm": thresholds.max_cross_scene_translation_p95_mm,
    }
    if not candidates:
        return CandidateDecision("CALIBRATION_INVALID", "AMBIGUOUS", None, "LOW", "no candidates", [], threshold_dict)
    if not viable:
        return CandidateDecision(
            "CALIBRATION_INVALID",
            "AMBIGUOUS",
            None,
            "LOW",
            "all candidates failed hard physical plausibility checks",
            [c.name for c in candidates],
            threshold_dict,
        )
    if physical_ids_from_both_sensors:
        best = viable[0]
        return CandidateDecision(
            "CALIBRATION_ACCEPTABLE",
            "PHYSICAL_ID",
            best.name,
            "HIGH",
            "both sensors provide physical feature IDs; no LiDAR-ID path exists for the current target",
            [c.name for c in viable],
            threshold_dict,
        )

    projection_available = []
    projected = []
    for c in viable:
        ok, median_px, p95_px, rate = projection_summary_score(c.projection_summary or c.projection_frame407)
        if ok and rate >= thresholds.min_match_rate:
            projection_available.append(c)
        if ok and rate >= thresholds.min_match_rate and cross_scene_ok(c, thresholds):
            projected.append((median_px, p95_px, -rate, c))
    if len(projected) >= 2:
        projected.sort(key=lambda row: (row[0], row[1], row[3].name))
        best_median, best_p95, best_neg_rate, best = projected[0]
        second_median, second_p95, second_neg_rate, _ = projected[1]
        best_rate = -best_neg_rate
        second_rate = -second_neg_rate
        clearly_better = (
            second_median - best_median >= thresholds.min_projection_margin_px
            and second_p95 - best_p95 >= thresholds.min_projection_p95_margin_px
            and best_rate >= second_rate * thresholds.min_match_rate_ratio
        )
        if clearly_better:
            return CandidateDecision(
                "CALIBRATION_ACCEPTABLE",
                "PRIOR_DISAMBIGUATION",
                best.name,
                "MEDIUM",
                "projection evidence is clearly better without match-rate collapse",
                [c.name for _, _, _, c in projected],
                threshold_dict,
            )

    seeded = [
        c for c in viable
        if c.delta_rotation_seed_deg is not None and c.delta_translation_seed_norm_mm is not None
    ]
    if len(seeded) >= 2:
        seeded.sort(key=lambda c: (c.delta_rotation_seed_deg, c.delta_translation_seed_norm_mm, c.name))
        best, second = seeded[0], seeded[1]
        d_rot_gap = (second.delta_rotation_seed_deg or 0.0) - (best.delta_rotation_seed_deg or 0.0)
        d_t_gap = (second.delta_translation_seed_norm_mm or 0.0) - (best.delta_translation_seed_norm_mm or 0.0)
        best_has_projection = projection_summary_score(best.projection_summary or best.projection_frame407)[0]
        if (
            (d_rot_gap >= 10.0 or d_t_gap >= 100.0)
            and not best.plausibility_warnings
            and best_has_projection
            and cross_scene_ok(best, thresholds)
        ):
            return CandidateDecision(
                "CALIBRATION_ACCEPTABLE",
                "PRIOR_DISAMBIGUATION",
                best.name,
                "MEDIUM",
                "seed/global proximity supports projection and cross-scene evidence; this is not physical ID observation",
                [c.name for c in seeded],
                threshold_dict,
            )

    reason = "no physical LiDAR IDs and no decisive projection/seed prior; RMSE/LOO are auxiliary only"
    if projection_available:
        reason = (
            "projection evidence is available, but cross-scene consistency evidence is missing or below gate; "
            "keeping calibration ambiguous"
        )
    return CandidateDecision(
        "CALIBRATION_AMBIGUOUS",
        "AMBIGUOUS",
        None,
        "LOW",
        reason,
        [c.name for c in viable],
        threshold_dict,
    )


def correspondence_from_scene_json(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lidar = np.asarray([p["lidar_xyz"] for p in data["pairs"]], np.float64)
    camera = np.asarray([p["camera_xyz"] for p in data["pairs"]], np.float64)
    return lidar, camera, data


def lightweight_projection_proxy(
    image_bgr: np.ndarray,
    projection_uv: np.ndarray,
    projection_depth: np.ndarray,
    params: ProjectionProxyParams = ProjectionProxyParams(),
) -> dict[str, Any]:
    """Distance-transform projection proxy, cheap enough for permutation sweeps.

    It deliberately uses a simpler matching rule than the full edge-alignment
    metric: projected LiDAR depth-edge points are compared to the nearest Canny
    image edge via a distance transform.  The proxy is for candidate comparison,
    not a replacement for the existing dense evaluator.
    """
    uv = np.asarray(projection_uv, np.float64).reshape(-1, 2)
    depth = np.asarray(projection_depth, np.float64).reshape(-1)
    if len(uv) == 0:
        return {"ok": False, "reason": "no projected points"}
    if len(uv) > params.max_projected_points:
        pick = np.linspace(0, len(uv) - 1, params.max_projected_points).astype(np.int64)
        uv = uv[pick]
        depth = depth[pick]

    edge_mask = _projected_depth_edges(uv, depth, image_bgr.shape[:2], params)
    edge_uv = uv[edge_mask]
    if len(edge_uv) > params.max_lidar_edge_points:
        pick = np.linspace(0, len(edge_uv) - 1, params.max_lidar_edge_points).astype(np.int64)
        edge_uv = edge_uv[pick]
    if len(edge_uv) == 0:
        return {"ok": False, "reason": "no lidar proxy edge points", "n_projected": int(len(uv))}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    edges = cv2.Canny(gray, params.canny_low, params.canny_high)
    if not edges.any():
        return {"ok": False, "reason": "no image edges", "n_projected": int(len(uv))}
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    h, w = dist.shape
    x = np.clip(np.rint(edge_uv[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(edge_uv[:, 1]).astype(np.int32), 0, h - 1)
    errors = dist[y, x].astype(np.float64)
    matched15 = errors <= max(params.thresholds_px)
    matched_errors = errors[matched15]
    return {
        "ok": True,
        "n_projected": int(len(uv)),
        "n_lidar_edge_points": int(len(edge_uv)),
        "matched_mean_px": float(np.mean(matched_errors)) if len(matched_errors) else float("nan"),
        "matched_median_px": float(np.median(matched_errors)) if len(matched_errors) else float("nan"),
        "matched_p95_px": float(np.percentile(matched_errors, 95)) if len(matched_errors) else float("nan"),
        "match_rate": float(np.mean(matched15)),
        "within_5px": float(np.mean(errors <= 5.0)),
        "within_10px": float(np.mean(errors <= 10.0)),
    }


def _projected_depth_edges(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: tuple[int, int],
    params: ProjectionProxyParams,
) -> np.ndarray:
    h, w = image_shape
    x = np.clip((uv[:, 0] / params.grid_px).astype(np.int32), 0, max(w // params.grid_px, 1))
    y = np.clip((uv[:, 1] / params.grid_px).astype(np.int32), 0, max(h // params.grid_px, 1))
    gw = max(w // params.grid_px + 1, 1)
    gh = max(h // params.grid_px + 1, 1)
    flat = y * gw + x
    n = gw * gh
    zmin = np.full(n, np.inf)
    zmax = np.full(n, -np.inf)
    np.minimum.at(zmin, flat, depth)
    np.maximum.at(zmax, flat, depth)
    # Dilate the min/max grids a little so a point can see neighbouring cells.
    zmin2 = zmin.reshape(gh, gw)
    zmax2 = zmax.reshape(gh, gw)
    valid = np.isfinite(zmin2) & np.isfinite(zmax2)
    sentinel = np.float32(1.0e6)
    zmin_img = np.where(valid, zmin2, sentinel).astype(np.float32)
    zmax_img = np.where(valid, zmax2, -sentinel).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    local_min = -cv2.dilate(-zmin_img, kernel)
    local_max = cv2.dilate(zmax_img, kernel)
    finite = np.isfinite(local_max) & np.isfinite(local_min)
    rng_img = np.zeros_like(local_max, dtype=np.float32)
    rng_img[finite] = local_max[finite] - local_min[finite]
    rng = rng_img.reshape(-1)
    return (rng[flat] > params.depth_jump_threshold_m) & (depth <= zmin[flat] + 1e-6)


def export_symmetry_diagnostic(
    scene_json: str | Path,
    out_dir: str | Path | None = None,
    seed_R: np.ndarray | None = None,
    seed_t: np.ndarray | None = None,
    include_all_permutations: bool = True,
    thresholds: PlausibilityThresholds = PlausibilityThresholds(),
) -> tuple[dict[str, Any], Path]:
    L, C, scene_data = correspondence_from_scene_json(scene_json)
    candidates = analyze_symmetry_candidates(L, C, seed_R, seed_t, include_all_permutations, thresholds)
    decision = decide_candidate(candidates, thresholds, physical_ids_from_both_sensors=False)
    report = {
        "source_json": str(scene_json),
        "ordering_notes": ORDERING_NOTES,
        "selection_policy": {
            "production_solve_changed": False,
            "yaml_or_extrinsic_written": False,
            "rmse_is_auxiliary_only": True,
            "current_target_lidar_physical_ids_available": False,
        },
        "candidate_decision": decision_to_dict(decision),
        "scene_pairs": scene_data.get("pairs", []),
        "camera_physical_ids": scene_data.get("camera_physical_ids", []),
        "camera_marker_ids": scene_data.get("camera_marker_ids", []),
        "camera_board_xy": scene_data.get("camera_board_xy", []),
        "camera_pca_xy": pca_plane_xy(C).tolist(),
        "lidar_pca_xy": pca_plane_xy(L).tolist(),
        "camera_distance_matrix_m": pairwise_distance_matrix(C).tolist(),
        "lidar_distance_matrix_m": pairwise_distance_matrix(L).tolist(),
        "candidates": [candidate_to_dict(c) for c in candidates],
        "named_candidates": {
            c.name: candidate_to_dict(c)
            for c in candidates
            if c.name in SYMMETRY_CANDIDATES
        },
        "diagnostic_status": diagnostic_status(candidates),
    }
    out = Path(out_dir) if out_dir is not None else RESULT_DIR / "correspondence_symmetry"
    out.mkdir(parents=True, exist_ok=True)
    (out / "symmetry_diagnostic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "candidate_selection_evidence.json").write_text(
        json.dumps({
            "decision": decision_to_dict(decision),
            "candidates": [candidate_to_dict(c) for c in candidates],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_candidate_csv(out / "symmetry_candidates.csv", candidates)
    _write_pca_svg(out / "symmetry_pca.svg", report["camera_pca_xy"], report["lidar_pca_xy"])
    return report, out


def export_candidate_selection_evidence(
    candidates: list[CandidateResult],
    decision: CandidateDecision,
    out_dir: str | Path | None = None,
    legacy_solution: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir) if out_dir is not None else RESULT_DIR / "candidate_selection"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": decision_to_dict(decision),
        "legacy_solution": legacy_solution,
        "selection_policy": {
            "production_solve_changed": False,
            "yaml_or_extrinsic_written": False,
            "rmse_and_loo_are_auxiliary_only": True,
            "current_target_lidar_physical_ids_available": False,
            "seed_is_prior_not_physical_id": True,
        },
        "candidates": [candidate_to_dict(c) for c in candidates],
    }
    (out / "candidate_selection_evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_candidate_csv(out / "candidate_selection_candidates.csv", candidates)
    return out


def diagnostic_status(candidates: list[CandidateResult]) -> str:
    if not candidates:
        return "INSUFFICIENT_EVIDENCE"
    best = candidates[0]
    close = [
        c for c in candidates
        if abs(c.fit_stats_mm["rmse"] - best.fit_stats_mm["rmse"]) <= max(0.5, best.fit_stats_mm["rmse"] * 0.05)
    ]
    if len(close) >= 4:
        return "MULTIPLE_SYMMETRIC_SOLUTIONS"
    if len(close) > 1:
        return "HALF_TURN_OR_MIRROR_AMBIGUITY"
    if best.plausibility_warnings:
        return "BEST_FIT_PHYSICALLY_IMPLAUSIBLE"
    return "CLEAR"


def candidate_to_dict(c: CandidateResult) -> dict[str, Any]:
    return {
        "name": c.name,
        "permutation": c.permutation,
        "mapping": c.mapping,
        "R": c.R,
        "t": c.t,
        "det_R": c.det_R,
        "orthogonality_error_fro": c.orthogonality_error_fro,
        "fit_stats_mm": c.fit_stats_mm,
        "residual_norms_mm": c.residual_norms_mm,
        "loo_errors_mm": c.loo_errors_mm,
        "loo_stats_mm": c.loo_stats_mm,
        "delta_rotation_seed_deg": c.delta_rotation_seed_deg,
        "delta_translation_seed_xyz_mm": c.delta_translation_seed_xyz_mm,
        "delta_translation_seed_norm_mm": c.delta_translation_seed_norm_mm,
        "plausibility_warnings": c.plausibility_warnings,
        "projection_frame407": c.projection_frame407,
        "projection_summary": c.projection_summary,
        "cross_scene_summary": c.cross_scene_summary,
        "status": c.status,
    }


def decision_to_dict(d: CandidateDecision) -> dict[str, Any]:
    return {
        "status": d.status,
        "selection_source": d.selection_source,
        "preferred_candidate_name": d.preferred_candidate_name,
        "selected_candidate": d.preferred_candidate_name,
        "confidence": d.confidence,
        "reason": d.reason,
        "candidate_names": d.candidate_names,
        "thresholds": d.thresholds,
    }


def _write_candidate_csv(path: Path, candidates: list[CandidateResult]) -> None:
    fields = [
        "rank", "name", "permutation", "rmse_mm", "loo_mean_mm", "loo_max_mm",
        "delta_rotation_seed_deg", "delta_translation_seed_norm_mm",
        "det_R", "orthogonality_error_fro", "status", "warnings", "t",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, c in enumerate(candidates, 1):
            writer.writerow({
                "rank": rank,
                "name": c.name,
                "permutation": c.permutation,
                "rmse_mm": c.fit_stats_mm["rmse"],
                "loo_mean_mm": c.loo_stats_mm["mean"],
                "loo_max_mm": c.loo_stats_mm["max"],
                "delta_rotation_seed_deg": c.delta_rotation_seed_deg,
                "delta_translation_seed_norm_mm": c.delta_translation_seed_norm_mm,
                "det_R": c.det_R,
                "orthogonality_error_fro": c.orthogonality_error_fro,
                "status": c.status,
                "warnings": ";".join(c.plausibility_warnings),
                "t": c.t,
            })


def _write_pca_svg(path: Path, camera_xy: list, lidar_xy: list) -> None:
    def panel(points, ox, oy, title):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        span_x = max(max(xs) - min(xs), 1e-9)
        span_y = max(max(ys) - min(ys), 1e-9)
        scale = min(260.0 / span_x, 220.0 / span_y)

        def tr(p):
            return ox + 40 + (p[0] - min(xs)) * scale, oy + 270 - (p[1] - min(ys)) * scale

        out = [f'<text x="{ox + 150}" y="{oy + 25}" text-anchor="middle" font-size="18" font-family="Arial">{title}</text>']
        cyc = list(range(len(points))) + [0]
        poly = " ".join(f"{tr(points[i])[0]:.1f},{tr(points[i])[1]:.1f}" for i in cyc)
        out.append(f'<polyline points="{poly}" fill="none" stroke="#777" stroke-width="2" stroke-dasharray="5 4"/>')
        for i, p in enumerate(points):
            x, y = tr(p)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#3b82f6"/>')
            out.append(f'<text x="{x:.1f}" y="{y - 13:.1f}" text-anchor="middle" font-size="16" font-weight="bold" font-family="Arial">{i}</text>')
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="340" viewBox="0 0 720 340">
<rect width="100%" height="100%" fill="white"/>
{panel(camera_xy, 20, 20, "Camera PCA plane")}
{panel(lidar_xy, 380, 20, "LiDAR PCA plane")}
<text x="360" y="325" text-anchor="middle" font-size="12" font-family="Arial" fill="#555">Dashed polygon: ordered_0 -> ordered_1 -> ordered_2 -> ordered_3 -> ordered_0</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")

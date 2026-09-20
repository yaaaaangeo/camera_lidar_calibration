"""Persistence and diagnostics for calibration correspondences.

This module is diagnostic-only.  It records the exact sorted/rolled point pairs
that enter the calibration solve, then computes residual, leave-one-out and
geometry checks from those same pairs.  It never changes the detector, the
production ROI, the extrinsic, or any correction path.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gui.core.project import RESULT_DIR
from gui.core.solve import PreparedScene, prepare_correspondences, solve_rigid


ORDERING_DESCRIPTION = (
    "Detector centers are unlabeled. Both LiDAR and camera centers are ordered by "
    "sort_centers(): fit the board plane, build in-plane axes around the centroid, "
    "sort by bearing, then rotate the cycle so a long rectangle edge starts the "
    "sequence. solve() may apply a per-scene half-turn roll of 0 or 2 using the "
    "upright check. Stored pairs are after that final roll."
)


@dataclass
class SceneMetadata:
    scene_index: str
    camera_timestamp_ns: int | None = None
    lidar_timestamp_ns: int | None = None
    camera_header_stamp_ns: int | None = None
    lidar_header_stamp_ns: int | None = None
    camera_physical_ids: list[str] = field(default_factory=list)
    camera_marker_ids: list[int] = field(default_factory=list)
    camera_board_xy: list[list[float]] = field(default_factory=list)


@dataclass
class PairDiagnostic:
    pair_index: int
    feature_id: str
    correspondence_order: int
    lidar_xyz: list[float]
    camera_xyz: list[float]
    pred_camera_xyz: list[float]
    residual_mm: list[float]
    residual_norm_mm: float


@dataclass
class SceneDiagnostic:
    scene_index: str
    camera_timestamp_ns: int | None
    lidar_timestamp_ns: int | None
    camera_header_stamp_ns: int | None
    lidar_header_stamp_ns: int | None
    pair_count: int
    assignment_shift: int
    ordering: str
    pairs: list[PairDiagnostic]
    fit_stats_mm: dict[str, float]
    loo_errors_mm: list[float]
    loo_stats_mm: dict[str, float]
    geometry: dict[str, Any]
    independent_R: list[list[float]] | None
    independent_t: list[float] | None
    delta_rotation_vs_global_deg: float | None
    delta_translation_vs_global_mm: float | None
    status: str
    camera_physical_ids: list[str] = field(default_factory=list)
    camera_marker_ids: list[int] = field(default_factory=list)
    camera_board_xy: list[list[float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def residual_stats_mm(values_mm: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_mm, np.float64).reshape(-1)
    if len(values) == 0:
        return {"mean": 0.0, "median": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def rotation_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = np.asarray(R_a, np.float64).T @ np.asarray(R_b, np.float64)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def geometry_condition(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return {
            "pairwise_distances_m": [],
            "centroid_spread_m": 0.0,
            "singular_values_m": [],
            "condition_indicator": float("inf"),
            "rank": 0,
            "warnings": ["NO_POINTS"],
        }

    pairwise = [
        float(np.linalg.norm(pts[j] - pts[i]))
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
    ]
    rel = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(rel, full_matrices=False)
    rank = int((s > 1e-6).sum())
    in_plane = s[1] if len(s) > 1 else 0.0
    condition = float(s[0] / max(in_plane, 1e-12)) if len(s) else float("inf")
    spread = float(np.sqrt(np.mean(np.sum(rel**2, axis=1))))

    warnings = []
    if spread < 0.05:
        warnings.append("NARROW_SPREAD")
    if rank < 2:
        warnings.append("NEARLY_COLLINEAR")
    if condition > 10.0:
        warnings.append("POOR_CONDITION")

    return {
        "pairwise_distances_m": pairwise,
        "centroid_spread_m": spread,
        "singular_values_m": [float(x) for x in s],
        "condition_indicator": condition,
        "rank": rank,
        "warnings": warnings,
    }


def leave_one_out_pair_errors(lidar: np.ndarray, camera: np.ndarray) -> tuple[list[float], list[str]]:
    L = np.asarray(lidar, np.float64).reshape(-1, 3)
    C = np.asarray(camera, np.float64).reshape(-1, 3)
    warnings: list[str] = []
    if len(L) < 4:
        return [], ["LOO_REQUIRES_AT_LEAST_4_PAIRS"]

    errors = []
    for holdout in range(len(L)):
        keep = np.ones(len(L), dtype=bool)
        keep[holdout] = False
        geom = geometry_condition(L[keep])
        if geom["rank"] < 2 or geom["condition_indicator"] > 10.0:
            warnings.append(f"LOO_{holdout}_GEOMETRY_WARNING")
        R, t = solve_rigid(L[keep], C[keep])
        pred = R @ L[holdout] + t
        errors.append(float(np.linalg.norm(pred - C[holdout]) * 1000.0))
    return errors, sorted(set(warnings))


def swapped_ordering_warning(lidar: np.ndarray, camera: np.ndarray, R: np.ndarray, t: np.ndarray) -> str | None:
    L = np.asarray(lidar, np.float64).reshape(-1, 3)
    C = np.asarray(camera, np.float64).reshape(-1, 3)
    if len(L) != len(C) or len(L) < 4:
        return None

    current = np.linalg.norm((R @ L.T).T + t - C, axis=1)
    best_other = float("inf")
    best_name = ""
    for shift in range(1, len(L)):
        rolled = np.roll(L, shift, axis=0)
        err = np.linalg.norm((R @ rolled.T).T + t - C, axis=1)
        rmse = float(np.sqrt(np.mean(err**2)))
        if rmse < best_other:
            best_other = rmse
            best_name = f"ROLL_{shift}"
    current_rmse = float(np.sqrt(np.mean(current**2)))
    if best_other + 1e-6 < current_rmse * 0.5:
        return f"POSSIBLE_ORDERING_SWAP_{best_name}"
    return None


def scene_status(scene: SceneDiagnostic) -> str:
    warnings = set(scene.warnings) | set(scene.geometry.get("warnings", []))
    loo_max = scene.loo_stats_mm.get("max", 0.0)
    d_rot = scene.delta_rotation_vs_global_deg or 0.0
    d_t = scene.delta_translation_vs_global_mm or 0.0
    if "POSSIBLE_ORDERING_SWAP" in " ".join(warnings):
        return "CORRESPONDENCE_SUSPECT"
    if warnings or loo_max > 50.0 or d_rot > 5.0 or d_t > 200.0:
        return "SUSPICIOUS"
    return "OK"


def _to_list(a: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(a, np.float64).reshape(-1)]


def analyze_prepared_scene(
    prepared: PreparedScene,
    sol_R: np.ndarray,
    sol_t: np.ndarray,
    metadata: SceneMetadata | None = None,
) -> SceneDiagnostic:
    L = np.asarray(prepared.lidar_sorted, np.float64)
    C = np.asarray(prepared.camera_sorted, np.float64)
    pred = (sol_R @ L.T).T + sol_t
    residual = pred - C
    residual_norm_mm = np.linalg.norm(residual, axis=1) * 1000.0

    pairs = [
        PairDiagnostic(
            pair_index=i,
            feature_id=f"ordered_{i}",
            correspondence_order=i,
            lidar_xyz=_to_list(L[i]),
            camera_xyz=_to_list(C[i]),
            pred_camera_xyz=_to_list(pred[i]),
            residual_mm=_to_list(residual[i] * 1000.0),
            residual_norm_mm=float(residual_norm_mm[i]),
        )
        for i in range(len(L))
    ]

    independent_R = independent_t = None
    d_rot = d_t = None
    if len(L) >= 3:
        independent_R_arr, independent_t_arr = solve_rigid(L, C)
        independent_R = independent_R_arr.tolist()
        independent_t = _to_list(independent_t_arr)
        d_rot = rotation_geodesic_deg(sol_R, independent_R_arr)
        d_t = float(np.linalg.norm(independent_t_arr - sol_t) * 1000.0)

    loo_errors, loo_warnings = leave_one_out_pair_errors(L, C)
    geom = geometry_condition(L)
    warnings = list(geom.get("warnings", [])) + loo_warnings
    swapped = swapped_ordering_warning(L, C, sol_R, sol_t)
    if swapped:
        warnings.append(swapped)

    md = metadata or SceneMetadata(scene_index=prepared.scene_id)
    diag = SceneDiagnostic(
        scene_index=prepared.scene_id,
        camera_timestamp_ns=md.camera_timestamp_ns,
        lidar_timestamp_ns=md.lidar_timestamp_ns,
        camera_header_stamp_ns=md.camera_header_stamp_ns,
        lidar_header_stamp_ns=md.lidar_header_stamp_ns,
        pair_count=len(L),
        assignment_shift=prepared.assignment_shift,
        ordering=ORDERING_DESCRIPTION,
        pairs=pairs,
        fit_stats_mm=residual_stats_mm(residual_norm_mm),
        loo_errors_mm=loo_errors,
        loo_stats_mm=residual_stats_mm(np.asarray(loo_errors, np.float64)),
        geometry=geom,
        independent_R=independent_R,
        independent_t=independent_t,
        delta_rotation_vs_global_deg=d_rot,
        delta_translation_vs_global_mm=d_t,
        camera_physical_ids=list(md.camera_physical_ids),
        camera_marker_ids=list(md.camera_marker_ids),
        camera_board_xy=[list(xy) for xy in md.camera_board_xy],
        status="",
        warnings=sorted(set(warnings)),
    )
    diag.status = scene_status(diag)
    return diag


def analyze_correspondences(
    scenes: list[tuple[str, np.ndarray, np.ndarray]],
    sol_R: np.ndarray,
    sol_t: np.ndarray,
    metadata: dict[str, SceneMetadata] | None = None,
) -> tuple[list[SceneDiagnostic], dict[str, Any]]:
    prepared, _ = prepare_correspondences(scenes)
    by_id = metadata or {}
    scene_diags = [
        analyze_prepared_scene(p, sol_R, sol_t, by_id.get(p.scene_id))
        for p in prepared
    ]
    all_residuals = [
        pair.residual_norm_mm
        for scene in scene_diags
        for pair in scene.pairs
    ]
    d_rots = [
        s.delta_rotation_vs_global_deg for s in scene_diags
        if s.delta_rotation_vs_global_deg is not None
    ]
    d_ts = [
        s.delta_translation_vs_global_mm for s in scene_diags
        if s.delta_translation_vs_global_mm is not None
    ]
    summary = {
        "scene_count": len(scene_diags),
        "pair_count": int(sum(s.pair_count for s in scene_diags)),
        "fit_stats_mm": residual_stats_mm(np.asarray(all_residuals, np.float64)),
        "delta_rotation_vs_global_deg": _distribution(d_rots),
        "delta_translation_vs_global_mm": _distribution(d_ts),
        "status_counts": _counts(s.status for s in scene_diags),
    }
    return scene_diags, summary


def _distribution(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, np.float64)
    if len(a) == 0:
        return {"median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def scene_to_dict(scene: SceneDiagnostic) -> dict[str, Any]:
    return {
        "scene_index": scene.scene_index,
        "camera_timestamp_ns": scene.camera_timestamp_ns,
        "lidar_timestamp_ns": scene.lidar_timestamp_ns,
        "camera_header_stamp_ns": scene.camera_header_stamp_ns,
        "lidar_header_stamp_ns": scene.lidar_header_stamp_ns,
        "pair_count": scene.pair_count,
        "assignment_shift": scene.assignment_shift,
        "ordering": scene.ordering,
        "pairs": [p.__dict__ for p in scene.pairs],
        "fit_stats_mm": scene.fit_stats_mm,
        "loo_errors_mm": scene.loo_errors_mm,
        "loo_stats_mm": scene.loo_stats_mm,
        "geometry": scene.geometry,
        "independent_R": scene.independent_R,
        "independent_t": scene.independent_t,
        "delta_rotation_vs_global_deg": scene.delta_rotation_vs_global_deg,
        "delta_translation_vs_global_mm": scene.delta_translation_vs_global_mm,
        "camera_physical_ids": scene.camera_physical_ids,
        "camera_marker_ids": scene.camera_marker_ids,
        "camera_board_xy": scene.camera_board_xy,
        "physical_id_note": (
            "Camera physical IDs come from ArUco board marker convention and are "
            "stored separately from ordered_i. Current production solve still "
            "uses sorted/rolled ordered_i correspondences."
        ),
        "status": scene.status,
        "warnings": scene.warnings,
    }


def export_correspondence_diagnostics(
    scenes: list[tuple[str, np.ndarray, np.ndarray]],
    sol_R: np.ndarray,
    sol_t: np.ndarray,
    metadata: dict[str, SceneMetadata] | None = None,
    out_dir: str | Path | None = None,
) -> tuple[list[SceneDiagnostic], dict[str, Any], Path]:
    out = Path(out_dir) if out_dir is not None else RESULT_DIR / "correspondences"
    out.mkdir(parents=True, exist_ok=True)

    scene_diags, summary = analyze_correspondences(scenes, sol_R, sol_t, metadata)

    for scene in scene_diags:
        name = _scene_filename(scene.scene_index)
        (out / f"{name}.json").write_text(
            json.dumps(scene_to_dict(scene), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (out / "summary.json").write_text(
        json.dumps({"summary": summary, "scenes": [scene_to_dict(s) for s in scene_diags]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(out / "all_correspondences.csv", scene_diags)
    _write_scene_summary_csv(out / "scene_quality_summary.csv", scene_diags)
    return scene_diags, summary, out


def _scene_filename(scene_id: str) -> str:
    digits = "".join(ch for ch in str(scene_id) if ch.isdigit())
    if digits:
        return f"scene_{int(digits):05d}"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(scene_id))
    return f"scene_{safe}"


def _write_csv(path: Path, scenes: list[SceneDiagnostic]) -> None:
    fields = [
        "scene_index", "camera_timestamp_ns", "lidar_timestamp_ns", "pair_index",
        "feature_id", "correspondence_order",
        "lidar_x", "lidar_y", "lidar_z",
        "camera_x", "camera_y", "camera_z",
        "pred_camera_x", "pred_camera_y", "pred_camera_z",
        "residual_dx_mm", "residual_dy_mm", "residual_dz_mm", "residual_norm_mm",
        "assignment_shift", "status", "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            for pair in scene.pairs:
                writer.writerow({
                    "scene_index": scene.scene_index,
                    "camera_timestamp_ns": scene.camera_timestamp_ns,
                    "lidar_timestamp_ns": scene.lidar_timestamp_ns,
                    "pair_index": pair.pair_index,
                    "feature_id": pair.feature_id,
                    "correspondence_order": pair.correspondence_order,
                    "lidar_x": pair.lidar_xyz[0],
                    "lidar_y": pair.lidar_xyz[1],
                    "lidar_z": pair.lidar_xyz[2],
                    "camera_x": pair.camera_xyz[0],
                    "camera_y": pair.camera_xyz[1],
                    "camera_z": pair.camera_xyz[2],
                    "pred_camera_x": pair.pred_camera_xyz[0],
                    "pred_camera_y": pair.pred_camera_xyz[1],
                    "pred_camera_z": pair.pred_camera_xyz[2],
                    "residual_dx_mm": pair.residual_mm[0],
                    "residual_dy_mm": pair.residual_mm[1],
                    "residual_dz_mm": pair.residual_mm[2],
                    "residual_norm_mm": pair.residual_norm_mm,
                    "assignment_shift": scene.assignment_shift,
                    "status": scene.status,
                    "warnings": ";".join(scene.warnings),
                })


def _write_scene_summary_csv(path: Path, scenes: list[SceneDiagnostic]) -> None:
    fields = [
        "scene", "pairs", "fit_rmse_mm", "loo_mean_mm", "loo_max_mm",
        "delta_rotation_vs_global_deg", "delta_translation_vs_global_mm",
        "condition_indicator", "status", "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            writer.writerow({
                "scene": scene.scene_index,
                "pairs": scene.pair_count,
                "fit_rmse_mm": scene.fit_stats_mm["rmse"],
                "loo_mean_mm": scene.loo_stats_mm["mean"],
                "loo_max_mm": scene.loo_stats_mm["max"],
                "delta_rotation_vs_global_deg": scene.delta_rotation_vs_global_deg,
                "delta_translation_vs_global_mm": scene.delta_translation_vs_global_mm,
                "condition_indicator": scene.geometry["condition_indicator"],
                "status": scene.status,
                "warnings": ";".join(scene.warnings),
            })

"""Temporal-offset / moving-board diagnostics.

Diagnostic-only: this module never writes production extrinsics, project YAML,
time corrections, deskewed clouds, or solver policy.  It asks whether a moving
calibration board can make a low-RMSE 4-point scene fit disagree with
cross-scene/projection evidence.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gui.core.detect_camera import board_model
from gui.core.evaluation.board_motion import BoardPoseSample, analyze_board_motion
from gui.core.evaluation.calibration_correspondence import residual_stats_mm, rotation_geodesic_deg
from gui.core.project import Project, Target
from gui.core.solve import solve_rigid


DELTA_T_SIGN_CONVENTION = (
    "delta_t_ms = t_camera_effective - t_lidar_board. "
    "Positive means the camera board pose is sampled later than the LiDAR board observation."
)

BRANCH_PERMUTATIONS = {
    "NORMAL": (0, 1, 2, 3),
    "HALF_TURN": (2, 3, 0, 1),
}


@dataclass
class CameraPose:
    timestamp_ns: int
    frame_index: int
    R_camera_board: np.ndarray
    t_camera_board: np.ndarray
    reproj_rms_px: float = float("nan")
    reproj_median_px: float = float("nan")
    reproj_p95_px: float = float("nan")
    reproj_max_px: float = float("nan")


@dataclass
class TemporalScene:
    scene_id: str
    frame_index: int | None
    camera_header_ns: int
    lidar_board_ns: int
    original_lidar_timestamp_ns: int
    timestamp_difference_ms: float
    lidar_points_ordered: np.ndarray
    stored_camera_points: np.ndarray
    board_point_count: int = 0
    board_span_ms: float = float("nan")
    camera_reproj_rms_px: float = float("nan")
    translation_velocity_mm_s: float = float("nan")
    rotation_velocity_deg_s: float = float("nan")
    motion_status: str = "UNKNOWN"


@dataclass
class TemporalSweepRow:
    branch: str
    delta_t_ms: float
    valid_scenes: int
    pair_count: int
    global_rmse_mm: float = float("nan")
    median_residual_mm: float = float("nan")
    p95_residual_mm: float = float("nan")
    max_residual_mm: float = float("nan")
    rotation_spread_median_deg: float = float("nan")
    rotation_spread_p90_deg: float = float("nan")
    rotation_spread_p95_deg: float = float("nan")
    rotation_spread_max_deg: float = float("nan")
    translation_spread_median_mm: float = float("nan")
    translation_spread_p90_mm: float = float("nan")
    translation_spread_p95_mm: float = float("nan")
    translation_spread_max_mm: float = float("nan")
    motion_translation_corr_pearson: float = float("nan")
    motion_translation_corr_spearman: float = float("nan")
    motion_rotation_corr_pearson: float = float("nan")
    motion_rotation_corr_spearman: float = float("nan")
    projection_matched_mean_px: float = float("nan")
    projection_p95_px: float = float("nan")
    projection_match_rate: float = float("nan")
    status: str = "OK"
    reason: str = ""


@dataclass
class TemporalMotionDiagnosticResult:
    valid_scenes: list[TemporalScene]
    camera_pose_count: int
    sweep_rows: list[TemporalSweepRow]
    branch_summaries: dict[str, dict[str, Any]]
    current_global_projection_baseline: dict[str, float]
    classification: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_camera_pose_trajectory(
    project: Project,
    frame_rows: list[dict[str, str]],
    max_frames: int = 0,
    include_all_camera_frames: bool = False,
) -> list[CameraPose]:
    """Detect camera board pose at the header-nearest frames listed in a CSV."""
    from gui.core.bag_reader import BagSource
    from gui.core.decode import image_to_bgr

    rows = frame_rows[:max_frames] if max_frames and max_frames > 0 else frame_rows
    out: list[CameraPose] = []
    seen: set[int] = set()
    if not project.bag_paths:
        return out
    with BagSource(project.bag_paths[0]) as src:
        from gui.core.evaluation.timestamp_diagnostic import header_stamp_ns
        from gui.core import detect_camera

        header_index: dict[int, tuple[int, object]] = {}
        times = src.timestamps(project.camera_topic)
        if times:
            for record_ns, msg in src.range(project.camera_topic, times[0], times[-1] + 1):
                stamp = header_stamp_ns(msg)
                if stamp is not None:
                    header_index[int(stamp)] = (int(record_ns), msg)
        if include_all_camera_frames:
            selected = [
                {
                    "camera_header_ns": str(header_ns),
                    "current_camera_index": str(i),
                }
                for i, header_ns in enumerate(sorted(header_index))
            ]
            if max_frames and max_frames > 0:
                selected = selected[:max_frames]
        else:
            selected = rows
        for row in selected:
            header_ns = _int_or_none(row.get("nearest_camera_header_ns") or row.get("camera_header_ns"))
            if header_ns is None or header_ns in seen or header_ns not in header_index:
                continue
            seen.add(header_ns)
            _, msg = header_index[header_ns]
            det = detect_camera.detect(image_to_bgr(msg), project.camera, project.target)
            if not det.ok or det.rvec is None or det.tvec is None:
                continue
            R, _ = cv2.Rodrigues(det.rvec)
            out.append(CameraPose(
                timestamp_ns=header_ns,
                frame_index=_int_or_none(row.get("nearest_camera_index") or row.get("current_camera_index")) or -1,
                R_camera_board=np.asarray(R, np.float64),
                t_camera_board=np.asarray(det.tvec, np.float64).reshape(3),
                reproj_rms_px=float(det.reproj_rms),
                reproj_median_px=float(det.reproj_median),
                reproj_p95_px=float(det.reproj_p95),
                reproj_max_px=float(det.reproj_max),
            ))
    return sorted(out, key=lambda p: p.timestamp_ns)


def run_temporal_motion_diagnostic(
    correspondence_dir: str | Path,
    board_pairing_csv: str | Path,
    target: Target,
    camera_poses: list[CameraPose],
    current_global_projection_json: str | Path | None = None,
    min_valid_scenes: int = 3,
) -> TemporalMotionDiagnosticResult:
    frame_rows = _read_csv_dicts(board_pairing_csv)
    scenes = load_temporal_scenes(correspondence_dir, frame_rows, camera_poses)
    deltas = temporal_sweep_values_ms()
    sweep_rows: list[TemporalSweepRow] = []
    for branch in BRANCH_PERMUTATIONS:
        for dt_ms in deltas:
            sweep_rows.append(evaluate_temporal_offset(branch, dt_ms, scenes, camera_poses, target))

    baseline = load_current_global_projection_baseline(current_global_projection_json)
    summaries = summarize_branches(sweep_rows)
    classification, reason = classify_temporal_result(scenes, summaries, min_valid_scenes)
    return TemporalMotionDiagnosticResult(
        valid_scenes=scenes,
        camera_pose_count=len(camera_poses),
        sweep_rows=sweep_rows,
        branch_summaries=summaries,
        current_global_projection_baseline=baseline,
        classification=classification,
        reason=reason,
        metadata={
            "delta_t_sign_convention": DELTA_T_SIGN_CONVENTION,
            "production_solve_changed": False,
            "yaml_or_extrinsic_written": False,
            "temporal_correction_applied": False,
            "deskew_applied": False,
            "min_valid_scenes": int(min_valid_scenes),
        },
    )


def load_temporal_scenes(
    correspondence_dir: str | Path,
    frame_rows: list[dict[str, str]],
    camera_poses: list[CameraPose],
) -> list[TemporalScene]:
    motion_by_header = _motion_by_header(camera_poses)
    rows_by_lidar = [
        (int(r["lidar_record_ns"]), r) for r in frame_rows
        if r.get("lidar_record_ns") and r.get("board_timestamp_valid", "1") not in {"0", "False", "false"}
    ]
    out = []
    for path in sorted(Path(correspondence_dir).glob("scene_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("pair_count") or 0) < 4:
            continue
        pairs = data.get("pairs") or []
        if len(pairs) < 4:
            continue
        lidar_ts = int(data.get("lidar_timestamp_ns") or data.get("lidar_header_stamp_ns") or 0)
        row = _nearest_lidar_row(lidar_ts, rows_by_lidar)
        board_ns = int(row.get("board_median_ns") or lidar_ts) if row else lidar_ts
        camera_header_ns = int(data.get("camera_header_stamp_ns") or data.get("camera_timestamp_ns") or 0)
        motion = motion_by_header.get(camera_header_ns, {})
        out.append(TemporalScene(
            scene_id=str(data.get("scene_index") or path.stem),
            frame_index=_int_or_none(row.get("frame_index")) if row else None,
            camera_header_ns=camera_header_ns,
            lidar_board_ns=board_ns,
            original_lidar_timestamp_ns=lidar_ts,
            timestamp_difference_ms=(camera_header_ns - board_ns) / 1e6,
            lidar_points_ordered=np.asarray([p["lidar_xyz"] for p in pairs[:4]], np.float64),
            stored_camera_points=np.asarray([p["camera_xyz"] for p in pairs[:4]], np.float64),
            board_point_count=(_int_or_none(row.get("board_point_count")) or 0) if row else 0,
            board_span_ms=_float_or_nan(row.get("board_span_ms")) if row else float("nan"),
            camera_reproj_rms_px=motion.get("reproj_rms_px", float("nan")),
            translation_velocity_mm_s=motion.get("translation_velocity_mm_s", float("nan")),
            rotation_velocity_deg_s=motion.get("rotation_velocity_deg_s", float("nan")),
            motion_status=motion.get("status", "UNKNOWN"),
        ))
    return out


def evaluate_temporal_offset(
    branch: str,
    delta_t_ms: float,
    scenes: list[TemporalScene],
    camera_poses: list[CameraPose],
    target: Target,
) -> TemporalSweepRow:
    model = board_model(target)[1]
    perm = BRANCH_PERMUTATIONS[branch]
    usable = []
    for scene in scenes:
        desired_ns = int(scene.lidar_board_ns + round(delta_t_ms * 1e6))
        pose = interpolate_camera_pose(camera_poses, desired_ns)
        if pose is None:
            continue
        camera_points = (pose[0] @ model.T).T + pose[1]
        lidar_points = scene.lidar_points_ordered[list(perm)]
        usable.append((scene, lidar_points, camera_points))
    if not usable:
        return TemporalSweepRow(branch, delta_t_ms, 0, 0, status="NO_INTERPOLATED_SCENES")

    L = np.vstack([u[1] for u in usable])
    C = np.vstack([u[2] for u in usable])
    R, t = solve_rigid(L, C)
    residuals = np.linalg.norm((R @ L.T).T + t - C, axis=1) * 1000.0
    stats = residual_stats_mm(residuals)

    scene_rots = []
    scene_trans = []
    motion_t = []
    motion_r = []
    for scene, lpts, cpts in usable:
        Rs, ts = solve_rigid(lpts, cpts)
        scene_rots.append(rotation_geodesic_deg(Rs, R))
        scene_trans.append(float(np.linalg.norm(ts - t) * 1000.0))
        motion_t.append(scene.translation_velocity_mm_s)
        motion_r.append(scene.rotation_velocity_deg_s)

    return TemporalSweepRow(
        branch=branch,
        delta_t_ms=float(delta_t_ms),
        valid_scenes=len(usable),
        pair_count=len(usable) * 4,
        global_rmse_mm=stats["rmse"],
        median_residual_mm=stats["median"],
        p95_residual_mm=stats["p95"],
        max_residual_mm=stats["max"],
        rotation_spread_median_deg=_pct(scene_rots, 50),
        rotation_spread_p90_deg=_pct(scene_rots, 90),
        rotation_spread_p95_deg=_pct(scene_rots, 95),
        rotation_spread_max_deg=_pct(scene_rots, 100),
        translation_spread_median_mm=_pct(scene_trans, 50),
        translation_spread_p90_mm=_pct(scene_trans, 90),
        translation_spread_p95_mm=_pct(scene_trans, 95),
        translation_spread_max_mm=_pct(scene_trans, 100),
        motion_translation_corr_pearson=_corr(motion_t, scene_trans, "pearson"),
        motion_translation_corr_spearman=_corr(motion_t, scene_trans, "spearman"),
        motion_rotation_corr_pearson=_corr(motion_r, scene_rots, "pearson"),
        motion_rotation_corr_spearman=_corr(motion_r, scene_rots, "spearman"),
    )


def interpolate_camera_pose(camera_poses: list[CameraPose], timestamp_ns: int, max_gap_ms: float = 250.0):
    if len(camera_poses) < 2:
        return None
    stamps = np.asarray([p.timestamp_ns for p in camera_poses], np.int64)
    i = int(np.searchsorted(stamps, timestamp_ns))
    if i <= 0 or i >= len(camera_poses):
        return None
    a, b = camera_poses[i - 1], camera_poses[i]
    gap_ms = (b.timestamp_ns - a.timestamp_ns) / 1e6
    if gap_ms <= 0 or gap_ms > max_gap_ms:
        return None
    alpha = (timestamp_ns - a.timestamp_ns) / (b.timestamp_ns - a.timestamp_ns)
    t = (1.0 - alpha) * a.t_camera_board + alpha * b.t_camera_board
    R = _slerp(a.R_camera_board, b.R_camera_board, float(alpha))
    return R, t


def temporal_sweep_values_ms() -> list[float]:
    coarse = list(np.arange(-50.0, 50.0 + 1e-9, 5.0))
    fine = list(np.arange(-10.0, 10.0 + 1e-9, 1.0))
    focus = [10.0, 15.0, 20.0]
    return sorted({round(v, 6) for v in coarse + fine + focus})


def summarize_branches(rows: list[TemporalSweepRow]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for branch in sorted({r.branch for r in rows}):
        valid = [r for r in rows if r.branch == branch and np.isfinite(r.global_rmse_mm)]
        best = min(valid, key=lambda r: r.global_rmse_mm) if valid else None
        at0 = min(valid, key=lambda r: abs(r.delta_t_ms)) if valid else None
        out[branch] = {
            "best_delta_t_ms": None if best is None else best.delta_t_ms,
            "rmse_at_0_mm": None if at0 is None else at0.global_rmse_mm,
            "rmse_at_best_mm": None if best is None else best.global_rmse_mm,
            "rotation_spread_at_0_deg": None if at0 is None else at0.rotation_spread_p95_deg,
            "rotation_spread_at_best_deg": None if best is None else best.rotation_spread_p95_deg,
            "translation_spread_at_0_mm": None if at0 is None else at0.translation_spread_p95_mm,
            "translation_spread_at_best_mm": None if best is None else best.translation_spread_p95_mm,
            "valid_scenes_at_best": 0 if best is None else best.valid_scenes,
        }
    return out


def classify_temporal_result(
    scenes: list[TemporalScene],
    branch_summaries: dict[str, dict[str, Any]],
    min_valid_scenes: int,
) -> tuple[str, str]:
    if len(scenes) < min_valid_scenes:
        return (
            "D. INSUFFICIENT_VALID_SCENES",
            f"Only {len(scenes)} valid persisted 4-correspondence scene(s); need at least {min_valid_scenes} "
            "for scene-to-scene consistency, motion/error correlation, and temporal-offset evidence.",
        )
    improved = []
    for name, s in branch_summaries.items():
        a = s.get("rmse_at_0_mm")
        b = s.get("rmse_at_best_mm")
        if a is not None and b is not None and b < a * 0.9:
            improved.append(name)
    if improved:
        return "B. TEMPORAL_OFFSET_WEAK", f"3D residual improves for {', '.join(improved)}, but projection/correlation gates must agree."
    return "C. NO_TEMPORAL_EVIDENCE", "Changing delta_t did not improve the combined evidence."


def load_current_global_projection_baseline(path: str | Path | None) -> dict[str, float]:
    if path is None or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    agg = (data.get("aggregate") or {}).get("CURRENT_GLOBAL") or {}
    return {
        "matched_mean_median_px": agg.get("matched_mean_median_px", float("nan")),
        "matched_p95_median_px": agg.get("matched_p95_median_px", float("nan")),
        "match_rate_median": agg.get("match_rate_median", float("nan")),
        "source": str(path),
    }


def export_temporal_motion_diagnostic(result: TemporalMotionDiagnosticResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": result.metadata,
        "camera_pose_count": result.camera_pose_count,
        "valid_scene_count": len(result.valid_scenes),
        "classification": result.classification,
        "reason": result.reason,
        "current_global_projection_baseline": result.current_global_projection_baseline,
        "branch_summaries": result.branch_summaries,
        "valid_scenes": [_scene_dict(s) for s in result.valid_scenes],
        "sweep_rows": [r.__dict__ for r in result.sweep_rows],
    }
    (out / "temporal_motion_diagnostic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    _write_scenes_csv(out / "valid_scenes.csv", result.valid_scenes)
    _write_sweep_csv(out / "temporal_sweep.csv", result.sweep_rows)
    _write_summary_text(out / "summary.txt", result)
    _write_plots(out, result.sweep_rows)
    return out


def _scene_dict(s: TemporalScene) -> dict[str, Any]:
    d = s.__dict__.copy()
    d.pop("lidar_points_ordered", None)
    d.pop("stored_camera_points", None)
    return d


def _motion_by_header(camera_poses: list[CameraPose]) -> dict[int, dict[str, Any]]:
    samples = [
        BoardPoseSample(
            timestamp_ns=p.timestamp_ns,
            frame_index=p.frame_index,
            R=p.R_camera_board,
            t=p.t_camera_board,
            reproj_rms=p.reproj_rms_px,
            reproj_median=p.reproj_median_px,
            reproj_p95=p.reproj_p95_px,
            reproj_max=p.reproj_max_px,
        )
        for p in camera_poses
    ]
    motion = analyze_board_motion(samples, attempted_count=len(samples))
    return {
        f.timestamp_ns: {
            "translation_velocity_mm_s": f.translation_velocity_mm_s,
            "rotation_velocity_deg_s": f.rotation_velocity_deg_s,
            "status": f.status,
            "reproj_rms_px": f.reproj_rms,
        }
        for f in motion.frames
    }


def _slerp(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    dR = np.asarray(R0, np.float64).T @ np.asarray(R1, np.float64)
    rvec, _ = cv2.Rodrigues(dR)
    R_delta, _ = cv2.Rodrigues(rvec.reshape(3) * alpha)
    return np.asarray(R0, np.float64) @ R_delta


def _nearest_lidar_row(lidar_ns: int, rows_by_lidar: list[tuple[int, dict[str, str]]]) -> dict[str, str] | None:
    if not rows_by_lidar:
        return None
    return min(rows_by_lidar, key=lambda item: abs(item[0] - lidar_ns))[1]


def _read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_scenes_csv(path: Path, rows: list[TemporalScene]) -> None:
    fields = [
        "scene_id", "frame_index", "camera_header_ns", "lidar_board_ns",
        "timestamp_difference_ms", "board_point_count", "board_span_ms",
        "camera_reproj_rms_px", "translation_velocity_mm_s",
        "rotation_velocity_deg_s", "motion_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: getattr(row, k) for k in fields})


def _write_sweep_csv(path: Path, rows: list[TemporalSweepRow]) -> None:
    fields = list(TemporalSweepRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row.__dict__)


def _write_summary_text(path: Path, result: TemporalMotionDiagnosticResult) -> None:
    lines = [
        "Phase 2 Temporal Offset / Moving Board Diagnostic",
        f"Sign convention: {DELTA_T_SIGN_CONVENTION}",
        f"Valid scenes: {len(result.valid_scenes)}",
        f"Camera pose samples: {result.camera_pose_count}",
        "",
    ]
    for branch, summary in result.branch_summaries.items():
        lines += [
            f"{branch}:",
            f"  Best candidate delta_t_ms: {summary.get('best_delta_t_ms')}",
            f"  3D RMSE @ 0: {summary.get('rmse_at_0_mm')}",
            f"  3D RMSE @ best: {summary.get('rmse_at_best_mm')}",
            f"  Rotation spread P95 @ 0/best: {summary.get('rotation_spread_at_0_deg')} / {summary.get('rotation_spread_at_best_deg')}",
            f"  Translation spread P95 @ 0/best: {summary.get('translation_spread_at_0_mm')} / {summary.get('translation_spread_at_best_mm')}",
            "",
        ]
    base = result.current_global_projection_baseline
    lines += [
        "CURRENT_GLOBAL projection baseline:",
        f"  Median matched mean: {base.get('matched_mean_median_px')}",
        f"  Median P95: {base.get('matched_p95_median_px')}",
        "",
        f"Final classification: {result.classification}",
        f"Reason: {result.reason}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(out: Path, rows: list[TemporalSweepRow]) -> None:
    specs = [
        ("global_rmse_mm", "Global 3D RMSE [mm]", "delta_t_vs_global_rmse.png"),
        ("rotation_spread_p95_deg", "Scene rotation spread P95 [deg]", "delta_t_vs_rotation_spread.png"),
        ("translation_spread_p95_mm", "Scene translation spread P95 [mm]", "delta_t_vs_translation_spread.png"),
    ]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        for field, ylabel, name in specs:
            _write_svg_plot(out / name.replace(".png", ".svg"), rows, field, ylabel)
        return
    for field, ylabel, name in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for branch in sorted({r.branch for r in rows}):
            br = [r for r in rows if r.branch == branch]
            ax.plot([r.delta_t_ms for r in br], [getattr(r, field) for r in br], marker="o", label=branch)
        ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
        ax.axvline(15, color="tab:red", linewidth=0.8, alpha=0.5, linestyle="--")
        ax.set_xlabel("delta_t [ms]")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / name, dpi=140)
        plt.close(fig)


def _write_svg_plot(path: Path, rows: list[TemporalSweepRow], field: str, ylabel: str) -> None:
    width, height = 760, 420
    margin = 52
    xs = np.asarray([r.delta_t_ms for r in rows], np.float64)
    ys = np.asarray([getattr(r, field) for r in rows], np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="white"/>'
            f'<text x="24" y="32" font-family="Arial" font-size="16">{ylabel}: no finite data</text>'
            "</svg>\n",
            encoding="utf-8",
        )
        return
    xmin, xmax = float(np.min(xs[ok])), float(np.max(xs[ok]))
    ymin, ymax = float(np.min(ys[ok])), float(np.max(ys[ok]))
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-12:
        pad = max(abs(ymin) * 0.05, 1.0)
        ymin -= pad
        ymax += pad

    def sx(x):
        return margin + (float(x) - xmin) / (xmax - xmin) * (width - 2 * margin)

    def sy(y):
        return height - margin - (float(y) - ymin) / (ymax - ymin) * (height - 2 * margin)

    colors = {"NORMAL": "#2563eb", "HALF_TURN": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="Arial" font-size="16">{ylabel}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#555"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#555"/>',
        f'<text x="{width/2-35}" y="{height-14}" font-family="Arial" font-size="12">delta_t [ms]</text>',
    ]
    for xmark, style in ((0.0, "#111"), (15.0, "#dc2626")):
        if xmin <= xmark <= xmax:
            x = sx(xmark)
            parts.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" y2="{height-margin}" stroke="{style}" stroke-dasharray="4 4" opacity="0.5"/>')
    for branch in sorted({r.branch for r in rows}):
        pts = [
            f"{sx(r.delta_t_ms):.2f},{sy(getattr(r, field)):.2f}"
            for r in rows
            if r.branch == branch and np.isfinite(getattr(r, field))
        ]
        if not pts:
            continue
        color = colors.get(branch, "#444")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for p in pts:
            x, y = p.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{color}"/>')
        y_leg = margin + 18 * (0 if branch == "NORMAL" else 1)
        parts.append(f'<text x="{width-margin-120}" y="{y_leg}" font-family="Arial" font-size="12" fill="{color}">{branch}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _pct(values, q: float) -> float:
    arr = np.asarray(values, np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def _corr(a, b, kind: str) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    if kind == "spearman":
        x = np.argsort(np.argsort(x)).astype(float)
        y = np.argsort(np.argsort(y)).astype(float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _int_or_none(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

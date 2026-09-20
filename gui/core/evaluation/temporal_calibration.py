"""Temporal calibration diagnostics with fixed spatial extrinsic.

This module estimates and validates only the Camera-LiDAR time offset.  It does
not modify or optimize the spatial extrinsic, detector logic, ROI logic, YAML,
bag timestamps, point timestamps, or deskewing.

Timestamp convention is fixed everywhere:

    delta_t_ms = t_camera - t_lidar_board

Positive means the camera observation occurs later than the LiDAR board
observation.  Camera time is message header time; LiDAR board time is the median
timestamp of board ROI points.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from gui.core.bag_reader import BagSource
from gui.core.decode import cloud_to_struct
from gui.core.detect_camera import board_model
from gui.core.detect_lidar import apply_box, detect as detect_lidar, point_spacing
from gui.core.evaluation.board_motion import BoardPoseSample, analyze_board_motion
from gui.core.evaluation.calibration_correspondence import residual_stats_mm
from gui.core.evaluation.temporal_motion_diagnostic import CameraPose, build_camera_pose_trajectory, interpolate_camera_pose
from gui.core.project import Project, Target
from gui.core.solve import solve_rigid


DELTA_T_CONVENTION = (
    "delta_t_ms = t_camera - t_lidar_board. Positive means camera observation "
    "occurs later than the LiDAR board observation. t_lidar_board is the "
    "median timestamp of board ROI points."
)


@dataclass
class TemporalCalibrationConfig:
    coarse_min_ms: float = -50.0
    coarse_max_ms: float = 50.0
    coarse_step_ms: float = 5.0
    fine_radius_ms: float = 10.0
    fine_step_ms: float = 1.0
    min_ready_scenes: int = 10
    min_marginal_scenes: int = 5
    max_interpolation_gap_ms: float = 250.0
    low_motion_translation_mm_s: float = 250.0
    high_motion_translation_mm_s: float = 750.0
    low_motion_rotation_deg_s: float = 2.0
    high_motion_rotation_deg_s: float = 6.0


@dataclass
class TemporalCalibrationScene:
    scene_id: str
    frame_index: int
    lidar_board_ns: int
    camera_header_ns: int
    lidar_points_xyz: np.ndarray
    camera_points_xyz: np.ndarray | None = None
    board_point_count: int = 0
    board_min_ns: int | None = None
    board_mean_ns: int | None = None
    board_max_ns: int | None = None
    board_span_ms: float = float("nan")
    camera_reproj_rms_px: float = float("nan")
    translation_velocity_mm_s: float = float("nan")
    rotation_velocity_deg_s: float = float("nan")
    motion_group: str = "UNKNOWN"


@dataclass
class TemporalSweepMetric:
    delta_t_ms: float
    valid_scene_count: int
    point_count: int
    mean_mm: float = float("nan")
    median_mm: float = float("nan")
    rmse_mm: float = float("nan")
    p95_mm: float = float("nan")
    max_mm: float = float("nan")
    low_motion_rmse_mm: float = float("nan")
    mid_motion_rmse_mm: float = float("nan")
    high_motion_rmse_mm: float = float("nan")


@dataclass
class SceneTemporalResidual:
    scene_id: str
    frame_index: int
    delta_t_ms: float
    motion_group: str
    mean_mm: float
    median_mm: float
    rmse_mm: float
    p95_mm: float
    max_mm: float
    translation_velocity_mm_s: float = float("nan")
    rotation_velocity_deg_s: float = float("nan")


@dataclass
class TemporalCalibrationResult:
    scenes: list[TemporalCalibrationScene]
    camera_pose_count: int
    sweep: list[TemporalSweepMetric]
    scene_residuals: list[SceneTemporalResidual]
    best_delta_t_ms: float | None
    classification: str
    reason: str
    projection_baseline: dict[str, float] = field(default_factory=dict)
    projection_validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_fixed_extrinsic(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "rotation_matrix" in data:
        return np.asarray(data["rotation_matrix"], np.float64), np.asarray(data["translation_m"], np.float64)
    T = np.asarray(data["T_cam_lidar"]["data"], np.float64).reshape(4, 4)
    return T[:3, :3], T[:3, 3]


def load_temporal_scenes_from_harvest(harvest_dir: str | Path) -> list[TemporalCalibrationScene]:
    scenes: list[TemporalCalibrationScene] = []
    for path in sorted(Path(harvest_dir).glob("scene_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        lidar = data.get("lidar_centers_xyz")
        if not lidar or len(lidar) < 4:
            continue
        scenes.append(TemporalCalibrationScene(
            scene_id=path.stem,
            frame_index=int(data.get("frame_index", -1)),
            lidar_board_ns=int(data.get("board_median_ns")),
            camera_header_ns=int(data.get("camera_header_ns", 0)),
            lidar_points_xyz=np.asarray(lidar[:4], np.float64),
            camera_points_xyz=np.asarray((data.get("camera_centers_xyz") or [])[:4], np.float64)
            if data.get("camera_centers_xyz") else None,
            board_point_count=int(data.get("board_point_count", data.get("lidar_plane_point_count", 0)) or 0),
            board_span_ms=float(data.get("board_span_ms", float("nan"))),
            camera_reproj_rms_px=float(data.get("camera_reproj_rms_px", float("nan"))),
            translation_velocity_mm_s=float(data.get("motion_translation_velocity_mm_s", float("nan"))),
            rotation_velocity_deg_s=float(data.get("motion_rotation_velocity_deg_s", float("nan"))),
            motion_group=str(data.get("motion_group", "UNKNOWN")),
        ))
    return scenes


def load_temporal_scenes_from_step6(
    project: Project,
    correspondence_dir: str | Path,
    scene_ids: list[str] | None = None,
    method: str = "original",
) -> list[TemporalCalibrationScene]:
    """Load current Step 6 Manual-ROI 4-pair scenes as temporal inputs.

    The persisted correspondences provide the exact ordered LiDAR/camera XYZ
    pairs used by Step 6.  The LiDAR board time is recomputed from the same
    project Manual ROI/filter and existing LiDAR detector plane points.  This is
    diagnostic-only and never changes ROI, detector, solver, timestamps, YAML or
    extrinsic files.
    """
    enabled = {s.id for s in project.scenes if s.enabled}
    if scene_ids is not None:
        enabled &= set(scene_ids)
    by_id = {s.id: s for s in project.scenes}
    out: list[TemporalCalibrationScene] = []
    sources: dict[str, BagSource] = {}
    try:
        for path in sorted(Path(correspondence_dir).glob("scene_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            sid = str(data.get("scene_index") or path.stem)
            if sid not in enabled:
                continue
            pairs = data.get("pairs") or []
            if len(pairs) != 4 or int(data.get("pair_count") or 0) != 4:
                continue
            scene = by_id.get(sid)
            if scene is None:
                continue
            bag = project.bag_for(scene)
            src = sources.get(bag)
            if src is None:
                src = BagSource(bag)
                src.open()
                sources[bag] = src
            board = compute_manual_roi_board_timestamp(project, scene, method=method, src=src)
            out.append(TemporalCalibrationScene(
                scene_id=sid,
                frame_index=_scene_numeric_index(sid),
                lidar_board_ns=board["board_median_ns"],
                camera_header_ns=int(data.get("camera_header_stamp_ns") or data.get("camera_timestamp_ns") or 0),
                lidar_points_xyz=np.asarray([p["lidar_xyz"] for p in pairs], np.float64),
                camera_points_xyz=np.asarray([p["camera_xyz"] for p in pairs], np.float64),
                board_point_count=int(board["board_point_count"]),
                board_min_ns=int(board["board_min_ns"]),
                board_mean_ns=int(board["board_mean_ns"]),
                board_max_ns=int(board["board_max_ns"]),
                board_span_ms=float(board["board_span_ms"]),
                camera_reproj_rms_px=float("nan"),
            ))
    finally:
        for src in sources.values():
            src.close()
    return out


def load_step6_reference_extrinsic(correspondence_dir: str | Path, scene_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the current Step 6 reference transform from persisted pairs."""
    wanted = set(scene_ids or [])
    lidar, camera = [], []
    for path in sorted(Path(correspondence_dir).glob("scene_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = str(data.get("scene_index") or path.stem)
        if wanted and sid not in wanted:
            continue
        pairs = data.get("pairs") or []
        if len(pairs) != 4 or int(data.get("pair_count") or 0) != 4:
            continue
        lidar.extend([p["lidar_xyz"] for p in pairs])
        camera.extend([p["camera_xyz"] for p in pairs])
    if len(lidar) < 3:
        raise ValueError("not enough Step 6 correspondences to recreate reference extrinsic")
    return solve_rigid(np.asarray(lidar, np.float64), np.asarray(camera, np.float64))


def compute_manual_roi_board_timestamp(project: Project, scene, method: str = "original", src: BagSource | None = None) -> dict[str, Any]:
    if src is None:
        bag = project.bag_for(scene)
        with BagSource(bag) as owned:
            cloud, point_ts, single, ring = _cloud_with_timestamps(owned, project.lidar_topic, scene.t_ns, scene.frames)
    else:
        cloud, point_ts, single, ring = _cloud_with_timestamps(src, project.lidar_topic, scene.t_ns, scene.frames)
    if point_ts is None or len(point_ts) != len(cloud):
        raise ValueError(f"{scene.id}: LiDAR point timestamp field missing or misaligned")
    inside_single = apply_box(single, scene.filter)
    params = _derive_params(point_spacing(inside_single) if len(inside_single) > 20 else 0.0)
    params.method = method
    det = detect_lidar(cloud, scene.filter, project.target, params, ring=ring)
    if det.plane_indices is None or len(det.plane_indices) == 0:
        raise ValueError(f"{scene.id}: LiDAR board plane points unavailable for timestamp diagnostic")
    board_ts = np.asarray(point_ts, np.float64)[np.asarray(det.plane_indices, np.int64)]
    board_ts = board_ts[np.isfinite(board_ts)]
    if len(board_ts) == 0:
        raise ValueError(f"{scene.id}: LiDAR board point timestamps are invalid")
    return {
        "board_median_ns": int(np.median(board_ts) * 1e9),
        "board_min_ns": int(np.min(board_ts) * 1e9),
        "board_mean_ns": int(np.mean(board_ts) * 1e9),
        "board_max_ns": int(np.max(board_ts) * 1e9),
        "board_span_ms": float((np.max(board_ts) - np.min(board_ts)) * 1000.0),
        "board_point_count": int(len(board_ts)),
    }


def annotate_scene_motion(scenes: list[TemporalCalibrationScene], camera_poses: list[CameraPose]) -> None:
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
    by_header = {
        f.timestamp_ns: f
        for f in motion.frames
    }
    for scene in scenes:
        frame = by_header.get(scene.camera_header_ns)
        if frame is None:
            continue
        scene.translation_velocity_mm_s = frame.translation_velocity_mm_s
        scene.rotation_velocity_deg_s = frame.rotation_velocity_deg_s
        scene.motion_group = motion_group_for_scene(scene, TemporalCalibrationConfig())


def estimate_temporal_offset(
    scenes: list[TemporalCalibrationScene],
    camera_poses: list[CameraPose],
    fixed_R_lidar_to_camera: np.ndarray,
    fixed_t_lidar_to_camera: np.ndarray,
    target: Target,
    config: TemporalCalibrationConfig = TemporalCalibrationConfig(),
    projection_baseline: dict[str, float] | None = None,
) -> TemporalCalibrationResult:
    rows: list[TemporalSweepMetric] = []
    residual_rows: list[SceneTemporalResidual] = []
    coarse_deltas = _coarse_sweep_values(config)
    for dt in coarse_deltas:
        metric, scene_metrics = evaluate_delta_t(
            scenes, camera_poses, fixed_R_lidar_to_camera, fixed_t_lidar_to_camera, target, dt, config
        )
        rows.append(metric)
        residual_rows.extend(scene_metrics)

    valid_rows = [r for r in rows if np.isfinite(r.rmse_mm)]
    coarse_best = min(valid_rows, key=lambda r: r.rmse_mm) if valid_rows else None
    if coarse_best is not None:
        existing = {r.delta_t_ms for r in rows}
        for dt in _fine_sweep_values(config, coarse_best.delta_t_ms):
            if dt in existing:
                continue
            metric, scene_metrics = evaluate_delta_t(
                scenes, camera_poses, fixed_R_lidar_to_camera, fixed_t_lidar_to_camera, target, dt, config
            )
            rows.append(metric)
            residual_rows.extend(scene_metrics)
        rows.sort(key=lambda r: r.delta_t_ms)
    valid_rows = [r for r in rows if np.isfinite(r.rmse_mm)]
    best = min(valid_rows, key=lambda r: r.rmse_mm) if valid_rows else None
    classification, reason = classify_temporal_calibration(scenes, rows, config)
    if classification == "INSUFFICIENT_VALID_SCENES":
        best_dt = None
    else:
        best_dt = None if best is None else best.delta_t_ms
    return TemporalCalibrationResult(
        scenes=scenes,
        camera_pose_count=len(camera_poses),
        sweep=rows,
        scene_residuals=residual_rows,
        best_delta_t_ms=best_dt,
        classification=classification,
        reason=reason,
        projection_baseline=projection_baseline or {},
        metadata={
            "delta_t_convention": DELTA_T_CONVENTION,
            "camera_time": "message header timestamp",
            "lidar_time": "median timestamp of board ROI points",
            "record_timestamp_used_for_pairing": False,
            "spatial_extrinsic_optimized": False,
            "spatial_extrinsic_fixed": True,
            "yaml_or_extrinsic_written": False,
            "timestamp_rewriting_applied": False,
            "deskew_applied": False,
            "max_interpolation_gap_ms": config.max_interpolation_gap_ms,
            "confidence": "MARGINAL" if config.min_marginal_scenes <= len(scenes) < config.min_ready_scenes else (
                "READY" if len(scenes) >= config.min_ready_scenes else "INSUFFICIENT"
            ),
            "coarse_best_delta_t_ms": None if coarse_best is None else coarse_best.delta_t_ms,
        },
    )


def evaluate_delta_t(
    scenes: list[TemporalCalibrationScene],
    camera_poses: list[CameraPose],
    R: np.ndarray,
    t: np.ndarray,
    target: Target,
    delta_t_ms: float,
    config: TemporalCalibrationConfig = TemporalCalibrationConfig(),
) -> tuple[TemporalSweepMetric, list[SceneTemporalResidual]]:
    model = board_model(target)[1]
    all_residuals = []
    per_scene = []
    by_group: dict[str, list[float]] = {"LOW": [], "MID": [], "HIGH": []}
    for scene in scenes:
        query_ns = int(scene.lidar_board_ns + round(delta_t_ms * 1e6))
        pose = interpolate_camera_pose(camera_poses, query_ns, max_gap_ms=config.max_interpolation_gap_ms)
        if pose is None:
            continue
        local_points = _scene_board_local_points(scene, camera_poses, model, config)
        if local_points is None:
            continue
        expected_camera = (pose[0] @ local_points.T).T + pose[1]
        observed_camera = (np.asarray(R) @ scene.lidar_points_xyz.T).T + np.asarray(t).reshape(3)
        residuals = np.linalg.norm(observed_camera - expected_camera, axis=1) * 1000.0
        stats = residual_stats_mm(residuals)
        all_residuals.extend(residuals.tolist())
        group = scene.motion_group if scene.motion_group in by_group else motion_group_for_scene(
            scene, config
        )
        if group in by_group:
            by_group[group].extend(residuals.tolist())
        per_scene.append(SceneTemporalResidual(
            scene_id=scene.scene_id,
            frame_index=scene.frame_index,
            delta_t_ms=float(delta_t_ms),
            motion_group=group,
            mean_mm=stats["mean"],
            median_mm=stats["median"],
            rmse_mm=stats["rmse"],
            p95_mm=stats["p95"],
            max_mm=stats["max"],
            translation_velocity_mm_s=scene.translation_velocity_mm_s,
            rotation_velocity_deg_s=scene.rotation_velocity_deg_s,
        ))
    stats = residual_stats_mm(np.asarray(all_residuals, np.float64))
    return TemporalSweepMetric(
        delta_t_ms=float(delta_t_ms),
        valid_scene_count=len(per_scene),
        point_count=len(all_residuals),
        mean_mm=stats["mean"],
        median_mm=stats["median"],
        rmse_mm=stats["rmse"],
        p95_mm=stats["p95"],
        max_mm=stats["max"],
        low_motion_rmse_mm=_rmse(by_group["LOW"]),
        mid_motion_rmse_mm=_rmse(by_group["MID"]),
        high_motion_rmse_mm=_rmse(by_group["HIGH"]),
    ), per_scene


def classify_temporal_calibration(
    scenes: list[TemporalCalibrationScene],
    sweep: list[TemporalSweepMetric],
    config: TemporalCalibrationConfig = TemporalCalibrationConfig(),
) -> tuple[str, str]:
    n = len(scenes)
    if n < config.min_marginal_scenes:
        return "INSUFFICIENT_VALID_SCENES", (
            f"{n} valid temporal scene(s); need at least {config.min_marginal_scenes}, "
            f"preferably {config.min_ready_scenes}+."
        )
    finite = [r for r in sweep if np.isfinite(r.rmse_mm)]
    if not finite:
        return "INSUFFICIENT_VALID_SCENES", "No delta_t candidate has enough camera-pose interpolation support."
    at0 = min(finite, key=lambda r: abs(r.delta_t_ms))
    best = min(finite, key=lambda r: r.rmse_mm)
    rel_improve = (at0.rmse_mm - best.rmse_mm) / max(at0.rmse_mm, 1e-9)
    if rel_improve > 0.05 and (
        np.isclose(best.delta_t_ms, config.coarse_min_ms) or np.isclose(best.delta_t_ms, config.coarse_max_ms)
    ):
        return "MARGINAL_BOUNDARY_LIMITED_TREND", (
            f"Best residual is at sweep boundary {best.delta_t_ms:.1f} ms. This supports only a directional "
            "trend inside the tested range, not a production-ready temporal offset."
        )
    high_improve = np.isfinite(at0.high_motion_rmse_mm) and np.isfinite(best.high_motion_rmse_mm) and (
        best.high_motion_rmse_mm < at0.high_motion_rmse_mm * 0.9
    )
    if n >= config.min_ready_scenes and rel_improve > 0.10 and high_improve:
        return "TEMPORAL_OFFSET_SUPPORTED", "Multi-scene and high-motion residuals support the same temporal offset region."
    if n < config.min_ready_scenes and rel_improve > 0.10:
        return "MARGINAL_TEMPORAL_OFFSET_SUPPORTED", (
            f"{n} valid temporal scenes support a residual minimum, but confidence remains MARGINAL until "
            f"{config.min_ready_scenes}+ scenes are available."
        )
    if rel_improve > 0.05:
        return "MARGINAL_TEMPORAL_OFFSET_WEAK" if n < config.min_ready_scenes else "TEMPORAL_OFFSET_WEAK", (
            "3D residual improves, but scene count or supporting evidence is limited."
        )
    return "NO_TEMPORAL_EVIDENCE", "Sweeping delta_t does not materially improve temporal residuals."


def export_temporal_calibration_result(
    result: TemporalCalibrationResult,
    out_dir: str | Path,
    camera_poses: list[CameraPose] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": result.metadata,
        "temporal_scenes": len(result.scenes),
        "camera_pose_count": result.camera_pose_count,
        "timestamp_definition": {
            "camera": "message header timestamp",
            "lidar": "board ROI point median timestamp",
            "delta_t": DELTA_T_CONVENTION,
        },
        "best_delta_t_ms": result.best_delta_t_ms,
        "classification": result.classification,
        "reason": result.reason,
        "projection_baseline": result.projection_baseline,
        "projection_validation": result.projection_validation,
        "sweep": [r.__dict__ for r in result.sweep],
    }
    (out / "temporal_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    _write_csv(out / "temporal_sweep.csv", TemporalSweepMetric, result.sweep)
    _write_csv(out / "scene_temporal_residuals.csv", SceneTemporalResidual, result.scene_residuals)
    _write_board_timestamp_csv(out / "board_timestamp_diagnostic.csv", result.scenes)
    if camera_poses is not None:
        _write_camera_pose_csv(out / "camera_pose_trajectory.csv", camera_poses)
    _write_svg_plot(out / "delta_t_vs_rmse.svg", result.sweep, "rmse_mm", "3D RMSE [mm]")
    _write_svg_plot(out / "delta_t_vs_p95.svg", result.sweep, "p95_mm", "3D P95 [mm]")
    _write_svg_plot(out / "delta_t_vs_motion_residual.svg", result.sweep, "high_motion_rmse_mm", "High-motion RMSE [mm]")
    return out


def load_projection_baseline(path: str | Path | None) -> dict[str, float]:
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


def build_camera_trajectory_for_project(project: Project, max_frames: int = 0) -> list[CameraPose]:
    return build_camera_pose_trajectory(project, [], max_frames=max_frames, include_all_camera_frames=True)


def motion_group_for_scene(scene: TemporalCalibrationScene, config: TemporalCalibrationConfig) -> str:
    tv = 0.0 if not np.isfinite(scene.translation_velocity_mm_s) else scene.translation_velocity_mm_s
    rv = 0.0 if not np.isfinite(scene.rotation_velocity_deg_s) else scene.rotation_velocity_deg_s
    if tv < config.low_motion_translation_mm_s and rv < config.low_motion_rotation_deg_s:
        return "LOW"
    if tv >= config.high_motion_translation_mm_s or rv >= config.high_motion_rotation_deg_s:
        return "HIGH"
    return "MID"


def _coarse_sweep_values(config: TemporalCalibrationConfig) -> list[float]:
    return [
        round(float(v), 6)
        for v in np.arange(config.coarse_min_ms, config.coarse_max_ms + 1e-9, config.coarse_step_ms)
    ]


def _fine_sweep_values(config: TemporalCalibrationConfig, center_ms: float) -> list[float]:
    lo = max(config.coarse_min_ms, center_ms - config.fine_radius_ms)
    hi = min(config.coarse_max_ms, center_ms + config.fine_radius_ms)
    return [
        round(float(v), 6)
        for v in np.arange(lo, hi + 1e-9, config.fine_step_ms)
    ]


def _scene_board_local_points(
    scene: TemporalCalibrationScene,
    camera_poses: list[CameraPose],
    default_model: np.ndarray,
    config: TemporalCalibrationConfig,
) -> np.ndarray | None:
    if scene.camera_points_xyz is None:
        return default_model
    pose0 = _camera_pose_at(camera_poses, scene.camera_header_ns, config)
    if pose0 is None:
        return None
    R0, t0 = pose0
    return (R0.T @ (np.asarray(scene.camera_points_xyz, np.float64) - t0).T).T


def _camera_pose_at(camera_poses: list[CameraPose], timestamp_ns: int, config: TemporalCalibrationConfig):
    for pose in camera_poses:
        if pose.timestamp_ns == timestamp_ns:
            return pose.R_camera_board, pose.t_camera_board
    return interpolate_camera_pose(camera_poses, timestamp_ns, max_gap_ms=config.max_interpolation_gap_ms)


def _scene_numeric_index(scene_id: str) -> int:
    digits = "".join(ch for ch in str(scene_id) if ch.isdigit())
    return int(digits) if digits else -1


def _cloud_with_timestamps(src: BagSource, topic: str, t_ns: int, frames: int):
    stamps = sorted(src.timestamps(topic))
    if not stamps:
        return np.empty((0, 3), np.float32), None, np.empty((0, 3), np.float32), None
    want = max(int(frames), 1)
    centre = min(range(len(stamps)), key=lambda i: abs(stamps[i] - t_ns))
    lo = max(0, centre - want // 2)
    hi = min(len(stamps), lo + want)
    lo = max(0, hi - want)
    xyz_chunks, ts_chunks, ring_chunks = [], [], []
    selected = stamps[lo:hi]
    for sweep_i, (_, msg) in enumerate(src.range(topic, selected[0], selected[-1] + 1)):
        rec = cloud_to_struct(msg)
        xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        xyz_chunks.append(xyz[finite])
        if "timestamp" in (rec.dtype.names or ()):
            ts_chunks.append(np.asarray(rec["timestamp"], np.float64)[finite])
        else:
            ts_chunks.append(np.full(int(finite.sum()), np.nan, np.float64))
        if "ring" in (rec.dtype.names or ()):
            ring_chunks.append(np.asarray(rec["ring"], np.int32)[finite])
        else:
            ring_chunks.append(None)
    if not xyz_chunks:
        return np.empty((0, 3), np.float32), None, np.empty((0, 3), np.float32), None
    xyz_all = np.vstack(xyz_chunks)
    ts_all = np.concatenate(ts_chunks)
    mid = len(xyz_chunks) // 2
    ring = None
    if all(r is not None for r in ring_chunks):
        span = max(int(r.max()) + 1 for r in ring_chunks if len(r))
        ring = np.concatenate([r + i * span for i, r in enumerate(ring_chunks)])
    return xyz_all, ts_all, xyz_chunks[mid], ring


def _derive_params(spacing: float):
    p = DetectParamsShim()
    if spacing > 0:
        p.sweep_spacing = spacing
        p.boundary_radius = max(p.boundary_radius, spacing * 4)
        p.cluster_tolerance = max(p.cluster_tolerance, spacing * 2.5)
    return p


class DetectParamsShim:
    """Small local copy of Step 5's derived parameters for diagnostic reruns."""

    def __init__(self):
        from gui.core.detect_lidar import DetectParams

        base = DetectParams()
        self.__dict__.update(base.__dict__)


def _rmse(values: list[float]) -> float:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")


def _write_csv(path: Path, cls, rows) -> None:
    fields = list(cls.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row.__dict__)


def _write_board_timestamp_csv(path: Path, scenes: list[TemporalCalibrationScene]) -> None:
    fields = [
        "scene_id", "frame_index", "camera_header_ns", "lidar_board_ns", "delta_t_observed_ms",
        "board_min_ns", "board_mean_ns", "board_max_ns", "board_span_ms", "board_point_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scenes:
            w.writerow({
                "scene_id": s.scene_id,
                "frame_index": s.frame_index,
                "camera_header_ns": s.camera_header_ns,
                "lidar_board_ns": s.lidar_board_ns,
                "delta_t_observed_ms": (s.camera_header_ns - s.lidar_board_ns) / 1e6,
                "board_min_ns": s.board_min_ns or "",
                "board_mean_ns": s.board_mean_ns or "",
                "board_max_ns": s.board_max_ns or "",
                "board_span_ms": s.board_span_ms,
                "board_point_count": s.board_point_count,
            })


def _write_camera_pose_csv(path: Path, poses: list[CameraPose]) -> None:
    fields = ["frame_index", "header_timestamp_ns", "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "tx", "ty", "tz", "reproj_rms_px"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in poses:
            R = np.asarray(p.R_camera_board)
            row = {
                "frame_index": p.frame_index,
                "header_timestamp_ns": p.timestamp_ns,
                "tx": p.t_camera_board[0],
                "ty": p.t_camera_board[1],
                "tz": p.t_camera_board[2],
                "reproj_rms_px": p.reproj_rms_px,
            }
            for i in range(3):
                for j in range(3):
                    row[f"r{i}{j}"] = R[i, j]
            w.writerow(row)


def _write_svg_plot(path: Path, rows: list[TemporalSweepMetric], field: str, ylabel: str) -> None:
    width, height, margin = 760, 420, 52
    xs = np.asarray([r.delta_t_ms for r in rows], np.float64)
    ys = np.asarray([getattr(r, field) for r in rows], np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="24" y="32" font-family="Arial" font-size="16">{ylabel}: no finite data</text></svg>\n', encoding="utf-8")
        return
    xmin, xmax = float(np.min(xs[ok])), float(np.max(xs[ok]))
    ymin, ymax = float(np.min(ys[ok])), float(np.max(ys[ok]))
    if abs(ymax - ymin) < 1e-12:
        pad = max(abs(ymin) * 0.05, 1.0)
        ymin -= pad
        ymax += pad

    def sx(x):
        return margin + (x - xmin) / max(xmax - xmin, 1e-9) * (width - 2 * margin)

    def sy(y):
        return height - margin - (y - ymin) / max(ymax - ymin, 1e-9) * (height - 2 * margin)

    pts = [f"{sx(x):.2f},{sy(y):.2f}" for x, y, good in zip(xs, ys, ok) if good]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="28" font-family="Arial" font-size="16">{ylabel}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#555"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#555"/>',
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="2"/>',
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")

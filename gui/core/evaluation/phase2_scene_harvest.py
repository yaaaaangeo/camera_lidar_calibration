"""Phase 2A valid-scene harvesting diagnostic.

This is diagnostic-only.  It scans a bag for camera/LiDAR frames that can form
real 4-correspondence calibration scenes, persists the evidence, and reports the
detection funnel.  It does not change production calibration, YAML, temporal
correction, deskew, or solver selection.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import yaml

from gui.core import roi_auto
from gui.core.bag_reader import BagSource
from gui.core.decode import cloud_to_struct, image_to_bgr
from gui.core.detect_camera import CIRCLE_PHYSICAL_IDS, board_circle_metadata, detect as detect_camera
from gui.core.detect_lidar import DetectParams, apply_box, detect as detect_lidar, point_spacing
from gui.core.evaluation.board_motion import BoardPoseSample, analyze_board_motion
from gui.core.evaluation.calibration_correspondence import residual_stats_mm, rotation_geodesic_deg
from gui.core.evaluation.timestamp_diagnostic import header_stamp_ns
from gui.core.project import Camera, Project, Target
from gui.core.solve import solve_rigid


FUNNEL_STAGES = [
    "TOTAL_LIDAR_FRAMES",
    "CAMERA_PAIR_FOUND",
    "CAMERA_DETECTION_VALID",
    "CAMERA_PNP_VALID",
    "CAMERA_PNP_QUALITY_PASS",
    "LIDAR_ROI_VALID",
    "LIDAR_PLANE_VALID",
    "LIDAR_BOUNDARY_VALID",
    "LIDAR_CIRCLE_GE_1",
    "LIDAR_CIRCLE_GE_2",
    "LIDAR_CIRCLE_GE_3",
    "LIDAR_CIRCLE_GE_4",
    "LIDAR_GEOMETRY_PASS",
    "LIDAR_FULL_TARGET_VALID",
    "TIMESTAMP_VALID",
    "FOUR_CORRESPONDENCES_CREATED",
    "FINAL_VALID_SCENES",
]

SYMMETRY_BRANCHES = {
    "NORMAL": (0, 1, 2, 3),
    "HALF_TURN": (2, 3, 0, 1),
}


@dataclass
class HarvestConfig:
    pnp_reproj_rms_max_px: float = 2.0
    max_camera_pair_dt_ms: float = 150.0
    duplicate_min_time_separation_s: float = 0.5
    duplicate_min_translation_separation_m: float = 0.10
    duplicate_min_rotation_separation_deg: float = 5.0
    roi_margins_m: tuple[float, ...] = (roi_auto.MARGIN_M, 1.5, 3.0, 5.0, 8.0)
    lidar_method: str = "original"


@dataclass
class CameraDetectionSample:
    header_ns: int
    record_ns: int
    index: int
    ok: bool
    det: Any = None
    reason: str = ""


@dataclass
class HarvestScene:
    frame_index: int
    camera_index: int
    lidar_index: int
    camera_header_ns: int
    lidar_record_ns: int
    sweep_start_ns: int
    sweep_mid_ns: int
    sweep_end_ns: int
    board_median_ns: int
    board_span_ms: float
    camera_minus_board_ms: float
    camera_reproj_rms_px: float
    camera_reproj_p95_px: float
    lidar_centers_xyz: np.ndarray
    camera_centers_xyz: np.ndarray
    lidar_plane_point_count: int
    lidar_boundary_count: int
    lidar_circle_count: int
    lidar_cluster_count: int
    lidar_plane_rms: float
    motion_translation_velocity_mm_s: float = float("nan")
    motion_rotation_velocity_deg_s: float = float("nan")
    motion_group: str = "UNKNOWN"
    pose_distance_m: float = float("nan")
    pose_x_m: float = float("nan")
    pose_y_m: float = float("nan")
    pose_yaw_deg: float = float("nan")
    pose_pitch_deg: float = float("nan")
    pose_roll_deg: float = float("nan")
    symmetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarvestResult:
    total_lidar_frames: int
    funnel_counts: dict[str, int]
    failure_reasons: dict[str, int]
    raw_scenes: list[HarvestScene]
    selected_scenes: list[HarvestScene]
    motion_distribution: dict[str, Any]
    pose_diversity: dict[str, Any]
    readiness: str
    readiness_reason: str
    metadata: dict[str, Any]


def load_seed_solution(path: str | Path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "rotation_matrix" in data:
        R = np.asarray(data["rotation_matrix"], np.float64)
        t = np.asarray(data["translation_m"], np.float64)
    else:
        T = np.asarray(data["T_cam_lidar"]["data"], np.float64).reshape(4, 4)
        R, t = T[:3, :3], T[:3, 3]
    return SimpleNamespace(ok=True, R=R, t=t, transform=lambda xyz: (R @ np.asarray(xyz).T).T + t)


def harvest_valid_scenes(
    project: Project,
    seed_solution,
    config: HarvestConfig = HarvestConfig(),
    max_lidar_frames: int = 0,
    progress=None,
) -> HarvestResult:
    if not project.bag_paths:
        raise ValueError("project has no bag_paths")
    funnel = {stage: 0 for stage in FUNNEL_STAGES}
    failures: dict[str, int] = {}
    raw_scenes: list[HarvestScene] = []
    with BagSource(project.bag_paths[0]) as src:
        camera_samples = _build_camera_detection_index(src, project.camera_topic, project.camera, project.target)
        motion_by_header = _motion_by_header(camera_samples)
        lidar_times = src.timestamps(project.lidar_topic)
        if max_lidar_frames and max_lidar_frames > 0:
            lidar_times = lidar_times[:max_lidar_frames]
        funnel["TOTAL_LIDAR_FRAMES"] = len(lidar_times)
        for lidar_index, lidar_record_ns in enumerate(lidar_times):
            if progress is not None:
                progress(lidar_index + 1, len(lidar_times))
            scene = _try_harvest_frame(
                src, project, seed_solution, config, lidar_index, int(lidar_record_ns),
                camera_samples, motion_by_header, funnel, failures,
            )
            if scene is not None:
                raw_scenes.append(scene)

    selected = remove_duplicate_scenes(raw_scenes, config)
    funnel["FINAL_VALID_SCENES"] = len(selected)
    motion = motion_distribution(selected)
    pose = pose_diversity(selected)
    readiness, reason = temporal_readiness(selected, motion, pose)
    return HarvestResult(
        total_lidar_frames=funnel["TOTAL_LIDAR_FRAMES"],
        funnel_counts=funnel,
        failure_reasons=failures,
        raw_scenes=raw_scenes,
        selected_scenes=selected,
        motion_distribution=motion,
        pose_diversity=pose,
        readiness=readiness,
        readiness_reason=reason,
        metadata={
            "production_solve_changed": False,
            "yaml_or_extrinsic_written": False,
            "temporal_correction_applied": False,
            "deskew_applied": False,
            "seed_extrinsic_is_diagnostic_only": True,
            "pairing": "LiDAR board median timestamp -> nearest camera header for final correspondence",
            "duplicate_thresholds": {
                "min_time_separation_s": config.duplicate_min_time_separation_s,
                "min_translation_separation_m": config.duplicate_min_translation_separation_m,
                "min_rotation_separation_deg": config.duplicate_min_rotation_separation_deg,
            },
        },
    )


def _build_camera_detection_index(src: BagSource, camera_topic: str, camera: Camera, target: Target) -> list[CameraDetectionSample]:
    out = []
    times = src.timestamps(camera_topic)
    for i, (record_ns, msg) in enumerate(src.range(camera_topic, times[0], times[-1] + 1) if times else []):
        h = header_stamp_ns(msg)
        if h is None:
            continue
        try:
            det = detect_camera(image_to_bgr(msg), camera, target)
        except Exception as exc:  # noqa: BLE001
            out.append(CameraDetectionSample(int(h), int(record_ns), i, False, reason=f"CAMERA_EXCEPTION:{type(exc).__name__}"))
            continue
        out.append(CameraDetectionSample(int(h), int(record_ns), i, bool(det.ok), det=det, reason=getattr(det, "reason", "")))
    return out


def _try_harvest_frame(
    src: BagSource,
    project: Project,
    seed_solution,
    config: HarvestConfig,
    lidar_index: int,
    lidar_record_ns: int,
    camera_samples: list[CameraDetectionSample],
    motion_by_header: dict[int, dict[str, Any]],
    funnel: dict[str, int],
    failures: dict[str, int],
) -> HarvestScene | None:
    _, lidar_msg = src.nearest(project.lidar_topic, lidar_record_ns, window_ns=1_000_000)
    if lidar_msg is None:
        _fail(failures, "LIDAR_MESSAGE_MISSING")
        return None
    rec = cloud_to_struct(lidar_msg)
    xyz_all = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
    finite = np.isfinite(xyz_all).all(axis=1)
    xyz = xyz_all[finite]
    point_ts = np.asarray(rec["timestamp"])[finite] if "timestamp" in (rec.dtype.names or ()) else None
    if point_ts is None or point_ts.size == 0 or not np.isfinite(point_ts).any():
        _fail(failures, "POINT_TIMESTAMP_MISSING")
        return None
    valid_ts = point_ts[np.isfinite(point_ts)]
    sweep_start_ns = int(float(np.min(valid_ts)) * 1e9)
    sweep_end_ns = int(float(np.max(valid_ts)) * 1e9)
    sweep_mid_ns = int((sweep_start_ns + sweep_end_ns) // 2)

    seed_cam = nearest_camera_sample(camera_samples, lidar_record_ns)
    if seed_cam is None:
        _fail(failures, "CAMERA_PAIR_NOT_FOUND")
        return None
    if abs(seed_cam.header_ns - lidar_record_ns) / 1e6 > config.max_camera_pair_dt_ms:
        _fail(failures, "CAMERA_PAIR_TOO_FAR")
        return None
    funnel["CAMERA_PAIR_FOUND"] += 1
    if not seed_cam.ok:
        _fail(failures, "CAMERA_DETECTION_INVALID")
        return None
    funnel["CAMERA_DETECTION_VALID"] += 1
    funnel["CAMERA_PNP_VALID"] += 1
    if getattr(seed_cam.det, "reproj_rms", float("inf")) > config.pnp_reproj_rms_max_px:
        _fail(failures, "CAMERA_PNP_QUALITY_FAIL")
        return None
    funnel["CAMERA_PNP_QUALITY_PASS"] += 1

    roi = _auto_roi(xyz, seed_solution, seed_cam.det, project.target, lidar_index, config)
    if roi is None:
        _fail(failures, "LIDAR_ROI_INVALID")
        return None
    funnel["LIDAR_ROI_VALID"] += 1
    inside = apply_box(xyz, roi)
    params = _derive_params(point_spacing(inside) if len(inside) > 20 else 0.0)
    params.method = config.lidar_method
    ring = np.asarray(rec["ring"])[finite].astype(np.int32) if "ring" in (rec.dtype.names or ()) else None
    det = detect_lidar(xyz, roi, project.target, params, ring=ring)
    if det.plane_indices is None or len(det.plane_indices) == 0:
        _fail(failures, _lidar_fail_reason(det, "PLANE_NOT_FOUND"))
        return None
    funnel["LIDAR_PLANE_VALID"] += 1
    if len(det.edges) > 0:
        funnel["LIDAR_BOUNDARY_VALID"] += 1
    if det.n_circles >= 1:
        funnel["LIDAR_CIRCLE_GE_1"] += 1
    if det.n_circles >= 2:
        funnel["LIDAR_CIRCLE_GE_2"] += 1
    if det.n_circles >= 3:
        funnel["LIDAR_CIRCLE_GE_3"] += 1
    if det.n_circles >= 4:
        funnel["LIDAR_CIRCLE_GE_4"] += 1
        funnel["LIDAR_GEOMETRY_PASS"] += 1
    if not det.ok or det.centers is None or len(det.centers) != 4:
        _fail(failures, _lidar_fail_reason(det, "CIRCLE_LT_4"))
        return None
    funnel["LIDAR_FULL_TARGET_VALID"] += 1

    board_ts = point_ts[np.asarray(det.plane_indices, dtype=np.int64)]
    board_ts = board_ts[np.isfinite(board_ts)]
    if len(board_ts) == 0:
        _fail(failures, "BOARD_TIMESTAMP_INVALID")
        return None
    funnel["TIMESTAMP_VALID"] += 1
    board_ns = int(np.median(board_ts) * 1e9)
    final_cam = nearest_camera_sample(camera_samples, board_ns)
    if final_cam is None or not final_cam.ok:
        _fail(failures, "BOARD_TIME_CAMERA_PAIR_INVALID")
        return None
    funnel["FOUR_CORRESPONDENCES_CREATED"] += 1
    motion = motion_by_header.get(final_cam.header_ns, {})
    R_board, _ = cv2.Rodrigues(final_cam.det.rvec)
    yaw, pitch, roll = _euler_zyx_deg(R_board)
    scene = HarvestScene(
        frame_index=lidar_index,
        camera_index=final_cam.index,
        lidar_index=lidar_index,
        camera_header_ns=final_cam.header_ns,
        lidar_record_ns=lidar_record_ns,
        sweep_start_ns=sweep_start_ns,
        sweep_mid_ns=sweep_mid_ns,
        sweep_end_ns=sweep_end_ns,
        board_median_ns=board_ns,
        board_span_ms=(float(np.max(board_ts)) - float(np.min(board_ts))) * 1000.0,
        camera_minus_board_ms=(final_cam.header_ns - board_ns) / 1e6,
        camera_reproj_rms_px=float(final_cam.det.reproj_rms),
        camera_reproj_p95_px=float(final_cam.det.reproj_p95),
        lidar_centers_xyz=np.asarray(det.centers, np.float64),
        camera_centers_xyz=np.asarray(final_cam.det.centers, np.float64),
        lidar_plane_point_count=len(det.plane),
        lidar_boundary_count=len(det.edges),
        lidar_circle_count=det.n_circles,
        lidar_cluster_count=det.n_clusters,
        lidar_plane_rms=float(det.plane_rms),
        motion_translation_velocity_mm_s=motion.get("translation_velocity_mm_s", float("nan")),
        motion_rotation_velocity_deg_s=motion.get("rotation_velocity_deg_s", float("nan")),
        motion_group=motion_group(motion.get("translation_velocity_mm_s", float("nan")), motion.get("rotation_velocity_deg_s", float("nan"))),
        pose_distance_m=float(np.linalg.norm(final_cam.det.tvec)),
        pose_x_m=float(np.asarray(final_cam.det.tvec).reshape(3)[0]),
        pose_y_m=float(np.asarray(final_cam.det.tvec).reshape(3)[1]),
        pose_yaw_deg=yaw,
        pose_pitch_deg=pitch,
        pose_roll_deg=roll,
    )
    scene.symmetry = symmetry_evidence(scene)
    return scene


def nearest_camera_sample(samples: list[CameraDetectionSample], timestamp_ns: int) -> CameraDetectionSample | None:
    if not samples:
        return None
    return min(samples, key=lambda s: abs(s.header_ns - int(timestamp_ns)))


def remove_duplicate_scenes(scenes: list[HarvestScene], config: HarvestConfig = HarvestConfig()) -> list[HarvestScene]:
    selected: list[HarvestScene] = []
    for scene in sorted(scenes, key=lambda s: s.board_median_ns):
        if not selected:
            selected.append(scene)
            continue
        keep = True
        for prev in selected:
            dt = abs(scene.board_median_ns - prev.board_median_ns) / 1e9
            trans = np.linalg.norm(scene.camera_centers_xyz.mean(axis=0) - prev.camera_centers_xyz.mean(axis=0))
            rot = _scene_rotation_gap_deg(scene, prev)
            if (
                dt < config.duplicate_min_time_separation_s
                and trans < config.duplicate_min_translation_separation_m
                and rot < config.duplicate_min_rotation_separation_deg
            ):
                keep = False
                break
        if keep:
            selected.append(scene)
    return selected


def symmetry_evidence(scene: HarvestScene) -> dict[str, Any]:
    out = {}
    for name, perm in SYMMETRY_BRANCHES.items():
        L = scene.lidar_centers_xyz[list(perm)]
        C = scene.camera_centers_xyz
        R, t = solve_rigid(L, C)
        residual_mm = np.linalg.norm((R @ L.T).T + t - C, axis=1) * 1000.0
        out[name] = {
            "permutation": list(perm),
            "R": R.tolist(),
            "t": t.tolist(),
            "fit_stats_mm": residual_stats_mm(residual_mm),
            "residual_norms_mm": residual_mm.tolist(),
        }
    return out


def export_harvest_result(result: HarvestResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for scene in result.raw_scenes:
        (out / f"scene_{scene.frame_index:05d}.json").write_text(
            json.dumps(scene_to_dict(scene), indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )
    _write_valid_scenes_csv(out / "valid_scenes.csv", result.selected_scenes)
    _write_funnel_csv(out / "detection_funnel.csv", result.funnel_counts)
    summary = {
        "metadata": result.metadata,
        "total_frames": result.total_lidar_frames,
        "raw_valid_scenes": len(result.raw_scenes),
        "after_duplicate_removal": len(result.selected_scenes),
        "detection_funnel": result.funnel_counts,
        "failure_reasons": dict(sorted(result.failure_reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "motion_distribution": result.motion_distribution,
        "pose_diversity": result.pose_diversity,
        "temporal_readiness": result.readiness,
        "readiness_reason": result.readiness_reason,
        "scene407_detected": any(s.frame_index == 407 for s in result.raw_scenes),
    }
    (out / "harvest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    return out


def scene_to_dict(scene: HarvestScene) -> dict[str, Any]:
    d = scene.__dict__.copy()
    d["lidar_centers_xyz"] = scene.lidar_centers_xyz.tolist()
    d["camera_centers_xyz"] = scene.camera_centers_xyz.tolist()
    d["camera_physical_ids"] = CIRCLE_PHYSICAL_IDS
    d["camera_marker_ids"] = [m["marker_id"] for m in board_circle_metadata(Target())]
    return d


def motion_distribution(scenes: list[HarvestScene]) -> dict[str, Any]:
    tv = [s.motion_translation_velocity_mm_s for s in scenes]
    rv = [s.motion_rotation_velocity_deg_s for s in scenes]
    groups = {"LOW": 0, "MID": 0, "HIGH": 0, "UNKNOWN": 0}
    for s in scenes:
        groups[s.motion_group] = groups.get(s.motion_group, 0) + 1
    return {
        "groups": groups,
        "translation_velocity_mm_s": _distribution(tv),
        "rotation_velocity_deg_s": _distribution(rv),
    }


def pose_diversity(scenes: list[HarvestScene]) -> dict[str, Any]:
    return {
        "distance_m": _distribution([s.pose_distance_m for s in scenes]),
        "horizontal_x_m": _distribution([s.pose_x_m for s in scenes]),
        "vertical_y_m": _distribution([s.pose_y_m for s in scenes]),
        "yaw_deg": _distribution([s.pose_yaw_deg for s in scenes]),
        "pitch_deg": _distribution([s.pose_pitch_deg for s in scenes]),
        "roll_deg": _distribution([s.pose_roll_deg for s in scenes]),
        "translation_range_m": _range([s.pose_x_m for s in scenes]),
        "yaw_range_deg": _range([s.pose_yaw_deg for s in scenes]),
        "pitch_range_deg": _range([s.pose_pitch_deg for s in scenes]),
        "roll_range_deg": _range([s.pose_roll_deg for s in scenes]),
    }


def temporal_readiness(scenes: list[HarvestScene], motion: dict[str, Any], pose: dict[str, Any]) -> tuple[str, str]:
    n = len(scenes)
    nonempty_motion_groups = sum(1 for k, v in motion.get("groups", {}).items() if k != "UNKNOWN" and v > 0)
    if n >= 10 and nonempty_motion_groups >= 2:
        return "READY", f"{n} independent valid scenes and {nonempty_motion_groups} motion groups."
    if 5 <= n <= 9:
        return "MARGINAL", f"{n} independent valid scenes; practical sweep possible but weak."
    return "INSUFFICIENT", f"{n} independent valid scenes after duplicate removal; need at least 5, preferably 10+."


def motion_group(translation_velocity_mm_s: float, rotation_velocity_deg_s: float) -> str:
    if not np.isfinite(translation_velocity_mm_s) and not np.isfinite(rotation_velocity_deg_s):
        return "UNKNOWN"
    t = 0.0 if not np.isfinite(translation_velocity_mm_s) else translation_velocity_mm_s
    r = 0.0 if not np.isfinite(rotation_velocity_deg_s) else rotation_velocity_deg_s
    score = max(t / 250.0, r / 2.0)
    if score < 1.0:
        return "LOW"
    if score < 3.0:
        return "MID"
    return "HIGH"


def _motion_by_header(samples: list[CameraDetectionSample]) -> dict[int, dict[str, Any]]:
    valid = []
    for s in samples:
        if not s.ok or s.det is None or s.det.rvec is None:
            continue
        R, _ = cv2.Rodrigues(s.det.rvec)
        valid.append(BoardPoseSample(
            timestamp_ns=s.header_ns,
            frame_index=s.index,
            R=R,
            t=np.asarray(s.det.tvec).reshape(3),
            reproj_rms=float(s.det.reproj_rms),
            reproj_p95=float(s.det.reproj_p95),
        ))
    motion = analyze_board_motion(valid, attempted_count=len(valid))
    return {
        f.timestamp_ns: {
            "translation_velocity_mm_s": f.translation_velocity_mm_s,
            "rotation_velocity_deg_s": f.rotation_velocity_deg_s,
        }
        for f in motion.frames
    }


def _auto_roi(xyz, sol, cam_det, target, idx, config: HarvestConfig):
    for margin in config.roi_margins_m:
        placement = roi_auto.place(xyz, sol, cam_det, target, scene_id=str(idx), margin=margin)
        if placement.ok and placement.box is not None:
            return placement.box
    return None


def _derive_params(spacing: float) -> DetectParams:
    p = DetectParams(sweep_spacing=spacing)
    if spacing > 0:
        p.boundary_radius = max(p.boundary_radius, spacing * 4)
        p.cluster_tolerance = max(p.cluster_tolerance, spacing * 2.5)
    return p


def _scene_rotation_gap_deg(a: HarvestScene, b: HarvestScene) -> float:
    Ra, _ = solve_rigid(a.lidar_centers_xyz, a.camera_centers_xyz)
    Rb, _ = solve_rigid(b.lidar_centers_xyz, b.camera_centers_xyz)
    return rotation_geodesic_deg(Ra, Rb)


def _euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), float(roll)


def _distribution(values) -> dict[str, float]:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {k: float("nan") for k in ("p10", "p25", "median", "p75", "p90", "p95", "min", "max")}
    return {
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _range(values) -> float:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    return float(np.max(a) - np.min(a)) if len(a) else float("nan")


def _fail(bucket: dict[str, int], reason: str) -> None:
    bucket[reason] = bucket.get(reason, 0) + 1


def _lidar_fail_reason(det, fallback: str) -> str:
    reason = getattr(det, "reason", "") or fallback
    if "4" in reason and ("필요" in reason or "꾩슂" in reason):
        return "CIRCLE_LT_4"
    return reason


def _write_valid_scenes_csv(path: Path, scenes: list[HarvestScene]) -> None:
    fields = [
        "frame_index", "camera_index", "lidar_index", "camera_header_ns", "lidar_record_ns",
        "board_median_ns", "camera_minus_board_ms", "camera_reproj_rms_px",
        "lidar_circle_count", "lidar_plane_point_count", "motion_group",
        "motion_translation_velocity_mm_s", "motion_rotation_velocity_deg_s",
        "pose_distance_m", "pose_yaw_deg", "pose_pitch_deg", "pose_roll_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scenes:
            w.writerow({k: getattr(s, k) for k in fields})


def _write_funnel_csv(path: Path, counts: dict[str, int]) -> None:
    prev = None
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "count", "pass_rate_from_previous"])
        w.writeheader()
        for stage in FUNNEL_STAGES:
            count = int(counts.get(stage, 0))
            rate = "" if prev in (None, 0) else count / prev
            w.writerow({"stage": stage, "count": count, "pass_rate_from_previous": rate})
            prev = count

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.core.evaluation.temporal_calibration import (
    TemporalCalibrationConfig,
    annotate_scene_motion,
    build_camera_trajectory_for_project,
    estimate_temporal_offset,
    export_temporal_calibration_result,
    load_fixed_extrinsic,
    load_projection_baseline,
    load_step6_reference_extrinsic,
    load_temporal_scenes_from_harvest,
    load_temporal_scenes_from_step6,
)
from gui.core.project import Project


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Temporal-only Camera-LiDAR offset diagnostic. Does not modify extrinsic/YAML."
    )
    ap.add_argument("--project", default="calib_data/torres.yaml")
    ap.add_argument("--input-mode", choices=("step6", "harvest"), default="step6")
    ap.add_argument("--correspondence-dir", default="calib_result/correspondences")
    ap.add_argument("--scene-ids", default="s01,s02,s04,s05,s06,s07")
    ap.add_argument("--harvest-dir", default="calib_result/phase2_scene_harvest")
    ap.add_argument("--reference-extrinsic", choices=("step6", "yaml"), default="step6")
    ap.add_argument("--fixed-extrinsic", default="calib_result/CA_County_01_bottom_center_bhj.yaml")
    ap.add_argument(
        "--projection-baseline-json",
        default="calib_result/scene407_projection_evidence_corrected/projection_evidence.json",
    )
    ap.add_argument("--out-dir", default="calib_result/temporal_calibration")
    ap.add_argument("--max-camera-frames", type=int, default=0)
    ap.add_argument("--coarse-min-ms", type=float, default=-50.0)
    ap.add_argument("--coarse-max-ms", type=float, default=50.0)
    ap.add_argument("--coarse-step-ms", type=float, default=5.0)
    ap.add_argument("--fine-radius-ms", type=float, default=10.0)
    ap.add_argument("--fine-step-ms", type=float, default=1.0)
    args = ap.parse_args()

    project = Project.load(args.project)
    scene_ids = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
    if args.input_mode == "step6":
        scenes = load_temporal_scenes_from_step6(project, args.correspondence_dir, scene_ids=scene_ids)
    else:
        scenes = load_temporal_scenes_from_harvest(args.harvest_dir)
    camera_poses = build_camera_trajectory_for_project(project, max_frames=args.max_camera_frames)
    annotate_scene_motion(scenes, camera_poses)
    if args.reference_extrinsic == "step6":
        R, t = load_step6_reference_extrinsic(args.correspondence_dir, scene_ids=scene_ids)
    else:
        R, t = load_fixed_extrinsic(args.fixed_extrinsic)
    config = TemporalCalibrationConfig(
        coarse_min_ms=args.coarse_min_ms,
        coarse_max_ms=args.coarse_max_ms,
        coarse_step_ms=args.coarse_step_ms,
        fine_radius_ms=args.fine_radius_ms,
        fine_step_ms=args.fine_step_ms,
    )
    result = estimate_temporal_offset(
        scenes,
        camera_poses,
        R,
        t,
        project.target,
        config=config,
        projection_baseline=load_projection_baseline(args.projection_baseline_json),
    )
    out = export_temporal_calibration_result(result, args.out_dir, camera_poses=camera_poses)
    finite = [r for r in result.sweep if r.rmse_mm == r.rmse_mm]
    best_row = min(finite, key=lambda r: r.rmse_mm) if finite else None
    by_delta = {r.delta_t_ms: r for r in result.sweep}
    scene_best = [
        r for r in result.scene_residuals
        if best_row is not None and r.delta_t_ms == best_row.delta_t_ms
    ]
    scene_zero = [r for r in result.scene_residuals if r.delta_t_ms == 0.0]
    print(json.dumps({
        "out_dir": str(out),
        "input_mode": args.input_mode,
        "scene_ids_requested": scene_ids,
        "timestamp_convention": result.metadata["delta_t_convention"],
        "record_timestamp_used_for_pairing": result.metadata["record_timestamp_used_for_pairing"],
        "spatial_extrinsic_fixed": result.metadata["spatial_extrinsic_fixed"],
        "yaml_or_extrinsic_written": result.metadata["yaml_or_extrinsic_written"],
        "temporal_scene_count": len(result.scenes),
        "valid_temporal_scenes": [
            {
                "scene_id": s.scene_id,
                "camera_minus_lidar_board_ms": (s.camera_header_ns - s.lidar_board_ns) / 1e6,
                "translation_velocity_mm_s": s.translation_velocity_mm_s,
                "rotation_velocity_deg_s": s.rotation_velocity_deg_s,
                "motion_group": s.motion_group,
                "board_point_count": s.board_point_count,
                "board_span_ms": s.board_span_ms,
            }
            for s in result.scenes
        ],
        "camera_pose_count": result.camera_pose_count,
        "classification": result.classification,
        "reason": result.reason,
        "confidence": result.metadata.get("confidence"),
        "best_delta_t_ms": result.best_delta_t_ms,
        "coarse_best_delta_t_ms": result.metadata.get("coarse_best_delta_t_ms"),
        "rmse_at_0_ms": None if 0.0 not in by_delta else by_delta[0.0].rmse_mm,
        "rmse_at_10_ms": None if 10.0 not in by_delta else by_delta[10.0].rmse_mm,
        "rmse_at_15_ms": None if 15.0 not in by_delta else by_delta[15.0].rmse_mm,
        "rmse_at_20_ms": None if 20.0 not in by_delta else by_delta[20.0].rmse_mm,
        "best_sweep_row": None if best_row is None else best_row.__dict__,
        "scene_residual_rmse_at_0_ms": {r.scene_id: r.rmse_mm for r in scene_zero},
        "scene_residual_rmse_at_best_ms": {r.scene_id: r.rmse_mm for r in scene_best},
        "projection_baseline": result.projection_baseline,
    }, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()

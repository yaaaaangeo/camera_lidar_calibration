from __future__ import annotations

import argparse
import json

from gui.core.evaluation.phase2_scene_harvest import (
    HarvestConfig,
    export_harvest_result,
    harvest_valid_scenes,
    load_seed_solution,
)
from gui.core.project import Project


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2A diagnostic-only valid scene harvester.")
    ap.add_argument("--project", default="calib_data/torres.yaml")
    ap.add_argument("--seed-extrinsic", default="calib_result/CA_County_01_bottom_center_bhj.yaml")
    ap.add_argument("--out-dir", default="calib_result/phase2_scene_harvest")
    ap.add_argument("--max-lidar-frames", type=int, default=0)
    ap.add_argument("--pnp-rms-max", type=float, default=2.0)
    ap.add_argument("--min-time-separation", type=float, default=0.5)
    ap.add_argument("--min-translation-separation", type=float, default=0.10)
    ap.add_argument("--min-rotation-separation", type=float, default=5.0)
    args = ap.parse_args()

    project = Project.load(args.project)
    seed = load_seed_solution(args.seed_extrinsic)
    config = HarvestConfig(
        pnp_reproj_rms_max_px=args.pnp_rms_max,
        duplicate_min_time_separation_s=args.min_time_separation,
        duplicate_min_translation_separation_m=args.min_translation_separation,
        duplicate_min_rotation_separation_deg=args.min_rotation_separation,
    )
    result = harvest_valid_scenes(project, seed, config=config, max_lidar_frames=args.max_lidar_frames)
    out = export_harvest_result(result, args.out_dir)
    largest_failure = next(iter(sorted(result.failure_reasons.items(), key=lambda kv: (-kv[1], kv[0]))), None)
    print(json.dumps({
        "out_dir": str(out),
        "total_frames": result.total_lidar_frames,
        "camera_pnp_valid": result.funnel_counts.get("CAMERA_PNP_VALID", 0),
        "lidar_plane_valid": result.funnel_counts.get("LIDAR_PLANE_VALID", 0),
        "circle_ge4": result.funnel_counts.get("LIDAR_CIRCLE_GE_4", 0),
        "geometry_pass": result.funnel_counts.get("LIDAR_GEOMETRY_PASS", 0),
        "four_correspondence_valid": result.funnel_counts.get("FOUR_CORRESPONDENCES_CREATED", 0),
        "raw_valid_scenes": len(result.raw_scenes),
        "after_duplicate_removal": len(result.selected_scenes),
        "motion_groups": result.motion_distribution.get("groups", {}),
        "pose_diversity": result.pose_diversity,
        "largest_failure_reason": largest_failure,
        "temporal_readiness": result.readiness,
        "readiness_reason": result.readiness_reason,
        "policy": result.metadata,
    }, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()

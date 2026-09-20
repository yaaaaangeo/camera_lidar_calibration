from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui.core.evaluation.temporal_motion_diagnostic import (
    build_camera_pose_trajectory,
    export_temporal_motion_diagnostic,
    run_temporal_motion_diagnostic,
)
from gui.core.project import Project


def main() -> None:
    ap = argparse.ArgumentParser(description="Run diagnostic-only Phase 2 temporal/moving-board analysis.")
    ap.add_argument("--project", default="calib_data/torres.yaml")
    ap.add_argument("--correspondence-dir", default="calib_result/correspondences")
    ap.add_argument("--board-pairing-csv", default="calib_result/torres_board_time_camera_pairing_100.csv")
    ap.add_argument("--current-global-projection-json", default="calib_result/scene407_projection_evidence_corrected/projection_evidence.json")
    ap.add_argument("--out-dir", default="calib_result/phase2_temporal_motion_diagnostic")
    ap.add_argument("--max-camera-frames", type=int, default=0)
    args = ap.parse_args()

    project = Project.load(args.project)
    with Path(args.board_pairing_csv).open(newline="", encoding="utf-8") as f:
        import csv
        frame_rows = list(csv.DictReader(f))

    poses = build_camera_pose_trajectory(
        project,
        frame_rows,
        max_frames=args.max_camera_frames,
        include_all_camera_frames=True,
    )
    result = run_temporal_motion_diagnostic(
        args.correspondence_dir,
        args.board_pairing_csv,
        project.target,
        poses,
        current_global_projection_json=args.current_global_projection_json,
    )
    out = export_temporal_motion_diagnostic(result, args.out_dir)
    print(json.dumps({
        "out_dir": str(out),
        "valid_scenes": len(result.valid_scenes),
        "camera_pose_count": result.camera_pose_count,
        "classification": result.classification,
        "reason": result.reason,
        "branch_summaries": result.branch_summaries,
        "current_global_projection_baseline": result.current_global_projection_baseline,
        "policy": {
            "production_solve_changed": False,
            "yaml_or_extrinsic_written": False,
            "temporal_correction_applied": False,
            "deskew_applied": False,
        },
    }, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()

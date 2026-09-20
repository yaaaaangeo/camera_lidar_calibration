from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from gui.core.evaluation.candidate_projection import (
    ProjectionValidationParams,
    evaluate_candidate_projection,
    export_projection_evidence,
    load_cached_validation_frames,
    select_validation_frames_from_board_csv,
)
from gui.core.project import Project


FRAME_IDS_17 = [505, 562, 527, 569, 688, 590, 646, 604, 463, 421, 393, 372, 498, 365, 260, 555, 400]
FULL_REFERENCE_FRAME_IDS = [400, 505, 260]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate scene 407 projection evidence with edge-first preprocessing."
    )
    ap.add_argument("--project", default="calib_data/torres.yaml")
    ap.add_argument("--board-csv", default="calib_result/torres_board_time_camera_pairing_100.csv")
    ap.add_argument("--symmetry-json", default="calib_result/scene407_safe_symmetry_policy/symmetry_diagnostic.json")
    ap.add_argument("--current-global-yaml", default="calib_result/CA_County_01_bottom_center_bhj.yaml")
    ap.add_argument("--out-dir", default="calib_result/scene407_projection_evidence_corrected")
    ap.add_argument("--full-reference-out-dir", default="calib_result/scene407_projection_evidence_corrected_full_reference_3frames")
    ap.add_argument("--old-evidence-dir", default="calib_result/scene407_projection_evidence")
    ap.add_argument("--max-edge-points", type=int, default=80_000)
    args = ap.parse_args()

    _mark_old_evidence_invalidated(Path(args.old_evidence_dir))

    project = Project.load(args.project)
    if not project.bag_paths:
        raise RuntimeError("project has no bag_paths")
    bag_path = project.bag_paths[0]
    candidates = _load_candidates(Path(args.symmetry_json), Path(args.current_global_yaml))
    specs = _select_exact_frames(Path(args.board_csv), FRAME_IDS_17)

    full_specs = _select_exact_frames(Path(args.board_csv), FULL_REFERENCE_FRAME_IDS)
    full_params = ProjectionValidationParams(n_frames=len(full_specs), evaluation_mode="FULL", max_cloud_points=0)
    full_frames = load_cached_validation_frames(
        bag_path, project.lidar_topic, project.camera_topic, full_specs, full_params
    )
    full_evidence = evaluate_candidate_projection(full_frames, candidates, project.camera, full_params)
    full_evidence.metadata.update(_decision_metadata(full_evidence, "FULL_REFERENCE_3FRAMES"))
    export_projection_evidence(full_evidence, args.full_reference_out_dir)

    edge_params = ProjectionValidationParams(
        n_frames=len(specs),
        evaluation_mode="EDGE_SUBSAMPLED",
        max_cloud_points=0,
        max_edge_points=args.max_edge_points,
    )
    frames = load_cached_validation_frames(
        bag_path, project.lidar_topic, project.camera_topic, specs, edge_params
    )
    evidence = evaluate_candidate_projection(frames, candidates, project.camera, edge_params)
    evidence.metadata.update({
        **_decision_metadata(evidence, "EDGE_SUBSAMPLED_17FRAMES"),
        "validation_frame_ids_requested": FRAME_IDS_17,
        "timestamp_source": "board_median_to_nearest_camera_header",
        "current_global_baseline": str(args.current_global_yaml),
        "symmetry_candidate_source": str(args.symmetry_json),
        "full_reference_output": str(args.full_reference_out_dir),
        "production_solve_changed": False,
        "yaml_or_extrinsic_written": False,
        "temporal_correction_applied": False,
        "deskew_applied": False,
    })
    export_projection_evidence(evidence, args.out_dir)

    summary = {
        "full_reference_out_dir": str(args.full_reference_out_dir),
        "corrected_out_dir": str(args.out_dir),
        "full_reference_aggregate": full_evidence.aggregate,
        "corrected_aggregate": evidence.aggregate,
        "corrected_pairwise_wins": evidence.pairwise_wins,
        "decision": evidence.metadata.get("decision"),
        "reason": evidence.metadata.get("reason"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _mark_old_evidence_invalidated(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    marker = {
        "audit_status": "PREPROCESSING_INVALIDATED",
        "usable_for_candidate_selection": False,
        "reason": "raw cloud linear subsampling occurred before LiDAR depth-edge extraction",
        "replacement": "calib_result/scene407_projection_evidence_corrected",
    }
    (out / "audit_status.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")


def _select_exact_frames(csv_path: Path, frame_ids: list[int]):
    specs = select_validation_frames_from_board_csv(csv_path, n_frames=0, max_abs_offset_ms=1e9)
    by_id = {s.frame_index: s for s in specs}
    missing = [idx for idx in frame_ids if idx not in by_id]
    if missing:
        raise RuntimeError(f"missing requested validation frames in {csv_path}: {missing}")
    return [by_id[idx] for idx in frame_ids]


def _load_candidates(symmetry_json: Path, global_yaml: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sym = json.loads(symmetry_json.read_text(encoding="utf-8"))
    named = sym.get("named_candidates") or {}
    out = {
        "NORMAL": _candidate_rt(named["NORMAL"]),
        "HALF_TURN": _candidate_rt(named["HALF_TURN"]),
        "CURRENT_GLOBAL": _global_rt(global_yaml),
    }
    return out


def _candidate_rt(data: dict):
    return np.asarray(data["R"], np.float64), np.asarray(data["t"], np.float64)


def _global_rt(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "rotation_matrix" in data:
        R = np.asarray(data["rotation_matrix"], np.float64)
        t = np.asarray(data["translation_m"], np.float64)
        return R, t
    mat = np.asarray(data["T_cam_lidar"]["data"], np.float64).reshape(4, 4)
    return mat[:3, :3], mat[:3, 3]


def _decision_metadata(evidence, source: str) -> dict:
    agg = evidence.aggregate
    cg = agg.get("CURRENT_GLOBAL", {})
    normal = agg.get("NORMAL", {})
    half = agg.get("HALF_TURN", {})
    cg_mean = cg.get("matched_mean_median_px", float("nan"))
    best_new = min(
        normal.get("matched_mean_median_px", float("inf")),
        half.get("matched_mean_median_px", float("inf")),
    )
    if np.isfinite(cg_mean) and np.isfinite(best_new) and cg_mean < best_new:
        decision = "NEW_CANDIDATES_UNDERPERFORM_GLOBAL"
        confidence = "HIGH"
        reason = "CURRENT_GLOBAL has lower median matched-mean projection error than NORMAL/HALF_TURN"
    else:
        decision = "CALIBRATION_AMBIGUOUS"
        confidence = "LOW"
        reason = "projection evidence alone is not treated as physical correspondence evidence"
    return {
        "decision_source": source,
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "cross_scene_consistency": "MISSING_NOT_EVALUATED",
        "final_status": "CALIBRATION_AMBIGUOUS" if source != "FULL_REFERENCE_3FRAMES" else "REFERENCE_ONLY",
    }


if __name__ == "__main__":
    main()

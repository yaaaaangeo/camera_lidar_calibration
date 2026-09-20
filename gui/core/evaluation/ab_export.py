"""Candidate A/B Test Package Export.

Packages an already-validated Independent Hold-out result (baseline vs. one
locked candidate) into a self-contained, reproducible directory that a real
vehicle / projection / E2E pipeline can use to A/B the two extrinsics --
never a new evaluation, never a new calibration, never a new candidate
search, and never a write to the validated baseline.

The one rule everything here exists to enforce, same spirit as
`gui.core.evaluation.holdout`'s: **export only repackages numbers this
module's callers already computed.** Nothing here re-projects a cloud, runs
Edge Alignment, fits T, searches a delta range, or reopens a bag to re-derive
a metric. The only bag access anywhere in this module is `manifest.json`'s
SHA-256 over the files THIS module itself just wrote -- never over a
recording. The two places this module comes close to "new work" are both
deliberately narrow and read-only:

  - Candidate R/t is *recomputed* from the locked baseline R/t plus
    axis/delta, via `perturbation.perturb_rotation`/`perturb_translation`
    (reused unchanged, never reimplemented) -- because `HoldoutResult` only
    ever stores a *fingerprint* of the candidate transform, by design (see
    `holdout.CandidateIdentity`'s docstring), not the transform itself. The
    recomputed transform's fingerprint is then checked against the locked
    one (`build_snapshot`) before anything is packaged; a mismatch aborts
    the export rather than risk shipping a stale result.
  - `dg.assess_holdout(holdout, status)` is called to obtain the Hold-out
    Diagnostic Evidence item -- a pure, deterministic re-derivation of
    already-computed HoldoutResult numbers through the existing diagnostics
    module, not a new judgment. No new diagnostic wording is authored here;
    `validation_report.md`'s Diagnostic Evidence section only formats that
    item's own fields.

Everything else -- per-frame P95/median, spatial/depth bins, bootstrap CI,
Discovery/Hold-out provenance -- is read straight off the `HoldoutResult`
`build_snapshot` is given.

`CandidateExportSnapshot` is the immutable boundary: built once, at the
moment "내보내기" is pressed, from plain values the caller (step_verify.py)
already holds -- never a live reference into UI state that could change
mid-export. `export_package` then only ever reads from that frozen snapshot.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from gui.core import verify
from gui.core.bag_reader import BagSignature
from gui.core.evaluation import diagnostics as dg
from gui.core.evaluation import holdout as ho
from gui.core.evaluation.perturbation import AXIS_LABELS, perturb_rotation, perturb_translation
from gui.core.project import Project
from gui.core.solve import Solution

PACKAGE_SCHEMA_VERSION = 1
VALIDATION_SUMMARY_SCHEMA_VERSION = 1

PACKAGE_DIR_PREFIX = "candidate_ab_test_"

PER_FRAME_CSV_COLUMNS = (
    "timeline_index", "lidar_timestamp_ns", "camera_timestamp_ns",
    "signed_sync_offset_ms", "abs_sync_offset_ms",
    "baseline_valid", "candidate_valid",
    "baseline_median_px", "baseline_p95_px", "baseline_match_rate",
    "candidate_median_px", "candidate_p95_px", "candidate_match_rate",
    "delta_median_px", "delta_p95_px",
    "comparison_status",
)
# baseline_match_rate/candidate_match_rate are always blank: match rate is
# only ever tracked pooled across a whole trial
# (perturbation.PerturbationPoint.match_rate), never per individual frame
# (perturbation.FrameMetric has no such field) -- inventing a per-frame
# match rate here would be new evaluation logic, not a reuse of an existing
# result, so these columns are left consistently blank rather than guessed.

SPATIAL_DEPTH_CSV_COLUMNS = (
    "analysis_type", "region",
    "baseline_p95_px", "candidate_p95_px", "delta_p95_px",
    "baseline_median_px", "candidate_median_px",
    "baseline_mean_px", "candidate_mean_px",
    "baseline_n_points", "candidate_n_points",
    "baseline_n_matched", "candidate_n_matched",
    "baseline_n_unmatched", "candidate_n_unmatched",
)


def _safe_float(v) -> Optional[float]:
    """`None` for anything not a finite number -- the one consistent way
    this whole module spells "no value here" (never 0.0, never NaN) in both
    JSON (`null`) and CSV (blank), per the export spec's own instruction."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _csv_cell(v):
    """Like `_safe_float`, but returns `""` (not `None`) for CSV rows, since
    csv.DictWriter would otherwise write the literal text "None"."""
    fv = _safe_float(v)
    return "" if fv is None else fv


# ------------------------------------------------------------------ snapshot


@dataclass(frozen=True)
class CandidateExportSnapshot:
    """Everything one export needs, frozen once at build time -- see the
    module docstring. `ok=False` (with `reason`) means the two mandatory
    pre-export checks (baseline fingerprint match, recomputed candidate
    fingerprint match) failed; nothing else on the object is meaningful in
    that case, and `export_package` refuses to write anything for it.
    """

    ok: bool
    reason: str = ""

    created_at: str = ""              # ISO 8601 UTC, frozen at build time
    project_name: str = ""

    baseline_R: Optional[np.ndarray] = None
    baseline_t: Optional[np.ndarray] = None
    baseline_fingerprint: str = ""
    baseline_rmse_m: float = 0.0
    baseline_scene_ids: tuple = ()
    baseline_n_pairs: int = 0

    candidate_R: Optional[np.ndarray] = None
    candidate_t: Optional[np.ndarray] = None
    candidate_fingerprint: str = ""
    candidate_spec: Optional[ho.CandidateSpec] = None
    candidate_provenance: Optional[ho.CandidateProvenance] = None

    camera: object = None
    target: object = None
    lidar_topic: str = ""
    camera_topic: str = ""
    bag_paths: tuple = ()
    method: str = ""

    holdout: Optional[ho.HoldoutResult] = None
    validation_status: str = ho.STATUS_INDEPENDENT
    holdout_evidence: Optional[dg.EvidenceItem] = None


def build_snapshot(
    *,
    project_name: str,
    baseline_solution: Solution,
    holdout: ho.HoldoutResult,
    validation_status: str,
    camera,
    target,
    lidar_topic: str,
    camera_topic: str,
    bag_paths,
    method: str = "",
    now: Optional[datetime] = None,
) -> CandidateExportSnapshot:
    """Build and validate the export snapshot -- see requirements 2/3/4 of
    the Candidate A/B Test Package Export spec: re-derive the candidate
    transform from the *current* baseline (never trust a stored one),
    verify both fingerprints against what the Hold-out result actually
    locked, and refuse (ok=False) rather than package a mismatch.

    `baseline_solution` must be the Solution currently displayed/effective
    (`VerifyStep._effective()`), read-only -- its R/t are copied, never
    mutated, and never written back anywhere.
    """
    if holdout is None or not holdout.ok or holdout.cancelled:
        return CandidateExportSnapshot(ok=False, reason="Hold-out result is not a completed, successful evaluation.")
    if holdout.candidate is None or holdout.candidate_identity is None or holdout.candidate_provenance is None:
        return CandidateExportSnapshot(
            ok=False, reason="Hold-out result is missing candidate identity/provenance -- cannot export."
        )
    if baseline_solution is None or not baseline_solution.ok or baseline_solution.R is None or baseline_solution.t is None:
        return CandidateExportSnapshot(ok=False, reason="No current extrinsic to export.")

    baseline_R = np.array(baseline_solution.R, dtype=np.float64, copy=True)
    baseline_t = np.array(baseline_solution.t, dtype=np.float64, copy=True)
    current_baseline_fp = ho.fingerprint_extrinsic(baseline_R, baseline_t)
    if current_baseline_fp != holdout.baseline_fingerprint:
        return CandidateExportSnapshot(
            ok=False,
            reason="Current extrinsic differs from the validated baseline. Run validation again before exporting.",
        )

    candidate = holdout.candidate
    if candidate.kind == "rotation":
        candidate_R = perturb_rotation(baseline_R, candidate.axis, candidate.delta)
        candidate_t = baseline_t.copy()
    else:
        candidate_R = baseline_R.copy()
        candidate_t = perturb_translation(baseline_t, candidate.axis, candidate.delta)
    recomputed_candidate_fp = ho.fingerprint_extrinsic(candidate_R, candidate_t)
    if recomputed_candidate_fp != holdout.candidate_identity.candidate_fingerprint:
        return CandidateExportSnapshot(
            ok=False,
            reason="Recomputed candidate transform does not match the locked candidate fingerprint. "
                   "Export aborted rather than package a stale or inconsistent result.",
        )

    holdout_evidence = dg.assess_holdout(holdout, validation_status)

    return CandidateExportSnapshot(
        ok=True,
        created_at=(now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        project_name=project_name,
        baseline_R=baseline_R, baseline_t=baseline_t, baseline_fingerprint=current_baseline_fp,
        baseline_rmse_m=float(baseline_solution.rmse), baseline_scene_ids=tuple(baseline_solution.scene_ids),
        baseline_n_pairs=int(baseline_solution.n_pairs),
        candidate_R=candidate_R, candidate_t=candidate_t, candidate_fingerprint=recomputed_candidate_fp,
        candidate_spec=candidate, candidate_provenance=holdout.candidate_provenance,
        camera=camera, target=target, lidar_topic=lidar_topic, camera_topic=camera_topic,
        bag_paths=tuple(bag_paths), method=method,
        holdout=holdout, validation_status=validation_status, holdout_evidence=holdout_evidence,
    )


# --------------------------------------------------------------- extrinsic YAML


def _extrinsic_yaml_text(snapshot: CandidateExportSnapshot, which: str) -> str:
    """Reuses `gui.core.verify.to_yaml` verbatim -- the exact writer/format
    the tool already uses for every other extrinsic export -- rather than
    invent a second, incompatible YAML schema (see requirement 5)."""
    if which == "baseline":
        R, t = snapshot.baseline_R, snapshot.baseline_t
        sol = Solution(ok=True, R=R, t=t, rmse=snapshot.baseline_rmse_m,
                        scene_ids=list(snapshot.baseline_scene_ids), n_pairs=snapshot.baseline_n_pairs)
        note = "Candidate A/B Test Export -- validated BASELINE T_cam_lidar."
    else:
        R, t = snapshot.candidate_R, snapshot.candidate_t
        cand = snapshot.candidate_spec
        sol = Solution(ok=True, R=R, t=t, rmse=snapshot.baseline_rmse_m,
                        scene_ids=list(snapshot.baseline_scene_ids), n_pairs=snapshot.baseline_n_pairs)
        note = (
            f"Candidate A/B Test Export -- CANDIDATE T_cam_lidar ({cand.describe()}). "
            "Result of Independent Extrinsic Parameter Sensitivity, NOT a full SE(3) optimisation. "
            "For A/B comparison only -- DO NOT overwrite the validated baseline automatically."
        )
    export_project = Project(
        name=snapshot.project_name, bag_paths=list(snapshot.bag_paths),
        lidar_topic=snapshot.lidar_topic, camera_topic=snapshot.camera_topic,
    )
    return verify.to_yaml(sol, snapshot.camera, snapshot.target, note, project=export_project, method=snapshot.method)


# ------------------------------------------------------------- validation_summary.json


def _signature_dict(sig: Optional[BagSignature]) -> Optional[dict]:
    if sig is None:
        return None
    return {
        "file_size_bytes": sig.file_size_bytes, "duration_s": sig.duration_s,
        "n_topics": sig.n_topics, "total_message_count": sig.total_message_count,
        "first_lidar_timestamp_ns": sig.first_lidar_timestamp_ns,
        "last_lidar_timestamp_ns": sig.last_lidar_timestamp_ns,
    }


def _discovery_provenance_dict(discovery: Optional[ho.DiscoveryProvenance]) -> Optional[dict]:
    if discovery is None:
        return None
    return {
        "bag": discovery.bag,
        "timestamps_ns": list(discovery.timestamps),
        "exclusion_margin_s": discovery.exclusion_margin_s,
        "signature": _signature_dict(discovery.signature),
    }


def _comparable_timestamps(holdout: ho.HoldoutResult) -> list:
    """LiDAR timestamps of frames where BOTH baseline and candidate scored
    successfully -- read off already-computed FrameMetric.ok flags
    (`holdout.frame_table` for the timestamp, `axis.baseline`/candidate
    `.per_frame` for validity), never recomputed."""
    bp, cp = holdout.baseline_point, holdout.candidate_point
    if bp is None or cp is None:
        return []
    out = []
    for rec in holdout.frame_table:
        b = bp.per_frame.get(rec.timeline_index)
        c = cp.per_frame.get(rec.timeline_index)
        if b is not None and c is not None and b.ok and c.ok:
            out.append(rec.lidar_timestamp_ns)
    return sorted(out)


def _holdout_provenance_dict(holdout: ho.HoldoutResult) -> Optional[dict]:
    hprov = holdout.provenance
    if hprov is None:
        return None
    # Source/slice identity recomputed via the *exact* (non-fuzzy) key an
    # empty registry would resolve to -- reusing resolve_source_identity/
    # slice_identity unchanged (see their own docstrings), not new logic.
    # Fuzzy "suspiciously similar" matching needs actual registry history as
    # context, which a standalone export snapshot does not carry.
    source_identity = ho.resolve_source_identity({}, hprov.bag, holdout.signature)
    slice_id = ho.slice_identity(source_identity, hprov.usable_timestamps)
    return {
        "bag": hprov.bag,
        "source_identity": source_identity,
        "slice_identity": slice_id,
        "signature": _signature_dict(holdout.signature),
        "requested_timestamps_ns": list(hprov.requested_timestamps),
        "excluded_near_discovery_timestamps_ns": list(hprov.excluded_near_discovery),
        "usable_timestamps_ns": list(hprov.usable_timestamps),
        "sync_rejected_timestamps_ns": list(hprov.sync_rejected_timestamps),
        "comparable_timestamps_ns": _comparable_timestamps(holdout),
        "exclusion_margin_s": holdout.discovery.exclusion_margin_s if holdout.discovery is not None else None,
        "signature_warning": holdout.signature_warning or None,
        "cross_bag_time_overlap_note": holdout.cross_bag_time_overlap_note or None,
    }


def _aggregate_metrics_dict(holdout: ho.HoldoutResult) -> Optional[dict]:
    bp, cp = holdout.baseline_point, holdout.candidate_point
    if bp is None or cp is None:
        return None
    return {
        "baseline_median_frame_p95_px": _safe_float(bp.median_frame_p95_px),
        "candidate_median_frame_p95_px": _safe_float(cp.median_frame_p95_px),
        "median_delta_frame_p95_px": _safe_float(cp.median_delta_frame_p95_px),
        "p95_delta_frame_p95_px": _safe_float(cp.p95_delta_frame_p95_px),
        "n_comparable_frames": cp.n_comparable_frames,
        "n_baseline_only_valid": cp.n_baseline_only_valid,
        "n_candidate_only_valid": cp.n_candidate_only_valid,
        "n_both_failed": cp.n_both_failed,
        "n_improved_frames": cp.n_improved_frames,
        "n_worsened_frames": cp.n_worsened_frames,
        "n_unchanged_frames": cp.n_unchanged_frames,
        "improved_frame_ratio": _safe_float(cp.improved_frame_ratio),
        "baseline_match_rate": _safe_float(bp.match_rate),
        "candidate_match_rate": _safe_float(cp.match_rate),
        "n_frames_requested": holdout.n_frames_requested,
        "n_frames_used": holdout.n_frames_used,
        "n_sync_rejected": holdout.n_sync_rejected,
    }


def _bootstrap_dict(bs: Optional[ho.BootstrapResult]) -> Optional[dict]:
    if bs is None:
        return None
    return {
        "ok": bs.ok, "reason": bs.reason or None, "method": bs.method or None,
        "iterations": bs.iterations, "block_s": bs.block_s, "n_blocks": bs.n_blocks,
        "n_comparable_frames": bs.n_comparable_frames,
        "median_frame_interval_s": _safe_float(bs.median_frame_interval_s),
        "frames_per_block_median": _safe_float(bs.frames_per_block_median),
        "frames_per_block_min": bs.frames_per_block_min, "frames_per_block_max": bs.frames_per_block_max,
        "sparse_sampling": bs.sparse_sampling,
        "median_delta_p95_px": _safe_float(bs.median_delta_p95_px),
        "median_delta_ci_low_px": _safe_float(bs.median_delta_ci_low_px),
        "median_delta_ci_high_px": _safe_float(bs.median_delta_ci_high_px),
        "improved_ratio": _safe_float(bs.improved_ratio),
        "improved_ratio_ci_low": _safe_float(bs.improved_ratio_ci_low),
        "improved_ratio_ci_high": _safe_float(bs.improved_ratio_ci_high),
    }


def _enriched_region_rows(rows: list, baseline_spatial, candidate_spatial, attr: str) -> list:
    """One row per already-computed `SpatialComparisonRow`, enriched with
    the fuller `spatial_analysis.BinStats` (median/mean/matched/unmatched)
    `PerturbationPoint.spatial` already computed but `SpatialComparisonRow`
    itself does not carry -- both sources come from the same
    `evaluate_holdout` call; nothing here is recomputed."""
    b_dict = getattr(baseline_spatial, attr, {}) if baseline_spatial is not None else {}
    c_dict = getattr(candidate_spatial, attr, {}) if candidate_spatial is not None else {}
    out = []
    for row in rows:
        b = b_dict.get(row.label)
        c = c_dict.get(row.label)
        out.append({
            "region": row.label,
            "baseline_p95_px": _safe_float(row.baseline_p95_px),
            "candidate_p95_px": _safe_float(row.candidate_p95_px),
            "delta_p95_px": _safe_float(row.delta_p95_px),
            "baseline_median_px": _safe_float(b.median_px) if b is not None else None,
            "candidate_median_px": _safe_float(c.median_px) if c is not None else None,
            "baseline_mean_px": _safe_float(b.mean_px) if b is not None else None,
            "candidate_mean_px": _safe_float(c.mean_px) if c is not None else None,
            "baseline_n_points": row.baseline_n_points,
            "candidate_n_points": row.candidate_n_points,
            "baseline_n_matched": b.n_matched if b is not None else None,
            "candidate_n_matched": c.n_matched if c is not None else None,
            "baseline_n_unmatched": b.n_unmatched if b is not None else None,
            "candidate_n_unmatched": c.n_unmatched if c is not None else None,
        })
    return out


def _spatial_summary_dict(holdout: ho.HoldoutResult) -> dict:
    bp_spatial = holdout.baseline_point.spatial if holdout.baseline_point is not None else None
    cp_spatial = holdout.candidate_point.spatial if holdout.candidate_point is not None else None
    return {
        "horizontal": _enriched_region_rows(holdout.horizontal, bp_spatial, cp_spatial, "horizontal"),
        "vertical": _enriched_region_rows(holdout.vertical, bp_spatial, cp_spatial, "vertical"),
    }


def _depth_summary_list(holdout: ho.HoldoutResult) -> list:
    bp_spatial = holdout.baseline_point.spatial if holdout.baseline_point is not None else None
    cp_spatial = holdout.candidate_point.spatial if holdout.candidate_point is not None else None
    return _enriched_region_rows(holdout.depth, bp_spatial, cp_spatial, "depth_bins")


def _evidence_dict(item: Optional["dg.EvidenceItem"]) -> Optional[dict]:
    if item is None:
        return None
    return {
        "category": item.category, "title": item.title, "strength": item.strength,
        "scope": item.scope, "mixed": item.mixed,
        "metrics": [{"label": k, "value": v} for k, v in item.metrics],
        "observations": list(item.observations),
        "interpretations": list(item.interpretations),
        "caveats": list(item.caveats),
    }


def _caveats(snapshot: CandidateExportSnapshot) -> list:
    holdout = snapshot.holdout
    out = [
        "Candidate is the result of Independent Extrinsic Parameter Sensitivity (a single-axis probe around "
        "baseline), not a full SE(3) optimisation -- see validation_summary.json's candidate.kind.",
        "Thresholds referenced anywhere in this package's diagnostic evidence are configurable heuristic "
        "defaults for this project, not sensor-validated correctness cut-offs.",
        "This package does not apply the candidate, overwrite the validated baseline calibration, or "
        "perform any automatic correction -- see manifest.json / validation_report.md.",
    ]
    if snapshot.validation_status in ho.EXPOSED_STATUSES:
        out.append(
            f"Validation status is {snapshot.validation_status!r} "
            f"({ho.STATUS_LABELS.get(snapshot.validation_status, '')}) -- this hold-out result was exposed "
            "during candidate tuning; read the comparison with that in mind."
        )
    if holdout is not None and holdout.provenance is not None and holdout.provenance.sync_rejected_timestamps:
        out.append(
            f"{len(holdout.provenance.sync_rejected_timestamps)} hold-out frame(s) were excluded for "
            "exceeding the configured sync offset."
        )
    if holdout is not None and holdout.bootstrap is not None and holdout.bootstrap.sparse_sampling:
        out.append(
            "Bootstrap sparse-sampling caveat: most temporal blocks hold only a single frame, so the block "
            "bootstrap's protection against temporal correlation is only weakly represented here."
        )
    if holdout is not None and (holdout.discovery is None or holdout.discovery.signature is None):
        out.append(
            "Discovery source BagSignature is unavailable (source file inaccessible when this Hold-out ran, "
            "or this result predates signature capture) -- discovery.signature is null."
        )
    return out


def build_validation_summary(snapshot: CandidateExportSnapshot) -> dict:
    """The machine-readable canonical record -- see requirement 13. Every
    number comes from `snapshot`/`snapshot.holdout`, already computed;
    nothing here is a new measurement."""
    holdout = snapshot.holdout
    cand = snapshot.candidate_spec
    prov = snapshot.candidate_provenance

    summary = {
        "schema_version": VALIDATION_SUMMARY_SCHEMA_VERSION,
        "created_at": snapshot.created_at,
        "tool": {
            "name": "Camera-LiDAR Calibration Tool",
            "feature": "Candidate A/B Test Package Export",
            "package_kind": "candidate_ab_test",
            "package_schema_version": PACKAGE_SCHEMA_VERSION,
        },
        "project": {"name": snapshot.project_name, "bag_paths": list(snapshot.bag_paths)},
        "transform": {
            "name": "T_cam_lidar",
            "direction": "lidar_to_camera",
            "definition": "p_camera = R @ p_lidar + t (see gui.core.solve.Solution.transform)",
            "camera_frame": {"x": "right", "y": "down", "z": "forward"},
        },
        "perturbation_convention": {
            "note": (
                "Independent Extrinsic Parameter Sensitivity -- each of the six parameters is wiggled on its "
                "own, holding every other parameter exactly at baseline; NOT a full SE(3) left composition. "
                "Axis names follow camera attitude, NOT the common robotics roll=X/pitch=Y/yaw=Z convention "
                "(see gui.core.evaluation.perturbation's module docstring)."
            ),
            "axes": dict(AXIS_LABELS),
        },
        "baseline": {
            "fingerprint": snapshot.baseline_fingerprint,
            "rotation_matrix": snapshot.baseline_R.tolist(),
            "translation_m": snapshot.baseline_t.tolist(),
            "rmse_m": _safe_float(snapshot.baseline_rmse_m),
            "n_scenes": len(snapshot.baseline_scene_ids),
            "n_pairs": snapshot.baseline_n_pairs,
        },
        "candidate": {
            "axis": cand.axis,
            "label": cand.label,
            "delta": cand.delta,
            "unit": "degree" if cand.unit == "deg" else "mm",
            "source": cand.source,
            "baseline_fingerprint": snapshot.baseline_fingerprint,
            "candidate_fingerprint": snapshot.candidate_fingerprint,
            "rotation_matrix": snapshot.candidate_R.tolist(),
            "translation_m": snapshot.candidate_t.tolist(),
            "kind": "Independent Extrinsic Parameter Sensitivity result -- not a full SE(3) optimisation.",
            "discovery_revision": prov.discovery_revision if prov is not None else None,
        },
        "discovery": _discovery_provenance_dict(holdout.discovery) if holdout is not None else None,
        "holdout": _holdout_provenance_dict(holdout) if holdout is not None else None,
        "validation_status": {
            "code": snapshot.validation_status,
            "label": ho.STATUS_LABELS.get(snapshot.validation_status, snapshot.validation_status),
            "is_exposed": snapshot.validation_status in ho.EXPOSED_STATUSES,
        },
        "aggregate_metrics": _aggregate_metrics_dict(holdout) if holdout is not None else None,
        "bootstrap": _bootstrap_dict(holdout.bootstrap) if holdout is not None else None,
        "spatial": _spatial_summary_dict(holdout) if holdout is not None else {"horizontal": [], "vertical": []},
        "depth": _depth_summary_list(holdout) if holdout is not None else [],
        "diagnostic_evidence": _evidence_dict(snapshot.holdout_evidence),
        "caveats": _caveats(snapshot),
    }
    return _json_safe(summary)


def _json_safe(obj):
    """Recursively converts to plain JSON-safe types -- NaN/Inf become
    `null`, never left as non-standard JSON tokens (see requirement 13)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        fv = float(obj)
        return fv if math.isfinite(fv) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


# ------------------------------------------------------------- per_frame_results.csv


def build_per_frame_rows(holdout: ho.HoldoutResult) -> list:
    """One row per Hold-out frame (`holdout.frame_table`), joined against
    the already-computed per-frame validity/P95/median
    (`axis.baseline`/candidate `.per_frame`) -- see requirement 11. No new
    scoring: a `comparison_status` is derived purely from the two `ok`
    flags already on record.
    """
    bp, cp = holdout.baseline_point, holdout.candidate_point
    rows = []
    for rec in holdout.frame_table:
        row = {c: "" for c in PER_FRAME_CSV_COLUMNS}
        row["timeline_index"] = rec.timeline_index
        row["lidar_timestamp_ns"] = rec.lidar_timestamp_ns
        row["camera_timestamp_ns"] = rec.camera_timestamp_ns
        row["signed_sync_offset_ms"] = _csv_cell(rec.signed_sync_offset_ms)
        row["abs_sync_offset_ms"] = _csv_cell(rec.abs_sync_offset_ms)

        if rec.sync_rejected:
            row["comparison_status"] = "sync_rejected"
            rows.append(row)
            continue

        b = bp.per_frame.get(rec.timeline_index) if bp is not None else None
        c = cp.per_frame.get(rec.timeline_index) if cp is not None else None
        b_ok = b is not None and b.ok
        c_ok = c is not None and c.ok
        row["baseline_valid"] = bool(b_ok)
        row["candidate_valid"] = bool(c_ok)
        if b is not None:
            row["baseline_median_px"] = _csv_cell(b.median_px)
            row["baseline_p95_px"] = _csv_cell(b.p95_px)
        if c is not None:
            row["candidate_median_px"] = _csv_cell(c.median_px)
            row["candidate_p95_px"] = _csv_cell(c.p95_px)
        if b_ok and c_ok:
            row["delta_median_px"] = _csv_cell(c.median_px - b.median_px)
            row["delta_p95_px"] = _csv_cell(c.p95_px - b.p95_px)
            row["comparison_status"] = "both_valid"
        elif b_ok and not c_ok:
            row["comparison_status"] = "baseline_only_valid"
        elif c_ok and not b_ok:
            row["comparison_status"] = "candidate_only_valid"
        else:
            row["comparison_status"] = "both_failed"
        rows.append(row)
    return rows


# --------------------------------------------------------- spatial_depth_results.csv


def build_spatial_depth_rows(holdout: ho.HoldoutResult) -> list:
    """Long-format rows (see requirement 12) -- one row per region per
    analysis type, reusing exactly the same enriched rows the JSON summary
    uses (`_enriched_region_rows`), never a second spatial/depth pass."""
    bp_spatial = holdout.baseline_point.spatial if holdout.baseline_point is not None else None
    cp_spatial = holdout.candidate_point.spatial if holdout.candidate_point is not None else None
    out = []
    for analysis_type, rows, attr in (
        ("horizontal", holdout.horizontal, "horizontal"),
        ("vertical", holdout.vertical, "vertical"),
        ("depth", holdout.depth, "depth_bins"),
    ):
        for r in _enriched_region_rows(rows, bp_spatial, cp_spatial, attr):
            out.append({
                "analysis_type": analysis_type,
                "region": r["region"],
                "baseline_p95_px": "" if r["baseline_p95_px"] is None else r["baseline_p95_px"],
                "candidate_p95_px": "" if r["candidate_p95_px"] is None else r["candidate_p95_px"],
                "delta_p95_px": "" if r["delta_p95_px"] is None else r["delta_p95_px"],
                "baseline_median_px": "" if r["baseline_median_px"] is None else r["baseline_median_px"],
                "candidate_median_px": "" if r["candidate_median_px"] is None else r["candidate_median_px"],
                "baseline_mean_px": "" if r["baseline_mean_px"] is None else r["baseline_mean_px"],
                "candidate_mean_px": "" if r["candidate_mean_px"] is None else r["candidate_mean_px"],
                "baseline_n_points": r["baseline_n_points"],
                "candidate_n_points": r["candidate_n_points"],
                "baseline_n_matched": "" if r["baseline_n_matched"] is None else r["baseline_n_matched"],
                "candidate_n_matched": "" if r["candidate_n_matched"] is None else r["candidate_n_matched"],
                "baseline_n_unmatched": "" if r["baseline_n_unmatched"] is None else r["baseline_n_unmatched"],
                "candidate_n_unmatched": "" if r["candidate_n_unmatched"] is None else r["candidate_n_unmatched"],
            })
    return out


def _rows_to_csv_text(rows: list, columns: tuple) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


# ------------------------------------------------------------- validation_report.md


def _fmt_px(v) -> str:
    fv = _safe_float(v)
    return "—" if fv is None else f"{fv:.2f} px"


def _fmt_signed_px(v) -> str:
    fv = _safe_float(v)
    return "—" if fv is None else f"{fv:+.2f} px"


def _md_region_table(rows: list) -> list:
    if not rows:
        return ["(no data)", ""]
    lines = ["| Region | Baseline P95 | Candidate P95 | Delta P95 |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.label} | {_fmt_px(r.baseline_p95_px)} | {_fmt_px(r.candidate_p95_px)} | "
                      f"{_fmt_signed_px(r.delta_p95_px)} |")
    lines.append("")
    return lines


def _md_evidence(item: "dg.EvidenceItem") -> list:
    """Formats `item`'s own already-computed fields as Markdown -- never
    authors new diagnostic text (see the module docstring)."""
    lines = [f"**{item.title}** -- Evidence: {item.strength_label}", ""]
    if item.metrics:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in item.metrics:
            lines.append(f"| {k} | {v} |")
        lines.append("")
    for heading, values in (
        ("Observed", item.observations),
        ("Possible interpretation", item.interpretations),
        ("Caveats", item.caveats),
    ):
        if values:
            lines.append(f"_{heading}:_")
            lines += [f"- {v}" for v in values]
            lines.append("")
    return lines


def build_validation_report_md(snapshot: CandidateExportSnapshot) -> str:
    holdout = snapshot.holdout
    cand = snapshot.candidate_spec
    bp, cp = holdout.baseline_point, holdout.candidate_point
    bs = holdout.bootstrap
    status = snapshot.validation_status
    status_label = ho.STATUS_LABELS.get(status, status)
    exposed = status in ho.EXPOSED_STATUSES

    lines = ["# Camera-LiDAR Candidate A/B Validation Report", ""]
    lines.append(f"**Validation Status: {'EXPLORATORY' if exposed else 'INDEPENDENT'}**")
    lines.append("")
    if exposed:
        lines.append(f"> {status_label}")
        lines.append("> This hold-out was exposed during candidate tuning.")
    else:
        lines.append(f"> {status_label}")
    lines += ["", f"Generated: {snapshot.created_at}", f"Project: {snapshot.project_name}", ""]

    lines += ["## Transform", ""]
    lines += [
        "**Baseline** (`T_cam_lidar`, LiDAR -> Camera)", "",
        f"- fingerprint: `{snapshot.baseline_fingerprint}`", "",
        "**Candidate**", "",
        f"- fingerprint: `{snapshot.candidate_fingerprint}`",
        "- NOT applied -- see `candidate_extrinsic.yaml`'s own note. Do not overwrite the validated baseline "
        "automatically.",
        "",
    ]

    lines += ["## Candidate Discovery", ""]
    unit = "°" if cand.unit == "deg" else " mm"
    prov = snapshot.candidate_provenance
    lines += [
        f"- Axis / delta: {cand.label} {cand.delta:+.3f}{unit}",
        f"- Discovery source: {cand.source.capitalize()}",
        f"- Discovery revision (locked): {prov.discovery_revision if prov is not None else '—'}",
        "",
    ]

    lines += ["## Independent Hold-out", ""]
    if holdout.provenance is not None:
        lines += [
            f"- Dataset (bag): `{holdout.provenance.bag}`",
            f"- Frames requested / usable / sync-rejected: "
            f"{holdout.n_frames_requested} / {holdout.n_frames_used} / {holdout.n_sync_rejected}",
            f"- Excluded near Discovery: {len(holdout.provenance.excluded_near_discovery)}",
            f"- Comparable (both baseline and candidate valid): {cp.n_comparable_frames if cp is not None else '—'}",
            "",
        ]

    lines += ["## A/B Result", ""]
    if bp is not None and cp is not None:
        lines += [
            f"- Baseline Median Frame-P95: {_fmt_px(bp.median_frame_p95_px)}",
            f"- Candidate Median Frame-P95: {_fmt_px(cp.median_frame_p95_px)}",
            f"- Median Delta Frame-P95: {_fmt_signed_px(cp.median_delta_frame_p95_px)}",
            f"- Improved / Worsened / Unchanged: "
            f"{cp.n_improved_frames} / {cp.n_worsened_frames} / {cp.n_unchanged_frames}",
            "",
        ]

    lines += ["## Bootstrap Uncertainty", ""]
    if bs is not None and bs.ok:
        lines += [
            f"- Method: {bs.method}",
            f"- 95% CI (median Delta Frame-P95): "
            f"[{_fmt_signed_px(bs.median_delta_ci_low_px)}, {_fmt_signed_px(bs.median_delta_ci_high_px)}]",
            f"- Effective temporal blocks: {bs.n_blocks}",
        ]
        if bs.sparse_sampling:
            lines.append("- **Sparse-sampling caveat**: most temporal blocks hold only a single frame.")
        lines.append("")
    elif bs is not None:
        lines += [f"- Unavailable: {bs.reason}", ""]
    else:
        lines += ["- Unavailable.", ""]

    lines += ["## Spatial Comparison", "", "### Horizontal", ""]
    lines += _md_region_table(holdout.horizontal)
    lines += ["### Vertical", ""]
    lines += _md_region_table(holdout.vertical)

    lines += ["## Depth Comparison", ""]
    lines += _md_region_table(holdout.depth)

    lines += ["## Diagnostic Evidence", ""]
    if snapshot.holdout_evidence is not None:
        lines += _md_evidence(snapshot.holdout_evidence)
    else:
        lines += ["(unavailable)", ""]

    lines += ["## Caveats", ""]
    lines += [f"- {c}" for c in _caveats(snapshot)]
    lines.append("")

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ manifest.json


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(files: "dict[str, Path]") -> dict:
    """SHA-256 here is for package reproducibility / change detection, not
    security authentication -- see requirement 15. `manifest.json` never
    hashes itself."""
    return {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "purpose": "reproducibility and change detection, not security authentication",
        "files": {name: {"sha256": _sha256_file(path)} for name, path in files.items()},
    }


# --------------------------------------------------------------------- orchestration


@dataclass
class ExportOutcome:
    ok: bool
    reason: str = ""
    out_dir: Optional[Path] = None
    files: tuple = ()


def _write_all_files(snapshot: CandidateExportSnapshot, tmp_root: Path) -> "dict[str, Path]":
    tmp_root.mkdir(parents=True, exist_ok=True)
    paths: "dict[str, Path]" = {}

    for name, text in (
        ("baseline_extrinsic.yaml", _extrinsic_yaml_text(snapshot, "baseline")),
        ("candidate_extrinsic.yaml", _extrinsic_yaml_text(snapshot, "candidate")),
    ):
        (tmp_root / name).write_text(text, encoding="utf-8")
        paths[name] = tmp_root / name

    summary = build_validation_summary(snapshot)
    (tmp_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["validation_summary.json"] = tmp_root / "validation_summary.json"

    (tmp_root / "per_frame_results.csv").write_text(
        _rows_to_csv_text(build_per_frame_rows(snapshot.holdout), PER_FRAME_CSV_COLUMNS), encoding="utf-8"
    )
    paths["per_frame_results.csv"] = tmp_root / "per_frame_results.csv"

    (tmp_root / "spatial_depth_results.csv").write_text(
        _rows_to_csv_text(build_spatial_depth_rows(snapshot.holdout), SPATIAL_DEPTH_CSV_COLUMNS), encoding="utf-8"
    )
    paths["spatial_depth_results.csv"] = tmp_root / "spatial_depth_results.csv"

    (tmp_root / "validation_report.md").write_text(build_validation_report_md(snapshot), encoding="utf-8")
    paths["validation_report.md"] = tmp_root / "validation_report.md"

    manifest = build_manifest(paths)
    (tmp_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Added to `paths` only now, after build_manifest(paths) ran -- manifest.json
    # is a written package file (must appear in outcome.files / self-validation),
    # but never lists its own hash inside itself (see build_manifest's docstring).
    paths["manifest.json"] = tmp_root / "manifest.json"

    return paths


def _self_validate(snapshot: CandidateExportSnapshot, tmp_root: Path) -> "tuple[bool, str]":
    """Requirement 17: re-read everything just written before it is ever
    promoted to the final directory. Any failure here means the temp
    directory is discarded and nothing final is ever produced.

    Baseline/candidate are compared with `np.allclose`, not
    `fingerprint_extrinsic` re-derived from the round-tripped text:
    `verify.to_yaml` (reused unchanged, see requirement 5 -- this module
    must not invent a new, higher-precision YAML schema just to make this
    check exact) prints rotation to 8 decimal places and translation to 6,
    while `fingerprint_extrinsic` rounds to 9 -- re-fingerprinting text at
    fewer decimal places than the fingerprint itself uses would almost
    always disagree with the in-memory value's fingerprint even for a
    perfectly correct round-trip. A tolerance well above that rounding
    (1e-5) checks what this step actually needs to guarantee: the file is
    readable and describes the same transform, not bit-for-bit precision
    beyond what the format ever promised to carry.
    """
    try:
        baseline_data = verify.from_yaml((tmp_root / "baseline_extrinsic.yaml").read_text(encoding="utf-8"))
        candidate_data = verify.from_yaml((tmp_root / "candidate_extrinsic.yaml").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - reported to the caller as a normal export failure
        return False, f"Baseline/candidate YAML failed to re-read: {exc}"
    if "R" not in baseline_data or "t" not in baseline_data:
        return False, "baseline_extrinsic.yaml round-trip lost the transform."
    if "R" not in candidate_data or "t" not in candidate_data:
        return False, "candidate_extrinsic.yaml round-trip lost the transform."
    if not (np.allclose(baseline_data["R"], snapshot.baseline_R, atol=1e-5)
            and np.allclose(baseline_data["t"], snapshot.baseline_t, atol=1e-5)):
        return False, "baseline_extrinsic.yaml round-trip does not match the snapshot's baseline transform."
    if not (np.allclose(candidate_data["R"], snapshot.candidate_R, atol=1e-5)
            and np.allclose(candidate_data["t"], snapshot.candidate_t, atol=1e-5)):
        return False, "candidate_extrinsic.yaml round-trip does not match the snapshot's candidate transform."

    try:
        summary = json.loads((tmp_root / "validation_summary.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"validation_summary.json failed to parse: {exc}"
    if not isinstance(summary, dict) or "schema_version" not in summary or "baseline" not in summary:
        return False, "validation_summary.json is missing required top-level keys."

    for name in ("per_frame_results.csv", "spatial_depth_results.csv"):
        try:
            lines = (tmp_root / name).read_text(encoding="utf-8").splitlines()
        except Exception as exc:  # noqa: BLE001
            return False, f"{name} failed to read: {exc}"
        if not lines:
            return False, f"{name} has no header row."

    try:
        manifest = json.loads((tmp_root / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"manifest.json failed to parse: {exc}"
    for rel_path, info in manifest.get("files", {}).items():
        actual = _sha256_file(tmp_root / rel_path)
        if actual != info.get("sha256"):
            return False, f"manifest.json hash mismatch for {rel_path}."

    return True, ""


def _package_dir_name(now: datetime) -> str:
    return f"{PACKAGE_DIR_PREFIX}{now.strftime('%Y%m%d_%H%M%S')}"


def _unique_dir(base_dir: Path, name: str) -> Path:
    """`name`, or `name_01`, `name_02`, ... -- the first that does not
    already exist (see requirement 1's collision policy)."""
    candidate = base_dir / name
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = base_dir / f"{name}_{i:02d}"
        if not candidate.exists():
            return candidate
        i += 1


def export_package(
    snapshot: CandidateExportSnapshot, out_base_dir: "str | Path", now: Optional[datetime] = None,
) -> ExportOutcome:
    """Build every file in a fresh temporary directory next to
    `out_base_dir`, self-validate it, then atomically rename it to a
    conflict-free final directory -- never writes anything into
    `out_base_dir` directly, never overwrites an existing package, and
    leaves nothing behind on any failure (requirement 16).
    """
    if not snapshot.ok:
        return ExportOutcome(ok=False, reason=snapshot.reason)

    out_base_dir = Path(out_base_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)
    final_dir = _unique_dir(out_base_dir, _package_dir_name(now or datetime.now()))

    tmp_root = Path(tempfile.mkdtemp(dir=out_base_dir, prefix=f".{final_dir.name}-"))
    try:
        files_written = _write_all_files(snapshot, tmp_root)
        ok, reason = _self_validate(snapshot, tmp_root)
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return ExportOutcome(ok=False, reason=reason)
        os.replace(tmp_root, final_dir)
    except BaseException:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return ExportOutcome(ok=True, out_dir=final_dir, files=tuple(sorted(files_written)))

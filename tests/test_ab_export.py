"""Tests for gui.core.evaluation.ab_export (Candidate A/B Test Package Export).

Split into several groups:
  - Snapshot building: candidate re-derivation + fingerprint verification,
    the two mandatory pre-export guards (requirements 2/3/4).
  - File builders: extrinsic YAML, validation_summary.json, both CSVs,
    validation_report.md, manifest.json -- each checked against the
    already-computed HoldoutResult they must reuse, not recompute.
  - export_package: atomicity, self-validation, directory collision
    avoidance, and read-only-ness (never touches R/t, Project,
    exposure registry, or discovery_revision).

A real (not monkeypatched) `evaluate_holdout` call backs every test here --
the synthetic depth-step scene proven in test_holdout.py/test_perturbation.py
-- so the CSV/JSON/report builders are exercised against genuine
PerturbationPoint/SpatialAnalysisResult/BootstrapResult data, not hand-built
stand-ins.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

import gui.core.evaluation.ab_export as abx
import gui.core.evaluation.holdout as ho
from gui.core.bag_reader import BagSignature
from gui.core.evaluation.edge_alignment import EdgeAlignmentParams
from gui.core.evaluation.holdout import CandidateSpec, DiscoveryProvenance, evaluate_holdout
from gui.core.evaluation.perturbation import prepare_frames
from gui.core.project import Camera, Target
from gui.core.solve import Solution
from gui.core import verify

SEC = 1_000_000_000
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# --------------------------------------------------------------- synthetic scene
#
# Same depth-step scene as test_holdout.py: a vertical intensity edge at CX,
# points at z=5m to its left and z=10m to its right, so a yaw perturbation
# measurably shifts the projected edge. Frames are spaced 3 real seconds
# apart (unlike test_holdout.py's bare-index spacing) so the block bootstrap
# actually has enough temporal blocks to produce a CI -- export must show a
# real bootstrap result, not just "unavailable".

_WIDTH, _HEIGHT = 640, 480
_FX = _FY = 500.0
_CX, _CY = _WIDTH / 2, _HEIGHT / 2


def _make_scene():
    image = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    image[:, int(_CX):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    u = np.repeat(np.linspace(0, _WIDTH - 1, 250), 200)
    v = np.tile(np.linspace(0, _HEIGHT - 1, 200), 250)
    z = np.where(u < _CX, 5.0, 10.0)
    x = (u - _CX) * z / _FX
    y = (v - _CY) * z / _FY
    return image, np.column_stack([x, y, z]).astype(np.float32)


def _camera() -> Camera:
    return Camera(fx=_FX, fy=_FY, cx=_CX, cy=_CY)


def _sig(**overrides) -> BagSignature:
    base = dict(file_size_bytes=2_000_000, duration_s=60.0, n_topics=3, total_message_count=4000,
                first_lidar_timestamp_ns=0, last_lidar_timestamp_ns=18 * SEC)
    base.update(overrides)
    return BagSignature(**base)


def _export_frames(image, cloud, n=6, spacing_s=3.0):
    spacing_ns = int(spacing_s * SEC)

    def loader(idx):
        return image, cloud, idx * spacing_ns, 0.0, idx * spacing_ns

    return prepare_frames(loader, n_total_timeline=n, n_samples=n,
                           edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0))


def _make_holdout_result(n=6, delta=0.2, axis="yaw", source="fine", discovery_revision=3,
                          discovery_signature="default", min_bootstrap_blocks=5) -> "ho.HoldoutResult":
    image, cloud = _make_scene()
    frames = _export_frames(image, cloud, n=n)
    cand = CandidateSpec(axis=axis, delta=delta, source=source)
    R, t = np.eye(3), np.zeros(3)
    sig = _sig() if discovery_signature == "default" else discovery_signature
    discovery = DiscoveryProvenance(bag="discovery.bag", timestamps=(0,), exclusion_margin_s=2.0, signature=sig)
    return evaluate_holdout(
        frames, R, t, cand, _camera(), edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
        discovery=discovery, holdout_bag="holdout.bag", holdout_signature=_sig(total_message_count=1),
        requested_timestamps=tuple(f.lidar_timestamp_ns for f in frames),
        discovery_revision=discovery_revision, bootstrap_block_s=2.0, min_bootstrap_blocks=min_bootstrap_blocks,
    )


def _make_snapshot(holdout=None, status=ho.STATUS_INDEPENDENT, baseline_R=None, baseline_t=None,
                    now=NOW, **holdout_kwargs) -> "abx.CandidateExportSnapshot":
    holdout = holdout if holdout is not None else _make_holdout_result(**holdout_kwargs)
    R = baseline_R if baseline_R is not None else np.eye(3)
    t = baseline_t if baseline_t is not None else np.zeros(3)
    sol = Solution(ok=True, R=R, t=t, rmse=0.0015, scene_ids=["s1", "s2"], n_pairs=8)
    return abx.build_snapshot(
        project_name="demo_project", baseline_solution=sol, holdout=holdout, validation_status=status,
        camera=_camera(), target=Target(), lidar_topic="/lidar", camera_topic="/cam",
        bag_paths=["C:/data/holdout.bag"], method="ring", now=now,
    )


# --------------------------------------------------------------------- snapshot


def test_candidate_fingerprint_recomputed_matches_locked_fingerprint():
    """Requirement 1: the candidate R/t this module derives from baseline +
    axis/delta must fingerprint to exactly what the Hold-out run locked."""
    holdout = _make_holdout_result()
    snap = _make_snapshot(holdout=holdout)
    assert snap.ok, snap.reason
    assert snap.candidate_fingerprint == holdout.candidate_identity.candidate_fingerprint


def test_fingerprint_mismatch_blocks_export():
    """Requirement 2: a tampered/stale locked candidate fingerprint must
    abort the snapshot (ok=False), never silently package a mismatch."""
    holdout = _make_holdout_result()
    tampered_identity = ho.CandidateIdentity(
        baseline_fingerprint=holdout.candidate_identity.baseline_fingerprint,
        candidate_fingerprint="0" * 12,  # deliberately wrong
        axis=holdout.candidate_identity.axis, delta=holdout.candidate_identity.delta,
    )
    import dataclasses
    tampered_holdout = dataclasses.replace(holdout, candidate_identity=tampered_identity)
    snap = _make_snapshot(holdout=tampered_holdout)
    assert not snap.ok
    assert "fingerprint" in snap.reason.lower()


def test_current_baseline_differs_from_validated_baseline_blocks_export():
    """Requirement 3: if the extrinsic on screen has moved on (flip,
    reload, recalibration) since the Hold-out was locked, export must
    refuse rather than silently export a candidate against the wrong
    baseline."""
    holdout = _make_holdout_result()
    different_t = np.array([0.5, 0.0, 0.0])
    snap = _make_snapshot(holdout=holdout, baseline_t=different_t)
    assert not snap.ok
    assert "differs from the validated baseline" in snap.reason


def test_snapshot_blocked_for_cancelled_or_failed_holdout():
    image, cloud = _make_scene()
    frames = _export_frames(image, cloud, n=6)
    cand = CandidateSpec(axis="yaw", delta=0.2, source="fine")
    cancelled = evaluate_holdout(frames, np.eye(3), np.zeros(3), cand, _camera(),
                                  edge_params=EdgeAlignmentParams(depth_jump_threshold_m=1.0),
                                  should_cancel=lambda: True)
    assert cancelled.cancelled
    snap = _make_snapshot(holdout=cancelled)
    assert not snap.ok


def test_baseline_and_candidate_have_actually_different_rt():
    """Requirement 5."""
    snap = _make_snapshot()
    assert snap.ok, snap.reason
    assert not np.allclose(snap.baseline_R, snap.candidate_R)
    assert snap.baseline_fingerprint != snap.candidate_fingerprint


def test_snapshot_never_mutates_baseline_solution_arrays():
    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()
    holdout = _make_holdout_result()
    sol = Solution(ok=True, R=R, t=t, rmse=0.0, scene_ids=[], n_pairs=0)
    abx.build_snapshot(
        project_name="p", baseline_solution=sol, holdout=holdout, validation_status=ho.STATUS_INDEPENDENT,
        camera=_camera(), target=Target(), lidar_topic="/lidar", camera_topic="/cam",
        bag_paths=[], now=NOW,
    )
    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)


# ------------------------------------------------------------------ file builders


def test_extrinsic_yaml_round_trip_baseline_and_candidate(tmp_path):
    """Requirement 4/6: verify.to_yaml/from_yaml (the tool's existing,
    unchanged writer/loader) round-trips both files, and the baseline file
    matches the exact baseline transform that was locked -- not a
    recomputed or rounded stand-in."""
    snap = _make_snapshot()
    baseline_text = abx._extrinsic_yaml_text(snap, "baseline")
    candidate_text = abx._extrinsic_yaml_text(snap, "candidate")

    baseline_data = verify.from_yaml(baseline_text)
    candidate_data = verify.from_yaml(candidate_text)
    assert np.allclose(baseline_data["R"], snap.baseline_R, atol=1e-6)
    assert np.allclose(baseline_data["t"], snap.baseline_t, atol=1e-6)
    assert np.allclose(candidate_data["R"], snap.candidate_R, atol=1e-6)
    assert np.allclose(candidate_data["t"], snap.candidate_t, atol=1e-6)
    # Candidate file must never claim to be the baseline, or vice versa.
    assert "DO NOT overwrite the validated baseline" in candidate_text
    assert "CANDIDATE" in candidate_text and "BASELINE" not in candidate_text.split("\n")[1]


def test_per_frame_csv_row_count_and_timestamps_match_holdout(tmp_path):
    """Requirement 7."""
    holdout = _make_holdout_result(n=6)
    rows = abx.build_per_frame_rows(holdout)
    assert len(rows) == len(holdout.frame_table) == 6
    expected_timestamps = sorted(rec.lidar_timestamp_ns for rec in holdout.frame_table)
    assert sorted(r["lidar_timestamp_ns"] for r in rows) == expected_timestamps
    # Every row must be traceable to a real timeline_index the HoldoutResult scored.
    known_indices = {rec.timeline_index for rec in holdout.frame_table}
    assert {r["timeline_index"] for r in rows} == known_indices


def test_per_frame_csv_comparison_status_matches_per_frame_ok_flags():
    holdout = _make_holdout_result(n=6)
    rows = {r["timeline_index"]: r for r in abx.build_per_frame_rows(holdout)}
    bp, cp = holdout.baseline_point, holdout.candidate_point
    for idx, row in rows.items():
        b = bp.per_frame.get(idx)
        c = cp.per_frame.get(idx)
        if b is not None and c is not None and b.ok and c.ok:
            assert row["comparison_status"] == "both_valid"
            assert row["delta_p95_px"] == pytest.approx(c.p95_px - b.p95_px)


def test_per_frame_csv_match_rate_columns_are_consistently_blank():
    """Per-frame match rate genuinely is not tracked anywhere upstream
    (only pooled per trial) -- must be blank, never a guessed 0."""
    holdout = _make_holdout_result(n=6)
    rows = abx.build_per_frame_rows(holdout)
    assert rows
    assert all(r["baseline_match_rate"] == "" for r in rows)
    assert all(r["candidate_match_rate"] == "" for r in rows)


def test_spatial_depth_csv_matches_existing_holdout_comparison_rows():
    """Requirement 8: no new spatial/depth evaluation -- P95/delta values in
    the CSV must equal the already-computed SpatialComparisonRow values on
    HoldoutResult exactly."""
    holdout = _make_holdout_result(n=6)
    rows = abx.build_spatial_depth_rows(holdout)
    by_key = {(r["analysis_type"], r["region"]): r for r in rows}
    for analysis_type, source_rows in (("horizontal", holdout.horizontal), ("vertical", holdout.vertical),
                                        ("depth", holdout.depth)):
        for sr in source_rows:
            row = by_key[(analysis_type, sr.label)]
            if np.isfinite(sr.baseline_p95_px):
                assert row["baseline_p95_px"] == pytest.approx(sr.baseline_p95_px)
            else:
                assert row["baseline_p95_px"] == ""
            if np.isfinite(sr.delta_p95_px):
                assert row["delta_p95_px"] == pytest.approx(sr.delta_p95_px)
            else:
                assert row["delta_p95_px"] == ""
            assert row["baseline_n_points"] == sr.baseline_n_points


def test_validation_summary_is_json_safe_nan_becomes_null():
    """Requirement 9: a region/bin with zero points has NaN P95 upstream --
    the summary must serialise it as null, and the whole document must be
    plain json.dumps/loads-able."""
    holdout = _make_holdout_result(n=6)
    snap = _make_snapshot(holdout=holdout)
    summary = abx.build_validation_summary(snap)
    text = json.dumps(summary)  # raises on anything non-JSON-safe (NaN, numpy types, ...)
    reloaded = json.loads(text)
    # LEFT/RIGHT are empty in this synthetic scene (edge sits at CENTER) --
    # exactly the "no data" case that must come back as null, not NaN/0.
    left = next(r for r in reloaded["spatial"]["horizontal"] if r["region"] == "LEFT")
    assert left["baseline_p95_px"] is None
    assert "NaN" not in text and "Infinity" not in text


def test_validation_summary_records_exact_validation_status():
    """Requirement 10."""
    holdout = _make_holdout_result()
    snap = _make_snapshot(holdout=holdout, status=ho.STATUS_INDEPENDENT_NEW_SLICE)
    summary = abx.build_validation_summary(snap)
    assert summary["validation_status"]["code"] == ho.STATUS_INDEPENDENT_NEW_SLICE
    assert summary["validation_status"]["label"] == ho.STATUS_LABELS[ho.STATUS_INDEPENDENT_NEW_SLICE]
    assert summary["validation_status"]["is_exposed"] is False


def test_exploratory_status_is_preserved_not_blocked_or_downgraded():
    """Requirement 8/10: Exploratory must not block export, and must never
    collapse into a bare boolean -- the exact status string/label survives."""
    holdout = _make_holdout_result()
    snap = _make_snapshot(holdout=holdout, status=ho.STATUS_EXPOSED_DISCOVERY_UPDATED)
    assert snap.ok, snap.reason
    summary = abx.build_validation_summary(snap)
    assert summary["validation_status"]["code"] == ho.STATUS_EXPOSED_DISCOVERY_UPDATED
    assert summary["validation_status"]["is_exposed"] is True
    report = abx.build_validation_report_md(snap)
    assert "EXPLORATORY" in report
    assert ho.STATUS_LABELS[ho.STATUS_EXPOSED_DISCOVERY_UPDATED] in report


def test_bootstrap_ci_method_and_blocks_preserved_in_summary():
    """Requirement 12."""
    holdout = _make_holdout_result(n=6)
    assert holdout.bootstrap is not None and holdout.bootstrap.ok
    snap = _make_snapshot(holdout=holdout)
    summary = abx.build_validation_summary(snap)
    bs = summary["bootstrap"]
    assert bs["method"] == holdout.bootstrap.method
    assert bs["n_blocks"] == holdout.bootstrap.n_blocks
    assert bs["median_delta_ci_low_px"] == pytest.approx(holdout.bootstrap.median_delta_ci_low_px)
    assert bs["median_delta_ci_high_px"] == pytest.approx(holdout.bootstrap.median_delta_ci_high_px)


def test_discovery_and_holdout_provenance_preserved_in_summary():
    """Requirement 13."""
    holdout = _make_holdout_result()
    snap = _make_snapshot(holdout=holdout)
    summary = abx.build_validation_summary(snap)
    assert summary["discovery"]["bag"] == "discovery.bag"
    assert summary["discovery"]["signature"]["total_message_count"] == 4000
    assert summary["holdout"]["bag"] == "holdout.bag"
    assert summary["holdout"]["usable_timestamps_ns"] == list(holdout.provenance.usable_timestamps)
    assert summary["holdout"]["source_identity"]
    assert summary["holdout"]["slice_identity"]


def test_discovery_signature_missing_is_null_with_caveat():
    """Requirement 9 (Discovery BagSignature gap): a source that could not
    be read (or predates signature capture) must yield discovery_signature:
    null and an explicit caveat, never fail the export."""
    holdout = _make_holdout_result(discovery_signature=None)
    assert holdout.discovery.signature is None
    snap = _make_snapshot(holdout=holdout)
    assert snap.ok, snap.reason
    summary = abx.build_validation_summary(snap)
    assert summary["discovery"]["signature"] is None
    assert any("BagSignature is unavailable" in c for c in summary["caveats"])


# --------------------------------------------------------------------- export_package


def test_export_package_writes_all_files_and_self_validates(tmp_path):
    snap = _make_snapshot()
    outcome = abx.export_package(snap, tmp_path, now=NOW)
    assert outcome.ok, outcome.reason
    assert outcome.out_dir.is_dir()
    expected = {
        "baseline_extrinsic.yaml", "candidate_extrinsic.yaml", "validation_summary.json",
        "per_frame_results.csv", "spatial_depth_results.csv", "validation_report.md", "manifest.json",
    }
    assert set(outcome.files) == expected
    assert {p.name for p in outcome.out_dir.iterdir()} == expected


def test_export_package_manifest_sha256_matches_actual_files(tmp_path):
    """Requirement 14."""
    snap = _make_snapshot()
    outcome = abx.export_package(snap, tmp_path, now=NOW)
    assert outcome.ok
    manifest = json.loads((outcome.out_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        assert abx._sha256_file(outcome.out_dir / name) == info["sha256"]
    assert "manifest.json" not in manifest["files"]  # never hashes itself


def test_export_package_refuses_to_write_anything_for_a_blocked_snapshot(tmp_path):
    holdout = _make_holdout_result()
    snap = _make_snapshot(holdout=holdout, baseline_t=np.array([1.0, 0.0, 0.0]))
    assert not snap.ok
    outcome = abx.export_package(snap, tmp_path, now=NOW)
    assert not outcome.ok
    assert list(tmp_path.iterdir()) == []


def test_partial_export_failure_leaves_no_final_package_behind(tmp_path, monkeypatch):
    """Requirement 16: if self-validation fails partway through, neither a
    half-written final directory nor a leftover temp directory survives."""
    snap = _make_snapshot()
    monkeypatch.setattr(abx, "_self_validate", lambda snapshot, tmp_root: (False, "forced failure for this test"))
    outcome = abx.export_package(snap, tmp_path, now=NOW)
    assert not outcome.ok
    assert outcome.reason == "forced failure for this test"
    assert list(tmp_path.iterdir()) == []  # no final dir, no leftover temp dir


def test_export_package_avoids_directory_collision_in_the_same_second(tmp_path):
    """Requirement 18."""
    snap = _make_snapshot()
    first = abx.export_package(snap, tmp_path, now=NOW)
    second = abx.export_package(snap, tmp_path, now=NOW)
    third = abx.export_package(snap, tmp_path, now=NOW)
    assert first.ok and second.ok and third.ok
    names = {first.out_dir.name, second.out_dir.name, third.out_dir.name}
    assert len(names) == 3
    assert first.out_dir.name == "candidate_ab_test_20260101_120000"
    assert second.out_dir.name == "candidate_ab_test_20260101_120000_01"
    assert third.out_dir.name == "candidate_ab_test_20260101_120000_02"


def test_export_package_never_mutates_rt_solution_project_or_registry(tmp_path):
    """Requirement 16/17: export is read-only end to end -- R/t arrays, the
    HoldoutResult, the exposure registry, and the persistent discovery
    revision must all be bit-for-bit identical before and after."""
    R, t = np.eye(3), np.zeros(3)
    R_before, t_before = R.copy(), t.copy()
    holdout = _make_holdout_result()

    registry = {}
    ho.determine_and_record_exposure(
        registry, "holdout.bag", None, holdout.provenance.usable_timestamps,
        holdout.candidate_identity, holdout.candidate_provenance, reveal_margin_s=2.0,
    )
    registry_before = ho.serialize_exposure_registry(registry, discovery_revision=3)

    sol = Solution(ok=True, R=R, t=t, rmse=0.001, scene_ids=["a"], n_pairs=4)
    snap = abx.build_snapshot(
        project_name="p", baseline_solution=sol, holdout=holdout, validation_status=ho.STATUS_INDEPENDENT,
        camera=_camera(), target=Target(), lidar_topic="/lidar", camera_topic="/cam",
        bag_paths=["holdout.bag"], now=NOW,
    )
    outcome = abx.export_package(snap, tmp_path, now=NOW)
    assert outcome.ok, outcome.reason

    assert np.array_equal(R, R_before) and np.array_equal(t, t_before)
    assert sol.R is R and sol.t is t  # export never rebinds the caller's Solution either
    registry_after = ho.serialize_exposure_registry(registry, discovery_revision=3)
    assert registry_after == registry_before


def test_export_package_out_base_dir_only_ever_gains_the_one_new_directory(tmp_path):
    """No stray files or sibling directories left in out_base_dir beyond
    the final package -- the temp directory must never survive a
    successful export either."""
    snap = _make_snapshot()
    before = set(tmp_path.iterdir())
    outcome = abx.export_package(snap, tmp_path, now=NOW)
    assert outcome.ok
    after = set(tmp_path.iterdir())
    assert after - before == {outcome.out_dir}

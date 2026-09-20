from __future__ import annotations

import json

import numpy as np

from gui.core.evaluation.candidate_projection import (
    CandidateFrameMetric,
    CandidateProjectionEvidence,
    ValidationFrameSpec,
    _aggregate_by_candidate,
    _organized_lidar_edge_indices,
    _pairwise_wins,
    _ring_lidar_edge_indices,
    _stratified_lidar_edge_subsample,
    export_projection_evidence,
    select_validation_frames_from_board_csv,
)


def test_select_validation_frames_uses_board_to_nearest_camera_header(tmp_path):
    path = tmp_path / "board_pairing.csv"
    path.write_text(
        "\n".join([
            "frame_index,status,lidar_record_ns,board_median_ns,nearest_camera_header_ns,nearest_offset_ms,board_timestamp_valid",
            "10,MOVING,1000,900,910,10.0,1",
            "11,MOVING,2000,1900,1935,35.0,1",
            "12,MOVING,3000,2900,2990,90.0,1",
            "13,MOVING,4000,3900,3901,1.0,0",
        ]),
        encoding="utf-8",
    )

    frames = select_validation_frames_from_board_csv(path, n_frames=2, max_abs_offset_ms=60.0)

    assert [f.frame_index for f in frames] == [10, 11]
    assert frames[0].timestamp_source == "board_median_to_nearest_camera_header"
    assert frames[0].camera_header_ns == 910
    assert frames[0].board_median_ns == 900


def test_projection_aggregate_and_pairwise_wins():
    rows = [
        CandidateFrameMetric("NORMAL", 1, True, matched_mean_px=2.0, matched_p95_px=5.0, match_rate=0.8,
                             within_5px=0.6, within_10px=0.9),
        CandidateFrameMetric("NORMAL", 2, True, matched_mean_px=3.0, matched_p95_px=6.0, match_rate=0.7,
                             within_5px=0.5, within_10px=0.8),
        CandidateFrameMetric("HALF_TURN", 1, True, matched_mean_px=4.0, matched_p95_px=8.0, match_rate=0.6,
                             within_5px=0.4, within_10px=0.7),
        CandidateFrameMetric("HALF_TURN", 2, True, matched_mean_px=2.5, matched_p95_px=7.0, match_rate=0.9,
                             within_5px=0.7, within_10px=0.95),
    ]

    agg = _aggregate_by_candidate(rows)
    wins = _pairwise_wins(rows, "NORMAL", "HALF_TURN")

    assert agg["NORMAL"]["matched_mean_median_px"] == 2.5
    assert wins["matched_mean"]["NORMAL"] == 1
    assert wins["matched_mean"]["HALF_TURN"] == 1
    assert wins["matched_p95"]["NORMAL"] == 2
    assert wins["match_rate"]["HALF_TURN"] == 1


def test_projection_evidence_export_writes_json_and_csv(tmp_path):
    evidence = CandidateProjectionEvidence(
        validation_frames=[
            ValidationFrameSpec(
                frame_index=1,
                lidar_record_ns=100,
                camera_record_ns=110,
                camera_header_ns=111,
                timestamp_source="board_median_to_nearest_camera_header",
                selection_reason="test",
            )
        ],
        per_frame=[
            CandidateFrameMetric("NORMAL", 1, True, matched_mean_px=2.0, matched_p95_px=5.0, match_rate=0.8),
        ],
        audits=[],
        aggregate={"NORMAL": {"matched_mean_median_px": 2.0}},
        pairwise_wins={},
        candidate_summaries={"NORMAL": {"ok": True}},
    )

    out = export_projection_evidence(evidence, tmp_path)
    data = json.loads((out / "projection_evidence.json").read_text(encoding="utf-8"))

    assert data["validation_frames"][0]["frame_index"] == 1
    assert (out / "projection_frame_metrics.csv").exists()
    assert (out / "projection_stage_audit.csv").exists()
    assert (out / "normal_vs_half_turn.csv").exists()


def test_organized_lidar_edge_indices_keep_near_side_before_subsampling():
    xyz = np.zeros((2, 4, 3), dtype=np.float32)
    for r in range(2):
        for c in range(4):
            x = 1.0 if c < 2 else 2.0
            xyz[r, c] = [x, float(c) * 0.01, float(r) * 0.01]
    flat = xyz.reshape(-1, 3)
    finite = np.ones(len(flat), dtype=bool)

    edge_idx = _organized_lidar_edge_indices(flat, (2, 4), finite, 0.3)

    assert set(edge_idx.tolist()) == {1, 5}


def test_stratified_lidar_edge_subsample_is_deterministic_and_balanced():
    pts = []
    for az in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        for radius in (1.0, 2.0, 3.0):
            pts.append([np.cos(az) * radius, np.sin(az) * radius, 0.1 * radius])
    xyz = np.asarray(pts, dtype=np.float32)
    idx = np.arange(len(xyz), dtype=np.int64)

    a = _stratified_lidar_edge_subsample(xyz, idx, 12, 12, 2, 123)
    b = _stratified_lidar_edge_subsample(xyz, idx, 12, 12, 2, 123)

    assert a.tolist() == b.tolist()
    assert len(a) == 12
    assert len(set(a.tolist())) == 12


def test_ring_lidar_edge_indices_reconstruct_unorganized_scan():
    rows = []
    rings = []
    for ring in range(2):
        for col in range(4):
            x = 1.0 if col < 2 else 2.0
            rows.append([x, col * 0.01, ring * 0.01])
            rings.append(ring)
    xyz = np.asarray(rows, dtype=np.float32)
    finite = np.ones(len(xyz), dtype=bool)

    edge_idx, shape = _ring_lidar_edge_indices(xyz, np.asarray(rings), finite, 0.3)

    assert shape == (2, 4)
    assert set(edge_idx.tolist()) == {1, 5}

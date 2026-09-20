from __future__ import annotations

import numpy as np

from gui.core.detect_lidar import DetectParams
from gui.core.evaluation.plane_rerank_diagnostic import (
    BoardSize,
    RerankConfig,
    _AllRegion,
    current_plane_selection_report,
    extract_plane_candidates,
    score_candidate,
)
from gui.core.project import Target


class _Det:
    def __init__(self, n_circles=0, n_edges=0):
        self.ok = n_circles == 4
        self.reason = "synthetic"
        self.edges = np.zeros((n_edges, 3), np.float32)
        self.n_clusters = n_circles
        self.radii = [0.1] * n_circles
        self.centers = np.zeros((n_circles, 3), np.float32) if n_circles else None

    @property
    def n_circles(self):
        return 0 if self.centers is None else len(self.centers)


def _plane(width, height, n=400, z=0.0, jitter=0.001):
    rng = np.random.default_rng(1)
    x = rng.uniform(-width / 2, width / 2, n)
    y = rng.uniform(-height / 2, height / 2, n)
    zz = np.full(n, z) + rng.normal(0, jitter, n)
    return np.column_stack([x, y, zz]).astype(np.float32)


def test_current_plane_selection_report_documents_single_largest_plane():
    report = current_plane_selection_report()

    assert report["ransac_runs_per_detection"] == 1
    assert report["largest_plane_selected"] is True
    assert report["extent_size_aspect_used_for_selection"] is False
    assert report["boundary_or_circle_used_for_selection"] is False


def test_wall_plane_plus_board_plane_scores_board_higher():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    wall = _plane(5.0, 2.0, n=1200)
    board = _plane(0.74, 0.64, n=250)

    wall_score = score_candidate(wall, _Det(n_circles=0, n_edges=8), [0, 0, 1], 0.0, 0.002, len(wall), cfg)
    board_score = score_candidate(board, _Det(n_circles=2, n_edges=60), [0, 0, 1], 0.0, 0.004, len(board), cfg)

    assert board_score["total_score"] > wall_score["total_score"]
    assert board_score["size_similarity"] > wall_score["size_similarity"]


def test_long_strip_gets_low_aspect_score():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    strip = _plane(3.74, 0.30, n=500)

    score = score_candidate(strip, _Det(), [0, 0, 1], 0.0, 0.002, len(strip), cfg)

    assert score["aspect_similarity"] < 0.2


def test_width_height_orientation_swap_scores_well():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    swapped = _plane(0.64, 0.74, n=400)

    score = score_candidate(swapped, _Det(n_circles=1, n_edges=40), [0, 0, 1], 0.0, 0.002, len(swapped), cfg)

    assert score["size_similarity"] > 0.85
    assert score["aspect_similarity"] > 0.85


def test_perfect_board_sized_plane_scores_high():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    board = _plane(0.74, 0.64, n=600)

    score = score_candidate(board, _Det(n_circles=4, n_edges=80), [0, 0, 1], 0.0, 0.001, len(board), cfg)

    assert score["total_score"] > 0.75


def test_flat_wall_does_not_beat_board_just_because_rms_is_good():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    wall = _plane(6.0, 2.5, n=2000, jitter=0.0001)
    board = _plane(0.74, 0.64, n=300, jitter=0.004)

    wall_score = score_candidate(wall, _Det(), [0, 0, 1], 0.0, 0.0001, len(wall), cfg)
    board_score = score_candidate(board, _Det(n_circles=1, n_edges=50), [0, 0, 1], 0.0, 0.004, len(board), cfg)

    assert board_score["total_score"] > wall_score["total_score"]


def test_circle_support_increases_score():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    board = _plane(0.74, 0.64, n=400)

    no_circle = score_candidate(board, _Det(n_circles=0, n_edges=40), [0, 0, 1], 0.0, 0.002, len(board), cfg)
    two_circles = score_candidate(board, _Det(n_circles=2, n_edges=40), [0, 0, 1], 0.0, 0.002, len(board), cfg)

    assert two_circles["total_score"] > no_circle["total_score"]


def test_no_valid_candidate_for_too_few_points():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64), min_remaining_points=50)
    target = Target()
    params = DetectParams(method="occupancy")

    current, candidates = extract_plane_candidates(_plane(0.74, 0.64, n=20), _AllRegion(), target, params, config=cfg)

    assert not current.ok
    assert candidates == []


def test_production_detector_result_is_not_mutated():
    cfg = RerankConfig(board_size=BoardSize(0.74, 0.64))
    target = Target()
    params = DetectParams(method="occupancy")
    cloud = _plane(0.74, 0.64, n=200)

    current, _ = extract_plane_candidates(cloud, _AllRegion(), target, params, config=cfg)

    assert len(current.filtered) <= len(cloud)
    assert cloud.shape == (200, 3)


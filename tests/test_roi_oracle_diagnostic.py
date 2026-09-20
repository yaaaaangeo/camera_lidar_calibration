from __future__ import annotations

import csv

import numpy as np

from gui.core.evaluation.roi_oracle_diagnostic import (
    RoiRunMetrics,
    classify_roi_case,
    export_manual_roi_template,
    load_manual_roi_csv,
    projection_metrics,
    summarize_comparisons,
    RoiComparisonFrame,
)
from gui.core.project import Camera, FilterBox
from gui.core.solve import Solution


def test_manual_roi_csv_reuses_filter_box_schema(tmp_path):
    path = tmp_path / "manual.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_index", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "yaw", "pitch", "roll",
        ])
        writer.writeheader()
        writer.writerow({
            "frame_index": 407,
            "x_min": 1, "x_max": 2, "y_min": -1, "y_max": 1,
            "z_min": 0, "z_max": 2, "yaw": 10, "pitch": 0, "roll": -5,
        })

    rois = load_manual_roi_csv(path)

    assert 407 in rois
    assert isinstance(rois[407], FilterBox)
    assert rois[407].yaw == 10


def test_manual_roi_template_prefills_auto_box(tmp_path):
    box = FilterBox(x_min=1, x_max=2, y_min=3, y_max=4, z_min=5, z_max=6, yaw=7, pitch=8, roll=9)
    path = export_manual_roi_template(tmp_path / "template.csv", [407], {407: box})

    text = path.read_text(encoding="utf-8")

    assert "407" in text
    assert "1.000000" in text
    assert "9.000000" in text


def test_classify_roi_a_when_manual_improves_from_bad_auto():
    auto = RoiRunMetrics(label="AUTO", board_like=False, best_circle_count=0, full_target_success=False)
    manual = RoiRunMetrics(label="MANUAL", board_like=True, best_boundary_count=20, best_circle_count=2)

    assert classify_roi_case(auto, manual) == "ROI-A_AUTO_ROI_MAIN_CAUSE"


def test_classify_roi_b_when_manual_board_like_but_circle_fails():
    auto = RoiRunMetrics(label="AUTO", board_like=False)
    manual = RoiRunMetrics(label="MANUAL", board_like=True, best_circle_count=0)

    assert classify_roi_case(auto, manual) == "ROI-B_POST_ROI_CROP_BOUNDARY_CIRCLE"


def test_classify_roi_c_when_manual_has_no_board_plane():
    auto = RoiRunMetrics(label="AUTO", board_like=False)
    manual = RoiRunMetrics(label="MANUAL", board_like=False)

    assert classify_roi_case(auto, manual) == "ROI-C_NO_BOARD_PLANE_IN_MANUAL_ROI"


def test_projection_metrics_reports_overlap_and_centroid_distance():
    camera = Camera(fx=100, fy=100, cx=50, cy=50)
    sol = Solution(ok=True, R=np.eye(3), t=np.zeros(3))
    roi = FilterBox(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1, z_min=1.0, z_max=1.2)

    metrics = projection_metrics((100, 100), camera, sol, roi, (40, 40, 60, 60))

    assert metrics.roi_bbox is not None
    assert metrics.centroid_distance_px < 1.0
    assert metrics.overlap_ratio > 0.0


def test_summarize_comparisons_counts_auto_and_manual():
    frames = [
        RoiComparisonFrame(
            frame_index=1,
            auto=RoiRunMetrics(label="AUTO", board_like=False, best_circle_count=0),
            manual=RoiRunMetrics(label="MANUAL", board_like=True, best_boundary_count=10, best_circle_count=4, full_target_success=True),
            roi_case="ROI-A_AUTO_ROI_MAIN_CAUSE",
        ),
        RoiComparisonFrame(
            frame_index=2,
            auto=RoiRunMetrics(label="AUTO", board_like=True, best_boundary_count=5, best_circle_count=1),
            manual=RoiRunMetrics(label="MANUAL", board_like=False),
            roi_case="ROI-C_NO_BOARD_PLANE_IN_MANUAL_ROI",
        ),
    ]

    summary = summarize_comparisons(frames)

    assert summary.auto_board_like == 1
    assert summary.manual_board_like == 1
    assert summary.manual_circle_ge4 == 1
    assert summary.case_counts["ROI-A_AUTO_ROI_MAIN_CAUSE"] == 1


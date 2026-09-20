from __future__ import annotations

import numpy as np

from gui.core.evaluation.target_geometry_diagnostic import (
    PREVIOUS_FAST_CALIB_TARGET,
    compare_to_previous,
    extent_distribution,
    parameter_trace,
    ratio_stats,
)
from gui.core.project import Target


def test_parameter_trace_documents_units_and_conventions():
    rows = parameter_trace(Target())
    by_name = {r.parameter: r for r in rows}

    assert by_name["circle_radius"].unit == "m"
    assert "radius" in by_name["circle_radius"].convention
    assert "center-to-center" in by_name["delta_width_circles"].convention


def test_previous_geometry_ratio_is_consistent():
    rows = compare_to_previous(Target(), PREVIOUS_FAST_CALIB_TARGET)
    stats = ratio_stats(rows)

    assert np.isclose(stats["mean"], 1.6835016835, rtol=1e-3)
    assert stats["max"] - stats["min"] < 0.002


def test_extent_distribution():
    dist = extent_distribution([
        {"width_m": 1.0, "height_m": 0.5, "area_m2": 0.5},
        {"width_m": 2.0, "height_m": 1.0, "area_m2": 2.0},
    ])

    assert dist["n"] == 2
    assert np.isclose(dist["width_m"]["median"], 1.5)

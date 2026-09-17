"""Breaking `edge_alignment`'s one frame-wide mean into where the error sits.

A single mean hides the difference between "consistently a little off everywhere"
and "fine up close, falling apart past 30 m" -- both can produce the same
number. This module re-groups `EdgeAlignmentResult`'s existing per-point arrays
by depth and by image region; it never touches the cloud, the image, or the
projection again; it only reduces arrays that `evaluate_edge_alignment` already
produced.

Horizontal (LEFT/CENTER/RIGHT) and vertical (TOP/CENTER/BOTTOM) position are
kept as two separate three-way splits rather than merged into one 3x3 grid,
because they answer different questions and merging them would need nine times
the points per cell to say as much: whether error concentrates on one side at
all (often a yaw-axis symptom) is visible from three numbers, and so is whether
it concentrates top or bottom (pitch, or a vertical offset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gui.core.evaluation.edge_alignment import EdgeAlignmentResult

DEPTH_BIN_EDGES = (0.0, 10.0, 20.0, 30.0, 50.0, float("inf"))
DEPTH_BIN_LABELS = ("0-10m", "10-20m", "20-30m", "30-50m", "50m+")

HORIZONTAL_REGIONS = ("LEFT", "CENTER", "RIGHT")
VERTICAL_REGIONS = ("TOP", "CENTER", "BOTTOM")


@dataclass
class BinStats:
    label: str
    mean_px: float
    median_px: float
    p95_px: float
    std_px: float
    n_points: int
    n_matched: int
    n_unmatched: int


def _bin_stats(label: str, errors: np.ndarray, matched: Optional[np.ndarray]) -> BinStats:
    n = errors.shape[0]
    if n == 0:
        return BinStats(label, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, 0)
    n_matched = int(matched.sum()) if matched is not None else n
    return BinStats(
        label=label,
        mean_px=float(np.mean(errors)),
        median_px=float(np.median(errors)),
        p95_px=float(np.percentile(errors, 95)),
        std_px=float(np.std(errors)),
        n_points=n,
        n_matched=n_matched,
        n_unmatched=n - n_matched,
    )


def bin_by_depth(depths_m: np.ndarray) -> np.ndarray:
    """Index into DEPTH_BIN_LABELS for each depth; right-open bins [0,10),
    [10,20), ..., [50, inf)."""
    idx = np.digitize(depths_m, DEPTH_BIN_EDGES[1:-1], right=False)
    return np.clip(idx, 0, len(DEPTH_BIN_LABELS) - 1)


def bin_by_horizontal(pixels_u: np.ndarray, image_width: int) -> np.ndarray:
    third = image_width / 3.0
    return np.clip((pixels_u // third).astype(np.int64), 0, 2)


def bin_by_vertical(pixels_v: np.ndarray, image_height: int) -> np.ndarray:
    third = image_height / 3.0
    return np.clip((pixels_v // third).astype(np.int64), 0, 2)


@dataclass
class SpatialAnalysisResult:
    depth_bins: dict       # label -> BinStats, DEPTH_BIN_LABELS order
    horizontal: dict       # label -> BinStats, HORIZONTAL_REGIONS order
    vertical: dict         # label -> BinStats, VERTICAL_REGIONS order


def analyze_spatial(result: EdgeAlignmentResult, image_width: int, image_height: int) -> Optional[SpatialAnalysisResult]:
    """Group `result`'s edge points by depth and by image region.

    Returns None when `result` has no per-point arrays to group (it failed, or
    predates this field) -- there is nothing to break down.
    """
    if not result.ok or result.edge_errors_px is None or result.edge_depths is None or result.edge_pixels is None:
        return None

    errors = result.edge_errors_px
    depths = result.edge_depths
    pixels = result.edge_pixels
    matched = result.edge_matched

    depth_idx = bin_by_depth(depths)
    h_idx = bin_by_horizontal(pixels[:, 0], image_width)
    v_idx = bin_by_vertical(pixels[:, 1], image_height)

    depth_bins = {}
    for i, label in enumerate(DEPTH_BIN_LABELS):
        sel = depth_idx == i
        depth_bins[label] = _bin_stats(label, errors[sel], matched[sel] if matched is not None else None)

    horizontal = {}
    for i, label in enumerate(HORIZONTAL_REGIONS):
        sel = h_idx == i
        horizontal[label] = _bin_stats(label, errors[sel], matched[sel] if matched is not None else None)

    vertical = {}
    for i, label in enumerate(VERTICAL_REGIONS):
        sel = v_idx == i
        vertical[label] = _bin_stats(label, errors[sel], matched[sel] if matched is not None else None)

    return SpatialAnalysisResult(depth_bins=depth_bins, horizontal=horizontal, vertical=vertical)

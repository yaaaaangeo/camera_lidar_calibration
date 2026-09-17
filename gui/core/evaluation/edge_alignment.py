"""How well the cloud's silhouettes line up with the image's, in pixels.

The target-fitting metrics in `gui.core.solve` (RMSE, leave-one-out, coverage)
only speak about the four scenes an extrinsic was fitted to. They cannot say
whether that extrinsic still holds on an ordinary stretch of road, because nothing
there has a known 3D position to compare against.

What an ordinary scene does have is edges: wherever a LiDAR sweep jumps from a
near surface to a far one, the near surface's outline is a silhouette, and a
silhouette is exactly what a camera photographs as an edge too. So this module
extracts both kinds of edge independently -- the cloud's from a depth
discontinuity, the image's from Canny -- and measures the pixel distance between
them. No target, no known-good 3D point: just "do the two sensors agree about
where things end", which is what the extrinsic is supposed to guarantee
everywhere, not only at the four holes it was solved from.

Takes plain `(pixels, depths)` arrays rather than a `gui.core.verify.Projection`
object on purpose, so this package never has to import `verify` (or PySide6)
and stays trivial to unit-test: the caller passes `projection.uv` and
`projection.depth`, which already carry every protection `project_cloud` applies
(behind-camera, off the sensor, past the distortion model's valid range, range
filter) -- this module trusts that and does not redo any of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class EdgeAlignmentParams:
    """Tunable knobs, gathered in one place so a caller can vary one without
    naming the rest. Defaults follow the values worked out on real scenes."""

    radius_px: float = 3.0
    depth_jump_threshold_m: float = 0.3
    min_neighbors: int = 3
    min_edge_points: int = 30
    canny_low: int = 50
    canny_high: int = 150
    correspondence_radii_px: tuple = (5.0, 10.0, 15.0)
    max_orientation_diff_deg: float = 30.0
    k_orientation: int = 6
    k_consistency: int = 5
    max_consistency_angle_deg: float = 45.0
    max_consistency_magnitude_ratio: float = 3.0
    min_consistency_magnitude_diff_px: float = 3.0
    min_consistency_displacement_for_angle_check_px: float = 1.5


@dataclass
class EdgeAlignmentResult:
    """Frame-level pixel-alignment numbers, plus every per-point array
    `spatial_analysis` needs to break them down further without redoing the
    projection or the edge extraction."""

    ok: bool
    reason: str = ""
    mean_px: float = float("nan")
    median_px: float = float("nan")
    p95_px: float = float("nan")
    max_px: float = float("nan")
    n_projected: int = 0
    n_edge_points: int = 0
    n_matched: int = 0
    n_unmatched: int = 0
    match_rate: float = float("nan")

    # Per LiDAR-edge-point, aligned arrays -- what spatial_analysis groups by
    # depth bin / image region without ever touching the cloud or image again.
    edge_pixels: Optional[np.ndarray] = None      # (K, 2) uv
    edge_depths: Optional[np.ndarray] = None      # (K,) camera-frame z, metres
    edge_errors_px: Optional[np.ndarray] = None   # (K,) penalised for unmatched
    edge_matched: Optional[np.ndarray] = None     # (K,) bool

    # Only populated when evaluate_edge_alignment(collect_timing=True) --
    # stage milliseconds, for judging whether a resolution needs a worker
    # thread rather than running on the GUI thread. Never touched otherwise,
    # so normal calls pay nothing for it.
    timing_ms: Optional[dict] = None


def extract_lidar_edge_points(
    pixels: np.ndarray,
    depths: np.ndarray,
    radius_px: float = 3.0,
    depth_jump_threshold_m: float = 0.3,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Which projected points sit on a depth discontinuity.

    A point qualifies when its `radius_px` neighbourhood (in image space, since
    that is what actually competes with an image edge) spans more depth than
    `depth_jump_threshold_m`, backed by at least `min_neighbors` other points --
    and only the near-side point of that neighbourhood is kept. The far side is
    occluded in the photo, so only the near surface's outline can ever match an
    image edge; keeping both would double-count the same discontinuity and let
    the far, physically-invisible half drag the error statistics around.

    Vectorised with `cKDTree.query_pairs` rather than a per-point
    `query_ball_point` loop: every within-radius pair is produced once as a
    plain array, and `np.maximum.at` / `np.minimum.at` reduce each point's
    neighbour depths in bulk. A Python loop here would run once per point per
    frame, and a frame can carry tens of thousands of candidates.
    """
    n = pixels.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    tree = cKDTree(pixels)
    pairs = tree.query_pairs(r=radius_px, output_type="ndarray")

    self_ids = np.arange(n, dtype=np.intp)
    if pairs.shape[0] == 0:
        point_ids = self_ids
        neighbor_ids = self_ids
    else:
        i, j = pairs[:, 0], pairs[:, 1]
        # Each unordered pair feeds both points' neighbour sets; every point is
        # also trivially its own neighbour (distance 0), matching what
        # query_ball_point would have included.
        point_ids = np.concatenate([i, j, self_ids])
        neighbor_ids = np.concatenate([j, i, self_ids])

    neighbor_counts = np.bincount(point_ids, minlength=n)
    neighbor_depths = depths[neighbor_ids]

    max_depth = np.full(n, -np.inf)
    min_depth = np.full(n, np.inf)
    np.maximum.at(max_depth, point_ids, neighbor_depths)
    np.minimum.at(min_depth, point_ids, neighbor_depths)

    depth_range = max_depth - min_depth
    return (
        (neighbor_counts >= min_neighbors)
        & (depth_range > depth_jump_threshold_m)
        & (depths <= min_depth + 1e-9)
    )


def extract_image_edges(image_bgr: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
    """Canny edge map. Accepts BGR or grayscale; returns 0/255 uint8."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return cv2.Canny(gray, canny_low, canny_high)


def _fail(reason: str, n_projected: int = 0, n_edge_points: int = 0) -> EdgeAlignmentResult:
    return EdgeAlignmentResult(
        ok=False, reason=reason, n_projected=n_projected, n_edge_points=n_edge_points,
    )


def evaluate_edge_alignment(
    image_bgr: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    params: EdgeAlignmentParams = EdgeAlignmentParams(),
    collect_timing: bool = False,
    cached_orientation_map=None,
) -> EdgeAlignmentResult:
    """The M2-style edge-alignment metric for one frame.

    `pixels`/`depths` are a projection's *visible* points -- e.g.
    `gui.core.verify.Projection.uv` / `.depth` -- already filtered by whatever
    protections that projection applies. This function does not re-derive them
    and does not know where they came from.

    A point that cannot find a correspondence is never dropped from the
    statistics: it is penalised at the largest search radius
    (`max(params.correspondence_radii_px)`), and counted in `n_unmatched` /
    reflected in `match_rate` -- silently excluding it would let a badly
    mismatched scene hide its worst points by having them simply not count.

    `collect_timing=True` attaches `result.timing_ms` (Canny / LiDAR edge
    extraction / correspondence / aggregation / total, in milliseconds) --
    meant for deciding whether a given resolution needs to move off the GUI
    thread, not for normal use, so it costs nothing when left off.

    `cached_orientation_map`, when given (an `edge_correspondence.EdgeOrientationMap`),
    skips both the Canny pass and the Sobel orientation/strength pass below --
    both depend only on `image_bgr`, never on `pixels`/`depths` or the
    extrinsic that produced them, so a caller re-evaluating the *same* image
    against many candidate extrinsics (perturbation sensitivity) computes it
    once, not once per candidate. `image_bgr` is still required in that case
    only because `match_lidar_edges_to_image`'s signature takes it; it does no
    image-processing work when a cached map is supplied.
    """
    from gui.core.evaluation.edge_correspondence import match_lidar_edges_to_image

    timing: dict = {}
    clock = time.perf_counter
    t_total0 = clock() if collect_timing else 0.0

    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    depths = np.asarray(depths, np.float64).reshape(-1)
    n_projected = pixels.shape[0]

    if n_projected == 0:
        return _fail("투영된 LiDAR 점이 없습니다.")

    t0 = clock() if collect_timing else 0.0
    edge_mask = extract_lidar_edge_points(
        pixels, depths,
        radius_px=params.radius_px,
        depth_jump_threshold_m=params.depth_jump_threshold_m,
        min_neighbors=params.min_neighbors,
    )
    if collect_timing:
        timing["lidar_edge_ms"] = (clock() - t0) * 1000
    n_edge_points = int(edge_mask.sum())
    if n_edge_points < params.min_edge_points:
        return _fail(
            f"LiDAR depth-edge 점이 {n_edge_points}개뿐입니다 (최소 {params.min_edge_points}개 필요).",
            n_projected=n_projected, n_edge_points=n_edge_points,
        )

    edge_pixels = pixels[edge_mask]
    edge_depths = depths[edge_mask]

    t0 = clock() if collect_timing else 0.0
    if cached_orientation_map is not None:
        edge_map = cached_orientation_map.edge_mask
    else:
        edge_map = extract_image_edges(image_bgr, params.canny_low, params.canny_high)
    if collect_timing:
        timing["canny_ms"] = (clock() - t0) * 1000
    if not edge_map.any():
        return _fail(
            "이미지에서 Canny edge를 찾지 못했습니다 (텍스처가 거의 없는 장면일 수 있습니다).",
            n_projected=n_projected, n_edge_points=n_edge_points,
        )

    t0 = clock() if collect_timing else 0.0
    correspondence = match_lidar_edges_to_image(
        edge_pixels, image_bgr,
        canny_low=params.canny_low, canny_high=params.canny_high,
        radii_px=params.correspondence_radii_px,
        max_orientation_diff_deg=params.max_orientation_diff_deg,
        k_orientation=params.k_orientation,
        k_consistency=params.k_consistency,
        max_consistency_angle_deg=params.max_consistency_angle_deg,
        max_consistency_magnitude_ratio=params.max_consistency_magnitude_ratio,
        min_consistency_magnitude_diff_px=params.min_consistency_magnitude_diff_px,
        min_consistency_displacement_for_angle_check_px=params.min_consistency_displacement_for_angle_check_px,
        edge_mask=edge_map,
        precomputed_orientation_map=cached_orientation_map,
    )
    if collect_timing:
        timing["correspondence_ms"] = (clock() - t0) * 1000
    errors_px = correspondence.distance_px
    matched = correspondence.matched
    n_matched = int(matched.sum())
    n_unmatched = n_edge_points - n_matched

    t0 = clock() if collect_timing else 0.0
    mean_px = float(np.mean(errors_px))
    median_px = float(np.median(errors_px))
    p95_px = float(np.percentile(errors_px, 95))
    max_px = float(np.max(errors_px))
    if collect_timing:
        timing["aggregation_ms"] = (clock() - t0) * 1000
        timing["total_ms"] = (clock() - t_total0) * 1000

    return EdgeAlignmentResult(
        ok=True,
        mean_px=mean_px,
        median_px=median_px,
        p95_px=p95_px,
        max_px=max_px,
        n_projected=n_projected,
        n_edge_points=n_edge_points,
        n_matched=n_matched,
        n_unmatched=n_unmatched,
        match_rate=n_matched / n_edge_points if n_edge_points else float("nan"),
        edge_pixels=edge_pixels,
        edge_depths=edge_depths,
        edge_errors_px=errors_px,
        edge_matched=matched,
        timing_ms=timing if collect_timing else None,
    )

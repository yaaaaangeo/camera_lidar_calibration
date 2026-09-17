"""Matching a LiDAR edge point to the *right* image edge, not just the nearest one.

The obvious way to score `extract_lidar_edge_points`' output is: distance-transform
the Canny map, sample it at every LiDAR edge pixel, done. That answers "how far
to the nearest edge of any kind" -- which is fooled the moment two edges sit a
few pixels apart with unrelated orientations, e.g. a lane-marking's edge next to
a shadow's. Nearest-distance cannot tell "the same physical boundary" from
"whatever happened to be a pixel closer", and a wrong extrinsic can hide behind
that: the cloud lands near *some* edge and reads as correct.

So a candidate is only accepted when it also agrees in orientation with the
LiDAR point's own local boundary direction, and among the survivors the
strongest image gradient wins -- a bold structural edge over a faint texture
edge that happens to be marginally closer. A last pass then checks each match's
displacement against its neighbours': a lone point pointing somewhere its
neighbours do not is more likely a coincidental snap than a real boundary, and
is demoted back to unmatched.

A point that survives none of this is *not* dropped -- see
`edge_alignment.evaluate_edge_alignment`'s docstring on why -- it is penalised
at the largest search radius and left for the caller to count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree

from gui.core.evaluation.edge_alignment import extract_image_edges


@dataclass
class EdgeOrientationMap:
    """Per-pixel edge candidates plus how each one runs and how strong it is."""

    edge_mask: np.ndarray        # (H, W) uint8, 0/255
    orientation_deg: np.ndarray  # (H, W) float, edge tangent mod 180 (undirected)
    strength: np.ndarray         # (H, W) float, Sobel gradient magnitude


def compute_edge_orientation_map(
    image_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_ksize: int = 3,
    edge_mask: Optional[np.ndarray] = None,
) -> EdgeOrientationMap:
    """Orientation is the edge's own tangent (the line's direction), not the
    gradient (which points across it, 90 degrees off) -- comparing tangents is
    what "does this LiDAR boundary run the same way as this image edge" means.
    Taken mod 180 since a line's direction is undirected: "up-right" and
    "down-left" at the same angle are the same edge.

    `edge_mask` lets a caller that already ran Canny on this image (as
    `evaluate_edge_alignment` does, to check for "no edges at all" up front)
    skip a second pass.
    """
    if edge_mask is None:
        edge_mask = extract_image_edges(image_bgr, canny_low, canny_high)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    gray_f = gray.astype(np.float64)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    strength = np.hypot(gx, gy)
    gradient_deg = np.degrees(np.arctan2(gy, gx))
    orientation_deg = np.mod(gradient_deg + 90.0, 180.0)

    return EdgeOrientationMap(edge_mask=edge_mask, orientation_deg=orientation_deg, strength=strength)


def estimate_lidar_edge_orientations(edge_pixels: np.ndarray, k: int = 6) -> np.ndarray:
    """Each LiDAR edge point's local boundary direction, from the layout of its
    k nearest other edge points: the principal axis of a small point
    neighbourhood lying along a boundary *is* that boundary's tangent, found
    here via SVD of the centred neighbourhood (steadier than eig(cov) for small
    counts). Degrees, mod 180 to match `compute_edge_orientation_map`.

    A point with fewer than two usable neighbours gets NaN. That is deliberate,
    not a placeholder: every comparison against NaN is False, so such a point
    can never pass the orientation filter below -- "we don't know this point's
    direction" must never be allowed to read as "matches everything".
    """
    n = edge_pixels.shape[0]
    if n <= 1:
        return np.full(n, np.nan)

    tree = cKDTree(edge_pixels)
    k_query = min(k + 1, n)  # +1: a point is always its own nearest neighbour
    if k_query < 2:
        return np.full(n, np.nan)
    _, neighbor_idx = tree.query(edge_pixels, k=k_query)

    orientations = np.full(n, np.nan)
    for i in range(n):
        neighbors = edge_pixels[neighbor_idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal = vt[0]
        orientations[i] = np.mod(np.degrees(np.arctan2(principal[1], principal[0])), 180.0)
    return orientations


def _angular_diff_mod180(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Smallest gap between two undirected (mod-180) angles, in [0, 90]."""
    diff = np.abs(a_deg - b_deg) % 180.0
    return np.minimum(diff, 180.0 - diff)


@dataclass
class CorrespondenceResult:
    """Per-LiDAR-edge-point outcome, arrays aligned with the input order."""

    matched: np.ndarray             # (N,) bool
    matched_pixels: np.ndarray      # (N, 2) NaN row where unmatched
    distance_px: np.ndarray         # (N,) real distance if matched, else the penalty
    orientation_diff_deg: np.ndarray
    strength: np.ndarray
    rejection_reason: list          # (N,) None | "no_candidate" | "consistency"


def find_correspondences(
    lidar_pixels: np.ndarray,
    lidar_orientations_deg: np.ndarray,
    orientation_map: EdgeOrientationMap,
    radii_px: tuple = (5.0, 10.0, 15.0),
    max_orientation_diff_deg: float = 30.0,
) -> CorrespondenceResult:
    """Candidate search + orientation filter + strongest-gradient pick.

    Radii grow from smallest to largest; a point already matched at a smaller
    radius is not re-queried at a larger one. Everything is batched per radius
    (one KD-tree query for every still-unmatched point, one flatten, one
    lexsort-based groupby-argmax) rather than looped point by point -- a frame's
    edge points number in the thousands, so a Python-level loop over them,
    running at every one of up to three radii, is not an option.
    """
    n = lidar_pixels.shape[0]
    matched = np.zeros(n, dtype=bool)
    matched_pixels = np.full((n, 2), np.nan)
    max_radius = max(radii_px)
    distance_px = np.full(n, max_radius)
    orientation_diff_deg = np.full(n, np.nan)
    strength = np.full(n, np.nan)
    rejection_reason: list = ["no_candidate"] * n

    edge_rows, edge_cols = np.nonzero(orientation_map.edge_mask)
    if edge_rows.size == 0 or n == 0:
        return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg, strength, rejection_reason)

    edge_xy = np.stack([edge_cols, edge_rows], axis=1).astype(np.float64)
    tree = cKDTree(edge_xy)

    # A point with unknown orientation fails the filter at every radius, so it
    # is dropped from the search up front rather than queried and rejected
    # three times over.
    remaining = np.nonzero(np.isfinite(lidar_orientations_deg))[0]

    for radius in sorted(radii_px):
        if remaining.size == 0:
            break

        neighbor_lists = tree.query_ball_point(lidar_pixels[remaining], r=radius)
        lengths = np.array([len(lst) for lst in neighbor_lists], dtype=np.int64)
        if lengths.sum() == 0:
            continue

        point_local = np.repeat(np.arange(remaining.size), lengths)
        cand_idx = np.concatenate([np.asarray(lst, dtype=np.intp) for lst in neighbor_lists if lst])
        point_idx = remaining[point_local]

        cand_xy = edge_xy[cand_idx]
        cand_rows = cand_xy[:, 1].astype(np.intp)
        cand_cols = cand_xy[:, 0].astype(np.intp)
        cand_orient = orientation_map.orientation_deg[cand_rows, cand_cols]
        cand_strength = orientation_map.strength[cand_rows, cand_cols]

        diff = _angular_diff_mod180(lidar_orientations_deg[point_idx], cand_orient)
        ok = diff <= max_orientation_diff_deg
        if not ok.any():
            continue

        pg = point_idx[ok]
        cxy = cand_xy[ok]
        cs = cand_strength[ok]
        cdiff = diff[ok]
        dist = np.linalg.norm(cxy - lidar_pixels[pg], axis=1)

        # Group by point (pg), strongest gradient first, closest distance as
        # the tiebreak -- np.lexsort's last key is primary.
        order = np.lexsort((dist, -cs, pg))
        pg_sorted = pg[order]
        winners, first_idx, _ = np.unique(pg_sorted, return_index=True, return_counts=True)
        winner_rows = order[first_idx]

        matched[winners] = True
        matched_pixels[winners] = cxy[winner_rows]
        distance_px[winners] = dist[winner_rows]
        orientation_diff_deg[winners] = cdiff[winner_rows]
        strength[winners] = cs[winner_rows]
        for idx in winners:
            rejection_reason[idx] = None

        remaining = np.setdiff1d(remaining, winners, assume_unique=True)

    return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg, strength, rejection_reason)


def apply_local_consistency_filter(
    lidar_pixels: np.ndarray,
    result: CorrespondenceResult,
    k: int = 5,
    max_angle_diff_deg: float = 45.0,
    max_magnitude_ratio: float = 3.0,
    min_magnitude_diff_px: float = 3.0,
    min_displacement_for_angle_check_px: float = 1.5,
    penalty_distance_px: Optional[float] = None,
) -> CorrespondenceResult:
    """Demote a match whose displacement disagrees with its matched neighbours'.

    Each matched point's displacement (matched pixel minus LiDAR pixel) is
    compared to the median displacement of its k nearest *matched* neighbours.
    A lone point pointing somewhere very different -- or wildly longer/shorter
    -- than everything around it is more likely a coincidental snap onto
    unrelated texture than a genuine boundary, even though it passed the
    orientation filter on its own.

    The magnitude check needs both the ratio *and* an absolute pixel gap before
    demoting: a ratio alone is unstable near zero (0.9px vs 3.9px is a ~4x
    ratio and also unremarkable sub-pixel noise). The angle check only applies
    once the point's own displacement exceeds
    `min_displacement_for_angle_check_px`: a near-zero vector's direction is
    noise, not signal, and demoting on it would penalise the *best* possible
    outcome (an almost-exact match) as if it were inconsistent.

    Returns a new `CorrespondenceResult`; does not mutate the input.
    """
    matched = result.matched.copy()
    matched_pixels = result.matched_pixels.copy()
    distance_px = result.distance_px.copy()
    orientation_diff_deg = result.orientation_diff_deg.copy()
    strength = result.strength.copy()
    rejection_reason = list(result.rejection_reason)

    if penalty_distance_px is None:
        penalty_distance_px = float(distance_px.max()) if distance_px.size else 0.0

    matched_idx = np.nonzero(matched)[0]
    m = matched_idx.size
    if m < 3:
        return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg, strength, rejection_reason)

    anchor_pixels = lidar_pixels[matched_idx]
    displacements = result.matched_pixels[matched_idx] - anchor_pixels

    tree = cKDTree(anchor_pixels)
    k_query = min(k + 1, m)
    _, neighbor_local = tree.query(anchor_pixels, k=k_query)
    if neighbor_local.ndim == 1:
        neighbor_local = neighbor_local[:, None]

    is_self = neighbor_local == np.arange(m)[:, None]
    neighbor_disps = displacements[neighbor_local]
    neighbor_mags = np.linalg.norm(neighbor_disps, axis=2)
    invalid = is_self | (neighbor_mags <= 1e-9)
    enough_neighbors = (~invalid).sum(axis=1) >= 2

    masked = np.where(invalid[:, :, None], np.nan, neighbor_disps)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows are excluded below
        median_disp = np.nanmedian(masked, axis=1)

    own_mag = np.linalg.norm(displacements, axis=1)
    median_mag = np.linalg.norm(median_disp, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.clip(
            np.einsum("ij,ij->i", displacements, median_disp) / (own_mag * median_mag), -1.0, 1.0
        )
        angle_diff_deg = np.degrees(np.arccos(cos_angle))
        mag_ratio = np.maximum(own_mag / median_mag, median_mag / own_mag)

    usable = enough_neighbors & (own_mag > 1e-9) & (median_mag > 1e-9)
    magnitude_diff = np.abs(own_mag - median_mag)
    demote_angle = usable & (own_mag >= min_displacement_for_angle_check_px) & (angle_diff_deg > max_angle_diff_deg)
    demote_magnitude = usable & (mag_ratio > max_magnitude_ratio) & (magnitude_diff > min_magnitude_diff_px)
    demote_local = demote_angle | demote_magnitude

    demote_global = matched_idx[demote_local]
    if demote_global.size:
        matched[demote_global] = False
        matched_pixels[demote_global] = np.nan
        distance_px[demote_global] = penalty_distance_px
        orientation_diff_deg[demote_global] = np.nan
        strength[demote_global] = np.nan
        for idx in demote_global:
            rejection_reason[idx] = "consistency"

    return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg, strength, rejection_reason)


def match_lidar_edges_to_image(
    lidar_pixels: np.ndarray,
    image_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    radii_px: tuple = (5.0, 10.0, 15.0),
    max_orientation_diff_deg: float = 30.0,
    k_orientation: int = 6,
    k_consistency: int = 5,
    max_consistency_angle_deg: float = 45.0,
    max_consistency_magnitude_ratio: float = 3.0,
    min_consistency_magnitude_diff_px: float = 3.0,
    min_consistency_displacement_for_angle_check_px: float = 1.5,
    edge_mask: Optional[np.ndarray] = None,
    precomputed_orientation_map: Optional[EdgeOrientationMap] = None,
) -> CorrespondenceResult:
    """The full pipeline `edge_alignment.evaluate_edge_alignment` calls:
    orientation/strength map -> LiDAR point orientations -> candidate search
    -> local consistency filtering.

    `precomputed_orientation_map`, when given, replaces the Canny + Sobel work
    `compute_edge_orientation_map` would otherwise redo -- it depends only on
    the image, never on the LiDAR points or the extrinsic, so a caller
    re-evaluating the *same* image against many candidate extrinsics (e.g.
    `evaluation.perturbation`) computes it once and passes it to every call.
    """
    orientation_map = (
        precomputed_orientation_map if precomputed_orientation_map is not None
        else compute_edge_orientation_map(image_bgr, canny_low, canny_high, edge_mask=edge_mask)
    )
    lidar_orientations = estimate_lidar_edge_orientations(lidar_pixels, k=k_orientation)
    raw = find_correspondences(
        lidar_pixels, lidar_orientations, orientation_map,
        radii_px=radii_px, max_orientation_diff_deg=max_orientation_diff_deg,
    )
    return apply_local_consistency_filter(
        lidar_pixels, raw, k=k_consistency,
        max_angle_diff_deg=max_consistency_angle_deg,
        max_magnitude_ratio=max_consistency_magnitude_ratio,
        min_magnitude_diff_px=min_consistency_magnitude_diff_px,
        min_displacement_for_angle_check_px=min_consistency_displacement_for_angle_check_px,
        penalty_distance_px=max(radii_px),
    )

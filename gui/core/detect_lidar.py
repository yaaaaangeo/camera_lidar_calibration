"""LiDAR side of the detection: point cloud -> four hole centres.

Two ways in, sharing everything up to the board plane:

    box filter -> (voxel) -> RANSAC plane -> rotate plane onto z=0
      -> "original":  ring walk, then RANSAC circles peeled off     [default]
      -> "occupancy": rasterise, holes are the empty regions
      -> "ring":      ring walk with crossings paired
      -> "boundary":  per-point angular gap, cluster, fit circles   [C++ port]

Where they stand, on ten consecutive sweeps of a stationary board
(test_3/1.bag, one ROI) and on the 96-condition synthetic grid `check_synth.py`
runs:

                real 10 sweeps    synthetic grid    centre error
    original        10/10            70 %          4.0 mm  (worst 42.2)
    occupancy       10/10            30 %          1.4 mm  (worst  4.5)
    ring             2/10            40 %          7.6 mm  (worst 92.4)
    boundary          0/10           22 %         19.0 mm  (worst 102.7)

Two things worth reading off that table. "original" finds four holes most often
but is the one that can be confidently wrong -- its worst case is an order of
magnitude off, because it fits circles to any collection of edge points without
requiring them to surround a centre. "occupancy" is the opposite: it declines
more often and is nearly always right when it does not.

The two written here to improve on the original both do worse than it. That only
became visible once the original was implemented alongside them, which is why it
stays rather than being replaced.

The boundary path is the port of `detect_solid_lidar` (`workflow.md` 1.4), meant
for solid-state units. It asks each point whether its neighbourhood is one-sided,
which only separates rim from interior while the search radius sits between the
point spacing and the hole size. On this rig's mechanical LiDAR that window is
shut, so its 0/10 is a mismatch of purpose rather than a failure.

Thresholds here are meant to be derived rather than tuned, because tuning against
one recording cannot distinguish a value that generalises from one that happens
to fit. That principle held less well than the comments below once claimed --
see the warning on DetectParams.

One deliberate deviation from the C++: the rotation that flattens the plane is
built from a *normalised* axis. The original passes `normal.cross(z)` straight
to `Eigen::AngleAxisd`, whose axis must be a unit vector; that cross product has
length sin(angle), so it is only correct when the plane sits at 90 degrees to z.
A vertical board seen by a level LiDAR is near that angle, which is why it goes
unnoticed -- but tilting the board, which is exactly what improves the camera
side, walks away from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from gui.core.project import FilterBox, Target

VOXEL = 0.005
PLANE_THRESHOLD = 0.01
BOUNDARY_RADIUS = 0.03
BOUNDARY_ANGLE = np.pi / 4
CLUSTER_TOLERANCE = 0.05
CLUSTER_MIN, CLUSTER_MAX = 50, 1000
CIRCLE_THRESHOLD = 0.01
CIRCLE_MAX_ERROR = 0.025


@dataclass
class DetectParams:
    """Detection thresholds.

    The first block reproduces the C++ constants and only affects the boundary
    method. The rest are meant to follow from sensor geometry or from error
    curves measured against synthetic ground truth.

    Read the derivations below with care. Several were measured with a generator
    that has since been found wrong in two ways: it modelled a beam as passing
    through a hole only when it fitted entirely inside (the opposite of what
    happens), and its panel was smaller than its own hole pattern, so holes ran
    through the board's edge. Both were fixed, and every number those comments
    quote predates the fix. They are kept because the shape of each argument
    still holds -- which value sits in the middle of a plateau, which threshold
    stops mattering above some size -- but the specific counts are stale until
    re-swept. `check_synth.py --sweep NAME=v1,v2,...` is what re-measures them.

    Anything expressed in metres also has to be read against the point spacing,
    which is why `sweep_spacing` is an input rather than something measured from
    the cloud handed in.
    """

    voxel: float = VOXEL
    plane_threshold: float = PLANE_THRESHOLD
    boundary_radius: float = BOUNDARY_RADIUS
    boundary_angle: float = BOUNDARY_ANGLE
    cluster_tolerance: float = CLUSTER_TOLERANCE
    cluster_min: int = CLUSTER_MIN
    cluster_max: int = CLUSTER_MAX
    circle_threshold: float = CIRCLE_THRESHOLD
    circle_max_error: float = CIRCLE_MAX_ERROR
    # Fallback band for the boundary method, which has no range to reason from.
    # The occupancy path uses expected_radius() instead.
    radius_min_ratio: float = 0.55
    radius_max_ratio: float = 1.35
    # Beam divergence, full angle. Decides how much *larger* than the drawing a
    # hole reads -- see expected_radius, which had this backwards until a tape
    # measure settled it. 3 mrad is typical for automotive spinning units; check
    # the datasheet to be exact.
    beam_divergence_mrad: float = 3.0
    # Fewest rings that must cross a hole. Three defines a circle in principle
    # (six rim points) but those points sit on two short arcs and the fit fails.
    # The counts this was chosen from came from the pre-fix generator; the
    # argument stands, the numbers need re-measuring.
    min_rings_per_hole: float = 4.0
    # A sanity guard, not a tuning knob. Counting points cannot tell "box is too
    # wide" from "board is close and densely sampled" -- and the second is the
    # good case. A 2 m capture legitimately reaches ~250k points and detects to
    # 2.6 mm; the old 60k ceiling rejected every capture closer than 4 m. The UI
    # warns on point count separately, where the user can override.
    max_points: int = 1_500_000
    # Which detector to run.
    #
    #   "original"      detect_mech_lidar as the C++ writes it, constants included
    #   "occupancy"     rasterise the plane, holes are the empty regions
    #   "ring"          ring walk with crossings paired, thresholds from hole size
    #   "boundary"      the C++ angular-gap test (detect_solid_lidar)
    #   "experimental"  a copy of "occupancy" to try changes in, so that the one
    #                   in use stays untouched while they are being tried
    #
    # "original" is the default: on ten consecutive sweeps of one stationary board
    # it found 4/4 every time. Two things it does that the reimplementations do
    # not look decisive -- it bounds the radius *before* scoring RANSAC hypotheses
    # rather than filtering afterwards, and it removes each circle's inliers
    # before looking for the next, so one hole cannot yield several candidates.
    #
    # "occupancy" now matches it on that recording (10/10) and is markedly more
    # accurate when it does answer -- 1.4 mm mean against 4.0 mm, and a worst case
    # of 4.5 mm against 42 mm. It declines more often on the synthetic grid, so it
    # is not yet the default, but it is the one to prefer when both succeed.
    method: str = "original"
    # Distance along a ring that counts as crossing a hole, as a fraction of the
    # hole diameter. The C++ hard-codes 0.10 m, chosen for the stock board; it
    # scales with the hole, so a resized target needs it moved. Too small and
    # ordinary sampling reads as a crossing; too large and only the chords through
    # the middle survive, which flattens the arc and inflates the fitted radius.
    # The sweep that picked 0.25 used the pre-fix generator -- and note that the
    # method this governs is the one that ended up losing to the C++ original.
    ring_gap_frac: float = 0.25
    ring_gap: float = 0.0  # absolute override, in metres; derived when zero
    # Grid resolution, as a fraction of the point spacing. A cell as coarse as the
    # sampling stops representing the surface at all; far finer than half wastes
    # memory without adding detail. 0.5 sat in the middle of a wide plateau when
    # swept, though that sweep predates the generator fix.
    cell_per_spacing: float = 0.5
    # Shape guards on a candidate empty region: loose backstops against something
    # absurd rather than tuning knobs. Sweeping them changed nothing except at the
    # very bottom of their range, which is the behaviour a backstop should have.
    area_ceiling: float = 2.5  # multiples of the ideal hole area
    aspect_max: float = 2.2  # a hole is round; reject long thin gaps
    # How wide a gap the occupancy raster treats as still being surface, as a
    # fraction of the hole diameter. See find_holes_by_occupancy for why this is
    # sized off the hole rather than off the point spacing.
    closing_per_diameter: float = 0.30
    # Likewise a backstop: it only matters when there are more than four
    # candidates to choose between, and swept across its range it changed nothing.
    rect_tolerance: float = 0.10
    # "experimental" only: restrict the rim to points the angular-gap test calls
    # a boundary. See refine_hole_rim's `eligible`.
    exp_rim_boundary_only: bool = True
    # Sample spacing of one sweep, in metres. The occupancy grid is sized from
    # this, and it cannot be measured from the cloud handed to detect(): that one
    # is voxelised and accumulated, so its nearest-neighbour distance reports the
    # voxel size, not how finely the surface was actually sampled.
    sweep_spacing: float = 0.0


@dataclass
class LidarDetection:
    ok: bool
    centers: np.ndarray | None = None  # (N, 3) hole centres in LiDAR frame
    reason: str = ""

    # intermediates, for display
    filtered: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    plane: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    plane_rms: float = 0.0
    n_clusters: int = 0  # boundary method: clusters fitted; occupancy: hole candidates
    spacing: float = 0.0  # median nearest-neighbour distance after voxelising
    radii: list[float] = field(default_factory=list)
    edge_counts: list[int] = field(default_factory=list)
    method: str = ""
    cell_size: float = 0.0  # occupancy grid resolution, when that method ran
    # The gap the ring walk actually used. Reported rather than recomputed by the
    # caller: the display had its own copy of the formula, kept the old 0.42
    # coefficient after the derived value moved to 0.25, and so said 60 mm while
    # detection ran at 36 mm. A number on screen that does not come from the run
    # it describes is worse than no number.
    ring_gap: float = 0.0

    @property
    def n_circles(self) -> int:
        return 0 if self.centers is None else len(self.centers)


def box_from_frame(centre, rot, half) -> FilterBox:
    """A FilterBox with the given centre, axes and half-widths.

    The Euler extraction has to match `FilterBox.rotation()`'s Rz @ Ry @ Rx
    order, and getting it wrong turns the box in a way that looks almost right --
    so it lives in one place and both the manual align and the automatic
    placement go through it.
    """
    centre = np.asarray(centre, float).reshape(3)
    half = np.asarray(half, float).reshape(3)
    rot = np.asarray(rot, float).reshape(3, 3)

    pitch = np.degrees(np.arcsin(-np.clip(rot[2, 0], -1, 1)))
    if abs(rot[2, 0]) < 0.9999:
        yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
        roll = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    else:  # gimbal lock
        yaw = np.degrees(np.arctan2(-rot[0, 1], rot[1, 1]))
        roll = 0.0

    return FilterBox(
        x_min=float(centre[0] - half[0]), x_max=float(centre[0] + half[0]),
        y_min=float(centre[1] - half[1]), y_max=float(centre[1] + half[1]),
        z_min=float(centre[2] - half[2]), z_max=float(centre[2] + half[2]),
        yaw=float(yaw), pitch=float(pitch), roll=float(roll),
    )


def align_box_to_board(cloud: np.ndarray, box: FilterBox, thickness: float = 0.12):
    """Turn a box to sit square with the largest plane inside it, and thin it down.

    Finding the plane is the easy part; the useful part is choosing which of the
    box's axes to point along the normal. Keeping the板 plane spanned by the two
    long sides means the third axis only has to cover the panel's thickness, so
    the wall behind falls outside.

    Returns (box, note) -- the box unchanged if no plane was found.
    """
    inside = cloud[box.mask(cloud)]
    if len(inside) < 200:
        return box, "박스 안에 점이 부족합니다"

    normal, d, inliers = fit_plane(inside.astype(np.float64), 0.02)
    if normal is None or inliers.sum() < 150:
        return box, "평면을 찾지 못했습니다"

    patch = inside[inliers]
    centre = patch.mean(axis=0)

    # Box frame: z along the plane normal, x and y spanning it.
    n = normal / np.linalg.norm(normal)
    if n @ centre > 0:  # point the normal back towards the sensor
        n = -n
    world_up = np.array([0.0, 0.0, 1.0])
    ex = np.cross(world_up, n)
    if np.linalg.norm(ex) < 1e-3:
        ex = np.cross(np.array([1.0, 0.0, 0.0]), n)
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    rot = np.column_stack([ex, ey, n])

    local = (patch - centre) @ rot
    lo, hi = local.min(axis=0), local.max(axis=0)
    pad = np.array([0.06, 0.06, 0.0])
    lo, hi = lo - pad, hi + pad
    lo[2], hi[2] = -thickness / 2, thickness / 2

    # `mask` measures against the box centre, and the centre is the midpoint of
    # the bounds -- so the bounds have to be symmetric about it or the two
    # disagree. Shift the centre onto the middle of the fitted extent and store
    # half-widths around it.
    mid = (lo + hi) / 2
    turned = box_from_frame(centre + rot @ mid, rot, (hi - lo) / 2)
    kept = int(turned.mask(cloud).sum())
    return turned, f"{kept:,}점 (이전 {len(inside):,}), 평면 잔차 {np.abs(patch @ n + d).mean() * 1000:.0f}mm"


def region_mask(xyz: np.ndarray, region) -> np.ndarray:
    """Mask for either kind of filter -- a plain box or a plane-aligned slab."""
    return region.mask(xyz)


def apply_region(xyz: np.ndarray, region) -> np.ndarray:
    return np.asarray(xyz)[region_mask(xyz, region)]


def apply_box(xyz: np.ndarray, box: FilterBox) -> np.ndarray:
    """Points inside `box`, rotation included.

    This used to ignore the rotation, which was silent and wrong in a specific
    way: step 6 measured point spacing through it, so on a turned box the spacing
    came from points that detection would never see. That spacing then sized the
    occupancy grid and the boundary radius, and steps 5 and 6 disagreed on the
    same scene. `region_mask` handles both cases, so there is no reason for a
    second, rotation-blind version to exist.
    """
    return apply_region(xyz, box)


def box_mask(xyz: np.ndarray, box: FilterBox) -> np.ndarray:
    """Mask of points inside `box`, rotation included. See apply_box."""
    return region_mask(xyz, box)


def voxel_down(xyz: np.ndarray, leaf: float = VOXEL) -> np.ndarray:
    """Keep one point per occupied cell, like pcl::VoxelGrid.

    Cell indices are packed into a single int64 before deduplicating. Calling
    np.unique on the Nx3 array instead makes it lexsort rows, which on a
    million points costs seconds rather than milliseconds.
    """
    if len(xyz) == 0:
        return xyz
    keys = np.floor(np.asarray(xyz, np.float64) / leaf).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    if dims.prod() < 2**62:
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    else:  # fall back to hashing if the grid is enormous
        flat = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, idx = np.unique(flat, return_index=True)
    return xyz[np.sort(idx)]


def fit_plane(xyz: np.ndarray, threshold: float = PLANE_THRESHOLD, iters: int = 400, rng=None):
    """RANSAC plane. Returns (unit normal, offset d, inlier mask) for n·p + d = 0."""
    rng = rng or np.random.default_rng(0)
    n = len(xyz)
    if n < 3:
        return None, 0.0, np.zeros(n, bool)

    # Score every hypothesis in one pass. Looping in Python and testing each
    # plane against all N points separately is the same arithmetic but spends
    # most of its time in interpreter overhead.
    tri = rng.integers(0, n, size=(iters, 3))
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-9
    if not good.any():
        return None, 0.0, np.zeros(n, bool)
    normals = normals[good] / lengths[good, None]
    offsets = -np.einsum("ij,ij->i", normals, a[good])

    # Count inliers on a subsample -- enough to rank hypotheses, and it keeps
    # the scoring matrix from being iters x N.
    probe = xyz if n <= 20000 else xyz[rng.choice(n, 20000, replace=False)]
    counts = (np.abs(probe @ normals.T + offsets) < threshold).sum(axis=0)
    best = int(np.argmax(counts))
    best_mask = np.abs(xyz @ normals[best] + offsets[best]) < threshold
    if best_mask.sum() < 3:
        return None, 0.0, best_mask

    # Least-squares refit on the inliers.
    pts = xyz[best_mask]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2] / np.linalg.norm(vt[2])
    d = -normal @ centroid
    return normal, float(d), np.abs(xyz @ normal + d) < threshold


def align_rotation(normal: np.ndarray) -> np.ndarray:
    """Rotation taking `normal` onto +z. Axis is normalised -- see module docstring."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3) if normal[2] > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    angle = np.arccos(np.clip(normal @ z, -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def boundary_points(xy: np.ndarray, radius: float = BOUNDARY_RADIUS, angle: float = BOUNDARY_ANGLE) -> np.ndarray:
    """Boolean mask of points sitting on an edge.

    For each point, take the neighbours inside `radius`, sort them by bearing,
    and look at the largest angular gap. Interior points are surrounded, so
    their gaps stay small; a point on a hole rim has a wide empty wedge.
    """
    n = len(xy)
    if n == 0:
        return np.zeros(0, bool)

    tree = cKDTree(xy)
    out = np.zeros(n, bool)
    # Neighbour lists are ragged, so this loop stays -- but the per-point work
    # is kept to sorting a small angle array, with the pair search vectorised.
    for i, neighbours in enumerate(tree.query_ball_point(xy, radius, workers=-1)):
        if len(neighbours) < 4:
            out[i] = True
            continue
        d = xy[neighbours] - xy[i]
        a = np.arctan2(d[:, 1], d[:, 0])
        a = np.sort(a[np.isfinite(a) & (d[:, 0] ** 2 + d[:, 1] ** 2 > 0)])
        if len(a) < 3:
            out[i] = True
            continue
        gaps = np.diff(a)
        wrap = a[0] + 2 * np.pi - a[-1]
        out[i] = max(gaps.max(initial=0.0), wrap) > angle
    return out


def boundary_points_by_ring(
    xyz: np.ndarray,
    ring: np.ndarray,
    normal: np.ndarray,
    offset: float,
    gap: float,
    plane_tolerance: float = 0.03,
    min_points_per_ring: int = 10,
    max_gap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary points from ring structure -- the port of `detect_mech_lidar`.

    Walk along each ring and mark the two points either side of a step longer
    than `gap`. A spinning sensor samples in rings, so a beam that crosses a hole
    leaves a jump in one dimension; that is a much sharper signal than the
    angular-gap test, which has to infer emptiness from a neighbourhood and only
    works while its radius sits between the point spacing and the hole.

    Returns (mask, pair_id). `pair_id` is -1 off the mask and otherwise numbers
    the crossing, so the two points share an id. That pairing is not a detail:
    the two rim points of one hole sit a hole-width apart (143 mm on the demo
    board) while the nearest rim point of the *next* hole is closer than that
    (96 mm), so no single-linkage distance can join a hole to itself without also
    joining it to its neighbour. Grouping has to follow the crossings.

    Points are used in acquisition order and must not be voxelised first -- that
    reorders them and the walk becomes meaningless.
    """
    n = len(xyz)
    out = np.zeros(n, bool)
    pair = np.full(n, -1, np.int64)
    if n == 0:
        return out, pair

    plane_dist = np.abs(xyz @ normal + offset)
    order = np.argsort(ring, kind="stable")
    ring_sorted = ring[order]
    starts = np.flatnonzero(np.r_[True, ring_sorted[1:] != ring_sorted[:-1]])
    bounds = np.r_[starts, n]

    next_id = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < min_points_per_ring:
            continue
        idx = order[a:b]
        pts = xyz[idx]
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # A crossing is a step too long to be sampling, but not so long that it
        # spans the panel's own edge or the emptiness beyond it.
        crossing = step > gap
        if max_gap > 0:
            crossing &= step <= max_gap
        # Ends of a ring are not holes, so a crossing needs a point either side.
        near_plane = plane_dist[idx] < plane_tolerance
        for j in np.flatnonzero(crossing):
            if not (near_plane[j] and near_plane[j + 1]):
                continue
            out[idx[j]] = out[idx[j + 1]] = True
            pair[idx[j]] = pair[idx[j + 1]] = next_id
            next_id += 1
    return out, pair


# --------------------------------------------------------------------------
# The original method, kept deliberately separate.
#
# `method="original"` reproduces `detect_mech_lidar` / `detect_solid_lidar` as
# they are, constants included, so it can be compared against rather than argued
# about. The constants below are absolute metres exactly as the C++ writes them
# -- they are NOT scaled to the board, and on a resized target that is the point:
# 0.10 m is 0.42 of the stock hole diameter but 0.70 of the demo board's, and
# seeing what that does is the reason this path exists.
#
# One acknowledged deviation: the C++ finishes with its own `Square` class for the
# geometric consistency check, and this uses `_select_rectangle` instead. Both
# pick the four candidates whose sides and diagonals match the target, and the
# difference is not what the comparison is about.
ORIG_NEIGHBOR_GAP = 0.10       # 邻近点距离阈值
ORIG_MIN_POINTS_PER_RING = 10
ORIG_EDGE_PLANE_MAX = 0.03
ORIG_CIRCLE_THRESHOLD = 0.02
ORIG_CIRCLE_MAX_ITER = 1000
ORIG_RADIUS_MARGIN = 0.03      # setRadiusLimits(r - 0.03, r + 0.03)
ORIG_MIN_CIRCLE_INLIERS = 5


def edge_points_original(
    xyz: np.ndarray,
    ring: np.ndarray,
    normal: np.ndarray,
    offset: float,
    gap: float = ORIG_NEIGHBOR_GAP,
    min_points_per_ring: int = ORIG_MIN_POINTS_PER_RING,
    plane_max: float = ORIG_EDGE_PLANE_MAX,
) -> np.ndarray:
    """Edge points exactly as `detect_mech_lidar` step 3 finds them.

    Group by ring, then for each interior point mark it when *either* neighbour
    is further than `gap`. Note what this is not: there is no pairing of the two
    sides of a crossing, and no upper bound on the jump, both of which the
    improved ring path adds. Kept faithful so the difference is measurable.
    """
    n = len(xyz)
    out = np.zeros(n, bool)
    if n == 0:
        return out

    plane_dist = np.abs(xyz @ normal + offset)
    order = np.argsort(ring, kind="stable")
    ring_sorted = ring[order]
    starts = np.flatnonzero(np.r_[True, ring_sorted[1:] != ring_sorted[:-1]])

    for a, b in zip(starts, np.r_[starts[1:], n]):
        idx = order[a:b]
        if len(idx) < min_points_per_ring:
            continue
        pts = xyz[idx]
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # Interior points only: the C++ loops k = 1 .. size-2.
        far_prev, far_next = step[:-1] > gap, step[1:] > gap
        hit = np.zeros(len(idx), bool)
        hit[1:-1] = far_prev | far_next
        out[idx] = hit & (plane_dist[idx] < plane_max)
    return out


def find_circles_peel(
    xy: np.ndarray,
    r_lo: float,
    r_hi: float,
    threshold: float = ORIG_CIRCLE_THRESHOLD,
    min_inliers: int = ORIG_MIN_CIRCLE_INLIERS,
    max_iter: int = ORIG_CIRCLE_MAX_ITER,
    rng=None,
):
    """Fit a circle, remove its inliers, repeat -- `detect_mech_lidar` step 5.

    No clustering anywhere: every remaining edge point is a candidate for the
    next circle. That is what lets a half rim become a circle of its own, since
    nothing requires the inliers to surround the centre.
    """
    work = np.asarray(xy, np.float64)
    out = []
    while len(work) > 3:
        c, r, mask = fit_circle(work, threshold, max_iter, rng, r_lo, r_hi)
        if c is None or int(mask.sum()) < min_inliers:
            break
        out.append((c, float(r), int(mask.sum())))
        work = work[~mask]
    return out


def cluster(xy: np.ndarray, tolerance: float = CLUSTER_TOLERANCE) -> list[np.ndarray]:
    """Single-linkage clustering, as pcl::EuclideanClusterExtraction does."""
    n = len(xy)
    if n == 0:
        return []
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(xy)
    pairs = tree.query_pairs(tolerance, output_type="ndarray")
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(v) for v in groups.values()]


def fit_circle(
    xy: np.ndarray,
    threshold: float = CIRCLE_THRESHOLD,
    iters: int = 1000,
    rng=None,
    r_lo: float = 0.0,
    r_hi: float = np.inf,
):
    """RANSAC 2D circle. Returns (centre, radius, inlier mask).

    `r_lo`/`r_hi` reject hypotheses outside a radius band before scoring, which
    is what `pcl::SACSegmentation::setRadiusLimits` does. Without it the winning
    hypothesis on a partial rim is often a much larger circle that happens to
    graze more points.
    """
    rng = rng or np.random.default_rng(0)
    n = len(xy)
    if n < 3:
        return None, 0.0, np.zeros(n, bool)

    # All hypotheses at once, same as fit_plane. Clusters are small, so the
    # iters x n scoring matrix is cheap.
    iters = min(iters, max(60, n * 6))
    tri = rng.integers(0, n, size=(iters, 3))
    p1, p2, p3 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    d12, d13 = p2 - p1, p3 - p1
    det = 2 * (d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0])
    ok = np.abs(det) > 1e-12
    if not ok.any():
        return None, 0.0, np.zeros(n, bool)

    s1 = (p2**2).sum(1) - (p1**2).sum(1)
    s2 = (p3**2).sum(1) - (p1**2).sum(1)
    cx = (s1 * d13[:, 1] - s2 * d12[:, 1])[ok] / det[ok]
    cy = (s2 * d12[:, 0] - s1 * d13[:, 0])[ok] / det[ok]
    centres = np.column_stack([cx, cy])
    radii = np.linalg.norm(p1[ok] - centres, axis=1)

    if r_lo > 0.0 or np.isfinite(r_hi):
        band = (radii >= r_lo) & (radii <= r_hi)
        if not band.any():
            return None, 0.0, np.zeros(n, bool)
        centres, radii = centres[band], radii[band]

    dists = np.linalg.norm(xy[:, None, :] - centres[None, :, :], axis=2)
    inlier_counts = (np.abs(dists - radii[None, :]) < threshold).sum(axis=0)
    best_i = int(np.argmax(inlier_counts))
    c, r = centres[best_i], float(radii[best_i])
    mask = np.abs(np.linalg.norm(xy - c, axis=1) - r) < threshold
    if mask.sum() < 3:
        return None, 0.0, np.zeros(n, bool)

    # Algebraic least-squares refit on the inliers (Kasa).
    pts = xy[mask]
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    sol, *_ = np.linalg.lstsq(A, (pts**2).sum(axis=1), rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
    mask = np.abs(np.linalg.norm(xy - c, axis=1) - r) < threshold
    return c, r, mask


def point_spacing(xyz: np.ndarray, sample: int = 2000) -> float:
    """Median nearest-neighbour distance -- the scale everything else must respect.

    Exact duplicates are dropped first. A merged cloud can carry a lot of them
    (about half the points in the rig this was written against), and they drag
    the median to zero, which would make the cloud look far denser than it is.
    """
    if len(xyz) < 2:
        return 0.0
    xyz = voxel_down(np.asarray(xyz, np.float32), 1e-4)  # drop exact duplicates
    if len(xyz) < 2:
        return 0.0
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), min(sample, len(xyz)), replace=False)
    d, _ = cKDTree(xyz).query(xyz[idx], k=2, workers=-1)
    return float(np.median(d[:, 1]))


@dataclass
class PlaneRegion:
    """A slab of space around the board, defined by its four corners.

    An axis-aligned box cannot isolate a board held at an angle: the volume it
    must span grows with the tilt, and the wall behind fills the corners it
    leaves empty. Four clicked corners give the plane and its outline at once,
    so what is kept is "on the panel", not "inside a cuboid".
    """

    corners: np.ndarray  # (4, 3) as clicked, any order
    thickness: float = 0.10  # kept either side of the plane
    margin: float = 0.05  # kept outside the outline, in-plane

    _basis: tuple | None = None

    def frame(self):
        """(origin, in-plane u, in-plane v, unit normal), fitted to the corners."""
        if self._basis is None:
            pts = np.asarray(self.corners, np.float64)
            origin = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
            u, v, n = vt[0], vt[1], vt[2]
            object.__setattr__(self, "_basis", (origin, u, v, n / np.linalg.norm(n)))
        return self._basis

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        origin, u, v, n = self.frame()
        rel = np.asarray(xyz, np.float64) - origin
        if np.abs(rel @ n).size == 0:
            return np.zeros(len(xyz), bool)
        near = np.abs(rel @ n) <= self.thickness / 2

        # In-plane bounds from the corners, with a margin.
        cu, cv = (np.asarray(self.corners, np.float64) - origin) @ u, (
            np.asarray(self.corners, np.float64) - origin
        ) @ v
        pu, pv = rel @ u, rel @ v
        return (
            near
            & (pu >= cu.min() - self.margin) & (pu <= cu.max() + self.margin)
            & (pv >= cv.min() - self.margin) & (pv <= cv.max() + self.margin)
        )

    def as_box(self) -> FilterBox:
        """The axis-aligned hull, for anything that still needs a plain box."""
        kept = np.asarray(self.corners, np.float64)
        lo, hi = kept.min(axis=0) - self.thickness, kept.max(axis=0) + self.thickness
        return FilterBox(
            x_min=float(lo[0]), x_max=float(hi[0]),
            y_min=float(lo[1]), y_max=float(hi[1]),
            z_min=float(lo[2]), z_max=float(hi[2]),
        )


def region_from_corners(
    cloud: np.ndarray,
    corners: np.ndarray,
    target: Target,
    snap_radius: float = 0.12,
    thickness: float = 0.10,
):
    """Build a PlaneRegion from four roughly-clicked corners.

    Clicks are only used to say *where*; the plane is then re-fitted to the
    points they enclose, so being a few centimetres out does not move the
    result. Returns (region, note) or (None, why not).
    """
    corners = np.asarray(corners, np.float64)
    if len(corners) != 4:
        return None, "모서리 4개가 필요합니다"

    rough = PlaneRegion(corners=corners, thickness=max(thickness, 4 * snap_radius), margin=snap_radius)
    inside = np.asarray(cloud, np.float64)[rough.mask(cloud)]
    if len(inside) < 100:
        return None, f"클릭한 영역 안에 점이 적습니다 ({len(inside)})"

    normal, d, inliers = fit_plane(inside, 0.02)
    if normal is None or inliers.sum() < 100:
        return None, "평면을 찾지 못했습니다"
    patch = inside[inliers]

    # Snap each clicked corner onto the fitted plane, keeping its in-plane spot.
    snapped = corners - ((corners @ normal + d))[:, None] * normal
    region = PlaneRegion(corners=snapped, thickness=thickness, margin=0.03)

    kept = int(region.mask(cloud).sum())
    span = 2 * (target.delta_width_qr_center + target.marker_size / 2)
    extent = np.ptp(snapped, axis=0)
    if np.linalg.norm(extent) > 2.5 * span:
        return None, "클릭한 사각형이 보드보다 큽니다 — 모서리를 다시 찍어보세요"
    return region, f"{kept:,}점, 평면 잔차 {np.abs(patch @ normal + d).mean() * 1000:.0f}mm"


def box_from_click(
    cloud: np.ndarray,
    click: np.ndarray,
    target: Target,
    search_radius: float = 0.7,
    pad: float = 0.10,
    plane_threshold: float = 0.02,
):
    """Grow a filter box from one clicked point on the board.

    An axis-aligned box cannot fit a board held at an angle: the volume it has
    to span grows with the tilt, and the surroundings come along with it. A sweep
    over box margin showed detection falling away sharply as the margin grew, and
    on a tilted board the margin cannot be made small by hand. (Those counts came
    from the pre-fix generator -- see DetectParams -- but the trend is the reason
    the box gained a rotation, which is the better answer to the same problem.)

    So the plane is found first and the box is fitted to the points that lie on
    it, rather than to a guess about where the board's extent is. The box is
    still axis-aligned, but it now hugs the panel instead of a bounding volume
    the user had to estimate.
    """
    if len(cloud) < 100:
        return None, "점이 부족합니다"

    near = cloud[np.linalg.norm(cloud - click, axis=1) < search_radius]
    if len(near) < 50:
        return None, "클릭 지점 주변에 점이 부족합니다"

    normal, d, inliers = fit_plane(near.astype(np.float64), plane_threshold)
    if normal is None or inliers.sum() < 50:
        return None, "평면을 찾지 못했습니다"

    # Keep only the inliers connected to the click, so a wall passing through
    # the same plane does not drag the box across the room.
    plane_pts = near[inliers]
    tree = cKDTree(plane_pts)
    seed = int(np.argmin(np.linalg.norm(plane_pts - click, axis=1)))
    reach = max(4 * point_spacing(plane_pts), 0.05)
    seen = {seed}
    frontier = [seed]
    while frontier:
        batch = plane_pts[frontier]
        nxt = set()
        for group in tree.query_ball_point(batch, reach):
            nxt.update(group)
        frontier = [i for i in nxt - seen]
        seen.update(frontier)
    patch = plane_pts[sorted(seen)]

    span = 2 * (target.delta_width_qr_center + target.marker_size / 2)
    extent = np.ptp(patch, axis=0)
    if extent.max() > 2.5 * span:
        return None, f"잡힌 면이 보드보다 큽니다 ({extent.max():.1f} m) — 다른 지점을 눌러보세요"

    lo, hi = patch.min(axis=0) - pad, patch.max(axis=0) + pad
    box = FilterBox(
        x_min=float(lo[0]), x_max=float(hi[0]),
        y_min=float(lo[1]), y_max=float(hi[1]),
        z_min=float(lo[2]), z_max=float(hi[2]),
    )
    return box, f"{len(patch):,}점, {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m"


def expected_radius(target: Target, spacing: float, distance: float, divergence_mrad: float = 3.0):
    """(centre, half-width) of the radius a hole should measure, in metres.

    A hole always reads *larger* than the drawing, by two effects that both grow
    with range:

      * Beam width. The beam is a disc, not a point -- about 10 mm across at 3 m.
        A beam whose centre is still on the panel but whose edge overhangs the
        hole returns too little energy to register, and the rim it clips is a
        wall rather than a face, so what light does hit is thrown sideways. The
        last surviving return therefore sits about a beam radius outside the true
        rim.
      * Sampling. The nearest return can be up to a point spacing beyond that
        again, and a circle fitted to those points follows them.

    This used to *subtract* the beam radius, on the reasoning that a beam is only
    lost through a hole if it fits entirely inside. The opposite dominates, and
    the error was not small: predicted 69.5 mm where the board measured 80.0 mm.
    Every fourth hole then failed the radius check and detection stopped at 3/4.

    What hid it was that `synth.py` traced rays under the same assumption -- the
    generator granted the premise the model was being checked against, so
    synthetic radii came out right while real ones did not. Confirmed against a
    tape measure: the board's holes are 142 mm across, matching the CAD 142.6 mm,
    while three independent detectors all reported about 160 mm.

    Tolerance is a spacing and a half plus a tenth of the radius, which is what
    covers real merged clouds -- their planes are thicker than a single sensor's,
    so the rim reads further out than the model alone predicts.
    """
    beam = distance * (divergence_mrad * 1e-3) / 2.0
    centre = target.circle_radius + beam + spacing / 2.0
    return centre, 1.5 * spacing + 0.1 * target.circle_radius


def find_holes_by_occupancy(
    xy: np.ndarray,
    spacing: float,
    target: Target,
    min_rings: float = 4.0,
    cell_per_spacing: float = 0.5,
    area_ceiling: float = 2.5,
    aspect_max: float = 2.2,
    closing_per_diameter: float = 0.30,
):
    """Locate holes as empty regions of the board plane.

    The angular-gap test asks each point whether its neighbourhood is one-sided.
    That is a local question, and it only separates rim from interior when the
    search radius sits comfortably between the point spacing and the hole size.
    On a merged cloud -- 20 mm spacing against a 71 mm hole radius -- that window
    is too narrow, and the test ends up marking most of the panel.

    A hole is a property of the space, not of any point, so this rasterises the
    plane instead: fill the board's outline, and whatever stays empty inside it
    is a hole.

    Returns (centre, radius) per candidate, in plane coordinates.
    """
    if len(xy) < 50 or spacing <= 0:
        return []

    cell = max(spacing * cell_per_spacing, 0.004)
    origin = xy.min(axis=0)
    idx = np.floor((xy - origin) / cell).astype(int)
    h, w = idx[:, 1].max() + 1, idx[:, 0].max() + 1
    if h < 8 or w < 8 or h * w > 4_000_000:
        return []

    occupied = np.zeros((h, w), bool)
    occupied[idx[:, 1], idx[:, 0]] = True

    # Close gaps between samples before deciding what is "empty", or the space
    # between neighbouring points reads as a hole.
    #
    # This kernel used to span one point spacing, on the reasoning that a solid
    # surface cannot show a gap wider than its own sampling. True on average, and
    # the average is the flaw: `spacing` is a global median, while density varies
    # across a single board. Where a ring grazes the panel at a shallow angle the
    # local gaps run several times the median, the outline fails to close there,
    # and a hole beside that edge leaks into the space outside the board. It stops
    # being enclosed and drops out of the subtraction entirely -- on ten sweeps of
    # one stationary board, exactly two of four holes survived every single time.
    #
    # Sizing off the hole removes the dependence on local density and brings its
    # own ceiling: the kernel must stay well under the hole or it fills what it is
    # meant to find.
    #
    # The ceiling has to be enforced last. Spacing belongs in here as a floor -- a
    # kernel narrower than the sampling closes nothing -- but written as a plain
    # max() the floor overrides the ceiling whenever sampling is coarse, and then
    # nothing bounds the kernel at all. At 2 m on a head-on board the spacing term
    # reached 92 mm against a 143 mm hole, closing the holes shut: zero candidates,
    # while a tilted board at the same range still found four. That failure needs
    # coarse sampling to appear, so the demo rig at 7.8 mm never showed it.
    #
    # `k` is the kernel's full width in cells, and must be odd so it has a centre.
    # It used to be written `round(spacing / cell) * 2 + 1`, doubling because the
    # term inside was read as a radius. `k_metres` is a width already, so that
    # doubling now silently returns twice what was asked -- 92 mm where 43 mm was
    # intended, which is where the holes were being closed shut.
    diameter = 2 * target.circle_radius
    k_metres = min(max(closing_per_diameter * diameter, spacing), 0.5 * diameter)
    k = max(int(round(k_metres / cell)), 1) | 1
    solid = ndimage.binary_closing(occupied, np.ones((k, k), bool))

    board = ndimage.binary_fill_holes(solid)
    empty = board & ~solid

    labels, n = ndimage.label(empty)
    if n == 0:
        return []

    # A hole covers a known area, and the floor is not a matter of taste: a gap
    # crossed by fewer than `min_rings` rings cannot be fitted, so a gap that
    # narrow is not a candidate however round it looks.
    ideal = np.pi * target.circle_radius**2
    min_extent = min_rings * spacing
    area_floor = max(0.10 * ideal, np.pi * (min_extent / 2) ** 2 * 0.5)
    out = []
    for label in range(1, n + 1):
        ys, xs = np.where(labels == label)
        area = len(ys) * cell * cell
        if not area_floor <= area <= area_ceiling * ideal:
            continue
        # Reject long thin gaps -- a hole is round. The floor is the same ring
        # requirement as a width; the ceiling lets a hole merged with an adjacent
        # scan gap through, which the rectangle check sorts out later.
        extent = np.array([np.ptp(xs) + 1, np.ptp(ys) + 1]) * cell
        if extent.min() < min_extent or extent.max() > 3.0 * target.circle_radius:
            continue
        if extent.max() / max(extent.min(), 1e-9) > aspect_max:
            continue
        centre = np.array([xs.mean(), ys.mean()]) * cell + origin
        out.append((centre, float(np.sqrt(area / np.pi))))
    return out


def find_holes_experimental(
    xy: np.ndarray,
    spacing: float,
    target: Target,
    min_rings: float = 4.0,
    cell_per_spacing: float = 0.5,
    area_ceiling: float = 2.5,
    aspect_max: float = 2.2,
    closing_per_diameter: float = 0.30,
):
    """Where changes to the plane-raster detector get tried.

    A copy of `find_holes_by_occupancy`, deliberately. That one is what actually
    gets used on this rig -- the merged topic carries no ring field, so neither
    ring walk can run, and the C++ angular-gap test finds too little -- which
    makes it the wrong place to try things. So it is left alone and this is
    edited instead, and the two can be run against each other on the same scene
    from the method dropdown.

    Read the original for why each number is what it is; the reasoning is not
    duplicated here, because a copy of an argument is a copy that goes stale.
    Right now the code below is identical to it.

    Returns (centre, radius) per candidate, in plane coordinates.
    """
    if len(xy) < 50 or spacing <= 0:
        return []

    cell = max(spacing * cell_per_spacing, 0.004)
    origin = xy.min(axis=0)
    idx = np.floor((xy - origin) / cell).astype(int)
    h, w = idx[:, 1].max() + 1, idx[:, 0].max() + 1
    if h < 8 or w < 8 or h * w > 4_000_000:
        return []

    occupied = np.zeros((h, w), bool)
    occupied[idx[:, 1], idx[:, 0]] = True

    diameter = 2 * target.circle_radius
    k_metres = min(max(closing_per_diameter * diameter, spacing), 0.5 * diameter)
    k = max(int(round(k_metres / cell)), 1) | 1
    solid = ndimage.binary_closing(occupied, np.ones((k, k), bool))

    board = ndimage.binary_fill_holes(solid)
    empty = board & ~solid

    labels, n = ndimage.label(empty)
    if n == 0:
        return []

    ideal = np.pi * target.circle_radius**2
    min_extent = min_rings * spacing
    area_floor = max(0.10 * ideal, np.pi * (min_extent / 2) ** 2 * 0.5)
    out = []
    for label in range(1, n + 1):
        ys, xs = np.where(labels == label)
        area = len(ys) * cell * cell
        if not area_floor <= area <= area_ceiling * ideal:
            continue
        extent = np.array([np.ptp(xs) + 1, np.ptp(ys) + 1]) * cell
        if extent.min() < min_extent or extent.max() > 3.0 * target.circle_radius:
            continue
        if extent.max() / max(extent.min(), 1e-9) > aspect_max:
            continue
        centre = np.array([xs.mean(), ys.mean()]) * cell + origin
        out.append((centre, float(np.sqrt(area / np.pi))))
    return out


def refine_hole_centre(xy: np.ndarray, centre: np.ndarray, r_hint: float, tree: cKDTree | None = None):
    """Fit a circle to a hole's rim, starting from the raster's estimate.

    The raster resolves the centre only to a cell, so the rim is measured
    directly: sweep around the estimate and, in each sector, take the nearest
    point. Those are the returns that stopped at the hole's edge.

    Sectors are only trusted near the expected radius. A sector that happens to
    contain no rim sample would otherwise contribute whatever point lies beyond
    the hole, which drags the fitted radius out well past the real one -- 90 mm
    against a 71 mm hole in this data, enough to fail the size check outright.
    """
    tree = tree or cKDTree(xy)
    near = tree.query_ball_point(centre, r_hint * 2.0)
    if len(near) < 12:
        return None, 0.0, 0

    pts = xy[near]
    delta = pts - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    angle = np.arctan2(delta[:, 1], delta[:, 0])

    sectors = 36
    bucket = ((angle + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    lo, hi = 0.55 * r_hint, 1.6 * r_hint
    rim = []
    for s in range(sectors):
        m = bucket == s
        if not m.any():
            continue
        j = int(np.argmin(radius[m]))
        if lo < radius[m][j] < hi:
            rim.append(pts[m][j])
    if len(rim) < 10:
        return None, 0.0, 0

    rim = np.array(rim)
    # Algebraic least squares (Kasa), then one trimmed pass so a stray sector
    # cannot pull the circle.
    for _ in range(2):
        a = np.column_stack([2 * rim, np.ones(len(rim))])
        sol, *_ = np.linalg.lstsq(a, (rim**2).sum(axis=1), rcond=None)
        c = sol[:2]
        r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
        residual = np.abs(np.linalg.norm(rim - c, axis=1) - r)
        keep = residual <= max(2.0 * residual.mean(), 1e-4)
        if keep.all() or keep.sum() < 8:
            break
        rim = rim[keep]

    if float(residual.mean()) > 0.35 * r_hint:
        return None, 0.0, 0
    return c, r, len(rim)


def refine_hole_rim(xy: np.ndarray, centre: np.ndarray, r_hint: float, tree: cKDTree | None = None,
                    sectors: int = 36, eligible: np.ndarray | None = None):
    """`refine_hole_centre`, but it also says which points it used.

    Same fit, same sectors, same trimming -- the only difference is that the rim
    comes back as indices into `xy` rather than being thrown away.

    That matters because the display had nothing true to draw. The occupancy path
    paints "검출 지점" from `query_ball_point(centre, r * 1.35)`, which is every
    point in a disc half again as wide as the hole -- hundreds of them, the hole's
    whole neighbourhood filled in solid. The fit meanwhile used at most 36 points,
    one per sector. So the yellow blob on screen was never what the circle was
    measured from, and there was no way to see a bad rim by looking at it.

    `eligible` is an optional mask over `xy` restricting which points may be
    taken as rim. The angular-gap test makes a good one: it marks far too much
    to locate a hole with -- 40 to 100% of the panel on this rig, which is why
    the method built on it alone finds nothing -- but it does not *miss* rim,
    and that is the half worth keeping. Measured here, 98.6% of the points this
    function already picks are flagged by it, against 40-100% of the panel. So
    it costs almost no rim and it can drop an interior point that a sector with
    no true rim sample would otherwise have settled for.

    Returns (centre, radius, indices into xy).
    """
    tree = tree or cKDTree(xy)
    near = np.asarray(tree.query_ball_point(centre, r_hint * 2.0), dtype=np.int64)
    if len(near) < 12:
        return None, 0.0, np.empty(0, np.int64)

    pts = xy[near]
    delta = pts - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    angle = np.arctan2(delta[:, 1], delta[:, 0])

    bucket = ((angle + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    lo, hi = 0.55 * r_hint, 1.6 * r_hint
    picked = []
    ok = np.ones(len(near), bool) if eligible is None else eligible[near]
    for sector in range(sectors):
        m = np.flatnonzero((bucket == sector) & ok)
        if not len(m):
            continue
        j = m[int(np.argmin(radius[m]))]
        if lo < radius[j] < hi:
            picked.append(int(near[j]))
    if len(picked) < 10:
        return None, 0.0, np.empty(0, np.int64)

    idx = np.array(picked, np.int64)
    for _ in range(2):
        rim = xy[idx]
        a = np.column_stack([2 * rim, np.ones(len(rim))])
        sol, *_ = np.linalg.lstsq(a, (rim**2).sum(axis=1), rcond=None)
        c = sol[:2]
        r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
        residual = np.abs(np.linalg.norm(rim - c, axis=1) - r)
        keep = residual <= max(2.0 * residual.mean(), 1e-4)
        if keep.all() or keep.sum() < 8:
            break
        idx = idx[keep]

    if float(residual.mean()) > 0.35 * r_hint:
        return None, 0.0, np.empty(0, np.int64)
    return c, r, idx


def _select_rectangle(centers, radii, counts, target: Target, tolerance: float = 0.08):
    """Of many circle candidates, the four matching the board's hole rectangle.

    Scored on how well the four side lengths and the diagonals match the target,
    so a set has to be the right shape *and* the right size to win.
    """
    from itertools import combinations

    pts = np.array(centers)
    want_w, want_h = target.delta_width_circles, target.delta_height_circles
    want_d = float(np.hypot(want_w, want_h))

    best = None
    for combo in combinations(range(len(pts)), 4):
        quad = pts[list(combo)]
        centre = quad.mean(axis=0)
        rel = quad - centre
        _, _, vt = np.linalg.svd(rel, full_matrices=False)
        order = np.argsort(np.arctan2(rel @ vt[1], rel @ vt[0]))
        ring = quad[order]

        sides = [float(np.linalg.norm(ring[(i + 1) % 4] - ring[i])) for i in range(4)]
        diags = [float(np.linalg.norm(ring[2] - ring[0])), float(np.linalg.norm(ring[3] - ring[1]))]

        # Either winding is fine, so try the rectangle both ways round.
        errors = []
        for w, h in ((want_w, want_h), (want_h, want_w)):
            errors.append(
                max(abs(sides[0] - w) / w, abs(sides[2] - w) / w,
                    abs(sides[1] - h) / h, abs(sides[3] - h) / h,
                    abs(diags[0] - want_d) / want_d, abs(diags[1] - want_d) / want_d)
            )
        error = min(errors)
        if error < tolerance and (best is None or error < best[0]):
            best = (error, [combo[i] for i in order])

    if best is None:
        return centers, radii, counts
    idx = best[1]
    return [centers[i] for i in idx], [radii[i] for i in idx], [counts[i] for i in idx]


def detect(
    xyz: np.ndarray,
    box: FilterBox,
    target: Target,
    params: DetectParams | None = None,
    ring: np.ndarray | None = None,
) -> LidarDetection:
    """Run the whole chain on one accumulated cloud."""
    params = params or DetectParams()
    xyz = np.asarray(xyz, np.float32)

    method = params.method
    if method in ("ring", "original") and ring is None:
        # `original` covers the mech path only. detect_solid_lidar needs no ring
        # and is already what `boundary` ports, so duplicating it here would give
        # two names for one code path.
        return LidarDetection(
            False, method=method,
            reason="이 토픽에는 ring 필드가 없습니다 (ring 없는 원본 경로는 '경계점 각도'와 같습니다)",
        )

    # "original" follows the C++ main(): a ring field selects detect_mech_lidar,
    # its absence detect_solid_lidar. Only the ring walks read acquisition order,
    # and only they must skip voxelising -- the original does not voxelise on the
    # mech path either, for the same reason.
    walks_rings = method == "ring" or (method == "original" and ring is not None)

    keep = region_mask(xyz, box)
    if walks_rings:
        # The ring walk reads points in acquisition order, so this path must not
        # voxelise -- that reorders them and the walk stops meaning anything.
        filtered = xyz[keep]
        ring_in = np.asarray(ring)[keep]
    else:
        filtered = voxel_down(xyz[keep], params.voxel)
        ring_in = None
    if len(filtered) < 50:
        return LidarDetection(False, filtered=filtered, method=params.method,
                              reason=f"필터 후 점이 너무 적습니다 ({len(filtered)})")

    # Detection is asked for explicitly, so a wide box is allowed -- it just
    # takes longer and is unlikely to help, since the board is a fraction of a
    # metre across and the rest of the box is wall and floor.
    if len(filtered) > params.max_points:
        return LidarDetection(
            False, filtered=filtered, spacing=point_spacing(filtered), method=params.method,
            reason=f"박스 안 {len(filtered):,} 점 — 보드 주변으로 좁히면 훨씬 빠르고 정확합니다",
        )

    spacing = point_spacing(filtered)
    normal, d, inliers = fit_plane(filtered.astype(np.float64), params.plane_threshold)
    if normal is None or inliers.sum() < 50:
        return LidarDetection(False, filtered=filtered, spacing=spacing, method=params.method,
                              reason="평면을 찾지 못했습니다")

    plane = filtered[inliers]
    residual = plane.astype(np.float64) @ normal + d
    plane_rms = float(np.sqrt((residual**2).mean()))

    R = align_rotation(normal)
    aligned = plane.astype(np.float64) @ R.T
    average_z = float(aligned[:, 2].mean())
    xy = aligned[:, :2]

    R_inv = R.T
    cell_size = 0.0
    gap_used = 0.0
    centers, radii, counts = [], [], []
    edges_world = np.empty((0, 3), np.float32)
    n_groups = 0

    if method == "ring":
        # The C++ hard-codes 0.10 m, tuned for the stock board. It has to sit
        # between the point spacing and the hole, so scale it with the hole.
        gap = params.ring_gap or params.ring_gap_frac * (2 * target.circle_radius)
        gap_used = gap
        edge_mask, pair_id = boundary_points_by_ring(
            plane.astype(np.float64), ring_in[inliers], normal, d, gap,
            max_gap=2.2 * target.circle_radius,
        )
        edges_xy = xy[edge_mask]
        edges_world = plane[edge_mask]
        pairs = pair_id[edge_mask]
        if len(edges_xy) < params.cluster_min:
            return LidarDetection(
                False, filtered=filtered, plane=plane, plane_rms=plane_rms, spacing=spacing,
                method=method, reason=f"경계점이 너무 적습니다 ({len(edges_xy)})",
            )
        # Group by crossing, not by distance. Each crossing's midpoint lies on
        # the hole's centre line, so midpoints of one hole land within a ring
        # step of each other while the next hole's sit a full hole spacing away
        # (238 mm here) -- the separation single-linkage needs, which the rim
        # points themselves do not have.
        first = np.full(pairs.max() + 1 if len(pairs) else 0, -1, np.int64)
        second = np.full_like(first, -1)
        for i, p in enumerate(pairs):
            if first[p] < 0:
                first[p] = i
            else:
                second[p] = i
        both = np.flatnonzero((first >= 0) & (second >= 0))
        if len(both) < max(int(params.min_rings_per_hole), 3):
            return LidarDetection(
                False, filtered=filtered, plane=plane, edges=edges_world, plane_rms=plane_rms,
                spacing=spacing, method=method,
                reason=f"구멍을 지나는 링이 너무 적습니다 ({len(both)})",
            )
        mids = (edges_xy[first[both]] + edges_xy[second[both]]) / 2.0

        # A rim seen by rings yields two points per crossing, so how big a real
        # rim is depends on how many rings cross the hole -- `min_rings_per_hole`
        # already states that. The C++ minimum of 50 assumes the voxelised clouds
        # the other path produces and throws real holes away here.
        ring_floor = max(int(params.min_rings_per_hole), 3)
        groups = []
        for g in cluster(mids, params.cluster_tolerance):
            if len(g) < ring_floor:
                continue
            rim = np.concatenate([first[both[g]], second[both[g]]])
            if len(rim) <= params.cluster_max:
                groups.append(rim)
        n_groups = len(groups)
        for g in groups:
            c, r, mask = fit_circle(edges_xy[g], params.circle_threshold)
            if c is None or mask.sum() < 3:
                continue
            pts = edges_xy[g][mask]
            if float(np.abs(np.linalg.norm(pts - c, axis=1) - r).mean()) >= params.circle_max_error:
                continue
            if not (params.radius_min_ratio * target.circle_radius
                    <= r <= params.radius_max_ratio * target.circle_radius):
                continue
            centers.append(R_inv @ np.array([c[0], c[1], average_z]))
            radii.append(r)
            counts.append(int(mask.sum()))

    elif method == "original":
        # detect_mech_lidar, constants and all.
        gap_used = ORIG_NEIGHBOR_GAP
        edge_mask = edge_points_original(
            plane.astype(np.float64), ring_in[inliers], normal, d, ORIG_NEIGHBOR_GAP
        )
        edges_xy = xy[edge_mask]
        edges_world = plane[edge_mask]
        if len(edges_xy) < 3:
            return LidarDetection(
                False, filtered=filtered, plane=plane, plane_rms=plane_rms, spacing=spacing,
                method=method, ring_gap=gap_used,
                reason=f"경계점이 너무 적습니다 ({len(edges_xy)})",
            )
        found = find_circles_peel(
            edges_xy,
            target.circle_radius - ORIG_RADIUS_MARGIN,
            target.circle_radius + ORIG_RADIUS_MARGIN,
        )
        n_groups = len(found)
        for c, r, count in found:
            centers.append(R_inv @ np.array([c[0], c[1], average_z]))
            radii.append(r)
            counts.append(count)

    elif method == "occupancy":
        # Holes first, then measure each rim precisely.
        tree = cKDTree(xy)
        rim_points = []
        grid_spacing = params.sweep_spacing or spacing
        cell_size = max(grid_spacing * params.cell_per_spacing, 0.004)
        candidates = find_holes_by_occupancy(
            xy, grid_spacing, target, params.min_rings_per_hole,
            params.cell_per_spacing, params.area_ceiling, params.aspect_max,
            params.closing_per_diameter,
        )
        n_candidates = len(candidates)
        # Range to the board, for the radius the holes should measure at.
        distance = float(np.median(np.linalg.norm(plane, axis=1)))
        r_expect, r_tol = expected_radius(
            target, grid_spacing, distance, params.beam_divergence_mrad
        )
        for approx, r_area in candidates:
            c, r, n_rim = refine_hole_centre(xy, approx, max(r_area, target.circle_radius * 0.5), tree)
            if c is None:
                continue
            if abs(r - r_expect) > r_tol:
                continue
            centers.append(R_inv @ np.array([c[0], c[1], average_z]))
            radii.append(r)
            counts.append(n_rim)
            ring = tree.query_ball_point(c, r * 1.35)
            rim_points.extend(ring)
        n_groups = n_candidates
        if rim_points:
            edges_world = plane[np.unique(rim_points)]
    elif method == "experimental":
        # A copy of the occupancy branch, kept separate on purpose: this is the
        # one that gets edited, and the one that works has to stay working while
        # it is. Everything below may diverge from the branch above -- that is
        # the point of it existing.
        tree = cKDTree(xy)
        rim_points = []
        # Recall from one test, precision from the other: the gap test says which
        # points could be rim, the raster says where the holes are.
        eligible = (boundary_points(xy, params.boundary_radius, params.boundary_angle)
                    if params.exp_rim_boundary_only else None)
        grid_spacing = params.sweep_spacing or spacing
        cell_size = max(grid_spacing * params.cell_per_spacing, 0.004)
        candidates = find_holes_experimental(
            xy, grid_spacing, target, params.min_rings_per_hole,
            params.cell_per_spacing, params.area_ceiling, params.aspect_max,
            params.closing_per_diameter,
        )
        n_candidates = len(candidates)
        distance = float(np.median(np.linalg.norm(plane, axis=1)))
        r_expect, r_tol = expected_radius(
            target, grid_spacing, distance, params.beam_divergence_mrad
        )
        for approx, r_area in candidates:
            c, r, rim_idx = refine_hole_rim(
                xy, approx, max(r_area, target.circle_radius * 0.5), tree, eligible=eligible)
            if c is None:
                continue
            if abs(r - r_expect) > r_tol:
                continue
            centers.append(R_inv @ np.array([c[0], c[1], average_z]))
            radii.append(r)
            counts.append(len(rim_idx))
            # The points the circle was actually fitted to, and only those.
            rim_points.extend(rim_idx.tolist())
        n_groups = n_candidates
        if rim_points:
            edges_world = plane[np.unique(rim_points)]

    else:
        cell_size = 0.0
        edge_mask = boundary_points(xy, params.boundary_radius, params.boundary_angle)
        edges_xy = xy[edge_mask]
        edges_world = plane[edge_mask]
        if len(edges_xy) < params.cluster_min:
            return LidarDetection(
                False, filtered=filtered, plane=plane, plane_rms=plane_rms, spacing=spacing,
                method=params.method, reason=f"경계점이 너무 적습니다 ({len(edges_xy)})",
            )

        # Only clusters that could be a hole rim are worth fitting; wall and floor
        # edges would otherwise become hundreds of candidates to run RANSAC on.
        # Rims do not always arrive separately -- on a tilted board they link up
        # through the panel's own edge -- so oversized blobs get split on a grid
        # of hole-sized cells rather than discarded.
        span = 2.6 * target.circle_radius
        groups = []
        for g in cluster(edges_xy, params.cluster_tolerance):
            if len(g) < params.cluster_min:
                continue
            pts = edges_xy[g]
            if np.ptp(pts, axis=0).max() <= span and len(g) <= params.cluster_max:
                groups.append(g)
                continue
            grid = 2.0 * target.circle_radius
            keys = np.floor(pts / grid).astype(np.int64)
            for key in np.unique(keys, axis=0):
                sub = g[(keys == key).all(axis=1)]
                if len(sub) >= params.cluster_min:
                    groups.append(sub)
        n_groups = len(groups)

        for g in groups:
            c, r, mask = fit_circle(edges_xy[g], params.circle_threshold)
            if c is None or mask.sum() < 3:
                continue
            pts = edges_xy[g][mask]
            # Judge the fit against the circle actually found, and check size
            # separately: beam width makes holes read smaller than the CAD value
            # (about 20 mm at 11 m here), which alone exceeded the old tolerance.
            # Centres survive that shrinkage, so only the test needed changing.
            if float(np.abs(np.linalg.norm(pts - c, axis=1) - r).mean()) >= params.circle_max_error:
                continue
            if not (params.radius_min_ratio * target.circle_radius
                    <= r <= params.radius_max_ratio * target.circle_radius):
                continue
            centers.append(R_inv @ np.array([c[0], c[1], average_z]))
            radii.append(r)
            counts.append(int(mask.sum()))

    # More candidates than holes is normal: rim fragments, the panel's own edge,
    # whatever else the box caught. Pick the four forming the target's rectangle.
    if len(centers) > 4:
        centers, radii, counts = _select_rectangle(
            centers, radii, counts, target, params.rect_tolerance
        )

    det = LidarDetection(
        ok=len(centers) == 4,
        centers=np.array(centers) if centers else None,
        filtered=filtered,
        plane=plane,
        edges=edges_world,
        plane_rms=plane_rms,
        spacing=spacing,
        n_clusters=n_groups,
        method=method,
        cell_size=cell_size,
        ring_gap=gap_used,
        radii=radii,
        edge_counts=counts,
    )
    if not det.ok:
        det.reason = f"원 {len(centers)}개 검출 (4개 필요)"
    return det

"""Synthetic scans of the calibration board, with ground truth.

Detection thresholds have to come from somewhere. Tuning them against a
recording means tuning against one distance, one sensor and one board pose, and
there is no way to tell a value that generalises from one that happens to work.
Here the answer is known, so a threshold can be derived from a measured error
curve instead of chosen.

The scan is ray-traced rather than sampled on a grid, because the two effects
that decide whether a hole is detectable are both properties of the rays:

  * A spinning sensor samples in rings. What matters is not how many points land
    on the board but how many rings *cross a hole* -- three rings give six rim
    points, and they may all sit on the same short arc.
  * A beam has width. One that clips a hole's edge still returns off the panel,
    so holes read smaller than they are, by roughly the beam radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gui.core.project import Target


@dataclass
class Sensor:
    """A spinning LiDAR, described the way a datasheet does."""

    v_res_deg: float = 0.11  # angle between neighbouring rings
    h_res_deg: float = 0.11  # angle between samples along a ring
    beam_divergence_mrad: float = 3.0  # full angle
    range_noise_m: float = 0.02
    angle_jitter_frac: float = 0.15  # sweep-to-sweep wobble, as a fraction of a step

    def beam_radius(self, distance: float) -> float:
        return distance * (self.beam_divergence_mrad * 1e-3) / 2.0

    def spacing(self, distance: float) -> float:
        return distance * np.radians(self.v_res_deg)

    def rings_across(self, distance: float, diameter: float) -> float:
        """How many rings cross a hole of this size at this range."""
        return diameter / max(self.spacing(distance), 1e-9)


@dataclass
class BoardPose:
    distance: float = 6.0
    azimuth_deg: float = 0.0  # bearing from the sensor's +x axis
    elevation_deg: float = 0.0
    yaw_deg: float = 25.0  # board turned about its own vertical
    pitch_deg: float = 12.0  # board tipped about its own horizontal

    def frame(self):
        """(centre, right, up, normal) of the board in sensor coordinates."""
        a, e = np.radians(self.azimuth_deg), np.radians(self.elevation_deg)
        centre = self.distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

        # Start facing the sensor, then turn and tip about the board's own axes.
        normal = -centre / np.linalg.norm(centre)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, normal)
        right /= np.linalg.norm(right)
        up = np.cross(normal, right)

        for angle, axis in ((self.yaw_deg, up), (self.pitch_deg, right)):
            t = np.radians(angle)
            k = axis / np.linalg.norm(axis)
            rot = (
                np.eye(3) * np.cos(t)
                + np.sin(t) * np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                + (1 - np.cos(t)) * np.outer(k, k)
            )
            normal, right, up = rot @ normal, rot @ right, rot @ up
        return centre, right, up, normal


def board_extent(target: Target, margin: float = 0.03) -> tuple[float, float]:
    """Half-width and half-height of a panel that fits the printed target.

    Whichever reaches further -- the hole pattern or the marker pattern -- sets
    the size, plus a margin of panel beyond it.
    """
    half_w = max(target.delta_width_circles / 2 + target.circle_radius,
                 target.delta_width_qr_center / 2 + target.marker_size / 2) + margin
    half_h = max(target.delta_height_circles / 2 + target.circle_radius,
                 target.delta_height_qr_center / 2 + target.marker_size / 2) + margin
    return half_w, half_h


def hole_centres(target: Target, pose: BoardPose) -> np.ndarray:
    """The four hole centres in sensor coordinates -- the ground truth."""
    centre, right, up, _ = pose.frame()
    return np.array([
        centre + su * target.delta_width_circles / 2 * right + sv * target.delta_height_circles / 2 * up
        for su in (-1, 1)
        for sv in (-1, 1)
    ])


def scan(
    target: Target,
    pose: BoardPose,
    sensor: Sensor,
    sweeps: int = 1,
    board_half_w: float | None = None,
    board_half_h: float | None = None,
    margin: float = 0.03,
    wall_distance: float = 3.0,
    seed: int = 0,
):
    """Ray-trace `sweeps` sweeps of the board, plus a wall behind it.

    Returns (points, ground-truth hole centres, ring numbers). The ring number
    is the elevation index, offset per sweep exactly as `accumulate_cloud` does
    when it stacks real sweeps -- otherwise a walk along "one ring" would jump
    between sweeps. Points come back in acquisition order (ring, then azimuth),
    which the ring walk depends on.
    """
    # The panel has to be big enough to hold what is printed on it, and it must
    # follow `target` when the board is rescaled. Fixed half-extents do not: at
    # the stock CAD the holes span 640 mm vertically, so a 600 mm panel puts them
    # through its own edge, the silhouette opens, and a hole stops being enclosed
    # by the board at all. Detection then fails for a reason the real board does
    # not have -- which is worth stating, because every threshold here is
    # justified by numbers this generator produced.
    half_w, half_h = board_extent(target, margin)
    board_half_w = half_w if board_half_w is None else board_half_w
    board_half_h = half_h if board_half_h is None else board_half_h

    rng = np.random.default_rng(seed)
    centre, right, up, normal = pose.frame()
    holes = np.array([
        (su * target.delta_width_circles / 2, sv * target.delta_height_circles / 2)
        for su in (-1, 1)
        for sv in (-1, 1)
    ])

    # Aim only where the board is, with a margin, so the scan stays cheap.
    corners = np.array([
        centre + sw * board_half_w * 1.25 * right + sh * board_half_h * 1.4 * up
        for sw in (-1, 1)
        for sh in (-1, 1)
    ])
    az = np.degrees(np.arctan2(corners[:, 1], corners[:, 0]))
    el = np.degrees(np.arcsin(corners[:, 2] / np.linalg.norm(corners, axis=1)))

    out, out_ring, out_az = [], [], []
    for s in range(sweeps):
        # Each sweep starts at a slightly different phase; this is what makes
        # stacking sweeps sample the surface more finely instead of repeating.
        jitter = sensor.angle_jitter_frac
        d_az = rng.uniform(-jitter, jitter) * sensor.h_res_deg
        d_el = rng.uniform(-jitter, jitter) * sensor.v_res_deg

        azimuths = np.radians(np.arange(az.min(), az.max() + sensor.h_res_deg, sensor.h_res_deg) + d_az)
        elevations = np.radians(np.arange(el.min(), el.max() + sensor.v_res_deg, sensor.v_res_deg) + d_el)
        aa, ee = np.meshgrid(azimuths, elevations)
        # meshgrid varies azimuth fastest, so raveling already gives acquisition
        # order: ring by ring, sweeping in azimuth within each.
        rings = np.repeat(np.arange(len(elevations)), len(azimuths)) + s * len(elevations)
        aa, ee = aa.ravel(), ee.ravel()

        dirs = np.column_stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)])

        denom = dirs @ normal
        live = np.abs(denom) > 1e-9
        t = np.full(len(dirs), np.inf)
        t[live] = (centre @ normal) / denom[live]
        hit = live & (t > 0)
        if not hit.any():
            continue

        pts = dirs[hit] * t[hit, None]
        local = pts - centre
        u, v = local @ right, local @ up
        on_board = (np.abs(u) <= board_half_w) & (np.abs(v) <= board_half_h)

        # A beam is lost as soon as it *touches* a hole, not only when it fits
        # inside one. The spot is a disc about 10 mm across at 3 m: overhang the
        # rim and too little energy comes back to register, and the rim itself is
        # a wall, so what light does hit it goes sideways. Holes therefore read
        # about a beam radius too large.
        #
        # This used to require the beam to fit entirely inside, which made holes
        # read small. That was the same assumption `expected_radius` made, so the
        # generator was granting the premise the model was being checked against
        # -- synthetic radii agreed with the model while real ones were 10 mm out.
        # A tape measure settled it: the board's holes are 142 mm across and the
        # detectors were reporting 160 mm.
        beam_r = sensor.beam_radius(pose.distance)
        through = np.zeros(len(pts), bool)
        for hu, hv in holes:
            through |= np.hypot(u - hu, v - hv) < target.circle_radius + beam_r

        hit_ring, hit_az = rings[hit], aa[hit]
        keep = on_board & ~through
        board_pts = pts[keep]
        if len(board_pts):
            r = np.linalg.norm(board_pts, axis=1)
            board_pts = board_pts * (1 + rng.normal(0, sensor.range_noise_m, len(r)) / r)[:, None]
            out.append(board_pts)
            out_ring.append(hit_ring[keep])
            out_az.append(hit_az[keep])

        # Rays that made it through the holes, or missed the panel, carry on to
        # a wall -- so the box filter sees something behind the board, as it does
        # in a real capture.
        missed = ~keep
        if missed.any() and wall_distance > 0:
            far = dirs[hit][missed] * (t[hit][missed] + wall_distance)[:, None]
            r = np.linalg.norm(far, axis=1)
            out.append(far * (1 + rng.normal(0, sensor.range_noise_m, len(r)) / r)[:, None])
            out_ring.append(hit_ring[missed])
            out_az.append(hit_az[missed])

    if not out:
        empty = np.empty((0, 3), np.float32)
        return empty, hole_centres(target, pose), np.empty(0, np.int32)

    # Board and wall returns were collected separately, so put them back into
    # acquisition order: the walk reads a ring's neighbours by their position in
    # the array, and a beam that went through a hole must land between the rim
    # points either side of it, not at the end of the ring.
    points = np.vstack(out).astype(np.float32)
    ring = np.concatenate(out_ring).astype(np.int32)
    order = np.lexsort((np.concatenate(out_az), ring))
    return points[order], hole_centres(target, pose), ring[order]


def bounding_box(centres: np.ndarray, pad: float = 0.35):
    """A filter box around the board, as a user would draw it."""
    from gui.core.project import FilterBox

    lo, hi = centres.min(axis=0) - pad, centres.max(axis=0) + pad
    return FilterBox(
        x_min=float(lo[0]), x_max=float(hi[0]),
        y_min=float(lo[1]), y_max=float(hi[1]),
        z_min=float(lo[2]), z_max=float(hi[2]),
    )


def centre_error(found: np.ndarray | None, truth: np.ndarray) -> float:
    """Mean distance from each detected centre to its nearest true one, in metres."""
    if found is None or len(found) != len(truth):
        return float("nan")
    return float(np.mean([np.min(np.linalg.norm(truth - c, axis=1)) for c in found]))

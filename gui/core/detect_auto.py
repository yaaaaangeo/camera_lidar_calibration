"""Trying a few sweep depths and keeping whichever saw the board best.

NOT WIRED UP, AND MUST NOT BE WITHOUT FIXING `choose` FIRST.

This is half a feature. It takes a `run(frames)` callback rather than reading
the bag itself, so that it stays testable headless -- which means something that
does read the bag has to drive it, and that half was never written. The split of
`bag_reader.accumulate_cloud` into `sweeps_around` / `middle_slice` /
`stack_sweeps` exists for exactly that caller, which is why `middle_slice` has no
other user either. The two are the same unfinished feature.

Leaving it unfinished is the right state for now, because connecting it as it
stands makes the calibration worse. Measured over eighteen scenes:

    one sweep everywhere       8 scenes found four holes, extrinsic RMSE  47 mm
    depth chosen by `choose`  11 scenes found four holes, extrinsic RMSE 117 mm

More scenes, a worse answer. `choose` ranks on hole count first
(`key = (-n_holes, shape_err, radius_err)`), so it takes four blurred holes over
three sharp ones. One scene there finds four holes at two sweeps whose own
residual against the fitted extrinsic is 311 mm -- the board had moved between
the sweeps. Finishing this means changing what `choose` optimises, not writing
the caller.

One sweep is the right default and usually enough, but not always: a board far
enough away that only a few beams cross each hole can come back with two of the
four, and stacking a couple more sweeps fills the rims in. It costs nothing to
find out -- the sweeps nest, so five are read once and the smaller depths are the
middle of them.

Only for detectors that do not read points in acquisition order. The ring-based
ones walk each ring looking for the jump across a hole's edge, and stacking
sweeps blurs exactly that: on ten consecutive sweeps of a stationary board the
original method found four holes out of four every time at one sweep.

A hand-held board drifts, so more sweeps can also smear it. This file used to
claim a smeared board "fails the shape check and loses on score". It does not:
the scene above measured 484 x 418 mm against a 500 x 400 target and passed
comfortably. A board that moved a few centimetres between sweeps still reads as
the right rectangle, in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gui.core.detect_lidar import expected_radius
from gui.core.project import Target

# Depths worth trying, in order. Beyond five the board has usually moved.
FRAME_CHOICES = (1, 2, 3, 4, 5)

# Detectors that read points in acquisition order and are hurt by stacking.
RING_METHODS = ("original", "ring")

# Two depths whose scores differ by less than this count as equally good, and
# then the shallower one wins. Stacking can genuinely sharpen a fit -- more
# points per rim -- so it is worth trying even when one sweep already found four
# holes; but a difference this small is not evidence of anything.
SCORE_TIE = 0.01

# Stacking a board that moved raises the plane residual, and that is the one
# failure the shape check cannot see: the rims blur, the fitted centres land on
# the board's *average* pose over the window, and the camera saw a single
# instant. The rectangle still comes out the right shape while the pair it forms
# is wrong. Anything this much worse than one sweep is treated as movement.
MAX_PLANE_GROWTH = 1.6


@dataclass
class Attempt:
    frames: int
    det: object
    n_holes: int
    shape_err: float
    radius_err: float

    @property
    def key(self):
        """Sort key: more holes first, then the better-shaped, then radii.

        Hole count dominates on purpose. Three well-placed centres carry more
        than four that only roughly form the rectangle, but four is what lets a
        scene stand on its own, and a partial set costs a candidate search later.
        """
        return (-self.n_holes, round(self.shape_err, 4), round(self.radius_err, 4))


def shape_error(centres, target: Target) -> float:
    """How far the found centres are from the board's own geometry, 0 is exact.

    Defined for two, three or four centres so a partial detection can still be
    ranked. With one there is nothing to compare and it returns 0 -- the hole
    count already says how little was found.
    """
    if centres is None:
        return 1.0
    pts = np.asarray(centres, float).reshape(-1, 3)
    w, h = target.delta_width_circles, target.delta_height_circles
    d = float(np.hypot(w, h))
    if len(pts) < 2:
        return 0.0

    dists = sorted(
        float(np.linalg.norm(pts[i] - pts[j]))
        for i in range(len(pts)) for j in range(i + 1, len(pts))
    )
    if len(pts) == 4:
        want = sorted([h, h, w, w, d, d])
    elif len(pts) == 3:
        want = sorted([w, h, d])
    else:
        # A pair is one of the sides or the diagonal; score against the closest,
        # since which one it is cannot be known from the distance alone.
        return float(min(abs(dists[0] - t) / t for t in (w, h, d)))
    return float(max(abs(a - b) / b for a, b in zip(dists, want)))


def radius_error(det, target: Target, distance: float) -> float:
    """Mean gap between the fitted circle radii and what that range should give."""
    radii = getattr(det, "radii", None)
    if not radii:
        return 0.0
    want, _ = expected_radius(target, getattr(det, "spacing", 0.0) or 0.0, distance)
    return float(np.mean([abs(r - want) for r in radii]) / max(want, 1e-6))


def rate(det, target: Target, distance: float, frames: int) -> Attempt:
    n = 0 if getattr(det, "centers", None) is None else len(det.centers)
    return Attempt(
        frames=frames,
        det=det,
        n_holes=n,
        shape_err=shape_error(getattr(det, "centers", None), target),
        radius_err=radius_error(det, target, distance),
    )


def best_frames(run, target: Target, distance: float, method: str,
                choices=FRAME_CHOICES) -> tuple[Attempt, list[Attempt]]:
    """Try each depth and return the best attempt, plus all of them for reporting.

    `run(frames)` does one detection at that depth and returns its
    `LidarDetection`; the caller owns the bag reading so this stays testable.
    """
    if method in RING_METHODS:
        choices = (1,)
    tried = []
    for n in choices:
        det = run(n)
        if det is None:
            continue
        tried.append(rate(det, target, distance, n))
    if not tried:
        return None, []
    return choose(tried), tried


def choose(tried: list[Attempt]) -> Attempt:
    """The shallowest depth that is as good as the best one found.

    Deeper is tried in full rather than stopped at the first four holes, because
    more points per rim can genuinely tighten the circle fits. But depth is only
    taken when it earns it: among attempts that score within a hair of the best,
    the one reading fewest sweeps wins, since every extra sweep is another chance
    for the board to have moved.
    """
    base = next((a for a in tried if a.frames == min(x.frames for x in tried)), None)
    ok = tried
    if base is not None:
        rms = getattr(base.det, "plane_rms", 0.0) or 0.0
        if rms > 0:
            steady = [a for a in tried
                      if (getattr(a.det, "plane_rms", 0.0) or 0.0) <= rms * MAX_PLANE_GROWTH]
            ok = steady or tried

    best = min(ok, key=lambda a: a.key)
    close = [
        a for a in ok
        if a.n_holes == best.n_holes
        and (a.shape_err + a.radius_err) <= (best.shape_err + best.radius_err) + SCORE_TIE
    ]
    return min(close, key=lambda a: a.frames) if close else best

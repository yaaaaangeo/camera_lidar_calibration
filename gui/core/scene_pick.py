"""Choosing which moments in a bag to calibrate from.

Scanning a recording is already cheap; deciding which frames to keep is the part
that was left to the eye, and it is the part repeated seven to eighteen times per
project. What follows is that decision written down.

Three stages, and they are kept apart on purpose.

A frame has to be *usable*: four markers, inside the range the distortion model
was fitted over, and a pose the markers actually support. That is a floor, not a
standard -- everything above it is a matter of degree.

It has to be *steady enough to pair*. The two sensors are read at different
instants, so a board still travelling has already been measured in two places
before the fit sees it, and no later step can undo that. What decides is how far
it moves in the gap, in millimetres, which makes a slow recording strict and a
fast one permissive without anything to configure.

And the set as a whole has to be *spread*, because that is the only thing that
conditions a rigid fit: ten views from one spot pin down far less than three
from different ones. Spread is measured as the fit sees it -- the singular
values of the correspondences -- rather than through a stand-in for it.

Nothing about how comfortable a frame looks enters the third stage. A sharper
copy of a pose already held adds nothing to the answer.

Kept free of Qt so it can be run against a bag from a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Only what makes a frame unusable is rejected. The scrubber's warning levels --
# 40 px markers, 12 degrees of tilt -- were tried here first and threw away 17 of
# the 18 scenes a person had chosen on this rig: at 9 m the markers are 18 px and
# the board is often held near head-on, and both still detect cleanly with under
# 1 px of reprojection error. Those numbers tell the eye "look closer", not
# "discard", and using them as a pass mark also collapsed the distance spread,
# because everything far away went out at once.
MIN_MARKERS = 4

# Kept only to score with, never to reject on.
GOOD_MARKER_PX = 40.0
GOOD_TILT_DEG = 12.0

# Reprojection error is reported in pixels, and a pixel is not a fixed amount of
# anything: the same one is 2.9 mm on the board at 4 m and 8.9 mm at 9 m. Scored
# in pixels it therefore *undoes* the marker-size term rather than adding to it
# -- on an eighteen-scene recording spanning 3.9 to 9.7 m it contributed -10% of
# the score's variance, rewarding distant frames for an error that is physically
# three times worse. Converted to millimetres on the board the sign comes right,
# but it then tracks marker size at +0.86 and says little the size term has not
# already said. So: converted, kept as a floor, and given a small weight.
#
#   mm on the board = reprojection px * distance / fx
GOOD_REPROJ_MM = 5.0
MAX_REPROJ_MM = 12.0

# Two samples belong to the same hold if the board barely moved between them.
# Generous, because a camera topic can be as slow as 1.7 Hz and the board drifts
# noticeably between frames even while someone is holding it still.
STILL_SHIFT_M = 0.06
STILL_TURN_DEG = 3.0

# How far the board may travel between the image and the sweep it gets paired
# with. The scene's instant comes off the camera clock, so the image is exact and
# the nearest sweep is up to half a sweep interval away; multiply that by how
# fast the board is moving and the answer is how far apart the two sensors think
# the same board is. Nothing downstream can recover it -- it is baked into the
# correspondence before the fit ever runs.
#
# 20 mm costs almost nothing and catches a great deal. On a 19 Hz recording the
# median frame is 6.4 mm out and the threshold keeps 211 of 264 stretches, with
# the conditioning of the chosen set moving 38.1 to 37.5. On one that came out
# at 0.7 Hz the median is 141 mm and the worst 1231 mm -- twice the spacing of
# the holes being matched -- and cutting at 20 mm *improves* the chosen set,
# 72.4 to 73.6, because what it removes is unusable.
#
# The threshold is in millimetres rather than seconds on purpose: a slow
# recording tightens itself and a fast one relaxes, with no setting to get wrong.
MAX_PAIR_MM = 20.0

# Deepest stack worth reporting. A board genuinely at rest gives an unbounded
# answer, and "41 sweeps" is not a useful thing to tell anyone -- past a handful
# the gain is gone and the risk of the board having drifted is not.
MAX_STACK = 15

# Requiring instead that the board hold still long enough to stack five sweeps
# was tried and is the wrong lever. Still moments cluster where the person
# paused, so demanding depth collapses the spread: on the 19 Hz recording the
# pool fell from 264 stretches to 33 and conditioning from 38.1 to 10.6, more
# than three times worse, in exchange for a depth most scenes have no need of.
# Stacking is per scene and optional -- `Candidate.safe_frames` says how deep
# each one may go, and a scene that can only take one sweep takes one and has no
# stacking error at all.

# Image is cut this many ways along each axis when reporting and rewarding
# coverage. Three is coarse on purpose -- it asks "did the board ever go left,
# or high", not "is the spread even".
COVER_CELLS = 3

# How much opening a new image cell, or a new band of tilt, is worth against the
# conditioning it costs. Conditioning cannot see either: two poses at opposite
# corners of the image with the same board geometry look identical to it.
COVER_BONUS = 0.25
TILT_BAND_DEG = 8.0


@dataclass
class Candidate:
    """One moment worth calibrating from."""

    t_ns: int
    distance: float
    tilt_deg: float
    reproj_rms: float  # pixels, as the detector reports it
    reproj_mm: float  # the same error as a length on the board
    marker_px: float
    centre: np.ndarray  # board centre in the camera frame
    normal: np.ndarray  # board normal in the camera frame, unit
    centers: np.ndarray  # the four hole centres, camera frame -- what gets fitted
    off_axis: float  # tan of the angle from the optical axis
    score: float = 0.0  # 0..1 comfort, for ranking within a pose
    hold: int = 0  # which still stretch it came from
    # How many sweeps could be stacked here before the board has moved too far.
    # Filled in by `measure_depths`, which reads it off the track.
    safe_frames: int = 1
    speed: float = 0.0  # m/s, from the frames either side
    pair_mm: float = 0.0  # how far the board moves in half a sweep, at that speed
    note: str = ""


@dataclass
class Coverage:
    """What the chosen set actually spans -- the part a better picker cannot fix.

    Both recordings measured here put every board within 18% of the image height,
    because nobody held it high or low. No choice among those frames recovers the
    other 82%; only recording again does. Saying so is the useful thing the tool
    can do about it.
    """

    x_frac: float = 0.0  # of the image width
    y_frac: float = 0.0
    cells: int = 0  # of COVER_CELLS^2
    distance: tuple[float, float] = (0.0, 0.0)
    tilt_range: float = 0.0
    condition: float = 0.0  # sigma2 * sigma3 of the fitted points, see `spread`
    pair_mm: float = 0.0  # the worst of the chosen set
    safe_frames: int = 1  # the shallowest of the chosen set


@dataclass
class PickResult:
    chosen: list[Candidate] = field(default_factory=list)
    holds: list[Candidate] = field(default_factory=list)  # middle frame of every hold
    coverage: Coverage = field(default_factory=Coverage)
    n_scanned: int = 0
    n_usable: int = 0
    rejected: dict[str, int] = field(default_factory=dict)


def _board_normal(rvec) -> np.ndarray:
    """Board normal in the camera frame, pointing back towards the camera."""
    import cv2

    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    n = R[:, 2]
    return -n if n[2] > 0 else n


def reproj_mm(det, fx: float) -> float:
    """Reprojection error as a length on the board rather than in the image.

    A pixel subtends more of the board the further away the board is, so the
    pixel figure flatters distant frames exactly where they are worst. Dividing
    by the focal length and multiplying by the range undoes that.
    """
    if not fx:
        return 0.0
    return float(det.reproj_rms * det.distance / fx * 1000.0)


def usable(det, radial_limit: float | None = None, fx: float = 0.0) -> str:
    """Empty string when the frame can be used, otherwise why not.

    Deliberately permissive: a frame is only thrown out when the board could not
    be measured at all, or when it sits where the distortion model is guessing.
    Everything else is a matter of degree and belongs in `quality`.
    """
    if det is None or not getattr(det, "ok", False):
        return "검출 실패"
    if getattr(det, "n_markers", 0) < MIN_MARKERS:
        return "마커 부족"
    if radial_limit is not None and det.tvec is not None:
        t = np.asarray(det.tvec, float).reshape(3)
        if t[2] > 1e-6 and np.hypot(t[0], t[1]) / t[2] > radial_limit:
            # Past here the distortion model is extrapolating, so the board's
            # measured centre is not to be trusted -- see verify.radial_limit.
            return "왜곡 모델 범위 밖"
    # A floor, not a standard. Across two recordings the usable frames all sat
    # between 2.9 and 11.1 mm, so tightening this cuts nothing but the far
    # frames -- and those are what give the set its range of depths, which is
    # what conditions the fit. Tightening here would make the answer worse.
    if fx and reproj_mm(det, fx) > MAX_REPROJ_MM:
        return "재투영 오차 과다"
    return ""


def quality(det, fx: float = 0.0) -> float:
    """0 to 1, how comfortable a frame is. Reported; it decides nothing.

    Each term saturates at the level the scrubber calls comfortable, so a frame
    that clears all three scores 1 and further improvement buys nothing.

    It used to break ties -- which frame of a hold to keep, where to start the
    diversity search. Both are now settled on their own terms, because a frame
    being pleasant to look at is not a reason to calibrate from it. What decides
    is stillness and the spread of the set.
    """
    size = min(det.marker_px / GOOD_MARKER_PX, 1.0)
    tilt = min(det.tilt_deg / GOOD_TILT_DEG, 1.0)
    mm = reproj_mm(det, fx) if fx else 0.0
    reproj = min(GOOD_REPROJ_MM / mm, 1.0) if mm > 0 else 1.0
    return float(size * 0.5 + tilt * 0.4 + reproj * 0.1)


def to_candidate(t_ns: int, det, fx: float = 0.0) -> Candidate:
    t = np.asarray(det.tvec, float).reshape(3)
    return Candidate(
        t_ns=int(t_ns),
        distance=float(det.distance),
        tilt_deg=float(det.tilt_deg),
        reproj_rms=float(det.reproj_rms),
        reproj_mm=reproj_mm(det, fx) if fx else 0.0,
        marker_px=float(det.marker_px),
        centre=t,
        normal=_board_normal(det.rvec),
        centers=np.asarray(det.centers, float).reshape(-1, 3),
        off_axis=float(np.hypot(t[0], t[1]) / max(t[2], 1e-9)),
        score=quality(det, fx),
    )


def measure_motion(cands: list[Candidate], sweep_s: float) -> None:
    """Fill in how fast the board was moving, and what that costs at pairing time.

    Speed comes from the frames either side rather than from a fitted track: the
    board is carried by hand, so there is no motion to model, only the distance
    it happened to cover. In place, because every caller wants it on the
    candidates it already has.
    """
    order = sorted(cands, key=lambda c: c.t_ns)
    for i, c in enumerate(order):
        a, b = order[max(0, i - 1)], order[min(len(order) - 1, i + 1)]
        dt = (b.t_ns - a.t_ns) / 1e9
        c.speed = float(np.linalg.norm(b.centre - a.centre) / dt) if dt > 0 else 0.0
        c.pair_mm = c.speed * sweep_s / 2 * 1000.0


def measure_depths(cands: list[Candidate], sweep_s: float,
                   limit_mm: float = MAX_PAIR_MM, cap: int = MAX_STACK) -> None:
    """Fill in how many sweeps each candidate could stack before the board moves.

    A stack of n is centred on the instant, so it spans (n-1)/2 sweep intervals
    either side and the board has travelled that much further by the outermost
    one. The same millimetres that decide whether a moment can be paired at all
    decide how deep it can go: one limit, applied twice.

    It has to be the speed, extrapolated, rather than the track read directly
    over the window -- which was tried, and cannot work. The window is a sweep or
    two wide, 53 ms on a 19 Hz LiDAR, while the scan that produced these
    candidates samples the camera a few hundred milliseconds apart to keep the
    button responsive. There is no frame inside the window to read. Measured that
    way every candidate in a 30 Hz recording came back at one sweep, not because
    the board was moving but because nothing had been sampled to say otherwise.

    Counting frames of the still stretch instead reads wrong the other way: the
    detector drops out on two frames in three, chopping a motionless board into
    stretches two samples long, and 222 of 264 stretches came out at one sweep
    while their boards were moving at 2 to 26 cm/s.

    So the speed over the sampling interval stands in for the speed over the
    window. It is an average, and it would miss a board set down and picked up
    again inside a single interval; nothing available at this sampling rate would
    catch that.
    """
    if sweep_s <= 0:
        for c in cands:
            c.safe_frames = 1
        return
    limit = limit_mm / 1000.0
    for c in cands:
        if c.speed <= 1e-6:
            c.safe_frames = cap
            continue
        half = limit / (c.speed * sweep_s)
        c.safe_frames = int(np.clip(int(half) * 2 + 1, 1, cap))


def group_holds(cands: list[Candidate]) -> list[Candidate]:
    """One frame per stretch where the board stayed put -- the middle of it.

    Without this the picker keeps returning neighbours of whatever moment scored
    best, since a board held still for two seconds gives twenty near-identical
    frames. It also drops the blurred ones taken while the board was being moved,
    because those sit alone in their own short stretch.

    The middle, not the sharpest. Step 5 stacks sweeps *centred* on whatever
    instant is chosen here, so a frame at the edge of a stretch has the stack
    reaching straight out into the motion that ended it. Measured on a 19 Hz
    recording, over the 42 stretches at least three frames long: taking the
    sharpest left a median of 3 sweeps that could safely be stacked and put 13
    stretches down to a single one, while taking the middle left a median of 11
    and none at 1. Same recording, same stretches -- only where in them the
    instant was taken.

    """
    if not cands:
        return []
    order = sorted(cands, key=lambda c: c.t_ns)
    holds: list[list[Candidate]] = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        shift = float(np.linalg.norm(cur.centre - prev.centre))
        turn = float(np.degrees(np.arccos(np.clip(prev.normal @ cur.normal, -1.0, 1.0))))
        if shift <= STILL_SHIFT_M and turn <= STILL_TURN_DEG:
            holds[-1].append(cur)
        else:
            holds.append([cur])

    best = []
    for i, group in enumerate(holds):
        span = (group[0].t_ns + group[-1].t_ns) / 2
        pick = min(group, key=lambda c: abs(c.t_ns - span))
        pick.hold = i
        held = (group[-1].t_ns - group[0].t_ns) / 1e9
        pick.note = f"{len(group)}프레임 {held:.1f}초 정지"
        best.append(pick)
    return best


def spread(cands: list[Candidate]) -> np.ndarray:
    """Singular values of the correspondences these candidates would contribute.

    This is not a proxy for how well conditioned the extrinsic will be; it is
    the thing itself. The fit is a rigid 3D-3D one, so the rotation is pinned
    down by how the matched points lie about their own centroid, and how they
    lie about it is what these three numbers say. The camera side alone is
    enough to compute it -- no LiDAR, no bag reading, so it can be evaluated
    once per candidate at selection time.

    The rotation that is worst determined is the one about the axis the points
    are most strung out along, and what pins *that* down is the two smaller
    values. Hence `sigma2 * sigma3` as the quantity to maximise, rather than the
    largest value or the condition number.
    """
    if not cands:
        return np.zeros(3)
    pts = np.vstack([c.centers for c in cands])
    if len(pts) < 3:
        return np.zeros(3)
    return np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)


def condition(cands: list[Candidate]) -> float:
    v = spread(cands)
    return float(v[1] * v[2]) if len(v) >= 3 else 0.0


def _bearing(c: Candidate) -> np.ndarray:
    """Where the board sits on the normalised image plane, before distortion.

    Enough to say which part of the frame it was in, and it needs no intrinsics
    and no image size -- both of which the picker would otherwise have to be
    handed just to bucket something into ninths.
    """
    z = max(c.centre[2], 1e-9)
    return np.array([c.centre[0] / z, c.centre[1] / z])


def _cells(cands: list[Candidate], n: int = COVER_CELLS) -> dict[int, tuple[int, int]]:
    """Which patch of the frame each candidate falls in, over the pool's own span.

    Relative to what the recording contains rather than to the sensor's field.
    The question the bonus needs answered is "does this open up somewhere the set
    has not been yet", and that is about the frames on offer. What the set covers
    of the *image* is a separate question, and `coverage` answers it in the
    honest units -- against the lens, where it can say 18% and mean it.
    """
    b = np.array([_bearing(c) for c in cands])
    lo, hi = b.min(axis=0), b.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    idx = np.clip(((b - lo) / span * n).astype(int), 0, n - 1)
    return {id(c): (int(i), int(j)) for c, (i, j) in zip(cands, idx)}


def select(cands: list[Candidate], n: int) -> list[Candidate]:
    """The n whose correspondences condition the fit best.

    Greedy: start from the furthest board, then repeatedly add whichever
    candidate lifts `condition` the most. Greedy rather than exhaustive because
    the choice is 15 from a few hundred, and each step only needs one small SVD.

    Two things conditioning cannot see, so they are nudged for. Where the board
    sat in the image: two poses at opposite corners with the same geometry look
    identical to it, and the corners are where the distortion model is doing its
    guessing, so a set that never goes there never finds out the model is wrong.
    And how far the board was tilted: the fit does not care, but a set at one
    tilt lets any systematic error in hole-finding be absorbed whole into the
    extrinsic instead of showing up as residual.

    Measured against the nearest-neighbour spread this replaces, picking 15:
    conditioning doubled on one recording (19.4 -> 38.1) and rose 23% on the
    other (63.0 -> 77.3), with the tilt range going up on both and the image
    coverage no worse. Conditioning alone did better still on the number it
    optimises and paid for it elsewhere -- 37 degrees of tilt down to 20 on one
    recording, image coverage from six patches to four on the other.
    """
    if n >= len(cands):
        return list(cands)
    if n <= 0 or not cands:
        return []

    cell = _cells(cands)
    # Furthest first: it is the one decision no later step can undo, since range
    # is the axis a hand-held board covers least willingly.
    taken = [max(cands, key=lambda c: c.distance)]
    rest = [c for c in cands if c is not taken[0]]
    while len(taken) < n and rest:
        seen_cell = {cell[id(c)] for c in taken}
        seen_tilt = {int(c.tilt_deg // TILT_BAND_DEG) for c in taken}
        best, best_score = None, -1.0
        for c in rest:
            score = condition(taken + [c])
            if cell[id(c)] not in seen_cell:
                score *= 1 + COVER_BONUS
            if int(c.tilt_deg // TILT_BAND_DEG) not in seen_tilt:
                score *= 1 + COVER_BONUS
            if score > best_score:
                best, best_score = c, score
        taken.append(best)
        rest.remove(best)
    return sorted(taken, key=lambda c: c.t_ns)


def coverage(cands: list[Candidate], camera=None, size=None) -> Coverage:
    """What the chosen set spans, for the report next to it.

    The image fractions need the lens and the frame size, since the whole point
    of them is to be absolute -- "the board never left the middle fifth" is only
    worth saying against the actual frame. Without those they are left at zero
    and the rest still reports.
    """
    if not cands:
        return Coverage()
    dist = [c.distance for c in cands]
    tilt = [c.tilt_deg for c in cands]
    cell = _cells(cands)
    cov = Coverage(
        cells=len(set(cell.values())),
        distance=(float(min(dist)), float(max(dist))),
        tilt_range=float(np.ptp(tilt)),
        condition=condition(cands),
        pair_mm=float(max(c.pair_mm for c in cands)),
        safe_frames=int(min(c.safe_frames for c in cands)),
    )
    if camera is not None and size:
        import cv2

        pts = np.array([c.centre for c in cands], float).reshape(-1, 1, 3)
        px = cv2.projectPoints(pts, np.zeros(3), np.zeros(3),
                               camera.matrix(), camera.dist())[0].reshape(-1, 2)
        cov.x_frac = float(np.ptp(px[:, 0]) / size[0])
        cov.y_frac = float(np.ptp(px[:, 1]) / size[1])
    return cov


def pick(samples, radial_limit: float | None = None, want: int = 12,
         camera=None, sweep_s: float = 0.0, size=None,
         max_pair_mm: float = MAX_PAIR_MM) -> PickResult:
    """Whole decision, from scanned frames to the moments worth keeping.

    In three separate stages, and deliberately so. What is unusable is thrown
    out; what remains is reduced to one instant per still stretch; and the final
    set is chosen from those on spread alone. Nothing about how comfortable a
    frame looks reaches the third stage -- a sharper duplicate of a pose already
    held is worth nothing, and the fit only ever sees the geometry.

    `samples` is (t_ns, CameraDetection) in any order.
    """
    fx = float(getattr(camera, "fx", 0.0) or 0.0)
    res = PickResult(n_scanned=len(samples))
    good = []
    for t_ns, det in samples:
        why = usable(det, radial_limit, fx)
        if why:
            res.rejected[why] = res.rejected.get(why, 0) + 1
            continue
        good.append(to_candidate(t_ns, det, fx))
    res.n_usable = len(good)
    measure_motion(good, sweep_s)
    measure_depths(good, sweep_s)
    res.holds = group_holds(good)

    # Drop the stretches where the board was still travelling far enough between
    # the image and its sweep to matter. Relaxed if it would leave too little to
    # choose from -- a recording bad enough for that has a problem no picker can
    # fix, and the report will say so.
    steady = [c for c in res.holds if c.pair_mm <= max_pair_mm]
    if sweep_s > 0 and len(steady) >= want:
        res.rejected["짝 어긋남 과다"] = len(res.holds) - len(steady)
        pool = steady
    else:
        pool = res.holds
    res.chosen = select(pool, want)
    res.coverage = coverage(res.chosen, camera, size)
    return res

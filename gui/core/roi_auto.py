"""Placing the step-5 box without drawing it.

Drawing a box per scene is the slowest part of a calibration -- eighteen times
of dragging bounds and turning them square with a board. It is also the part the
tool has the most information about: the camera has already found the board and
knows its full pose, and `align_box_to_board` already snaps a rough box onto the
plane inside it. The only thing missing was a way to get from the camera's answer
to a rough box in the LiDAR's frame, and that needs the extrinsic.

Which is circular, since the extrinsic is what the whole thing is for -- except
that one scene is enough to solve it. Four hole pairs against six unknowns. So a
single box drawn by hand seeds the rest, and the seed does not have to be
accurate: it only has to land the board inside a box far larger than its own
error, after which the plane fit and the hole detection start again from the
points themselves. Nothing of the seed's error survives into the answer.

The half turn a one-scene solve cannot see does matter, though, and badly. The
two solutions differ by a turn about the seed board's own normal; that axis runs
through the seed board's centre, so the seed scene is unaffected -- but every
other board sits somewhere else, and the turn throws its predicted position by
twice its distance from the axis. Measured on eighteen scenes: seeds that took
the wrong turn missed by 7 to 18 metres, seeds that took the right one by a
quarter of a metre. `solve.is_upright` is what tells them apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gui.core.detect_lidar import align_box_to_board, box_from_frame
from gui.core.project import FilterBox, Target

# How much wider than the board the first guess is drawn. It has to cover the
# error of whatever extrinsic is placing the box, and the extrinsic from a single
# hand-drawn scene is worth a good deal less than the finished one: measured on
# eighteen scenes it put board centres a median 0.19 to 0.56 m out, worst case
# 1.03 m, where the finished extrinsic manages 0.42 m worst case.
#
# 0.45 m was set against the finished figure and is too tight for the seed. From
# each of eight possible seed scenes in turn, 0.45 m left one seed stuck at five
# scenes and an extrinsic 5.6 degrees off; 0.70 m brought every one of the eight
# home to nine or more scenes, all within 0.6 degrees of the hand-drawn answer.
# 1.00 m was no better and sometimes worse -- a box that wide starts catching
# whatever else is in the room.
MARGIN_M = 0.70

# Slab depth of the first guess, before the snap thins it to the panel.
THICKNESS_M = 0.30

# A snapped plane facing more than this away from the camera's prediction is not
# the board -- most likely a wall that happened to be inside the guess.
MAX_TILT_DISAGREE_DEG = 25.0


@dataclass
class Placement:
    scene_id: str
    ok: bool
    box: FilterBox | None = None
    note: str = ""


def board_frame_in_lidar(sol, cam_det):
    """Board centre and normal, carried from the camera's frame into the LiDAR's.

    `sol.transform` maps LiDAR to camera, so the inverse is what is wanted here.
    The centre comes from the four hole centres rather than the marker pose: it
    is the middle of the rectangle the detection actually measures, which is what
    the box has to sit on.
    """
    R = np.asarray(sol.R, float)
    t = np.asarray(sol.t, float).reshape(3)

    centres_cam = np.asarray(cam_det.centers, float).reshape(-1, 3)
    centre_cam = centres_cam.mean(axis=0)

    import cv2

    R_board = cv2.Rodrigues(np.asarray(cam_det.rvec, float).reshape(3))[0]
    normal_cam = R_board[:, 2]

    centre = R.T @ (centre_cam - t)
    normal = R.T @ normal_cam
    normal /= max(np.linalg.norm(normal), 1e-12)
    return centre, normal


def guess_box(sol, cam_det, target: Target,
              margin: float = MARGIN_M, thickness: float = THICKNESS_M) -> FilterBox:
    """A box around where the board should be, before looking at the points."""
    centre, normal = board_frame_in_lidar(sol, cam_det)

    # Point the box's third axis along the board normal, so the thin dimension is
    # the panel's depth and the wall behind falls outside -- the same reasoning
    # `align_box_to_board` uses.
    n = normal if normal @ centre < 0 else -normal
    world_up = np.array([0.0, 0.0, 1.0])
    ex = np.cross(world_up, n)
    if np.linalg.norm(ex) < 1e-3:
        ex = np.cross(np.array([1.0, 0.0, 0.0]), n)
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    rot = np.column_stack([ex, ey, n])

    reach = float(np.hypot(target.delta_width_circles, target.delta_height_circles)) / 2
    reach += target.circle_radius + margin
    return box_from_frame(centre, rot, (reach, reach, thickness / 2))


def place(cloud: np.ndarray, sol, cam_det, target: Target, scene_id: str = "",
          margin: float = MARGIN_M) -> Placement:
    """Guess where the board is, then snap the guess onto the points."""
    if cam_det is None or not getattr(cam_det, "ok", False) or cam_det.centers is None:
        return Placement(scene_id, False, note="카메라가 보드를 못 찾았습니다")
    if sol is None or not getattr(sol, "ok", False):
        return Placement(scene_id, False, note="출발점 extrinsic 이 없습니다")

    rough = guess_box(sol, cam_det, target, margin)
    inside = int(rough.mask(cloud).sum())
    if inside < 200:
        return Placement(scene_id, False, rough,
                         f"예측한 자리에 점이 {inside}개뿐입니다")

    snapped, note = align_box_to_board(cloud, rough)
    # Did it snap onto the board or onto something else that was in the way?
    want = rough.rotation()[:, 2]
    got = snapped.rotation()[:, 2]
    tilt = float(np.degrees(np.arccos(abs(np.clip(want @ got, -1.0, 1.0)))))
    if tilt > MAX_TILT_DISAGREE_DEG:
        return Placement(scene_id, False, rough,
                         f"찾은 평면이 예측과 {tilt:.0f}° 어긋납니다 — 벽일 수 있습니다")
    return Placement(scene_id, True, snapped, f"{note} · 예측과 {tilt:.0f}° 차이")


# How much room to leave when a caller keeps only the part of a cloud the sweep
# could possibly need. The box is placed from an extrinsic that improves between
# rounds, so the board can sit up to a metre from where the first round put it,
# and the crop has to still contain it after the box has moved.
CROP_SLACK_M = 1.5


def crop_near_board(cloud: np.ndarray, sol, cam_det, target: Target,
                    margin: float = MARGIN_M, slack: float = CROP_SLACK_M) -> np.ndarray:
    """The part of a cloud a later `place` could reach, and nothing else.

    Holding every scene's full cloud so the rounds can revisit them is fine at
    a hundred megabytes and not fine at a gigabyte, which is where eighteen
    scenes of five stacked sweeps from a 128-channel unit lands. Detection
    crops to the box as its first act anyway, so keeping a ball around where
    the board is predicted to be changes nothing about the answer -- as long as
    the ball is wide enough to survive the box moving as the extrinsic sharpens.
    """
    centre, _ = board_frame_in_lidar(sol, cam_det)
    reach = float(np.hypot(target.delta_width_circles, target.delta_height_circles)) / 2
    radius = reach + target.circle_radius + margin + slack
    xyz = np.asarray(cloud, np.float32)
    keep = np.linalg.norm(xyz - centre.astype(np.float32), axis=1) <= radius
    return xyz[keep]


@dataclass
class SweepScene:
    """One scene as the sweep sees it: a cloud, the camera's answer, and -- once
    something has found them -- the four hole centres in the LiDAR frame."""

    scene_id: str
    cloud: np.ndarray
    cam_det: object
    centres: np.ndarray | None = None
    box: FilterBox | None = None
    note: str = ""


@dataclass
class SweepReport:
    solution: object = None
    rounds: int = 0
    placed: list[str] = None  # scene ids the sweep placed and detected, in order

    def __post_init__(self):
        if self.placed is None:
            self.placed = []


def sweep(scenes: list[SweepScene], target: Target, detect,
          rounds: int = 4, margin: float = MARGIN_M, progress=None) -> SweepReport:
    """Place, detect, refit, repeat -- until no scene new to the answer turns up.

    One box drawn by hand is enough to start, but only just: that extrinsic can
    put a board a metre from where it really is, and boards it misses stay
    missed. Every scene it does land, though, is a scene the next fit gets to
    use, and the fit sharpens fast -- from a single seed to the finished answer
    is a factor of five or so in placement error. So the ones missed on the first
    pass are usually caught on the second or third.

    Two things must not happen and do not. A scene already holding centres is
    never re-placed, so a box drawn by hand is never overwritten by a guess. And
    a scene whose detection comes back short is left alone rather than recorded
    as failed: the next round's better extrinsic may well place it.

    `detect(scene, box)` is the caller's, so this file needs to know nothing
    about detector settings or ring fields. Anything with `.ok` and `.centers`
    will do as its return.
    """
    from gui.core.solve import solve

    known = {s.scene_id: s for s in scenes if s.centres is not None}
    if not known:
        return SweepReport()

    def refit():
        return solve([(s.scene_id, s.centres, s.cam_det.centers) for s in known.values()])

    report = SweepReport(solution=refit())
    for rnd in range(rounds):
        report.rounds = rnd + 1
        fresh = []
        for s in scenes:
            if s.scene_id in known:
                continue
            if progress is not None:
                progress(rnd + 1, s.scene_id)
            p = place(s.cloud, report.solution, s.cam_det, target, s.scene_id, margin)
            if not p.ok:
                s.note = p.note
                continue
            det = detect(s, p.box)
            if not getattr(det, "ok", False) or det.centers is None:
                s.note = getattr(det, "reason", "") or "검출 실패"
                continue
            s.box, s.centres, s.note = p.box, det.centers, p.note
            known[s.scene_id] = s
            fresh.append(s.scene_id)
        if not fresh:
            break
        report.placed += fresh
        report.solution = refit()
    return report

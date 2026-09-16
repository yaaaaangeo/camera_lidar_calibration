"""Checking an extrinsic by looking at it, and writing it out.

The residual cannot answer the question this step asks. Four hole centres form a
rectangle, and a rectangle is unchanged by a half turn -- so the correspondence
that swaps every hole with the one diagonally opposite fits *exactly as well*
while describing an extrinsic rotated 180 degrees about the board normal. Both
solutions report the same RMSE. Only putting the cloud back on the image
separates them, and then it is obvious.

`solve()` resolves the ambiguity when several scenes disagree about it, but a
single scene cannot: there is nothing to disagree with. That is the case this
module exists for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Projection:
    """Cloud points that landed on the image, with what to colour them by."""

    uv: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float64))
    depth: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))
    intensity: np.ndarray | None = None
    # The surviving points in the LiDAR's own frame, so they can be coloured by
    # height the way RViz's Axis Color does. Depth is distance from the camera
    # and cannot show height: on a forward view the road and the wall beside it
    # sit at the same depth.
    axis: np.ndarray | None = None
    n_input: int = 0
    n_behind: int = 0  # dropped for sitting behind the camera
    n_outside: int = 0  # projected off the sensor
    n_folded: int = 0  # direction outside the distortion model's valid range

    @property
    def n_visible(self) -> int:
        return len(self.uv)


_LIMIT_CACHE: dict[tuple, float] = {}


def radial_limit(camera, r_max: float = 6.0, steps: int = 20001) -> float:
    """Largest off-axis ratio the distortion model still maps outward.

    The model is fitted from board photos, so it only describes the angles the
    camera actually sees. Past that it keeps returning numbers, and at some point
    those numbers start getting *smaller* -- a direction further off-axis is
    reported as landing nearer the image centre. `cv2.projectPoints` does not
    check, so directions the camera could never see are drawn inside the frame.

    Measured on 31CAM_HFOV54, a 55-degree lens: the curve turns at 57.8 degrees
    incidence and 63-65 degrees lands back inside the image. Dropping everything
    past the turn is what stops it. Points between the real field of view and the
    turn are past the image edge anyway and the bounds test already removes them.

    Returned as tan(angle), which is what the projection works in. Radial terms
    only -- the tangential p1, p2 shift a point sideways rather than change how
    far out it lands, and are ~0.001 on these cameras.
    """
    d = np.asarray(camera.dist(), float)
    key = (d.tobytes(), r_max, steps)
    cached = _LIMIT_CACHE.get(key)
    if cached is not None:
        return cached

    k1, k2, k3 = d[0], d[1], d[4]
    k4, k5, k6 = (d[5], d[6], d[7]) if len(d) >= 8 else (0.0, 0.0, 0.0)
    r = np.linspace(0.0, r_max, steps)
    r2 = r * r
    num = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    den = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    rd = r * num / den

    falling = np.diff(rd) <= 0
    if not falling.any():
        limit = float(r_max)
    else:
        # Fail closed. An index of 0 would mean the curve never rises at all,
        # which no real lens does -- but if it happens, clamp hard rather than
        # let everything through.
        limit = float(r[int(np.argmax(falling))])
    _LIMIT_CACHE[key] = limit
    return limit


def project_cloud(
    cloud: np.ndarray,
    sol,
    camera,
    width: int,
    height: int,
    intensity: np.ndarray | None = None,
    max_range: float = 0.0,
    min_range: float = 0.0,
) -> Projection:
    """LiDAR points -> pixels, keeping only what the camera could actually see.

    Points behind the camera are dropped first. Without that they come back
    through the projection mirrored, and a wrong extrinsic can look plausible
    because the wall behind the sensor lands neatly on the image.
    """
    import cv2

    pts = np.asarray(cloud, np.float64).reshape(-1, 3)
    n_input = len(pts)
    if n_input == 0 or sol is None or not getattr(sol, "ok", False):
        return Projection(n_input=n_input)

    cam_pts = sol.transform(pts)
    keep = cam_pts[:, 2] > 1e-3
    # Judged on the incoming direction, not on the pixel the model computes for
    # it: past the model's valid range that pixel is the thing that is wrong, so
    # the bounds test below cannot catch these.
    off_axis = np.hypot(cam_pts[:, 0], cam_pts[:, 1]) / np.maximum(cam_pts[:, 2], 1e-9)
    folded = keep & (off_axis > radial_limit(camera))
    keep &= ~folded
    if max_range > 0 or min_range > 0:
        dist = np.linalg.norm(cam_pts, axis=1)
        if max_range > 0:
            keep &= dist <= max_range
        if min_range > 0:
            keep &= dist >= min_range
    n_behind = int((cam_pts[:, 2] <= 1e-3).sum())
    n_folded = int(folded.sum())
    cam_pts = cam_pts[keep]
    if intensity is not None:
        intensity = np.asarray(intensity).reshape(-1)[keep]
    if not len(cam_pts):
        return Projection(n_input=n_input, n_behind=n_behind, n_folded=n_folded)

    uv, _ = cv2.projectPoints(
        cam_pts, np.zeros(3), np.zeros(3), camera.matrix(), camera.dist()
    )
    uv = uv.reshape(-1, 2)
    on = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return Projection(
        uv=uv[on],
        depth=cam_pts[on, 2],
        intensity=None if intensity is None else intensity[on],
        axis=pts[keep][on],
        n_input=n_input,
        n_behind=n_behind,
        n_outside=int((~on).sum()),
        n_folded=n_folded,
    )


# The flat colour. Green sits far from road grey, foliage and sky in hue, and
# stays legible on both a bright and a dark photo.
SOLID_RGB = (60, 255, 90)


def make_overlay(
    image: np.ndarray,
    pr: Projection,
    colour_by: str = "depth",
    colormap: str = "turbo",
    point_size: float = 1.0,
    dim: float = 0.45,
    lo: float = 0.0,
    hi: float = 0.0,
) -> np.ndarray:
    """Draw the projected cloud onto a copy of the image.

    The photo is dimmed underneath so the points read against it. Judging an
    extrinsic means seeing whether *edges* line up -- the board's outline, the
    holes -- and at full brightness the image texture competes with the points.

    `colour_by` is "depth", "intensity", "axis_x"/"axis_y"/"axis_z" (RViz's Axis
    Color, in the LiDAR's own frame), or "solid" for one flat colour. Solid gives
    up depth for legibility: a ramp puts a second pattern on top of the photo's
    own, and when the question is only whether a line falls where it should, that
    pattern is in the way.
    """
    from gui.core.colorize import _ramp, auto_range

    out = np.asarray(image).astype(np.float32)
    out *= max(0.0, min(1.0, dim))
    out = out.astype(np.uint8)
    if pr.n_visible == 0:
        return out

    if colour_by == "solid":
        rgb = np.tile(np.asarray(SOLID_RGB, np.uint8), (pr.n_visible, 1))
    else:
        values = pr.depth
        if colour_by == "intensity" and pr.intensity is not None:
            values = pr.intensity
        elif colour_by.startswith("axis_") and pr.axis is not None:
            values = pr.axis[:, {"x": 0, "y": 1, "z": 2}[colour_by[-1]]]
        if hi <= lo:
            lo, hi = auto_range(values)
        t = np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        rgb = (_ramp(t, colormap) * 255).astype(np.uint8)

    u = np.clip(pr.uv[:, 0].astype(int), 0, out.shape[1] - 1)
    v = np.clip(pr.uv[:, 1].astype(int), 0, out.shape[0] - 1)
    # BGR for OpenCV-ordered images.
    bgr = rgb[:, ::-1]

    # Whole-pixel radius, as before. The 0.1 steps of the size box therefore only
    # take effect where the rounded radius changes -- around 1.5, 3.0 and 5.1 --
    # and do nothing in between. Drawing by area coverage instead did follow every
    # step, but a true 1.5 px disc is thinner and softer than this, and the dense
    # clouds read as faded next to it. Left as it was until that trade is worth
    # revisiting.
    if point_size < 1.5:
        out[v, u] = bgr
    else:
        radius = max(int(round(point_size / 2.0)), 1)
        _stamp_points(out, u, v, bgr, radius)
    return out


_DISC_CACHE: dict[int, np.ndarray] = {}


def _disc_offsets(radius: int) -> np.ndarray:
    """Which pixels a filled circle of this radius covers, as (dy, dx) pairs.

    Taken from OpenCV itself rather than derived, so the dots keep exactly the
    shape they had when each one was drawn by its own `cv2.circle` call.
    """
    cached = _DISC_CACHE.get(radius)
    if cached is not None:
        return cached
    import cv2

    k = 2 * radius + 3
    stamp = np.zeros((k, k), np.uint8)
    cv2.circle(stamp, (k // 2, k // 2), radius, 255, -1)
    ys, xs = np.nonzero(stamp)
    offsets = np.column_stack([ys - k // 2, xs - k // 2]).astype(np.int64)
    _DISC_CACHE[radius] = offsets
    return offsets


def _stamp_points(out, u, v, bgr, radius: int):
    """Draw every dot at once, giving the same picture as a call per point.

    A call per point costs 67 ms for 55k points, and the default distance now
    reaches 50 m so that many is normal. Writing the same pixels as a few whole
    array operations -- five passes at radius 1, thirteen at 2 -- is several
    times quicker.

    The catch is who wins where dots overlap. Drawing point by point, the later
    point overwrote the earlier one's whole dot; deciding it per pixel instead
    mixes the two along the seam. With depth or height that is invisible, since
    points overlapping on screen are neighbours with nearly the same value -- but
    intensity jumps from point to point (lane paint against asphalt) and the
    mixing shows as speckle, off by 39 grey levels on average.

    So the winner is resolved first: each pixel remembers the highest-numbered
    point covering it, and only then takes that point's colour. Same result as
    the loop, one gather and one scatter per dot pixel instead of a draw call per
    point.
    """
    h, w = out.shape[:2]
    n = len(u)
    idx = np.arange(n, dtype=np.int32)
    owner = np.full(h * w, -1, np.int32)
    for dy, dx in _disc_offsets(radius):
        yy = v + dy
        xx = u + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        if not ok.any():
            continue
        flat = yy[ok] * w + xx[ok]
        who = idx[ok]
        # Read before writing, so duplicates in `flat` all compare against the
        # same value; the scatter then leaves the largest, because `who` rises.
        better = who > owner[flat]
        owner[flat[better]] = who[better]

    hit = owner >= 0
    if not hit.any():
        return
    out.reshape(-1, 3)[hit] = bgr[owner[hit]]


def draw_centres(image: np.ndarray, sol, camera, lidar_centres, camera_centres) -> np.ndarray:
    """Mark both sensors' hole centres, so a mismatch is visible directly.

    Circles are the camera's answer, crosses the LiDAR's brought through the
    extrinsic. When they sit on top of each other the extrinsic is right; when
    each cross sits on the *diagonally opposite* circle, the correspondence took
    the half turn and the extrinsic is 180 degrees out.
    """
    import cv2

    out = np.asarray(image).copy()
    if sol is None or not getattr(sol, "ok", False):
        return out

    cam_uv, _ = cv2.projectPoints(
        np.asarray(camera_centres, np.float64), np.zeros(3), np.zeros(3),
        camera.matrix(), camera.dist(),
    )
    lid_uv, _ = cv2.projectPoints(
        sol.transform(np.asarray(lidar_centres, np.float64)), np.zeros(3), np.zeros(3),
        camera.matrix(), camera.dist(),
    )
    for (u, v) in cam_uv.reshape(-1, 2):
        cv2.circle(out, (int(u), int(v)), 13, (255, 90, 255), 2)
    for (u, v) in lid_uv.reshape(-1, 2):
        u, v = int(u), int(v)
        cv2.line(out, (u - 11, v), (u + 11, v), (60, 255, 255), 2)
        cv2.line(out, (u, v - 11), (u, v + 11), (60, 255, 255), 2)
    # Join each pair so a swap is unmistakable even when both land on the board.
    for (cu, cv_), (lu, lv) in zip(cam_uv.reshape(-1, 2), lid_uv.reshape(-1, 2)):
        cv2.line(out, (int(cu), int(cv_)), (int(lu), int(lv)), (255, 255, 255), 1)
    return out


def centre_agreement(lidar_centres, camera_centres, sol) -> dict:
    """How far apart the two sensors put the same four holes, in 3D and in pixels.

    Reported per hole rather than as one number: a half-turn error leaves the
    mean roughly where it was while sending each hole to its diagonal opposite,
    so the spread is what gives it away.
    """
    if sol is None or not getattr(sol, "ok", False):
        return {}
    lid = sol.transform(np.asarray(lidar_centres, np.float64))
    cam = np.asarray(camera_centres, np.float64)

    # Pair each transformed LiDAR centre with its nearest camera centre; a good
    # extrinsic makes this a bijection, a half-turn makes it one too -- but a
    # much worse one.
    d = np.linalg.norm(lid[:, None, :] - cam[None, :, :], axis=2)
    pick = d.argmin(axis=1)
    per_hole = d[np.arange(len(lid)), pick]
    return {
        "per_hole_mm": per_hole * 1000,
        "mean_mm": float(per_hole.mean() * 1000),
        "max_mm": float(per_hole.max() * 1000),
        "bijective": len(set(pick.tolist())) == len(lid),
    }


def half_turn_check(lidar_centres, camera_centres, sol) -> dict:
    """Compare the fit against its own half-turn, on this scene alone.

    Rotating the LiDAR correspondence by two positions is exactly the diagonal
    swap. If that alternative fits as well as the current one, this scene cannot
    tell them apart and the extrinsic needs a second scene or a look at the
    overlay. The numbers are reported rather than judged -- there is no threshold
    that makes this decidable from residuals, which is the whole point.
    """
    from gui.core.solve import solve_rigid, sort_centers

    lid = sort_centers(np.asarray(lidar_centres, np.float64))
    cam = sort_centers(np.asarray(camera_centres, np.float64))
    fits = []
    for shift in (0, 2):
        rolled = np.roll(lid, shift, axis=0)
        R, t = solve_rigid(rolled, cam)
        resid = np.linalg.norm((R @ rolled.T).T + t - cam, axis=1)
        fits.append((float(np.sqrt((resid**2).mean()) * 1000), R))

    # Which of the two `sol` currently is cannot be read off it, so name them by
    # residual instead of guessing: the point is only whether they differ.
    best = min(fits, key=lambda f: f[0])
    other = max(fits, key=lambda f: f[0])
    shown = min(fits, key=lambda f: np.linalg.norm(f[1] - sol.R)) if sol is not None else best
    alternative = max(fits, key=lambda f: np.linalg.norm(f[1] - sol.R)) if sol is not None else other
    return {
        "current": shown[0],
        "half_turn": alternative[0],
        # Below a couple of millimetres the two are a coin toss and the residual
        # cannot be used to choose. That is the normal case for a single scene.
        "separable": (other[0] - best[0]) > 2.0,
    }


def to_yaml(sol, camera, target, note: str = "", project=None, method: str = "") -> str:
    """The extrinsic, with enough context to audit it later.

    This file gets committed and read by other people, so the numbers alone are
    not enough: which vehicle, from which recording, with which detector, and
    when. Six months on, "is this value still right?" has to be answerable
    without finding whoever produced it.
    """
    from datetime import date

    R, t = sol.R, sol.t
    lines = ["# camera <- lidar extrinsic"]
    if note:
        lines.append(f"# {note}")
    if project is not None:
        lines += [
            f"vehicle: {project.name}",
            f"calibrated_on: {date.today().isoformat()}",
        ]
        if project.bag_paths:
            names = [Path(b).name for b in project.bag_paths]
            lines.append("bags:")
            lines += [f"  - {n}" for n in names]
        lines += [
            f"lidar_topic: {project.lidar_topic}",
            f"camera_topic: {project.camera_topic}",
        ]
    if method:
        lines.append(f"detector: {method}")
    lines += [
        f"rmse_mm: {sol.rmse * 1000:.3f}",
        f"scenes: {len(sol.scene_ids)}",
        f"correspondences: {sol.n_pairs}",
        "",
        "T_cam_lidar:",
        "  rows: 4",
        "  cols: 4",
        "  data: [",
    ]
    T = sol.matrix
    for row in T:
        lines.append("    " + ", ".join(f"{v:12.8f}" for v in row) + ",")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("  ]")
    lines += [
        "",
        "translation_m: [{:.6f}, {:.6f}, {:.6f}]".format(*t),
        "rotation_matrix:",
    ]
    for row in R:
        lines.append("  - [{:12.8f}, {:12.8f}, {:12.8f}]".format(*row))

    # The intrinsics travel with the extrinsic. They are a pair: projecting with
    # this transform and someone else's focal length gives a wrong answer that
    # looks plausible, and the file has to be self-contained for anyone checking
    # it later to get the same picture.
    lines += [
        "",
        "camera:",
        f"  fx: {camera.fx:.8f}",
        f"  fy: {camera.fy:.8f}",
        f"  cx: {camera.cx:.8f}",
        f"  cy: {camera.cy:.8f}",
        f"  distortion_model: {'rational_polynomial' if camera.rational else 'plumb_bob'}",
        "  # " + ", ".join(camera.dist_names()),
        "  distortion: [" + ", ".join(f"{v:.10f}" for v in camera.dist()) + "]",
    ]
    if target is not None:
        lines.append(f"  # target hole radius: {target.circle_radius * 1000:.2f} mm")
    return "\n".join(lines) + "\n"


def from_yaml(text: str) -> dict:
    """Read back what `to_yaml` wrote: transform, camera, and where it came from.

    Kept deliberately tolerant. The point of loading an extrinsic is to check
    someone else's answer against a recording, and that has to work on a file
    hand-edited or produced by an older version of this tool. Anything absent
    comes back missing rather than raising, and the caller decides whether it can
    proceed without it.
    """
    import yaml as _yaml

    d = _yaml.safe_load(text) or {}
    out: dict = {}

    block = d.get("T_cam_lidar")
    flat = None
    if isinstance(block, dict) and block.get("data") is not None:
        flat = list(block["data"])
    elif isinstance(block, (list, tuple)):
        flat = list(np.asarray(block, float).ravel())
    if flat is not None and len(flat) >= 12:
        T = np.eye(4)
        T[:3, :4] = np.asarray(flat[:16], float).reshape(4, 4)[:3, :4] if len(flat) >= 16 \
            else np.asarray(flat[:12], float).reshape(3, 4)
        out["R"] = T[:3, :3]
        out["t"] = T[:3, 3]
    elif d.get("rotation_matrix") is not None and d.get("translation_m") is not None:
        out["R"] = np.asarray(d["rotation_matrix"], float).reshape(3, 3)
        out["t"] = np.asarray(d["translation_m"], float).reshape(3)

    cam = d.get("camera")
    if isinstance(cam, dict) and cam.get("fx"):
        dist = [float(v) for v in (cam.get("distortion") or [])]
        # Pad to eight so a plumb_bob file (5) and a rational one (8) both read,
        # and the missing k4..k6 come back as zero -- which is what keeps such a
        # camera on the 5-coefficient path.
        dist += [0.0] * (8 - len(dist))
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        out["camera"] = dict(
            fx=float(cam["fx"]), fy=float(cam.get("fy", cam["fx"])),
            cx=float(cam.get("cx", 0.0)), cy=float(cam.get("cy", 0.0)),
            **dict(zip(names, dist[:8])),
        )

    for key in ("vehicle", "lidar_topic", "camera_topic", "detector", "calibrated_on"):
        if d.get(key):
            out[key] = d[key]
    if d.get("bags"):
        out["bags"] = list(d["bags"])
    if d.get("rmse_mm") is not None:
        out["rmse_mm"] = float(d["rmse_mm"])
    return out


def to_rt_matrix(sol, camera, project=None, method: str = "") -> str:
    """Flat row-major strings: 3x4 extrinsic, 3x3 intrinsic, 1x5 distortion.

    One line per matrix, comma separated, quoted. Parsers that read this only have
    to split on commas -- no nesting, no indentation to get wrong -- which is why
    several in-house tools take their calibration this way.

    The 3x4 is the top three rows of T_cam_lidar: rotation then translation, so
    `RT @ [x, y, z, 1]` maps a LiDAR point into camera coordinates.
    """
    from datetime import date

    R, t = sol.R, sol.t
    rt = np.hstack([R, t.reshape(3, 1)]).ravel()
    K = camera.matrix().ravel()
    D = camera.dist()

    def row(values, fmt):
        return ", ".join(format(v, fmt) for v in values)

    lines = []
    if project is not None:
        lines += [f"# {project.name}", f"# calibrated_on: {date.today().isoformat()}"]
        if project.bag_paths:
            lines.append(f"# bag: {', '.join(Path(b).name for b in project.bag_paths)}")
    if method:
        lines.append(f"# detector: {method}")
    lines += [
        f"# rmse: {sol.rmse * 1000:.3f} mm, scenes: {len(sol.scene_ids)}",
        "",
        "  # 3 x 4  LiDAR(vehicle) → Camera (행 우선, 12개)",
        f'  RT_Matrix: "{row(rt, ".8f")}"',
        "",
        "  # 3 x 3  내부 파라미터 (행 우선, 9개)",
        f'  cameraMatrix: "{row(K, ".8e")}"',
        "",
        f"  # 1 x {len(D)}  왜곡계수 ({', '.join(camera.dist_names())})",
        f'  distCoeffs: "{row(D, ".8e")}"',
    ]
    return "\n".join(lines) + "\n"


def to_static_transform(sol, parent: str = "camera", child: str = "lidar") -> str:
    """One ROS command line, ready to paste.

    `tf2_ros static_transform_publisher` takes the child's pose in the parent
    frame, which is what this extrinsic already is: it maps LiDAR points into
    camera coordinates.
    """
    t = sol.t
    R = sol.R
    # Rotation matrix -> quaternion (x, y, z, w), branch-free enough for a matrix
    # that is orthonormal by construction.
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = q / np.linalg.norm(q)
    return (
        "ros2 run tf2_ros static_transform_publisher "
        f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
        f"{parent} {child}"
    )

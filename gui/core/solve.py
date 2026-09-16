"""Solving for the extrinsic, and judging how much to trust it.

Both sensors give four 3D points per scene, so this is a rigid 3D-3D fit rather
than PnP -- which is why no initial guess is needed.

The two point sets carry no labels, though. Nothing marks which LiDAR hole is
the top-left one. Correspondence comes from sorting both sets the same way, and
a sort that disagrees between the two sides produces a confident, completely
wrong extrinsic. `sort_centers` is therefore the most delicate part of this
file, not the least.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sort_centers(points: np.ndarray, frame: str = "") -> np.ndarray:
    """Order four coplanar centres so both sensors agree on which is which.

    Bearings are taken in the board's own plane, about its centre, with the
    in-plane axes oriented against the sensor's line of sight. Both sensors see
    the same face of the board, so that single rule makes the two orderings run
    the same way round -- no per-frame axis swap, and no assumption about where
    the target sits relative to the sensor.

    The original does swap axes by a fixed rule, which only yields a head-on
    view when the board lies on the LiDAR's +x axis. With the board off to one
    side -- the common case, and what happens as soon as you move it around to
    get pose variety -- the four bearings collapse together and the ordering
    comes out of numerical noise.

    `frame` is accepted for readability at call sites; the rule is the same
    either way.
    """
    points = np.asarray(points, np.float64)
    if len(points) != 4:
        raise ValueError(f"expected 4 centres, got {len(points)}")

    centroid = points.mean(axis=0)
    if np.linalg.norm(centroid) < 1e-6:
        raise ValueError("target centroid sits on the sensor origin")

    rel = points - centroid
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    u, v = vt[0], vt[1]
    if np.cross(u, v) @ centroid < 0:
        v = -v
    ring = points[np.argsort(np.arctan2(rel @ v, rel @ u))]

    # Start the cycle on a long edge. The holes sit on a rectangle -- 297 x 238
    # mm on the stock board -- so "long edge first" is decidable from the points
    # alone and lands on the same edge from either sensor. What it cannot decide
    # is *which* long edge: the rectangle looks identical after a half turn.
    # solve() resolves that ambiguity, where more than one scene is available.
    if np.linalg.norm(ring[1] - ring[0]) < np.linalg.norm(ring[2] - ring[1]):
        ring = np.roll(ring, -1, axis=0)
    return ring


def is_upright(R: np.ndarray) -> bool:
    """Does this extrinsic keep the world's up direction pointing up in the image?

    The one thing a single scene cannot decide is the half turn: swapping every
    hole with the one diagonally opposite fits exactly as well and describes an
    extrinsic turned 180 degrees about the board's normal. Since the board faces
    the sensors, that normal runs roughly along the line of sight, so the turn
    barely moves the LiDAR origin -- neither the residual nor the size of the
    translation can tell the two apart.

    What the turn does do is put the picture upside down. It is a half turn about
    an axis pointing at the camera, so up becomes down. Taking the LiDAR's +z as
    roughly upward and the camera's +y as downward -- the first is how a LiDAR is
    mounted, the second is the OpenCV convention the intrinsics already follow --
    the real solution has up landing on negative y.

    Measured on eighteen scenes the two came out at -0.96 and +0.94; the sign is
    not close to the fence. It would only mislead a LiDAR mounted on its side.
    """
    return float((np.asarray(R, np.float64) @ np.array([0.0, 0.0, 1.0]))[1]) < 0.0


@dataclass
class Solution:
    ok: bool
    R: np.ndarray | None = None  # 3x3, LiDAR -> camera
    t: np.ndarray | None = None  # 3,
    rmse: float = 0.0
    per_point: np.ndarray | None = None  # residual per correspondence, metres
    scene_ids: list[str] = field(default_factory=list)
    per_scene_rmse: dict[str, float] = field(default_factory=dict)
    n_pairs: int = 0
    reason: str = ""

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        if self.R is not None:
            T[:3, :3] = self.R
            T[:3, 3] = self.t
        return T

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        return (self.R @ np.asarray(xyz, np.float64).T).T + self.t


def solve_rigid(lidar: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: the rotation and translation taking `lidar` points onto `camera`."""
    mu_l, mu_c = lidar.mean(axis=0), camera.mean(axis=0)
    H = (lidar - mu_l).T @ (camera - mu_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_c - R @ mu_l


def solve(scenes: list[tuple[str, np.ndarray, np.ndarray]]) -> Solution:
    """Fit one extrinsic to every scene at once.

    `scenes` is (id, lidar centres, camera centres); each pair gets sorted here
    so the caller does not have to think about correspondence.

    Sorting alone cannot say which hole the cycle starts at -- the choice drifts
    with board orientation, and a rule that disagrees between the two sensors
    yields a confident, badly wrong extrinsic. So all eight relabellings of the
    cycle (four rotations, each with or without a flip) are tried.

    Residual alone cannot pick the winner. The holes sit on a rectangle, which
    is symmetric under a half turn, so the correspondence that swaps every hole
    with the one diagonally opposite fits *exactly as well* while describing an
    extrinsic rotated by 180 degrees. Physics breaks the tie: the board is in
    front of the camera, so the solution that puts it there is the real one.
    """
    if not scenes:
        return Solution(False, reason="scene이 없습니다")

    ids, L_sets, C_sets = [], [], []
    for sid, lidar_pts, cam_pts in scenes:
        try:
            L_sets.append(sort_centers(np.asarray(lidar_pts, np.float64), "lidar"))
            C_sets.append(sort_centers(np.asarray(cam_pts, np.float64), "camera"))
            ids.append(sid)
        except ValueError:
            continue

    if len(ids) < 1:
        return Solution(False, reason="사용할 수 있는 scene이 없습니다")

    def fit(assignment):
        L_ = np.vstack([np.roll(s_, sh, axis=0) for s_, sh in zip(L_sets, assignment)])
        C_ = np.vstack(C_sets)
        R_, t_ = solve_rigid(L_, C_)
        return L_, C_, R_, t_

    def rmse_of(assignment) -> float:
        L_, C_, R_, t_ = fit(assignment)
        return float(np.sqrt((np.linalg.norm((R_ @ L_.T).T + t_ - C_, axis=1) ** 2).mean()))

    # Each scene independently starts on one of the two long edges, and its own
    # residual cannot say which: the rectangle is unchanged by the half turn, so
    # the two fit to the same digit. Which way up the answer comes out does say
    # -- see is_upright -- and that reads off a single scene, so every scene can
    # settle its own turn before any of them are combined.
    #
    # Leaving it to a joint search instead is what this used to do, and it seeds
    # the search with an answer it has no reason to believe: one scene's turn
    # chosen, every other scene left at zero. The extrinsic fitted to that is
    # meaningless, and the scenes then agree with the meaningless one and stay
    # there. On three scenes here that converged on an assignment fitting at
    # 382 mm while the upright assignment fits the same points at 25 mm.
    assignment, undecided = [], []
    for i, (s_, c_) in enumerate(zip(L_sets, C_sets)):
        up = [sh for sh in (0, 2)
              if is_upright(solve_rigid(np.roll(s_, sh, axis=0), c_)[0])]
        if len(up) == 1:
            assignment.append(up[0])
        else:
            assignment.append(0)
            undecided.append(i)

    # The test abstains on a board lying flat, or a LiDAR mounted on its side --
    # then up says nothing about the turn. Those scenes fall back to agreeing
    # with the rest, and only those: a scene the test did settle must not be
    # talked out of it, least of all by the residual, which is the same to the
    # last digit either way and so flips on nothing but rounding.
    best_rmse = rmse_of(assignment)
    for _ in range(3):
        changed = False
        for i in undecided:
            trial = list(assignment)
            trial[i] = 2 - trial[i]
            err = rmse_of(trial)
            if err < best_rmse:
                assignment, best_rmse, changed = trial, err, True
        if not changed:
            break

    L, C, R, t = fit(assignment)
    residual = np.linalg.norm((R @ L.T).T + t - C, axis=1)

    per_scene = {
        sid: float(np.sqrt((residual[i * 4 : i * 4 + 4] ** 2).mean())) for i, sid in enumerate(ids)
    }
    return Solution(
        ok=True,
        R=R,
        t=t,
        rmse=float(np.sqrt((residual**2).mean())),
        per_point=residual,
        scene_ids=ids,
        per_scene_rmse=per_scene,
        n_pairs=len(L),
    )


def leave_one_out(scenes: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, dict]:
    """Refit without each scene in turn, and see how far the answer moves.

    A scene that is quietly wrong often keeps the overall RMSE low while pulling
    the extrinsic somewhere else. Dropping it and watching the result jump finds
    that, where the residual alone does not.
    """
    if len(scenes) < 3:
        return {}

    full = solve(scenes)
    if not full.ok:
        return {}

    out = {}
    for i, (sid, _, _) in enumerate(scenes):
        rest = scenes[:i] + scenes[i + 1 :]
        sub = solve(rest)
        if not sub.ok:
            continue
        d_rot = np.degrees(
            np.arccos(np.clip((np.trace(full.R.T @ sub.R) - 1) / 2, -1.0, 1.0))
        )
        out[sid] = {
            "rmse_without": sub.rmse,
            "rmse_delta": sub.rmse - full.rmse,
            "shift_mm": float(np.linalg.norm(sub.t - full.t) * 1000),
            "rotation_deg": float(d_rot),
        }
    return out


def coverage(scenes: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, float]:
    """How much the board actually moved between scenes.

    Extrinsic rotation is only well determined when the correspondences are
    spread out; repeats of the same pose add points without adding information.
    """
    if not scenes:
        return {}
    centroids = np.array([np.asarray(c, np.float64).mean(axis=0) for _, _, c in scenes])
    normals = []
    for _, _, cam in scenes:
        pts = np.asarray(cam, np.float64)
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        n = vt[2]
        view = pts.mean(axis=0)
        normals.append(n if n @ view < 0 else -n)
    normals = np.array(normals)

    tilts = np.degrees(
        np.arccos(np.clip(np.abs(np.sum(normals * (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)), axis=1)), 0, 1))
    )
    return {
        "x_range": float(np.ptp(centroids[:, 0])),
        "y_range": float(np.ptp(centroids[:, 1])),
        "z_range": float(np.ptp(centroids[:, 2])),
        "tilt_min": float(tilts.min()),
        "tilt_max": float(tilts.max()),
        "tilt_range": float(np.ptp(tilts)),
    }


def to_fast_livo2(sol: Solution, camera, width: int = 0, height: int = 0) -> str:
    """The text format the original tool writes, so results stay drop-in."""
    lines = ["# FAST-LIVO2 calibration format"]
    if width and height:
        lines += [
            "cam_model: Pinhole",
            f"cam_width: {width}",
            f"cam_height: {height}",
            "scale: 1.0",
        ]
    lines += [
        f"cam_fx: {camera.fx:.6f}",
        f"cam_fy: {camera.fy:.6f}",
        f"cam_cx: {camera.cx:.6f}",
        f"cam_cy: {camera.cy:.6f}",
        f"cam_d0: {camera.k1:.8f}",
        f"cam_d1: {camera.k2:.8f}",
        f"cam_d2: {camera.p1:.8f}",
        f"cam_d3: {camera.p2:.8f}",
        f"cam_d4: {camera.k3:.8f}",
    ]
    # The format has five slots and no more. A rational calibration cannot be
    # written here without changing what it means, so say so in the file rather
    # than let the missing denominator pass for a complete camera.
    if camera.rational:
        lines += [
            f"# 주의: 이 카메라는 rational_polynomial(8계수)입니다 —"
            f" k4={camera.k4:.8f}, k5={camera.k5:.8f}, k6={camera.k6:.8f} 는",
            "#       이 형식에 담을 자리가 없어 빠졌습니다. 그대로 쓰면 화면 주변부가 틀어집니다.",
        ]
    lines += [""]
    R, t = sol.R, sol.t
    lines.append(
        "Rcl: [ {:9.6f}, {:9.6f}, {:9.6f},\n"
        "       {:9.6f}, {:9.6f}, {:9.6f},\n"
        "       {:9.6f}, {:9.6f}, {:9.6f}]".format(*R.ravel())
    )
    lines.append("Pcl: [ {:9.6f}, {:9.6f}, {:9.6f}]".format(*t))
    return "\n".join(lines) + "\n"

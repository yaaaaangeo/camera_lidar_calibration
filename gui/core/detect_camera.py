"""Camera side of the detection: image -> four circle centres in camera frame.

Port of `src/qr_detect.hpp`. The geometry is unchanged; the OpenCV calls are
not, because `estimatePoseSingleMarkers` and `estimatePoseBoard` were removed
after 4.7 and this machine runs 4.11. `solvePnP` over every detected marker
corner is what those functions did internally anyway.

Worth remembering when reading numbers out of here: the four circle centres are
*derived* from one board pose, not measured independently. They share a single
6-DOF error -- if the pose is off, all four move together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import cv2.aruco as aruco
import numpy as np

from gui.core.project import Camera, Target

DICTIONARY = aruco.DICT_6X6_250

# Quadrant -> ArUco id, matching qr_detect.hpp's boardIds{1, 2, 4, 3}.
# Clockwise from top-left the ids read 1, 2, 4, 3 -- not 1, 2, 3, 4.
MARKERS = [(-1, +1, 1), (+1, +1, 2), (+1, -1, 4), (-1, -1, 3)]

# Corner order within a marker, matching what detectMarkers returns:
# top-left, top-right, bottom-right, bottom-left (board frame has +y up).
_CORNERS = [(-1, +1), (+1, +1), (+1, -1), (-1, -1)]


def board_model(t: Target) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """(marker id -> its four 3D corners, four 3D circle centres) on the board plane."""
    corners: dict[int, np.ndarray] = {}
    centers = []
    for sx, sy, mid in MARKERS:
        cx, cy = sx * t.delta_width_qr_center, sy * t.delta_height_qr_center
        corners[mid] = np.array(
            [[cx + dx * t.marker_size / 2, cy + dy * t.marker_size / 2, 0.0] for dx, dy in _CORNERS],
            dtype=np.float64,
        )
        centers.append([sx * t.delta_width_circles / 2, sy * t.delta_height_circles / 2, 0.0])
    return corners, np.array(centers, dtype=np.float64)


@dataclass
class CameraDetection:
    ok: bool
    ids: list[int] = field(default_factory=list)
    image_corners: np.ndarray | None = None  # (N, 4, 2)
    centers: np.ndarray | None = None  # (4, 3) circle centres, camera frame
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reproj_rms: float = 0.0
    reproj_max: float = 0.0
    marker_px: float = 0.0  # mean marker side length in pixels
    distance: float = 0.0  # metres to board centre
    tilt_deg: float = 0.0  # angle between board normal and the viewing ray
    reason: str = ""

    @property
    def n_markers(self) -> int:
        return len(self.ids)


_detector: aruco.ArucoDetector | None = None


def _get_detector() -> aruco.ArucoDetector:
    global _detector
    if _detector is None:
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        _detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(DICTIONARY), params)
    return _detector


def detect_markers(image: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """Just the marker pass -- cheap enough to run while scrubbing."""
    corners, ids, _ = _get_detector().detectMarkers(image)
    if ids is None or len(ids) == 0:
        return [], []
    return [int(i) for i in ids.ravel()], [c.reshape(4, 2) for c in corners]


def detect(image: np.ndarray, camera: Camera, target: Target, min_markers: int = 3) -> CameraDetection:
    """Full camera-side detection: markers -> board pose -> circle centres."""
    ids, corners = detect_markers(image)
    if not ids:
        return CameraDetection(False, reason="마커 없음")

    marker_px = float(
        np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for c in corners for k in range(4)])
    )
    known = [(i, c) for i, c in zip(ids, corners) if i in {m[2] for m in MARKERS}]
    if len(known) < min_markers:
        return CameraDetection(
            False, ids=ids, marker_px=marker_px,
            reason=f"보드 마커 {len(known)}개 (최소 {min_markers}개 필요)",
        )
    if not camera.is_set:
        return CameraDetection(
            False, ids=ids, marker_px=marker_px, reason="카메라 파라미터 미입력"
        )

    model, circle_model = board_model(target)
    obj = np.vstack([model[i] for i, _ in known])
    img = np.vstack([c for _, c in known]).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, camera.matrix(), camera.dist(), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return CameraDetection(False, ids=ids, marker_px=marker_px, reason="pose 계산 실패")

    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera.matrix(), camera.dist())
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)

    R, _ = cv2.Rodrigues(rvec)
    centers = (R @ circle_model.T).T + tvec.ravel()

    normal = R @ np.array([0.0, 0.0, 1.0])
    view = tvec.ravel() / np.linalg.norm(tvec)
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal @ view), 0.0, 1.0))))

    return CameraDetection(
        ok=True,
        ids=[i for i, _ in known],
        image_corners=np.array([c for _, c in known]),
        centers=centers,
        rvec=rvec,
        tvec=tvec,
        reproj_rms=float(np.sqrt((err**2).mean())),
        reproj_max=float(err.max()),
        marker_px=marker_px,
        distance=float(np.linalg.norm(tvec)),
        tilt_deg=tilt,
    )


def draw_overlay(image: np.ndarray, det: CameraDetection, camera: Camera) -> np.ndarray:
    """Annotate a copy of the image with what was found."""
    out = image.copy()
    good = (0, 220, 0)
    bad = (0, 0, 235)
    accent = (255, 0, 255)

    if det.image_corners is not None:
        for pts, mid in zip(det.image_corners, det.ids):
            poly = pts.astype(np.int32)
            cv2.polylines(out, [poly], True, good if det.ok else bad, 2)
            cv2.putText(
                out, str(mid), tuple(poly.mean(axis=0).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, good if det.ok else bad, 2,
            )

    if det.ok and det.centers is not None:
        proj, _ = cv2.projectPoints(det.centers, np.zeros(3), np.zeros(3), camera.matrix(), camera.dist())
        for (u, v) in proj.reshape(-1, 2):
            cv2.circle(out, (int(u), int(v)), 6, accent, -1)
        cv2.drawFrameAxes(out, camera.matrix(), camera.dist(), det.rvec, det.tvec, 0.2, 2)

    return out

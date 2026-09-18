"""Step 7 — put the cloud back on the image and decide whether to believe it.

Every number before this page can look right while the answer is wrong. Four
hole centres form a rectangle, a rectangle is unchanged by a half turn, and so
an extrinsic rotated 180 degrees about the board normal produces *the same
residual to the decimal* -- measured at 2.78 mm against 2.78 mm on a real scene.
With several scenes disagreeing about it, `solve()` resolves the ambiguity. With
one scene there is nothing to disagree with, and only the overlay separates them.

So the page shows the overlay, and offers the swap as a button. Deciding by eye
is not a weakness here: it is the only thing that can decide.
"""

from __future__ import annotations

import html
import threading

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gui.core import verify
from gui.core.bag_reader import BagSource, accumulate_cloud, pick_camera_for_frame
from gui.core.decode import image_to_bgr
from gui.core.detect_camera import detect as detect_camera
from gui.core.detect_lidar import DetectParams, apply_box, detect as detect_lidar, point_spacing
from pathlib import Path

from gui.core.evaluation import diagnostics as dg
from gui.core.evaluation import edge_alignment as ea
from gui.core.evaluation import multiframe_consistency as mc
from gui.core.evaluation import perturbation as pert
from gui.core.evaluation import spatial_analysis as sa
from gui.core.project import RESULT_DIR, Project
from gui.core.solve import Solution, solve_rigid, sort_centers, to_fast_livo2
from gui.ui.steps import StepPage
from gui.ui.steps.step_filter import derive_params


class _Worker(QtCore.QObject):
    opened = QtCore.Signal(int, object)  # gen, LiDAR message times
    ready = QtCore.Signal(int, object, object, object, object, object)  # gen, image, cloud, intensity, lidar c, cam c
    failed = QtCore.Signal(int, str)
    scanning = QtCore.Signal(int, int, int)  # gen, messages read, total

    # Target-independent validation (Multi-frame Consistency) -- separate from
    # the single-frame load/ready pair above so a long multi-frame run cannot
    # be mistaken for, or interrupted by, ordinary timeline scrubbing.
    mf_progress = QtCore.Signal(int, int, int)  # gen, done, total
    mf_ready = QtCore.Signal(int, object)       # gen, MultiFrameConsistencyResult
    mf_failed = QtCore.Signal(int, str)

    # Current-frame quantitative evaluation -- measured at ~0.4-1.0s on a 4K
    # frame with a realistic point count (see bench notes in the PR), which is
    # long enough to visibly stall the GUI thread on a button click, so this
    # also runs here rather than inline in the main thread.
    eval_ready = QtCore.Signal(int, object)     # gen, EdgeAlignmentResult
    eval_failed = QtCore.Signal(int, str)

    # Perturbation Sensitivity -- diagnostic-only, never writes back to a
    # Solution or a project file (see gui.core.evaluation.perturbation).
    pert_progress = QtCore.Signal(int, int, int)  # gen, done, total
    pert_ready = QtCore.Signal(int, object)       # gen, PerturbationResult
    pert_failed = QtCore.Signal(int, str)
    # Fine Scan -- a single-axis, custom-range re-run of the same machinery.
    fine_progress = QtCore.Signal(int, int, int)  # gen, done, total
    fine_ready = QtCore.Signal(int, object)       # gen, AxisSensitivity
    fine_failed = QtCore.Signal(int, str)

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None
        self._mf_cancel = threading.Event()
        self._pert_cancel = threading.Event()
        self._fine_cancel = threading.Event()

    def _source(self, bag: str) -> BagSource:
        if self._src is None or str(self._src.path) != bag:
            if self._src is not None:
                self._src.close()
            self._src = BagSource(bag)
            self._src.open()
        return self._src

    def open_bag(self, gen: int, bag: str, lidar_topic: str):
        """Read the LiDAR message times so the timeline has something to span.

        Separate from `load` because it is the slow part -- seventeen seconds on a
        36 GB recording -- and it only has to happen once per bag. The reader keeps
        the result, so switching back to a bag already visited is instant.
        """
        try:
            src = self._source(bag)
            times = src.timestamps(
                lidar_topic,
                progress=lambda done, total: self.scanning.emit(gen, done, total),
            )
            self.opened.emit(gen, sorted(times))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def load(self, gen: int, bag: str, lidar_topic: str, camera_topic: str,
             t_ns: int, frames: int, scene, camera, target, method: str,
             pinned_camera_t_ns: "int | None" = None):
        """One moment's cloud and image, and the hole centres when asked for.

        `scene` is None while scrubbing freely. Detection needs the scene's filter
        box to know where to look, so away from a captured moment only the
        projection is produced -- which is still enough to judge an extrinsic by
        eye, and often more telling than four points, since the whole scene has to
        line up rather than just the board.

        `pinned_camera_t_ns` is set for exactly one caller: a Worst-Frame jump
        from Multi-frame Consistency, which paired this LiDAR moment with its
        *nearest* camera image rather than the first one after it (see
        `gui.core.bag_reader.pick_camera_for_frame`). Left at None -- every
        other caller -- this is byte-for-byte the pairing Step 7 has always used.
        """
        try:
            src = self._source(bag)

            _, img_msg = pick_camera_for_frame(src, camera_topic, t_ns, pinned_camera_t_ns)
            if img_msg is None:
                self.failed.emit(gen, "이 시점에 카메라 이미지가 없습니다")
                return
            image = image_to_bgr(img_msg)

            cloud, _, single, intensity, ring = accumulate_cloud(
                src, lidar_topic, t_ns, frames,
                progress=lambda done, total: self.scanning.emit(gen, done, total),
            )

            lidar_c = cam_c = None
            if scene is not None:
                inside = apply_box(single, scene.filter)
                params = derive_params(point_spacing(inside) if len(inside) > 20 else 0.0)
                params.method = method
                lid = detect_lidar(cloud, scene.filter, target, params, ring=ring)
                cam = detect_camera(image, camera, target)
                lidar_c = lid.centers if lid.ok else None
                cam_c = cam.centers if cam.ok else None

            self.ready.emit(gen, image, cloud, intensity, lidar_c, cam_c)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def run_multiframe(self, gen: int, bag: str, lidar_topic: str, camera_topic: str,
                        camera, sol, times: list[int], n_samples: int, min_range: float, max_range: float,
                        max_sync_offset_ms: float):
        """Sample `n_samples` moments evenly across `times` and run Edge
        Alignment on each -- an ordinary frame, not a captured scene, so this
        is the only place that answers "does the extrinsic hold generally"
        rather than "did it fit the four holes it was solved from".

        Pairs each sampled LiDAR moment with its *nearest* camera image, not
        the first one at or after it: on a moving vehicle, a loose pairing
        shows up in the pixel error exactly like a rotation error would, so
        the evaluation path needs the closest timestamp match it can get, not
        whatever the display path's `first_after` happens to land on -- and it
        reports how far off that pairing was so a wrong-looking frame can be
        told apart from a badly-synced one instead of blaming the extrinsic.

        One sweep per frame (not the scene's own `frames` count, since these
        moments were never captured scenes) -- accumulating several sweeps per
        sampled frame would multiply the read cost for no benefit here.
        """
        self._mf_cancel.clear()
        try:
            src = self._source(bag)
            frame_loader = self._nearest_frame_loader(src, lidar_topic, camera_topic, times, max_sync_offset_ms)

            result = mc.evaluate_multiframe_consistency(
                frame_loader=frame_loader,
                n_total_timeline=len(times),
                n_samples=n_samples,
                sol=sol,
                camera=camera,
                min_range=min_range,
                max_range=max_range,
                max_sync_offset_ms=max_sync_offset_ms if max_sync_offset_ms > 0 else None,
                progress=lambda done, total: self.mf_progress.emit(gen, done, total),
                should_cancel=self._mf_cancel.is_set,
            )
            self.mf_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.mf_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def cancel_multiframe(self):
        self._mf_cancel.set()

    def _nearest_frame_loader(self, src, lidar_topic: str, camera_topic: str,
                               times: list[int], max_sync_offset_ms: float):
        """Shared by run_multiframe and run_perturbation: pairs each sampled
        LiDAR moment with its *nearest* camera image, not the first one at or
        after it (Step 7's display path) -- on a moving vehicle, a loose
        pairing shows up in the pixel error exactly like a rotation error
        would, so both evaluation paths need the closest timestamp match they
        can get. `img_stamp` is handed back raw so the caller can derive
        signed/absolute sync offset itself (see `pick_camera_for_frame` for
        how a Worst-Frame jump later pins the display to this same timestamp).
        """
        base_t = times[0] if times else 0
        # Wide enough to still find a pairing at whatever sync tolerance the
        # user configured, without turning into an unbounded bag scan.
        window_ns = max(500_000_000, int((max_sync_offset_ms or 0) * 2_000_000))

        def frame_loader(idx: int):
            # A single unreadable frame (a truncated message, a decode error)
            # must count as one failed frame, not abort a run that may cover
            # hundreds of others -- so failures are swallowed here rather
            # than left to escape and cancel the whole batch.
            try:
                t_ns = times[idx]
                img_stamp, img_msg = src.nearest(camera_topic, t_ns, window_ns=window_ns)
                if img_msg is None:
                    return None
                image = image_to_bgr(img_msg)
                cloud, _, _, _, _ = accumulate_cloud(src, lidar_topic, t_ns, frames=1)
                if len(cloud) == 0:
                    return None
                return image, cloud, t_ns, (t_ns - base_t) / 1e9, img_stamp
            except Exception:  # noqa: BLE001 - counted as a failed frame, see above
                return None

        return frame_loader

    def run_eval_current(self, gen: int, image, cloud, sol, camera, min_range: float, max_range: float):
        """The current-frame "정량 평가" button's work, off the GUI thread.

        Projects `cloud` fresh with the caller-supplied evaluation range --
        deliberately not reusing whatever Projection the display last drew,
        so this metric cannot silently move when someone nudges the display's
        range sliders (see the Evaluation Range control this is paired with).
        """
        try:
            h, w = image.shape[:2]
            pr = verify.project_cloud(cloud, sol, camera, w, h, min_range=min_range, max_range=max_range)
            if pr.n_visible == 0:
                self.eval_failed.emit(gen, "평가 범위 안에 투영된 점이 없습니다.")
                return
            result = ea.evaluate_edge_alignment(image, pr.uv, pr.depth, ea.EdgeAlignmentParams())
            self.eval_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.eval_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def _build_perturbation_frames(self, mode: str, bag: str, lidar_topic: str, camera_topic: str,
                                    camera, times: list[int], n_frames: int,
                                    max_sync_offset_ms: float, edge_params, pinned_image, pinned_cloud,
                                    should_cancel):
        """Shared by run_perturbation and run_fine_scan: build the fixed frame
        set a run scores every trial against.

        `mode == "current_frame"` reuses `pinned_image`/`pinned_cloud` exactly
        (no bag read at all -- the same identity-based pairing guarantee
        `run_eval_current` already relies on). `mode == "multi_frame"` samples
        `n_frames` moments from `times` with the same nearest-pairing policy
        Multi-frame Consistency uses, loading them once up front (see
        `perturbation.prepare_frames`'s docstring for why: the same frames get
        re-scored dozens of times here, not loaded-scored-discarded once each).
        A Fine Scan run with the same `n_frames` as the preceding Quick/Full
        run samples the identical timeline indices again (`sample_frame_indices`
        is a deterministic linspace), so the two stay comparable without
        needing to explicitly carry the frame list between UI actions.
        """
        if mode == "current_frame":
            return [pert.frame_from_current(pinned_image, pinned_cloud, edge_params)]
        src = self._source(bag)
        frame_loader = self._nearest_frame_loader(src, lidar_topic, camera_topic, times, max_sync_offset_ms)
        return pert.prepare_frames(
            frame_loader, n_total_timeline=len(times), n_samples=n_frames,
            max_sync_offset_ms=max_sync_offset_ms if max_sync_offset_ms > 0 else None,
            edge_params=edge_params, should_cancel=should_cancel,
        )

    def run_perturbation(self, gen: int, mode: str, bag: str, lidar_topic: str, camera_topic: str,
                          camera, R, t, times: list[int], n_frames: int,
                          rotation_deltas: tuple, translation_deltas: tuple,
                          min_range: float, max_range: float, max_sync_offset_ms: float,
                          pinned_image, pinned_cloud):
        """Score the current T, then every requested small rotation/translation
        nudge around it, purely as a diagnostic -- see
        `gui.core.evaluation.perturbation`'s module docstring. `R`/`t` are read
        here, never written to; the caller's `Solution`/project state is
        untouched regardless of what this finds.
        """
        self._pert_cancel.clear()
        edge_params = ea.EdgeAlignmentParams()
        try:
            frames = self._build_perturbation_frames(
                mode, bag, lidar_topic, camera_topic, camera, times, n_frames,
                max_sync_offset_ms, edge_params, pinned_image, pinned_cloud, self._pert_cancel.is_set,
            )
            result = pert.evaluate_perturbation_grid(
                frames, R, t, camera, min_range=min_range, max_range=max_range, mode=mode,
                rotation_deltas_deg=rotation_deltas, translation_deltas_mm=translation_deltas,
                edge_params=edge_params,
                progress=lambda done, total: self.pert_progress.emit(gen, done, total),
                should_cancel=self._pert_cancel.is_set,
            )
            self.pert_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.pert_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def run_fine_scan(self, gen: int, mode: str, bag: str, lidar_topic: str, camera_topic: str,
                       camera, R, t, times: list[int], n_frames: int, axis: str, deltas: tuple,
                       min_range: float, max_range: float, max_sync_offset_ms: float,
                       pinned_image, pinned_cloud):
        """Fine Scan: the same trial-scoring machinery as run_perturbation, but
        for one caller-chosen axis and delta range (see
        `gui.core.evaluation.perturbation.evaluate_single_axis`)."""
        self._fine_cancel.clear()
        edge_params = ea.EdgeAlignmentParams()
        try:
            frames = self._build_perturbation_frames(
                mode, bag, lidar_topic, camera_topic, camera, times, n_frames,
                max_sync_offset_ms, edge_params, pinned_image, pinned_cloud, self._fine_cancel.is_set,
            )
            axis_result = pert.evaluate_single_axis(
                frames, R, t, camera, axis, deltas,
                min_range=min_range, max_range=max_range, mode=mode, edge_params=edge_params,
                progress=lambda done, total: self.fine_progress.emit(gen, done, total),
                should_cancel=self._fine_cancel.is_set,
            )
            self.fine_ready.emit(gen, axis_result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.fine_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def cancel_fine_scan(self):
        self._fine_cancel.set()

    def cancel_perturbation(self):
        self._pert_cancel.set()

    def shutdown(self):
        if self._src is not None:
            self._src.close()
            self._src = None


class _FrameSlider(QtWidgets.QSlider):
    """The timeline: one wheel notch is one frame.

    A plain QSlider moves by the desktop's scroll-lines setting -- three frames a
    notch here -- which overshoots whatever you were trying to land on. Stepping
    one at a time makes the wheel the natural way to walk a recording.
    """

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        if not steps or not self.isEnabled():
            super().wheelEvent(ev)
            return
        # Wheel-up reads as "forward in time", the direction the frames advance.
        self.setValue(self.value() + int(round(steps)) * self.singleStep())
        ev.accept()


class _ZoomLabel(QtWidgets.QLabel):
    """The overlay: wheel to zoom about the cursor, left-drag to pan.

    Buttons alone meant losing your place on every step -- the view grows about
    its own centre, so a hole being inspected at the edge slides off and has to be
    found again. Zooming about the pointer keeps whatever is under it still, and
    dragging is how you move once zoomed in past the window.
    """

    zoomed = QtCore.Signal(float, QtCore.QPointF)
    panned = QtCore.Signal(QtCore.QPoint)  # movement in pixels, to subtract from scroll
    stepped = QtCore.Signal(int)           # Shift+wheel: frames to advance
    sized = QtCore.Signal(int)             # Ctrl+wheel: notches of point size
    picked = QtCore.Signal(QtCore.QPointF)  # a click that did not drag, in label coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("6단계 캘리브레이션을 완료해주세요.")
        self._drag_from: QtCore.QPoint | None = None
        # Where the button went down, so a click can be told from a drag: panning
        # and picking share the left button, and only movement separates them.
        self._press_at: QtCore.QPointF | None = None

    def _has_image(self) -> bool:
        pm = self.pixmap()
        return pm is not None and not pm.isNull()

    def wheelEvent(self, ev):
        if not self._has_image():
            super().wheelEvent(ev)
            return
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            super().wheelEvent(ev)
            return
        # Three things one wants to sweep without leaving the image: the zoom, the
        # frame, and how fat the points are. Bare wheel is zoom because it is the
        # one you reach for most; the other two take a modifier.
        mods = ev.modifiers()
        if mods & QtCore.Qt.ShiftModifier:
            self.stepped.emit(int(round(steps)))
        elif mods & QtCore.Qt.ControlModifier:
            self.sized.emit(int(round(steps)))
        else:
            self.zoomed.emit(1.25 ** steps, ev.position())
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._has_image():
            self._press_at = ev.position()
            self._drag_from = ev.position().toPoint()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_from is not None:
            now = ev.position().toPoint()
            self.panned.emit(now - self._drag_from)
            # The scroll offset moves under the widget, so the grab point stays
            # where it is in widget coordinates -- do not advance it.
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._drag_from is not None:
            start = self._press_at
            self._press_at = None
            self._drag_from = None
            if start is not None:
                moved = (ev.position() - start).manhattanLength()
                if moved <= 3:
                    self.picked.emit(ev.position())
            self.unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class VerifyStep(StepPage):
    title = "7. 검증"
    subtitle = "이미지에 투영해 확인"

    request_open = QtCore.Signal(int, str, str)
    # ... method, pinned camera t_ns (None for ordinary navigation)
    request_load = QtCore.Signal(int, str, str, str, object, int, object, object, object, str, object)
    # gen, bag, lidar topic, camera topic, camera, sol, times, n_samples, eval min, eval max, max sync offset ms
    request_multiframe = QtCore.Signal(int, str, str, str, object, object, object, int, float, float, float)
    # gen, image, cloud, sol, camera, eval min, eval max
    request_eval_current = QtCore.Signal(int, object, object, object, object, float, float)
    # gen, mode, bag, lidar topic, camera topic, camera, R, t, times, n_frames,
    # eval min, eval max, max sync offset ms, pinned image, pinned cloud
    # gen, mode, bag, lidar topic, camera topic, camera, R, t, times, n_frames,
    # rotation deltas, translation deltas, eval min, eval max, max sync offset ms, pinned image, pinned cloud
    request_perturbation = QtCore.Signal(
        int, str, str, str, str, object, object, object, object, int,
        object, object, float, float, float, object, object,
    )
    # gen, mode, bag, lidar topic, camera topic, camera, R, t, times, n_frames,
    # axis, deltas, eval min, eval max, max sync offset ms, pinned image, pinned cloud
    request_fine_scan = QtCore.Signal(
        int, str, str, str, str, object, object, object, object, int, str, object,
        float, float, float, object, object,
    )

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self._gen = 0
        self._image: np.ndarray | None = None
        self._cloud: np.ndarray | None = None
        self._last_projection = None
        self._picked: int | None = None
        self._intensity: np.ndarray | None = None
        self._lidar_c: np.ndarray | None = None
        self._cam_c: np.ndarray | None = None
        self._sol: Solution | None = None
        self._flipped = False
        self._zoom = 1.0
        self._fit = True
        self._times: list[int] = []
        self._loaded_for: tuple | None = None
        self._scene = None  # the captured scene at the current time, if any
        self._loaded_from: str | None = None  # extrinsic opened from a file
        self._loaded_meta: dict = {}
        self._mf_gen = 0
        self._mf_result: mc.MultiFrameConsistencyResult | None = None
        self._eval_gen = 0
        self._eval_frame_shape: tuple[int, int] = (0, 0)
        # One-shot: consumed and cleared by the very next _request(), so only
        # the jump that set it is affected -- normal scrubbing right after a
        # Worst-Frame jump goes straight back to ordinary first_after pairing.
        self._pinned_camera_t_ns: "int | None" = None
        self._pert_gen = 0
        self._pert_result: pert.PerturbationResult | None = None
        self._fine_gen = 0
        self._fine_result: pert.AxisSensitivity | None = None
        # What each stored result was computed against -- (R, t) copies plus
        # the run settings the diagnostics need to describe it. Diagnostics
        # compare these to the extrinsic shown *now* and leave a stale result
        # out rather than interpret numbers that belong to a different T.
        self._eval_result: ea.EdgeAlignmentResult | None = None
        self._eval_spatial: sa.SpatialAnalysisResult | None = None
        self._eval_T: tuple | None = None
        self._mf_T: tuple | None = None
        self._mf_sync_limit_ms: float = 0.0
        self._pert_T: tuple | None = None
        self._pert_sync_limit_ms: float = 0.0
        self._fine_T: tuple | None = None
        self._fine_mode: str = "multi_frame"
        self._diag_report: dg.DiagnosticReport | None = None

        # --- controls --------------------------------------------------------
        # Checking the extrinsic only at the scenes it was fitted to is marking
        # your own work: those four points are what the fit minimised, so they
        # agree by construction. A timeline over the whole recording -- and over
        # every bag in the project, including one added purely to check against --
        # is what actually tests it.
        self.bag_combo = QtWidgets.QComboBox()
        self.bag_combo.currentIndexChanged.connect(self._switch_bag)

        self.slider = _FrameSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimumHeight(24)  # a wheel target you can hit without aiming
        self.slider.setToolTip("드래그하거나 마우스 휠로 한 프레임씩 이동합니다")
        self.slider.valueChanged.connect(self._on_slider)
        self.time_label = QtWidgets.QLabel("—")
        self.time_label.setStyleSheet("font-family: monospace;")
        self.time_label.setFixedWidth(150)

        # Reloading on every slider tick would queue a decode per pixel dragged.
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._request)

        self.scene_combo = QtWidgets.QComboBox()
        self.scene_combo.setToolTip("캡처한 scene 으로 이동합니다")
        self.scene_combo.currentIndexChanged.connect(self._goto_scene)

        self.colour_combo = QtWidgets.QComboBox()
        self.colour_combo.addItem("거리", "depth")
        self.colour_combo.addItem("Intensity", "intensity")
        # RViz's Axis Color. Height separates road from wall, which distance
        # cannot: on a forward view both sit at the same depth. It is the default
        # because that separation is what one is usually looking for.
        self.colour_combo.addItem("높이 z", "axis_z")
        self.colour_combo.addItem("전후 x", "axis_x")
        self.colour_combo.addItem("좌우 y", "axis_y")
        # One flat colour, for when the only question is whether lines fall where
        # they should and a ramp just adds a second pattern to read past.
        self.colour_combo.addItem("단색", "solid")
        self.colour_combo.setCurrentIndex(self.colour_combo.findData("axis_z"))
        self.colour_combo.currentIndexChanged.connect(self._redraw)

        self.size_spin = QtWidgets.QDoubleSpinBox()
        self.size_spin.setRange(0.5, 7.0)
        self.size_spin.setSingleStep(0.1)
        self.size_spin.setDecimals(1)
        self.size_spin.setValue(1.5)
        self.size_spin.setSuffix(" px")
        self.size_spin.setToolTip(
            "점 하나를 몇 픽셀로 그릴지. 1.5 미만은 한 픽셀로 찍습니다.\n"
            "실제로 굵어지는 지점은 1.5, 3.0, 5.1 이고 그 사이는 같게 보입니다"
        )
        self.size_spin.valueChanged.connect(self._redraw)

        self.dim_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dim_slider.setRange(0, 100)
        self.dim_slider.setValue(40)
        self.dim_slider.setFixedWidth(90)
        self.dim_slider.valueChanged.connect(self._redraw)

        # Both ends, not just the far one. Points a few centimetres from the lens
        # -- vehicle body, a mount, the sensor's own housing -- project across the
        # whole frame and bury the board, so a near cut matters as much as a far one.
        self.near_spin = QtWidgets.QDoubleSpinBox()
        self.near_spin.setRange(0.0, 200.0)
        self.near_spin.setValue(0.0)
        self.near_spin.setSingleStep(0.5)
        self.near_spin.setSuffix(" m")
        self.near_spin.setSpecialValueText("제한 없음")
        self.near_spin.setToolTip("이 거리보다 가까운 점은 그리지 않습니다")
        self.near_spin.valueChanged.connect(self._on_range)

        self.far_spin = QtWidgets.QDoubleSpinBox()
        self.far_spin.setRange(0.0, 200.0)
        # 10 m showed only the road right in front. Most of what tells you an
        # extrinsic is right -- walls, poles, the far edge of the carriageway --
        # sits beyond that.
        self.far_spin.setValue(50.0)
        self.far_spin.setSingleStep(0.5)
        self.far_spin.setSuffix(" m")
        self.far_spin.setSpecialValueText("제한 없음")
        self.far_spin.setToolTip("이 거리보다 먼 점은 그리지 않습니다")
        self.far_spin.valueChanged.connect(self._on_range)

        # Sliders alongside the boxes: sweeping a cut through the scene to see what
        # each depth contains is a different action from typing a known number, and
        # typing cannot do it. Tenths of a metre, which is the useful resolution
        # here -- 0.5 m steps jump straight past a board.
        self.near_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # Matched to the spin box's 0-200 m at 1/100 m. They were 0-2000 (20 m)
        # while the boxes went to 200, so any value past 20 m pinned the slider
        # at its end and the two disagreed.
        self.near_slider.setRange(0, 20000)
        self.near_slider.setFixedWidth(110)
        self.near_slider.setToolTip("가까운 쪽 자르기")
        self.near_slider.valueChanged.connect(
            lambda v: self._sync_range(self.near_spin, v / 100.0)
        )
        self.far_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.far_slider.setRange(0, 20000)
        self.far_slider.setValue(5000)
        self.far_slider.setFixedWidth(110)
        self.far_slider.setToolTip("먼 쪽 자르기")
        self.far_slider.valueChanged.connect(
            lambda v: self._sync_range(self.far_spin, v / 100.0)
        )

        self.markers_check = QtWidgets.QCheckBox("원 중심 표시")
        self.markers_check.setChecked(True)
        self.markers_check.setToolTip(
            "원 = 카메라가 계산한 구멍 중심\n십자 = LiDAR 가 찾은 중심을 extrinsic 으로 옮긴 것\n"
            "둘이 겹치면 맞고, 대각선으로 어긋나면 반바퀴 뒤집힌 것입니다."
        )
        self.markers_check.toggled.connect(self._redraw)

        self.flip_btn = QtWidgets.QPushButton("반바퀴 뒤집기")
        self.flip_btn.setToolTip(
            "구멍 사각형은 반 바퀴 돌려도 같아 보여서, 180도 뒤집힌 extrinsic 도 잔차가 똑같이 나옵니다.\n"
            "scene 이 하나뿐이면 계산으로는 구분할 수 없으니 화면을 보고 고르세요."
        )
        self.flip_btn.clicked.connect(self._flip)

        self.fit_btn = QtWidgets.QPushButton("전체 보기")
        self.fit_btn.clicked.connect(self._fit_view)
        self.zoom_in = QtWidgets.QPushButton("＋")
        self.zoom_out = QtWidgets.QPushButton("－")
        for b in (self.zoom_in, self.zoom_out):
            b.setFixedWidth(32)
        self.zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.4))
        self.zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.4))

        # --- image -----------------------------------------------------------
        self.canvas = _ZoomLabel()
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setStyleSheet("background: #101216; color: palette(mid);")
        self.canvas.setToolTip(
            "휠: 확대·축소 (커서 기준)\n"
            "Shift+휠: 한 프레임씩 이동\n"
            "Ctrl+휠: 점 크기\n"
            "왼쪽 드래그: 화면 이동"
        )
        self.canvas.zoomed.connect(self._on_wheel)
        self.canvas.stepped.connect(self._step_frames)
        self.canvas.sized.connect(self._step_size)
        self.canvas.panned.connect(self._on_pan)
        self.canvas.picked.connect(self._pick_point)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(360)

        # --- readouts --------------------------------------------------------
        self.verdict = QtWidgets.QLabel("—")
        self.verdict.setTextFormat(QtCore.Qt.RichText)
        self.verdict.setWordWrap(True)

        self.numbers = QtWidgets.QLabel("—")
        self.numbers.setTextFormat(QtCore.Qt.RichText)
        self.numbers.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.export_combo = QtWidgets.QComboBox()
        self.export_combo.addItem("FAST-LIVO2 형식", "livo2")
        self.export_combo.addItem("YAML (4x4 변환)", "yaml")
        self.export_combo.addItem("ROS static_transform_publisher", "tf")
        self.export_combo.addItem("RT_Matrix (평문 3x4 + K + D)", "rt")
        self.export_combo.currentIndexChanged.connect(self._update_export)
        self.export_text = QtWidgets.QPlainTextEdit()
        self.export_text.setReadOnly(True)
        self.export_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.export_text.setMaximumHeight(150)
        # Clicking a projected dot answers "what was this before it became a
        # pixel" -- the question that comes up every time something looks wrong in
        # the overlay and cannot be settled from the picture alone.
        self.pick_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>점을 클릭하면 원래 3D 좌표를 봅니다.</span>"
        )
        self.pick_label.setWordWrap(True)
        self.pick_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.capture_btn = QtWidgets.QPushButton("캡처 저장…")
        self.capture_btn.setToolTip(
            "지금 보고 있는 겹침 그림을 원본 해상도로 저장합니다.\n"
            "같은 이름의 .txt 에 어떤 조건으로 만든 그림인지 함께 기록합니다"
        )
        self.capture_btn.clicked.connect(self._save_capture)

        copy_btn = QtWidgets.QPushButton("복사")
        copy_btn.clicked.connect(self._copy_export)
        save_btn = QtWidgets.QPushButton("파일로 저장…")
        save_btn.clicked.connect(self._save_export)

        # Checking someone else's answer is the common case on a team, and until
        # now this page only opened after step 6 had run here. A saved extrinsic
        # carries the vehicle, the bag and the topics with it, so the file says
        # what it should be checked against.
        self.load_btn = QtWidgets.QPushButton("extrinsic 불러와 검증…")
        self.load_btn.setToolTip(
            "다른 사람이 계산한 extrinsic 파일(YAML)을 열어 이 bag 에 투영해 봅니다.\n"
            "6단계를 돌리지 않아도 되고, 파일에 적힌 bag·토픽·카메라를 그대로 씁니다."
        )
        self.load_btn.clicked.connect(self._load_extrinsic)

        # --- quantitative validation (target-independent) --------------------
        # Step 6's RMSE/LOO/coverage judge fit to the four scenes an extrinsic
        # was solved from. These judge the same extrinsic against ordinary
        # frames it never saw -- a different question, so they live on their
        # own tab rather than mixed into "판정"/"수치" above.
        self.eval_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>Target 없이, 일반 장면에서의 정렬 오차(px)를 봅니다. "
            "6단계 RMSE와는 별개의 지표입니다.</span>"
        )
        self.eval_info_label.setWordWrap(True)

        # Deliberately separate from near_spin/far_spin above (the display
        # range): reusing the display's projection would mean nudging that
        # slider silently changes the metric, and two calibrations checked at
        # different display ranges are not a fair comparison. This range
        # applies to both the current-frame button and every Multi-frame run.
        self.eval_range_note = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>표시(Display) range와 별개입니다 — "
            "평가에는 항상 이 범위만 적용됩니다.</span>"
        )
        self.eval_range_note.setWordWrap(True)

        self.eval_min_range_spin = QtWidgets.QDoubleSpinBox()
        self.eval_min_range_spin.setRange(0.0, 200.0)
        self.eval_min_range_spin.setValue(0.0)
        self.eval_min_range_spin.setSingleStep(0.5)
        self.eval_min_range_spin.setSuffix(" m")
        self.eval_min_range_spin.setSpecialValueText("제한 없음")
        self.eval_min_range_spin.setToolTip("평가 범위: 이 거리보다 가까운 점은 제외합니다")

        self.eval_max_range_spin = QtWidgets.QDoubleSpinBox()
        self.eval_max_range_spin.setRange(0.0, 200.0)
        self.eval_max_range_spin.setValue(0.0)
        self.eval_max_range_spin.setSingleStep(0.5)
        self.eval_max_range_spin.setSuffix(" m")
        self.eval_max_range_spin.setSpecialValueText("제한 없음")
        self.eval_max_range_spin.setToolTip("평가 범위: 이 거리보다 먼 점은 제외합니다")

        self.eval_run_btn = QtWidgets.QPushButton("현재 프레임 평가")
        self.eval_run_btn.setToolTip(
            "지금 화면의 LiDAR depth-edge 와 카메라 Canny edge 사이 pixel 거리를 계산합니다.\n"
            "위 평가 범위로 새로 투영합니다 (화면 표시용 range와는 다를 수 있습니다).\n"
            "4K 해상도에서 최대 1초 가까이 걸릴 수 있어 백그라운드에서 실행합니다."
        )
        self.eval_run_btn.clicked.connect(self._eval_current_frame)

        self.eval_edge_label = QtWidgets.QLabel("—")
        self.eval_edge_label.setTextFormat(QtCore.Qt.RichText)
        self.eval_edge_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.eval_spatial_label = QtWidgets.QLabel("—")
        self.eval_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.eval_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.eval_depth_label = QtWidgets.QLabel("—")
        self.eval_depth_label.setTextFormat(QtCore.Qt.RichText)
        self.eval_depth_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.mf_count_spin = QtWidgets.QSpinBox()
        self.mf_count_spin.setRange(5, 5000)
        self.mf_count_spin.setValue(100)
        self.mf_count_spin.setToolTip("전체 timeline 에서 균등하게 뽑아 평가할 frame 개수")

        # On a moving vehicle, a camera/LiDAR pairing looser than this reads
        # as a rotation error in the pixel statistics -- such a frame is kept
        # out of the geometric aggregate entirely (see sync_rejected below)
        # rather than quietly counted as evidence about the extrinsic.
        self.max_sync_spin = QtWidgets.QSpinBox()
        self.max_sync_spin.setRange(0, 1000)
        self.max_sync_spin.setValue(50)
        self.max_sync_spin.setSuffix(" ms")
        self.max_sync_spin.setSpecialValueText("제한 없음")
        self.max_sync_spin.setToolTip(
            "camera/LiDAR 페어링 시간 차이가 이 값을 넘는 frame은 geometric 평가에서 제외하고\n"
            "sync_rejected 로 따로 집계합니다. 0 = 제한 없음(그래도 오차는 계속 보고합니다)."
        )

        self.mf_run_btn = QtWidgets.QPushButton("Multi-frame 평가 실행")
        self.mf_run_btn.setToolTip(
            "scene 이 아닌 일반 frame을 timeline 전체에서 균등 추출해, camera/LiDAR nearest timestamp 로\n"
            "짝지은 뒤 각각 Edge Alignment 를 계산합니다. 시간이 걸릴 수 있어 백그라운드에서 실행합니다."
        )
        self.mf_run_btn.clicked.connect(self._run_multiframe)

        self.mf_cancel_btn = QtWidgets.QPushButton("취소")
        self.mf_cancel_btn.setVisible(False)
        self.mf_cancel_btn.clicked.connect(self._cancel_multiframe)

        self.mf_progress_label = QtWidgets.QLabel("")
        self.mf_progress_label.setStyleSheet("color: palette(mid);")

        self.mf_summary_label = QtWidgets.QLabel("—")
        self.mf_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.mf_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.mf_worst_table = QtWidgets.QTableWidget(0, 5)
        self.mf_worst_table.setHorizontalHeaderLabels(["시간", "Error", "Match", "Sync Δt", "이동"])
        self.mf_worst_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.mf_worst_table.verticalHeader().setVisible(False)

        # --- perturbation sensitivity (diagnostic only) -----------------------
        # Answers "is the current T sitting at a local minimum of Edge
        # Alignment error, or would a tiny nudge do better" -- strictly by
        # re-evaluating the existing metric at small, temporary copies of T.
        # Nothing here ever writes to Solution/Project; see
        # gui.core.evaluation.perturbation's module docstring.
        self.pert_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>진단 전용 — T_cam_lidar를 자동으로 수정하거나 저장하지 않습니다.</span>"
        )
        self.pert_info_label.setWordWrap(True)

        self.pert_mode_current_radio = QtWidgets.QRadioButton("Current Frame")
        self.pert_mode_multi_radio = QtWidgets.QRadioButton("Multi-frame")
        self.pert_mode_multi_radio.setChecked(True)
        self.pert_mode_group = QtWidgets.QButtonGroup(self)
        self.pert_mode_group.addButton(self.pert_mode_current_radio)
        self.pert_mode_group.addButton(self.pert_mode_multi_radio)
        self.pert_mode_current_radio.toggled.connect(self._update_pert_trial_count)

        # Quick (13 trials) is the default so a first look stays fast; Full
        # (43 trials, the original grid) is unchanged and still available.
        self.pert_search_quick_radio = QtWidgets.QRadioButton("Quick")
        self.pert_search_full_radio = QtWidgets.QRadioButton("Full")
        self.pert_search_quick_radio.setChecked(True)
        self.pert_search_group = QtWidgets.QButtonGroup(self)
        self.pert_search_group.addButton(self.pert_search_quick_radio)
        self.pert_search_group.addButton(self.pert_search_full_radio)
        self.pert_search_quick_radio.setToolTip(
            f"Roll/Pitch/Yaw {pert.QUICK_ROTATION_DELTAS_DEG}, Tx/Ty/Tz {pert.QUICK_TRANSLATION_DELTAS_MM} -- 13 trial"
        )
        self.pert_search_full_radio.setToolTip(
            f"Roll/Pitch/Yaw {pert.DEFAULT_ROTATION_DELTAS_DEG}, Tx/Ty/Tz {pert.DEFAULT_TRANSLATION_DELTAS_MM} -- 43 trial"
        )
        self.pert_search_quick_radio.toggled.connect(self._update_pert_trial_count)

        self.pert_frames_spin = QtWidgets.QSpinBox()
        self.pert_frames_spin.setRange(5, 500)
        self.pert_frames_spin.setValue(30)
        self.pert_frames_spin.setToolTip(
            "Multi-frame 모드에서 사용할 frame 수.\n"
            "축(6개) x perturbation 값마다 다시 평가하므로 큰 값은 비용이 매우 커집니다."
        )
        self.pert_frames_spin.valueChanged.connect(self._update_pert_trial_count)

        self.pert_trial_count_label = QtWidgets.QLabel("")
        self.pert_trial_count_label.setStyleSheet("color: palette(mid);")

        self.pert_run_btn = QtWidgets.QPushButton("분석 실행")
        self.pert_run_btn.setToolTip(
            "현재 T 주변에서 Roll/Pitch/Yaw/Tx/Ty/Tz를 각각 조금씩 바꿔가며\n"
            "Edge Alignment가 어떻게 변하는지 봅니다. T는 절대 변경되지 않습니다."
        )
        self.pert_run_btn.clicked.connect(self._run_perturbation)

        self.pert_cancel_btn = QtWidgets.QPushButton("취소")
        self.pert_cancel_btn.setVisible(False)
        self.pert_cancel_btn.clicked.connect(self._cancel_perturbation)

        self.pert_progress_label = QtWidgets.QLabel("")
        self.pert_progress_label.setStyleSheet("color: palette(mid);")

        self.pert_axis_combo = QtWidgets.QComboBox()
        for key in (*pert.ROTATION_AXES, *pert.TRANSLATION_AXES):
            self.pert_axis_combo.addItem(pert.AXIS_LABELS[key], key)
        self.pert_axis_combo.currentIndexChanged.connect(self._display_perturbation_axis)
        self.pert_axis_combo.currentIndexChanged.connect(self._on_pert_axis_changed_for_fine_scan)

        self.pert_table_label = QtWidgets.QLabel("—")
        self.pert_table_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.pert_frame_table_label = QtWidgets.QLabel("—")
        self.pert_frame_table_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_frame_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.pert_frame_table_label.setToolTip(
            "Multi-frame 모드의 기본 판단 기준: pooled(합산) 통계는 edge 점이 많은 frame이\n"
            "결과를 지배할 수 있어, frame마다 동일 가중치로 비교한 값을 별도로 보여줍니다."
        )

        self.pert_summary_label = QtWidgets.QLabel("—")
        self.pert_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_summary_label.setWordWrap(True)
        self.pert_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.pert_spatial_label = QtWidgets.QLabel("—")
        self.pert_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- fine scan (one axis, custom range) -------------------------------
        self.fine_axis_label = QtWidgets.QLabel("—")
        self.fine_axis_label.setStyleSheet("color: palette(mid);")

        self.fine_min_spin = QtWidgets.QDoubleSpinBox()
        self.fine_min_spin.setRange(-100.0, 100.0)
        self.fine_min_spin.setDecimals(3)
        self.fine_min_spin.setValue(-0.2)

        self.fine_max_spin = QtWidgets.QDoubleSpinBox()
        self.fine_max_spin.setRange(-100.0, 100.0)
        self.fine_max_spin.setDecimals(3)
        self.fine_max_spin.setValue(0.2)

        self.fine_step_spin = QtWidgets.QDoubleSpinBox()
        self.fine_step_spin.setRange(0.001, 100.0)
        self.fine_step_spin.setDecimals(3)
        self.fine_step_spin.setValue(0.05)

        self.fine_run_btn = QtWidgets.QPushButton("Fine Scan 실행")
        self.fine_run_btn.setToolTip("선택한 축(위 콤보박스)만, 지정한 범위/간격으로 다시 촘촘히 평가합니다.")
        self.fine_run_btn.clicked.connect(self._run_fine_scan)

        self.fine_cancel_btn = QtWidgets.QPushButton("취소")
        self.fine_cancel_btn.setVisible(False)
        self.fine_cancel_btn.clicked.connect(self._cancel_fine_scan)

        self.fine_progress_label = QtWidgets.QLabel("")
        self.fine_progress_label.setStyleSheet("color: palette(mid);")

        self.fine_table_label = QtWidgets.QLabel("—")
        self.fine_table_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.fine_frame_table_label = QtWidgets.QLabel("—")
        self.fine_frame_table_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_frame_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.fine_summary_label = QtWidgets.QLabel("—")
        self.fine_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_summary_label.setWordWrap(True)
        self.fine_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.fine_spatial_label = QtWidgets.QLabel("—")
        self.fine_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- diagnostic evidence (interpretation only) -------------------------
        # Reads the results above as they already are; never re-projects,
        # re-scores, re-perturbs, or re-solves, and never offers to change T --
        # see gui.core.evaluation.diagnostics.
        self.diag_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>이미 계산된 결과만 해석합니다 — 원인을 확정하거나 "
            "T_cam_lidar를 수정하지 않습니다.<br>"
            "<b>Evidence strength, not a calibration correctness score.</b></span>"
        )
        self.diag_info_label.setWordWrap(True)

        self.diag_run_btn = QtWidgets.QPushButton("진단 요약 생성")
        self.diag_run_btn.setToolTip(
            "Multi-frame / Perturbation / Fine Scan / 현재 프레임 평가 / 6단계 Leave-One-Out 결과를\n"
            "evidence category 별로 묶어 보여줍니다. 새로운 평가는 실행하지 않습니다."
        )
        self.diag_run_btn.clicked.connect(self._generate_diagnostics)

        self.diag_copy_btn = QtWidgets.QPushButton("진단 결과 복사")
        self.diag_copy_btn.setEnabled(False)
        self.diag_copy_btn.clicked.connect(self._copy_diagnostics)

        self.diag_label = QtWidgets.QLabel("—")
        self.diag_label.setTextFormat(QtCore.Qt.RichText)
        self.diag_label.setWordWrap(True)
        self.diag_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- layout ----------------------------------------------------------
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("bag"))
        top.addWidget(self.bag_combo, 1)
        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("scene 이동"))
        top.addWidget(self.scene_combo)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("색상"))
        top.addWidget(self.colour_combo)
        top.addWidget(QtWidgets.QLabel("점"))
        top.addWidget(self.size_spin)
        top.addWidget(QtWidgets.QLabel("배경"))
        top.addWidget(self.dim_slider)
        top.addWidget(QtWidgets.QLabel("거리"))
        top.addWidget(self.near_spin)
        top.addWidget(self.near_slider)
        top.addWidget(QtWidgets.QLabel("~"))
        top.addWidget(self.far_spin)
        top.addWidget(self.far_slider)
        top.addWidget(self.markers_check)
        top.addStretch(1)
        top.addWidget(self.zoom_out)
        top.addWidget(self.zoom_in)
        top.addWidget(self.fit_btn)

        timeline = QtWidgets.QHBoxLayout()
        timeline.addWidget(self.slider, 1)
        timeline.addWidget(self.time_label)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("<b>판정</b>"))
        side.addWidget(self.verdict)
        side.addWidget(self.flip_btn)
        side.addSpacing(10)
        side.addWidget(QtWidgets.QLabel("<b>수치</b>"))
        side.addWidget(self.numbers)
        side.addStretch(1)
        side.addWidget(QtWidgets.QLabel("<b>클릭한 점</b>"))
        side.addWidget(self.pick_label)
        side.addSpacing(10)
        side.addWidget(self.capture_btn)
        side.addWidget(self.load_btn)
        side.addSpacing(10)
        side.addWidget(QtWidgets.QLabel("<b>내보내기</b>"))
        side.addWidget(self.export_combo)
        side.addWidget(self.export_text)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(copy_btn)
        row.addWidget(save_btn)
        side.addLayout(row)
        side_box = QtWidgets.QWidget()
        side_box.setLayout(side)

        eval_layout = QtWidgets.QVBoxLayout()
        eval_layout.addWidget(self.eval_info_label)
        eval_layout.addSpacing(6)
        eval_layout.addWidget(QtWidgets.QLabel("<b>평가 범위 (Evaluation Range)</b>"))
        eval_layout.addWidget(self.eval_range_note)
        eval_range_row = QtWidgets.QHBoxLayout()
        eval_range_row.addWidget(QtWidgets.QLabel("Min"))
        eval_range_row.addWidget(self.eval_min_range_spin)
        eval_range_row.addWidget(QtWidgets.QLabel("Max"))
        eval_range_row.addWidget(self.eval_max_range_spin)
        eval_layout.addLayout(eval_range_row)
        eval_layout.addSpacing(10)
        eval_layout.addWidget(self.eval_run_btn)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Edge Alignment</b>"))
        eval_layout.addWidget(self.eval_edge_label)
        eval_layout.addSpacing(8)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Spatial</b>"))
        eval_layout.addWidget(self.eval_spatial_label)
        eval_layout.addSpacing(8)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Depth</b>"))
        eval_layout.addWidget(self.eval_depth_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Multi-frame 평가</b>"))
        mf_row = QtWidgets.QHBoxLayout()
        mf_row.addWidget(QtWidgets.QLabel("평가 프레임 수"))
        mf_row.addWidget(self.mf_count_spin)
        eval_layout.addLayout(mf_row)
        sync_row = QtWidgets.QHBoxLayout()
        sync_row.addWidget(QtWidgets.QLabel("Max Sync Offset"))
        sync_row.addWidget(self.max_sync_spin)
        eval_layout.addLayout(sync_row)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.mf_run_btn, 1)
        run_row.addWidget(self.mf_cancel_btn)
        eval_layout.addLayout(run_row)
        eval_layout.addWidget(self.mf_progress_label)
        eval_layout.addWidget(self.mf_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Worst Frames"))
        eval_layout.addWidget(self.mf_worst_table)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Perturbation Sensitivity</b>"))
        eval_layout.addWidget(self.pert_info_label)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self.pert_mode_current_radio)
        mode_row.addWidget(self.pert_mode_multi_radio)
        eval_layout.addLayout(mode_row)
        search_row = QtWidgets.QHBoxLayout()
        search_row.addWidget(QtWidgets.QLabel("Search Mode"))
        search_row.addWidget(self.pert_search_quick_radio)
        search_row.addWidget(self.pert_search_full_radio)
        eval_layout.addLayout(search_row)
        pert_frames_row = QtWidgets.QHBoxLayout()
        pert_frames_row.addWidget(QtWidgets.QLabel("Frames"))
        pert_frames_row.addWidget(self.pert_frames_spin)
        eval_layout.addLayout(pert_frames_row)
        eval_layout.addWidget(self.pert_trial_count_label)
        pert_run_row = QtWidgets.QHBoxLayout()
        pert_run_row.addWidget(self.pert_run_btn, 1)
        pert_run_row.addWidget(self.pert_cancel_btn)
        eval_layout.addLayout(pert_run_row)
        eval_layout.addWidget(self.pert_progress_label)
        pert_axis_row = QtWidgets.QHBoxLayout()
        pert_axis_row.addWidget(QtWidgets.QLabel("축"))
        pert_axis_row.addWidget(self.pert_axis_combo)
        eval_layout.addLayout(pert_axis_row)
        eval_layout.addWidget(QtWidgets.QLabel("Pooled (diagnostic)"))
        eval_layout.addWidget(self.pert_table_label)
        eval_layout.addWidget(QtWidgets.QLabel("Frame-balanced (Multi-frame 판단 기준)"))
        eval_layout.addWidget(self.pert_frame_table_label)
        eval_layout.addWidget(self.pert_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Spatial (Baseline vs Lowest tested)"))
        eval_layout.addWidget(self.pert_spatial_label)
        eval_layout.addSpacing(10)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Fine Scan</b>"))
        eval_layout.addWidget(self.fine_axis_label)
        fine_range_row = QtWidgets.QHBoxLayout()
        fine_range_row.addWidget(QtWidgets.QLabel("Min"))
        fine_range_row.addWidget(self.fine_min_spin)
        fine_range_row.addWidget(QtWidgets.QLabel("Max"))
        fine_range_row.addWidget(self.fine_max_spin)
        fine_range_row.addWidget(QtWidgets.QLabel("Step"))
        fine_range_row.addWidget(self.fine_step_spin)
        eval_layout.addLayout(fine_range_row)
        fine_run_row = QtWidgets.QHBoxLayout()
        fine_run_row.addWidget(self.fine_run_btn, 1)
        fine_run_row.addWidget(self.fine_cancel_btn)
        eval_layout.addLayout(fine_run_row)
        eval_layout.addWidget(self.fine_progress_label)
        eval_layout.addWidget(QtWidgets.QLabel("Pooled (diagnostic)"))
        eval_layout.addWidget(self.fine_table_label)
        eval_layout.addWidget(QtWidgets.QLabel("Frame-balanced"))
        eval_layout.addWidget(self.fine_frame_table_label)
        eval_layout.addWidget(self.fine_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Spatial (Baseline vs Lowest tested)"))
        eval_layout.addWidget(self.fine_spatial_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Diagnostic Evidence</b>"))
        eval_layout.addWidget(self.diag_info_label)
        diag_row = QtWidgets.QHBoxLayout()
        diag_row.addWidget(self.diag_run_btn, 1)
        diag_row.addWidget(self.diag_copy_btn)
        eval_layout.addLayout(diag_row)
        eval_layout.addWidget(self.diag_label)
        eval_layout.addStretch(1)
        eval_box = QtWidgets.QWidget()
        eval_box.setLayout(eval_layout)
        eval_scroll = QtWidgets.QScrollArea()
        eval_scroll.setWidget(eval_box)
        eval_scroll.setWidgetResizable(True)

        self.side_tabs = QtWidgets.QTabWidget()
        self.side_tabs.addTab(side_box, "검증")
        self.side_tabs.addTab(eval_scroll, "정량 평가")
        self.side_tabs.setFixedWidth(360)

        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(self.scroll, 1)
        middle.addWidget(self.side_tabs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(timeline)
        layout.addLayout(middle, 1)

        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.opened.connect(self._on_opened)
        self._worker.scanning.connect(self._on_scanning)
        self._worker.mf_progress.connect(self._on_mf_progress)
        self._worker.mf_ready.connect(self._on_mf_ready)
        self._worker.mf_failed.connect(self._on_mf_failed)
        self._worker.eval_ready.connect(self._on_eval_ready)
        self._worker.eval_failed.connect(self._on_eval_failed)
        self._worker.pert_progress.connect(self._on_pert_progress)
        self._worker.pert_ready.connect(self._on_pert_ready)
        self._worker.pert_failed.connect(self._on_pert_failed)
        self._worker.fine_progress.connect(self._on_fine_progress)
        self._worker.fine_ready.connect(self._on_fine_ready)
        self._worker.fine_failed.connect(self._on_fine_failed)
        self.request_load.connect(self._worker.load)
        self.request_open.connect(self._worker.open_bag)
        self.request_multiframe.connect(self._worker.run_multiframe)
        self.request_eval_current.connect(self._worker.run_eval_current)
        self.request_perturbation.connect(self._worker.run_perturbation)
        self.request_fine_scan.connect(self._worker.run_fine_scan)
        self._thread.start()
        self._update_pert_trial_count()

    # ------------------------------------------------------------------ enter

    def _calibrate_step(self):
        for page in getattr(self.window(), "pages", []):
            if hasattr(page, "solution"):
                return page
        return None

    def _method(self) -> str:
        for page in getattr(self.window(), "pages", []):
            combo = getattr(page, "method_combo", None)
            if combo is not None:
                return combo.currentData()
        return DetectParams.method

    def _load_extrinsic(self):
        """Open a saved extrinsic and verify it against a recording.

        The loaded transform replaces whatever step 6 produced, and the camera and
        topics come from the file too -- an extrinsic and the intrinsics it was
        solved with are a pair, and projecting with a different focal length gives
        a wrong answer that still looks reasonable.

        Hole markers cannot be drawn: those need a scene's filter box, which the
        file does not carry. The overlay is what does the work here anyway.
        """
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "extrinsic 불러오기", str(RESULT_DIR), "extrinsic (*.yaml *.yml);;모든 파일 (*)"
        )
        if not path:
            return
        try:
            data = verify.from_yaml(Path(path).read_text())
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.critical(self, "불러오기 실패", f"{type(exc).__name__}: {exc}")
            return
        if "R" not in data:
            QtWidgets.QMessageBox.warning(
                self, "불러오기 실패",
                "이 파일에서 변환 행렬을 찾지 못했습니다.\n"
                "T_cam_lidar 또는 rotation_matrix + translation_m 이 있어야 합니다.",
            )
            return

        self._sol = Solution(
            ok=True, R=data["R"], t=data["t"],
            rmse=data.get("rmse_mm", 0.0) / 1000.0,
            scene_ids=[], n_pairs=0,
        )
        self._flipped = False
        self._loaded_from = Path(path).name
        self._loaded_meta = data

        p = self.project
        if data.get("camera"):
            for key, value in data["camera"].items():
                setattr(p.camera, key, value)
        # Follow the file's topics when the project has none, or when they differ
        # -- verifying against topics other than the ones it was solved on is a
        # different measurement.
        for key in ("lidar_topic", "camera_topic"):
            if data.get(key):
                setattr(p, key, data[key])

        # Offer the bag the file names, if the project does not already hold it.
        missing = []
        for name in data.get("bags") or []:
            if not any(Path(b).name == name for b in p.bag_paths):
                missing.append(name)
        if missing:
            QtWidgets.QMessageBox.information(
                self, "bag 을 지정하세요",
                "이 extrinsic 은 아래 bag 으로 계산됐습니다. 1단계에서 추가하면 그 구간으로 검증할 수 있습니다.\n\n"
                + "\n".join(missing),
            )

        self._fill_bags()
        self._fill_scenes()
        self._open_current_bag()
        self.window().statusBar().showMessage(
            f"extrinsic 불러옴: {Path(path).name}"
            + (f"  ({data['vehicle']})" if data.get("vehicle") else ""), 10000
        )

    def on_enter(self):
        # A loaded extrinsic stays until step 6 is run again -- it was opened on
        # purpose, and recomputing behind the user's back would discard it.
        if self._loaded_from is not None:
            if self._times:
                self._fill_bags()
                self._fill_scenes()
            return

        step = self._calibrate_step()
        sol = getattr(step, "solution", None) if step else None
        if sol is None or not sol.ok:
            self._sol = None
            self.canvas.setText(
                "6단계 캘리브레이션을 완료하거나, 오른쪽에서 저장된 extrinsic 을 불러오세요."
            )
            self.verdict.setText("—")
            self.numbers.setText("—")
            self.export_text.clear()
            return

        if sol is not self._sol:
            self._sol = sol
            self._flipped = False
        self._fill_bags()
        self._fill_scenes()
        self._open_current_bag()

    def _current_bag(self) -> str:
        i = self.bag_combo.currentIndex()
        paths = self.project.bag_paths
        return paths[i] if 0 <= i < len(paths) else (paths[0] if paths else "")

    def _fill_bags(self):
        """Every bag in the project, whether or not it holds a captured scene.

        A recording added only to check the result has no scenes at all, and that
        is the most useful kind to look at -- it had no say in the answer.
        """
        from pathlib import Path

        names = [Path(b).name for b in self.project.bag_paths]
        if [self.bag_combo.itemText(i) for i in range(self.bag_combo.count())] == names:
            return
        keep = self.bag_combo.currentIndex()
        self.bag_combo.blockSignals(True)
        self.bag_combo.clear()
        self.bag_combo.addItems(names)
        self.bag_combo.setCurrentIndex(max(0, min(keep, len(names) - 1)))
        self.bag_combo.blockSignals(False)

    def _fill_scenes(self):
        """Jump targets: the scenes captured from the bag currently shown."""
        bag = self._current_bag()
        current = self.scene_combo.currentData()
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItem("— 직접 이동 —", None)
        # Relative to the start of the recording. Raw t_ns is a Unix timestamp,
        # which prints as 1785224859.29 and matches nothing else on screen.
        base = self._times[0] if self._times else 0
        for sc in self.project.scenes:
            if self.project.bag_for(sc) == bag:
                offset = (sc.t_ns - base) / 1e9 if base else 0.0
                self.scene_combo.addItem(f"{sc.id}  ({offset:.2f}s)", sc.id)
        idx = self.scene_combo.findData(current)
        self.scene_combo.setCurrentIndex(max(idx, 0))
        self.scene_combo.blockSignals(False)

    def _switch_bag(self):
        self._fill_scenes()
        self._open_current_bag()

    def _open_current_bag(self):
        p = self.project
        bag = self._current_bag()
        if not bag or not p.lidar_topic or not p.camera_topic:
            self.canvas.setText("1단계에서 bag 과 토픽을 지정하세요.")
            return
        key = (bag, p.lidar_topic)
        if key == self._loaded_for and self._times:
            self._request()
            return
        self._loaded_for = key
        self._times = []
        self.slider.setEnabled(False)
        self._gen += 1
        self.canvas.setText("bag 을 읽는 중…")
        self.request_open.emit(self._gen, bag, p.lidar_topic)

    def _on_opened(self, gen: int, times):
        if gen != self._gen:
            return
        self._times = list(times)
        if not self._times:
            self.canvas.setText("이 bag 에 LiDAR 메시지가 없습니다.")
            return
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self._times) - 1)
        # Start where a scene was captured when there is one, since that is the
        # moment with hole centres to compare against.
        first = next(
            (i for sc in self.project.scenes if self.project.bag_for(sc) == self._current_bag()
             for i in [min(range(len(self._times)), key=lambda k: abs(self._times[k] - sc.t_ns))]),
            len(self._times) // 2,
        )
        self.slider.setValue(first)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        self._fill_scenes()  # labels need self._times to show relative seconds
        self._request()

    def _on_slider(self):
        self._update_time_label()
        self._debounce.start()

    def _update_time_label(self):
        if not self._times:
            self.time_label.setText("—")
            return
        i = self.slider.value()
        t = (self._times[i] - self._times[0]) / 1e9
        tag = f"  {self._scene.id}" if self._scene is not None else ""
        self.time_label.setText(f"{t:7.2f} s  [{i + 1}/{len(self._times)}]{tag}")

    def _scene_at(self, t_ns: int):
        """The captured scene at this moment, if the slider is sitting on one.

        Detection needs a scene's filter box, so the hole markers and the numbers
        only appear here. Within half a sweep counts as the same moment.
        """
        bag = self._current_bag()
        tol = 60_000_000  # 60 ms, comfortably inside one sweep at 19 Hz
        for sc in self.project.scenes:
            if self.project.bag_for(sc) == bag and abs(sc.t_ns - t_ns) <= tol:
                return sc
        return None

    def _request(self):
        p = self.project
        if not self._times:
            return
        t_ns = self._times[self.slider.value()]
        self._scene = self._scene_at(t_ns)
        frames = self._scene.frames if self._scene is not None else 1
        self._gen += 1
        self._update_time_label()
        self.canvas.setText("불러오는 중…")
        # One-shot: only the request a Worst-Frame jump itself triggers should
        # use it, so it is cleared immediately -- any further scrubbing (even
        # before this request's result comes back) must fall through to
        # ordinary first_after pairing again.
        pinned = self._pinned_camera_t_ns
        self._pinned_camera_t_ns = None
        self.request_load.emit(
            self._gen, self._current_bag(), p.lidar_topic, p.camera_topic,
            t_ns, frames, self._scene, p.camera, p.target, self._method(), pinned,
        )

    def _goto_scene(self):
        sid = self.scene_combo.currentData()
        if sid is None or not self._times:
            return
        scene = next((sc for sc in self.project.scenes if sc.id == sid), None)
        if scene is None:
            return
        idx = min(range(len(self._times)), key=lambda i: abs(self._times[i] - scene.t_ns))
        if idx == self.slider.value():
            self._request()
        else:
            self.slider.setValue(idx)

    def _on_scanning(self, gen: int, done: int, total: int):
        if gen != self._gen:
            return
        pct = f" ({done / total * 100:.0f}%)" if total else ""
        self.canvas.setText(f"bag 인덱스 읽는 중…  {done:,} / {total:,}{pct}")

    def _on_failed(self, gen: int, msg: str):
        if gen == self._gen:
            self.canvas.setText(msg)

    def _on_ready(self, gen, image, cloud, intensity, lidar_c, cam_c):
        if gen != self._gen:
            return
        self._image, self._cloud, self._intensity = image, cloud, intensity
        self._lidar_c, self._cam_c = lidar_c, cam_c
        # Point indices mean nothing once a different frame is projected.
        self._picked = None
        self.pick_label.setText(
            "<span style='color:palette(mid)'>점을 클릭하면 원래 3D 좌표를 봅니다.</span>"
        )
        self._update_time_label()
        self._redraw()
        self._update_numbers()
        self._update_export()

    # ------------------------------------------------------------------- draw

    def _effective(self):
        """The solution as currently shown -- flipped or not.

        Which of the two half-turn choices `solve()` settled on is not recorded,
        and assuming one produces a "flip" that changes nothing half the time.
        So both are built here and the one furthest from the current rotation is
        taken as the alternative. Distance between rotation matrices is a fine
        test for this: the two differ by 180 degrees, so there is no near miss.
        """
        if self._sol is None:
            return None
        if not self._flipped or self._lidar_c is None or self._cam_c is None:
            return self._sol

        L, C = sort_centers(self._lidar_c), sort_centers(self._cam_c)
        options = [solve_rigid(np.roll(L, sh, axis=0), C) for sh in (0, 2)]
        far = max(options, key=lambda Rt: np.linalg.norm(Rt[0] - self._sol.R))
        R, t = far
        return Solution(
            ok=True, R=R, t=t, rmse=self._sol.rmse,
            scene_ids=list(self._sol.scene_ids), n_pairs=self._sol.n_pairs,
        )

    def _redraw(self):
        sol = self._effective()
        if self._image is None or self._cloud is None or sol is None:
            return
        h, w = self._image.shape[:2]
        pr = verify.project_cloud(
            self._cloud, sol, self.project.camera, w, h,
            intensity=self._intensity,
            min_range=self.near_spin.value(), max_range=self.far_spin.value(),
        )
        self._last_projection = pr
        img = verify.make_overlay(
            self._image, pr,
            colour_by=self.colour_combo.currentData(),
            point_size=self.size_spin.value(),
            dim=self.dim_slider.value() / 100.0,
        )
        if self._picked is not None and self._picked < pr.n_visible:
            import cv2 as _cv2

            pu, pv = pr.uv[self._picked]
            _cv2.circle(img, (int(pu), int(pv)), 14, (255, 255, 255), 2)
            _cv2.circle(img, (int(pu), int(pv)), 15, (0, 0, 0), 1)
        if self.markers_check.isChecked() and self._lidar_c is not None and self._cam_c is not None:
            img = verify.draw_centres(img, sol, self.project.camera, self._lidar_c, self._cam_c)

        rgb = np.ascontiguousarray(img[:, :, ::-1])
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self._apply_zoom()
        self._visible = pr.n_visible

    def _img_origin(self) -> QtCore.QPointF:
        """Where the image's top-left sits in the label, which centres its pixmap."""
        pm = self.canvas.pixmap()
        if pm is None or pm.isNull():
            return QtCore.QPointF(0.0, 0.0)
        return QtCore.QPointF(
            max((self.canvas.width() - pm.width()) / 2.0, 0.0),
            max((self.canvas.height() - pm.height()) / 2.0, 0.0),
        )

    def _pick_point(self, pos: QtCore.QPointF):
        """Report the 3D point behind the dot nearest the click."""
        def nothing(msg: str):
            # Saying why beats leaving the previous point's numbers on screen,
            # which reads as if the click had found that point again.
            self._picked = None
            self.pick_label.setText(f"<span style='color:palette(mid)'>{msg}</span>")

        pr = self._last_projection
        if pr is None or pr.n_visible == 0 or pr.axis is None:
            nothing("아직 그려진 점이 없습니다.")
            return
        scale = self._effective_scale()
        if scale <= 0:
            nothing("화면 배율을 읽을 수 없습니다.")
            return
        origin = self._img_origin()
        x = (pos.x() - origin.x()) / scale
        y = (pos.y() - origin.y()) / scale

        d = np.hypot(pr.uv[:, 0] - x, pr.uv[:, 1] - y)
        i = int(np.argmin(d))
        # Generous in image pixels but tighter the further in you are zoomed, so
        # the pick follows what is actually under the cursor on screen.
        if d[i] > max(12.0 / max(scale, 1e-6), 3.0):
            nothing(f"그 자리에 점이 없습니다 (가장 가까운 점이 {d[i]:.0f} px 떨어져 있습니다).")
            self._redraw()
            return

        self._picked = i
        sol = self._effective()
        p3 = pr.axis[i]
        cam_pt = sol.transform(p3.reshape(1, 3))[0]
        rng = float(np.linalg.norm(cam_pt))
        off = float(np.hypot(cam_pt[0], cam_pt[1]) / max(cam_pt[2], 1e-9))
        limit = verify.radial_limit(self.project.camera)
        inten = "—" if pr.intensity is None else f"{pr.intensity[i]:.1f}"
        rows = [
            ("화면", f"({pr.uv[i, 0]:.1f}, {pr.uv[i, 1]:.1f}) px"),
            ("LiDAR 좌표", f"앞 {p3[0]:+.3f}  좌 {p3[1]:+.3f}  위 {p3[2]:+.3f} m"),
            ("카메라 거리", f"{rng:.2f} m"),
            ("입사각", f"{np.degrees(np.arctan(off)):.2f}°"
                      f"  <span style='color:palette(mid)'>(한계 {np.degrees(np.arctan(limit)):.1f}°)</span>"),
            ("intensity", inten),
        ]
        self.pick_label.setText(
            "<table cellspacing='3'>"
            + "".join(
                f"<tr><td style='color:palette(mid)'>{k}</td><td>{v}</td></tr>"
                for k, v in rows
            )
            + "</table>"
        )
        self._redraw()

    def _apply_zoom(self):
        if not hasattr(self, "_pix"):
            return
        if self._fit:
            area = self.scroll.viewport().size()
            pm = self._pix.scaled(area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        else:
            pm = self._pix.scaled(
                self._pix.size() * self._zoom, QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        self.canvas.setPixmap(pm)
        self.canvas.setMinimumSize(pm.size() if not self._fit else QtCore.QSize(0, 0))

    def _sync_range(self, spin, value: float):
        """Slider moved: push the value into its spin box, which redraws."""
        if abs(spin.value() - value) < 1e-9:
            return
        spin.setValue(value)  # triggers _on_range

    def _push_range_sliders(self):
        """Mirror the spin boxes back onto the sliders without a signal loop."""
        for spin, slider in ((self.near_spin, self.near_slider),
                             (self.far_spin, self.far_slider)):
            want = int(round(spin.value() * 100))
            if slider.value() != want:
                slider.blockSignals(True)
                slider.setValue(want)
                slider.blockSignals(False)

    def _on_range(self):
        """Keep the near cut below the far one before redrawing.

        Crossed values would silently show nothing, which looks like a detection
        failure rather than a filter set inside out.
        """
        near, far = self.near_spin.value(), self.far_spin.value()
        if far > 0 and near > 0 and near >= far:
            sender = self.sender()
            blocked = self.far_spin if sender is self.near_spin else self.near_spin
            blocked.blockSignals(True)
            if sender is self.near_spin:
                self.far_spin.setValue(near + self.far_spin.singleStep())
            else:
                self.near_spin.setValue(max(0.0, far - self.near_spin.singleStep()))
            blocked.blockSignals(False)
        self._push_range_sliders()
        self._redraw()

    def _on_pan(self, delta: QtCore.QPoint):
        """Drag the view: move the scroll offset opposite to the pointer."""
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        h.setValue(h.value() - delta.x())
        v.setValue(v.value() - delta.y())

    def _step_frames(self, notches: int):
        """Shift+wheel over the image: walk the timeline without leaving it."""
        if not self._times:
            return
        self.slider.setValue(self.slider.value() + notches * self.slider.singleStep())

    def _step_size(self, notches: int):
        """Ctrl+wheel over the image: thicken the cloud until the shape reads."""
        self.size_spin.setValue(self.size_spin.value() + notches * self.size_spin.singleStep())

    def _on_wheel(self, factor: float, pos: QtCore.QPointF):
        """Zoom about the pointer, keeping what is under it in place.

        The scroll offset has to move with the scale: a point at `pos` sits at
        (offset + pos) in the scaled image, and after scaling by `factor` that
        lands at factor * (offset + pos). Subtracting `pos` again gives the offset
        that puts it back under the cursor.
        """
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        before = self._effective_scale()
        # Wheeling out of "fit" has to carry on from what is on screen. `_zoom` is
        # still 1.0 while fitted, so using it as-is would jump to full size on the
        # first notch.
        if self._fit and before > 0:
            self._zoom = before
        self._fit = False
        self._zoom = max(0.1, min(self._zoom * factor, 12.0))
        self._apply_zoom()
        after = self._effective_scale()
        if before > 0:
            ratio = after / before
            h.setValue(int(ratio * (h.value() + pos.x()) - pos.x()))
            v.setValue(int(ratio * (v.value() + pos.y()) - pos.y()))

    def _effective_scale(self) -> float:
        """Pixels drawn per source pixel, whichever sizing mode is active."""
        if not hasattr(self, "_pix") or self._pix.isNull():
            return 0.0
        shown = self.canvas.pixmap()
        if shown is None or shown.isNull():
            return 0.0
        return shown.width() / self._pix.width()

    def _set_zoom(self, z: float):
        self._fit = False
        self._zoom = max(0.1, min(z, 12.0))
        self._apply_zoom()

    def _fit_view(self):
        self._fit = True
        self._apply_zoom()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit:
            self._apply_zoom()

    # ---------------------------------------------------------------- numbers

    def _flip(self):
        if self._lidar_c is None or self._cam_c is None:
            self.window().statusBar().showMessage(
                "뒤집기는 구멍 중심이 있어야 계산됩니다 — 캡처한 scene 시점으로 이동하세요.", 6000
            )
            return
        self._flipped = not self._flipped
        self._redraw()
        self._update_numbers()
        self._update_export()

    def _update_numbers(self):
        sol = self._effective()
        if sol is None:
            return
        if self._lidar_c is None or self._cam_c is None:
            if self._loaded_from is not None:
                meta = self._loaded_meta
                bits = [f"<b>{self._loaded_from}</b> 의 extrinsic 으로 투영하고 있습니다."]
                detail = " · ".join(
                    x for x in (
                        meta.get("vehicle"),
                        f"계산 {meta['calibrated_on']}" if meta.get("calibrated_on") else "",
                        f"검출 {meta['detector']}" if meta.get("detector") else "",
                        f"RMSE {meta['rmse_mm']:.2f}mm" if meta.get("rmse_mm") else "",
                    ) if x
                )
                if detail:
                    bits.append(f"<span style='color:palette(mid)'>{detail}</span>")
                bits.append(
                    "구멍 중심 비교는 이 scene 의 박스가 필요해 불가능합니다. "
                    "<b>클라우드가 이미지 위에 맞게 얹히는지</b>로 판단하세요 — "
                    "벽·바닥·기둥은 extrinsic 을 구할 때 쓰이지 않았으니 독립된 근거입니다."
                )
                self.verdict.setText("<br>".join(bits))
                self.numbers.setText(
                    "<table cellspacing='3'>"
                    f"<tr><td style='color:palette(mid)'>출처</td><td><b>{self._loaded_from}</b></td></tr>"
                    f"<tr><td style='color:palette(mid)'>화면에 그린 점</td>"
                    f"<td><b>{getattr(self, '_visible', 0):,}</b></td></tr>"
                    "</table>"
                )
                return
            if self._scene is None:
                self.verdict.setText(
                    "캡처한 scene 이 아닌 시점입니다. <b>구멍 중심 비교는 scene 에서만</b> "
                    "가능하니, 여기서는 클라우드가 이미지 위에 제대로 얹히는지 보세요.<br>"
                    "보드 구멍뿐 아니라 벽·바닥·기둥의 선이 맞는지가 오히려 더 확실한 근거입니다."
                )
            else:
                self.verdict.setText(
                    f"{self._scene.id} 에서 검출에 실패했습니다. 5단계에서 이 scene 의 "
                    "박스와 검출 방식을 확인하세요."
                )
            self.numbers.setText(
                "<span style='color:palette(mid)'>이 시점에는 비교할 구멍 중심이 없습니다.</span>"
            )
            return

        ag = verify.centre_agreement(self._lidar_c, self._cam_c, sol)
        ht = verify.half_turn_check(self._lidar_c, self._cam_c, sol)
        n_scenes = len(self._sol.scene_ids)

        if ht["separable"]:
            verdict = (
                "<span style='color:#2e7d32'><b>잔차로 구분됩니다.</b></span> "
                f"현재 {ht['current']:.2f} mm, 뒤집으면 {ht['half_turn']:.2f} mm."
            )
        elif n_scenes >= 2:
            verdict = (
                f"이 scene 만으로는 구분되지 않지만 scene 이 {n_scenes}개라 "
                "6단계에서 함께 결정했습니다. 그래도 아래 겹침을 한 번 확인하세요."
            )
        else:
            verdict = (
                "<span style='color:#b26a00'><b>계산으로는 구분할 수 없습니다.</b></span> "
                f"현재 {ht['current']:.2f} mm, 뒤집어도 {ht['half_turn']:.2f} mm 로 같습니다.<br>"
                "화면에서 <b>십자와 원이 겹치는지</b> 보고 판단하세요. "
                "대각선으로 어긋나 보이면 뒤집기를 누르세요."
            )
        if self._flipped:
            verdict += "<br><span style='color:#b26a00'>현재 <b>뒤집은</b> 해를 보고 있습니다.</span>"
        self.verdict.setText(verdict)

        rows = [
            ("원 중심 일치", f"평균 {ag['mean_mm']:.1f} mm · 최대 {ag['max_mm']:.1f} mm"),
            ("구멍별", " / ".join(f"{v:.1f}" for v in ag["per_hole_mm"]) + " mm"),
            ("일대일 대응", "예" if ag["bijective"] else "아니오 — 두 중심이 겹칩니다"),
            ("화면에 그린 점", f"{getattr(self, '_visible', 0):,}"),
            ("scene 수", f"{n_scenes}"),
            ("6단계 RMSE", f"{self._sol.rmse * 1000:.2f} mm"),
        ]
        self.numbers.setText(
            "<table cellspacing='3'>"
            + "".join(
                f"<tr><td style='color:palette(mid)'>{k}</td><td><b>{v}</b></td></tr>"
                for k, v in rows
            )
            + "</table>"
        )

    # ------------------------------------------------------ target-independent
    #
    # Step 6's numbers above answer "how well did this fit the four scenes it
    # was solved from". Everything below answers a different question -- "does
    # the resulting T_cam_lidar still hold on an ordinary frame it never saw"
    # -- using LiDAR depth-edges vs. camera Canny edges instead of the target's
    # holes, so it works on any frame, captured scene or not.

    @staticmethod
    def _rows_table(rows) -> str:
        return (
            "<table cellspacing='3'>"
            + "".join(
                f"<tr><td style='color:palette(mid)'>{k}</td><td><b>{v}</b></td></tr>"
                for k, v in rows
            )
            + "</table>"
        )

    @staticmethod
    def _bin_text(stats) -> str:
        if stats.n_points == 0:
            return "—"
        return f"{stats.mean_px:.2f} px  (n={stats.n_points:,})"

    def _eval_current_frame(self):
        """Score the frame already on screen.

        Projects fresh with the Evaluation Range above -- not `self._last_projection`,
        which was built from the *display* range sliders. Reusing that would
        mean nudging the display's near/far controls silently moves this
        metric too, which is exactly the hidden coupling that makes two
        calibrations "checked" at different display ranges incomparable.

        Measured at up to ~1s on a 4K frame with a realistic point count, so
        this runs on the worker thread rather than blocking the GUI here.
        """
        sol = self._effective()
        if self._image is None or self._cloud is None or sol is None or not sol.ok:
            self.window().statusBar().showMessage("먼저 화면에 프레임을 불러오세요.", 4000)
            return
        self._eval_gen += 1
        # Pinned to the frame being sent, not read back from self._image later:
        # if the user scrubs to a different (possibly different-resolution)
        # frame before this result comes back, self._image will have already
        # moved on by the time _display_edge_alignment runs.
        self._eval_frame_shape = self._image.shape[:2]
        self._eval_result, self._eval_spatial = None, None
        self._eval_T = (sol.R.copy(), sol.t.copy())
        self.eval_run_btn.setEnabled(False)
        self.eval_edge_label.setText("<span style='color:palette(mid)'>평가 중…</span>")
        self.request_eval_current.emit(
            self._eval_gen, self._image, self._cloud, sol, self.project.camera,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
        )

    def _on_eval_ready(self, gen: int, result: "ea.EdgeAlignmentResult"):
        if gen != self._eval_gen:
            return
        self.eval_run_btn.setEnabled(True)
        self._display_edge_alignment(result)

    def _on_eval_failed(self, gen: int, msg: str):
        if gen != self._eval_gen:
            return
        self.eval_run_btn.setEnabled(True)
        self.eval_edge_label.setText(f"<span style='color:#d9534f'>{msg}</span>")
        self.eval_spatial_label.setText("—")
        self.eval_depth_label.setText("—")

    def _display_edge_alignment(self, result: "ea.EdgeAlignmentResult"):
        self._eval_result = result
        if not result.ok:
            self.eval_edge_label.setText(f"<span style='color:palette(mid)'>{result.reason}</span>")
            self.eval_spatial_label.setText("—")
            self.eval_depth_label.setText("—")
            return

        self.eval_edge_label.setText(self._rows_table([
            ("Mean", f"{result.mean_px:.2f} px"),
            ("Median", f"{result.median_px:.2f} px"),
            ("P95", f"{result.p95_px:.2f} px"),
            ("Max", f"{result.max_px:.2f} px"),
            ("Match Rate", f"{result.match_rate * 100:.1f} %"),
            ("Edge Points", f"{result.n_edge_points:,}"),
            ("Matched / Unmatched", f"{result.n_matched:,} / {result.n_unmatched:,}"),
        ]))

        h, w = self._eval_frame_shape
        spatial = sa.analyze_spatial(result, w, h)
        self._eval_spatial = spatial
        if spatial is None:
            self.eval_spatial_label.setText("—")
            self.eval_depth_label.setText("—")
            return

        self.eval_spatial_label.setText(
            "수평<br>"
            + self._rows_table([(label, self._bin_text(spatial.horizontal[label])) for label in sa.HORIZONTAL_REGIONS])
            + "수직<br>"
            + self._rows_table([(label, self._bin_text(spatial.vertical[label])) for label in sa.VERTICAL_REGIONS])
        )
        self.eval_depth_label.setText(
            self._rows_table([(label, self._bin_text(spatial.depth_bins[label])) for label in sa.DEPTH_BIN_LABELS])
        )

    def _run_multiframe(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        if not self._times:
            self.window().statusBar().showMessage("bag 이 아직 로드되지 않았습니다.", 4000)
            return

        n_samples = min(self.mf_count_spin.value(), len(self._times))
        self._mf_gen += 1
        self.mf_run_btn.setEnabled(False)
        self.mf_cancel_btn.setVisible(True)
        self.mf_progress_label.setText(f"평가 중 0 / {n_samples}")
        self.mf_summary_label.setText("—")
        self.mf_worst_table.setRowCount(0)
        self._mf_result = None
        self._mf_T = (sol.R.copy(), sol.t.copy())
        self._mf_sync_limit_ms = float(self.max_sync_spin.value())

        p = self.project
        self.request_multiframe.emit(
            self._mf_gen, self._current_bag(), p.lidar_topic, p.camera_topic,
            p.camera, sol, list(self._times), n_samples,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
            float(self.max_sync_spin.value()),
        )

    def _cancel_multiframe(self):
        self._worker.cancel_multiframe()
        self.mf_progress_label.setText(self.mf_progress_label.text() + "  (취소 중…)")

    def _on_mf_progress(self, gen: int, done: int, total: int):
        if gen != self._mf_gen:
            return
        self.mf_progress_label.setText(f"평가 중 {done} / {total}")

    def _on_mf_ready(self, gen: int, result: "mc.MultiFrameConsistencyResult"):
        if gen != self._mf_gen:
            return
        self.mf_run_btn.setEnabled(True)
        self.mf_cancel_btn.setVisible(False)
        self._mf_result = result
        self._display_multiframe(result)

    def _on_mf_failed(self, gen: int, msg: str):
        if gen != self._mf_gen:
            return
        self.mf_run_btn.setEnabled(True)
        self.mf_cancel_btn.setVisible(False)
        self.mf_progress_label.setText("")
        self.mf_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _display_multiframe(self, result: "mc.MultiFrameConsistencyResult"):
        self.mf_progress_label.setText(f"완료 — {result.n_total} frame")
        if result.reason:
            # Still worth showing what was collected -- e.g. a high
            # Sync Rejected count is often *why* too few frames were valid.
            self.mf_summary_label.setText(
                f"<span style='color:palette(mid)'>{result.reason}</span>"
                + self._rows_table([
                    ("Frames", f"{result.n_total}"),
                    ("Failed", f"{result.n_failed}"),
                    ("Sync Rejected", f"{result.n_sync_rejected}"),
                    ("Median Sync Offset (abs)", self._ms_text(result.sync_offset_median_ms)),
                    ("Max Sync Offset (abs)", self._ms_text(result.sync_offset_max_ms)),
                    ("Signed Sync Median", self._signed_ms_text(result.sync_offset_signed_median_ms)),
                ])
            )
            self.mf_worst_table.setRowCount(0)
            return

        self.mf_summary_label.setText(self._rows_table([
            ("Frames", f"{result.n_total}"),
            ("Valid", f"{result.n_valid}"),
            ("Failed", f"{result.n_failed}"),
            ("Outlier", f"{result.n_outlier}"),
            ("Sync Rejected", f"{result.n_sync_rejected}"),
            ("Mean", f"{result.mean_px:.2f} px"),
            ("Median", f"{result.median_px:.2f} px"),
            ("STD", f"{result.std_px:.2f} px"),
            ("P95", f"{result.p95_px:.2f} px"),
            ("Max", f"{result.max_px:.2f} px"),
            ("MAD", f"{result.mad_px:.2f} px"),
            ("IQR", f"{result.iqr_px:.2f} px"),
            ("Valid Ratio", f"{result.valid_ratio * 100:.1f} %"),
            ("Failure Ratio", f"{result.failure_ratio * 100:.1f} %"),
            ("Outlier Ratio", f"{result.outlier_ratio * 100:.1f} %"),
            ("Sync Rejected Ratio", f"{result.sync_rejected_ratio * 100:.1f} %"),
            ("Median Sync Offset (abs)", self._ms_text(result.sync_offset_median_ms)),
            ("P95 Sync Offset (abs)", self._ms_text(result.sync_offset_p95_ms)),
            ("Max Sync Offset (abs)", self._ms_text(result.sync_offset_max_ms)),
            ("Signed Sync Median", self._signed_ms_text(result.sync_offset_signed_median_ms)),
        ]))

        self.mf_worst_table.setRowCount(len(result.worst_frames))
        for r, f in enumerate(result.worst_frames):
            self.mf_worst_table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{f.offset_s:.2f}s"))
            self.mf_worst_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{f.mean_px:.2f} px"))
            self.mf_worst_table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{f.match_rate * 100:.0f}%"))
            self.mf_worst_table.setItem(r, 3, QtWidgets.QTableWidgetItem(self._ms_text(f.abs_sync_offset_ms)))
            btn = QtWidgets.QPushButton("이동")
            # Pinning the exact camera timestamp this frame was evaluated with
            # is what guarantees the jump shows the same Camera/LiDAR pair the
            # Multi-frame run scored -- see pick_camera_for_frame's docstring.
            btn.clicked.connect(
                lambda _checked=False, idx=f.timeline_index, cam_t=f.camera_timestamp_ns:
                    self._goto_timeline_index(idx, cam_t)
            )
            self.mf_worst_table.setCellWidget(r, 4, btn)
        self.mf_worst_table.resizeColumnsToContents()

    @staticmethod
    def _ms_text(value_ms: float) -> str:
        return f"{value_ms:.1f} ms" if np.isfinite(value_ms) else "—"

    @staticmethod
    def _signed_ms_text(value_ms: float) -> str:
        return f"{value_ms:+.1f} ms" if np.isfinite(value_ms) else "—"

    def _goto_timeline_index(self, idx: int, camera_t_ns: "int | None" = None):
        """Where a Worst-Frame row's "이동" button lands: reuse the same
        slider -> debounce -> _request path a manual drag already takes.

        `camera_t_ns`, when given, pins `_request()`'s next load to the exact
        camera message Multi-frame evaluation scored this frame with (see
        `pick_camera_for_frame`) -- without it, `_request()` would pair this
        LiDAR moment with `first_after` instead, which is not necessarily the
        same image Multi-frame used (it pairs by nearest timestamp).
        """
        if not self._times or not (0 <= idx < len(self._times)):
            return
        self.side_tabs.setCurrentIndex(0)
        self._pinned_camera_t_ns = camera_t_ns
        if idx == self.slider.value():
            self._request()
        else:
            self.slider.setValue(idx)

    # ----------------------------------------------------- perturbation sensitivity
    #
    # Diagnostic only: re-evaluates the existing Edge Alignment metric at
    # small, temporary copies of the current T. Never writes back to
    # self._sol, the project, or any file -- see
    # gui.core.evaluation.perturbation's module docstring for the guarantee
    # this whole section rests on.

    def _run_perturbation(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return

        mode = "current_frame" if self.pert_mode_current_radio.isChecked() else "multi_frame"
        if mode == "current_frame":
            if self._image is None or self._cloud is None:
                self.window().statusBar().showMessage("먼저 화면에 프레임을 불러오세요.", 4000)
                return
        elif not self._times:
            self.window().statusBar().showMessage("bag 이 아직 로드되지 않았습니다.", 4000)
            return

        self._pert_gen += 1
        self.pert_run_btn.setEnabled(False)
        self.pert_cancel_btn.setVisible(True)
        self.pert_progress_label.setText("분석 준비 중…")
        self.pert_table_label.setText("—")
        self.pert_frame_table_label.setText("—")
        self.pert_summary_label.setText("—")
        self.pert_spatial_label.setText("—")
        self._pert_result = None
        self._pert_T = (sol.R.copy(), sol.t.copy())
        self._pert_sync_limit_ms = float(self.max_sync_spin.value())

        rotation_deltas, translation_deltas = self._pert_search_deltas()
        p = self.project
        self.request_perturbation.emit(
            self._pert_gen, mode, self._current_bag(), p.lidar_topic, p.camera_topic,
            p.camera, sol.R, sol.t, list(self._times), self.pert_frames_spin.value(),
            rotation_deltas, translation_deltas,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
            float(self.max_sync_spin.value()), self._image, self._cloud,
        )

    def _pert_search_deltas(self) -> "tuple[tuple, tuple]":
        if self.pert_search_quick_radio.isChecked():
            return pert.QUICK_ROTATION_DELTAS_DEG, pert.QUICK_TRANSLATION_DELTAS_MM
        return pert.DEFAULT_ROTATION_DELTAS_DEG, pert.DEFAULT_TRANSLATION_DELTAS_MM

    def _update_pert_trial_count(self):
        """Trial/evaluation count shown before running, per the earlier
        request to see the cost up front (no time estimate -- just the count,
        since past-run timing isn't tracked)."""
        rotation_deltas, translation_deltas = self._pert_search_deltas()
        n_trials = 3 * (len(rotation_deltas) - 1) + 3 * (len(translation_deltas) - 1) + 1
        n_frames = 1 if self.pert_mode_current_radio.isChecked() else self.pert_frames_spin.value()
        self.pert_trial_count_label.setText(
            f"Frames: {n_frames}   Trials: {n_trials}   Total Evaluations: {n_frames * n_trials}"
        )

    def _cancel_perturbation(self):
        self._worker.cancel_perturbation()
        self.pert_progress_label.setText(self.pert_progress_label.text() + "  (취소 중…)")

    def _on_pert_progress(self, gen: int, done: int, total: int):
        if gen != self._pert_gen:
            return
        self.pert_progress_label.setText(f"평가 중 {done} / {total}")

    def _on_pert_ready(self, gen: int, result: "pert.PerturbationResult"):
        if gen != self._pert_gen:
            return
        self.pert_run_btn.setEnabled(True)
        self.pert_cancel_btn.setVisible(False)
        self._pert_result = result
        self._display_perturbation(result)

    def _on_pert_failed(self, gen: int, msg: str):
        if gen != self._pert_gen:
            return
        self.pert_run_btn.setEnabled(True)
        self.pert_cancel_btn.setVisible(False)
        self.pert_progress_label.setText("")
        self.pert_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _display_perturbation(self, result: "pert.PerturbationResult"):
        self.pert_progress_label.setText(
            f"완료 — {result.n_frames_used} frame 사용"
            + (f", sync 제외 {result.n_sync_rejected}" if result.n_sync_rejected else "")
            + (" (취소됨 — 일부 축만 계산됨)" if result.cancelled else "")
        )
        if result.reason or not result.axes:
            self.pert_table_label.setText(
                f"<span style='color:palette(mid)'>{result.reason or '결과가 없습니다.'}</span>"
            )
            self.pert_summary_label.setText("—")
            self.pert_spatial_label.setText("—")
            return
        self._display_perturbation_axis()

    def _display_perturbation_axis(self):
        """Re-render the table/summary/spatial panels for whichever axis is
        currently selected -- called after a run finishes and every time the
        axis combo changes, with no new computation either time (everything
        for every axis was already computed by `evaluate_perturbation_grid`)."""
        result = self._pert_result
        if result is None or not result.axes:
            return
        axis = result.axes.get(self.pert_axis_combo.currentData())
        if axis is None:
            self.pert_table_label.setText(
                "<span style='color:palette(mid)'>이 축은 취소로 인해 계산되지 않았습니다.</span>"
            )
            self.pert_frame_table_label.setText("—")
            self.pert_summary_label.setText("—")
            self.pert_spatial_label.setText("—")
            return
        self.pert_table_label.setText(self._perturbation_table_html(axis))
        self.pert_frame_table_label.setText(self._perturbation_frame_table_html(axis))
        self.pert_summary_label.setText(self._perturbation_summary_html(axis))
        self.pert_spatial_label.setText(self._perturbation_spatial_html(axis))

    @staticmethod
    def _perturbation_table_html(axis: "pert.AxisSensitivity") -> str:
        unit = "°" if axis.unit == "deg" else "mm"
        header = "".join(f"<th style='text-align:right;padding:0 6px'>{h}</th>" for h in ("Δ", "Median", "P95", "ΔP95", "Match"))
        rows = [f"<tr>{header}</tr>"]
        for p in axis.points:
            weight = "font-weight:bold;" if p.delta == 0.0 else ""
            delta_text = f"{p.delta:+.2f}{unit}" if p.delta != 0.0 else f"0{unit} (baseline)"
            if not p.ok:
                rows.append(
                    f"<tr style='{weight}'><td>{delta_text}</td>"
                    f"<td colspan='4' style='color:#d9534f'>{p.reason or '평가 실패'}</td></tr>"
                )
                continue
            dp95 = f"{p.delta_p95_px:+.2f}" if np.isfinite(p.delta_p95_px) else "—"
            rows.append(
                f"<tr style='{weight}'>"
                f"<td>{delta_text}</td>"
                f"<td align='right'>{p.median_px:.2f}</td>"
                f"<td align='right'>{p.p95_px:.2f}</td>"
                f"<td align='right'>{dp95}</td>"
                f"<td align='right'>{p.match_rate * 100:.0f}%</td>"
                f"</tr>"
            )
        html = "<table cellspacing='4'>" + "".join(rows) + "</table>"
        if any(p.coverage_warning for p in axis.points):
            html += "<div style='color:#b26a00'>Comparable frame coverage differs from baseline for some perturbations.</div>"
        return html

    @staticmethod
    def _perturbation_frame_table_html(axis: "pert.AxisSensitivity") -> str:
        """Frame-balanced view: every frame counts once regardless of how many
        edge points it contributed, and Δ is always paired same-frame-to-same-
        frame against baseline (see `_pair_frame_metrics`) -- this is what
        Multi-frame mode's Lowest-Tested/Local-Minimum judgment is based on,
        not the pooled table above."""
        unit = "°" if axis.unit == "deg" else "mm"
        header = "".join(
            f"<th style='text-align:right;padding:0 6px'>{h}</th>"
            for h in ("Δ", "Frame P95(med)", "ΔFrame-P95(med)", "Improved", "Worsened", "Unchanged", "Ratio", "Both valid")
        )
        rows = [f"<tr>{header}</tr>"]
        baseline_valid = axis.baseline.n_valid_frames
        for p in axis.points:
            weight = "font-weight:bold;" if p.delta == 0.0 else ""
            delta_text = f"{p.delta:+.2f}{unit}" if p.delta != 0.0 else f"0{unit} (baseline)"
            if p.delta == 0.0:
                rows.append(
                    f"<tr style='{weight}'><td>{delta_text}</td>"
                    f"<td align='right'>{p.median_frame_p95_px:.2f}</td>"
                    f"<td colspan='5' align='center' style='color:palette(mid)'>—</td>"
                    f"<td align='right'>{baseline_valid}/{baseline_valid}</td></tr>"
                )
                continue
            if not p.ok:
                rows.append(
                    f"<tr style='{weight}'><td>{delta_text}</td>"
                    f"<td colspan='7' style='color:#d9534f'>{p.reason or '평가 실패'}</td></tr>"
                )
                continue
            d_frame_p95 = f"{p.median_delta_frame_p95_px:+.2f}" if np.isfinite(p.median_delta_frame_p95_px) else "—"
            ratio = f"{p.improved_frame_ratio * 100:.0f}%" if np.isfinite(p.improved_frame_ratio) else "—"
            # Named by which side actually failed -- a frame baseline scored
            # but this candidate could not is "Candidate failed", never lumped
            # in with "Candidate-only valid" (see frame_coverage_breakdown).
            coverage = f"{p.n_comparable_frames}/{baseline_valid}"
            extra = [f"{label} {n}" for label, n in pert.frame_coverage_breakdown(p)[1:] if n]
            if extra:
                coverage += "<br>" + "<br>".join(extra)
            rows.append(
                f"<tr style='{weight}'>"
                f"<td>{delta_text}</td>"
                f"<td align='right'>{p.median_frame_p95_px:.2f}</td>"
                f"<td align='right'>{d_frame_p95}</td>"
                f"<td align='right'>{p.n_improved_frames}</td>"
                f"<td align='right'>{p.n_worsened_frames}</td>"
                f"<td align='right'>{p.n_unchanged_frames}</td>"
                f"<td align='right'>{ratio}</td>"
                f"<td align='right'>{coverage}</td>"
                f"</tr>"
            )
        legend = (
            "<div style='color:palette(mid)'>Both valid = baseline·candidate 모두 평가된 frame (paired 비교 대상) / "
            "baseline 유효 frame. Candidate failed = baseline만 유효, Candidate-only valid = candidate만 유효, "
            "Both failed = 둘 다 실패.</div>"
        )
        return "<table cellspacing='4'>" + "".join(rows) + "</table>" + legend

    @staticmethod
    def _perturbation_summary_html(axis: "pert.AxisSensitivity") -> str:
        unit = "°" if axis.unit == "deg" else "mm"
        name = axis.ranking_metric_name
        prefix = "Frame-balanced — " if axis.ranking_metric == "median_frame_p95" else ""
        lines = [f"<span style='color:palette(mid)'>Ranking basis: {prefix}{name}</span>"]
        lines.append(f"Baseline ({name}): {axis.ranking_baseline_px:.2f} px")
        if axis.lowest_point is axis.baseline:
            lines.append("Lowest tested: baseline (0)")
        else:
            lines.append(f"Lowest tested: {axis.lowest_point.delta:+.2f}{unit} → {axis.ranking_best_px:.2f} px")
            lines.append(f"Improvement ({name}): {axis.ranking_improvement_px:.2f} px")
        lines.append(f"Baseline local minimum: <b>{'Yes' if axis.is_local_minimum else 'No'}</b>")
        return "<br>".join(lines)

    @staticmethod
    def _perturbation_spatial_html(axis: "pert.AxisSensitivity") -> str:
        b_spatial, l_spatial = axis.baseline.spatial, axis.lowest_point.spatial
        if b_spatial is None or l_spatial is None:
            return "<span style='color:palette(mid)'>—</span>"
        # HORIZONTAL_REGIONS and VERTICAL_REGIONS both contain "CENTER" (a
        # different statistic in each), so the two dicts must stay separate
        # here rather than merged into one lookup that would collide on it.
        rows = ["<tr><th></th><th>Baseline</th><th>Lowest tested</th></tr>"]
        for group, b_dict, l_dict in (
            ("수평", b_spatial.horizontal, l_spatial.horizontal),
            ("수직", b_spatial.vertical, l_spatial.vertical),
        ):
            rows.append(f"<tr><td colspan='3' style='color:palette(mid)'>{group}</td></tr>")
            for label, b in b_dict.items():
                l = l_dict[label]
                rows.append(
                    f"<tr><td>{label}</td>"
                    f"<td align='right'>{VerifyStep._bin_text(b)}</td>"
                    f"<td align='right'>{VerifyStep._bin_text(l)}</td></tr>"
                )
        return "<table cellspacing='4'>" + "".join(rows) + "</table>"

    # --------------------------------------------------------------- fine scan
    #
    # Zooms into whichever axis is selected in the grid's own axis combo, over
    # a caller-chosen range/step -- shares evaluate_single_axis with the grid's
    # evaluate_perturbation_grid (see gui.core.evaluation.perturbation), so a
    # delta scored here and one scored by the grid are never computed two
    # different ways.

    def _on_pert_axis_changed_for_fine_scan(self):
        axis_key = self.pert_axis_combo.currentData()
        if axis_key is None:
            return
        self.fine_axis_label.setText(f"Selected Axis: {pert.AXIS_LABELS[axis_key]}")
        is_rotation = axis_key in pert.ROTATION_AXES
        suffix = " deg" if is_rotation else " mm"
        for spin in (self.fine_min_spin, self.fine_max_spin, self.fine_step_spin):
            spin.setSuffix(suffix)
        if is_rotation:
            self.fine_min_spin.setValue(-0.2)
            self.fine_max_spin.setValue(0.2)
            self.fine_step_spin.setValue(0.05)
        else:
            self.fine_min_spin.setValue(-10.0)
            self.fine_max_spin.setValue(10.0)
            self.fine_step_spin.setValue(2.0)

    def _run_fine_scan(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        axis_key = self.pert_axis_combo.currentData()
        if axis_key is None:
            return

        mode = "current_frame" if self.pert_mode_current_radio.isChecked() else "multi_frame"
        if mode == "current_frame":
            if self._image is None or self._cloud is None:
                self.window().statusBar().showMessage("먼저 화면에 프레임을 불러오세요.", 4000)
                return
        elif not self._times:
            self.window().statusBar().showMessage("bag 이 아직 로드되지 않았습니다.", 4000)
            return

        if self.fine_max_spin.value() < self.fine_min_spin.value():
            self.window().statusBar().showMessage("Fine Scan Max가 Min보다 작습니다.", 4000)
            return
        deltas = pert.make_delta_range(
            self.fine_min_spin.value(), self.fine_max_spin.value(), self.fine_step_spin.value(),
        )

        self._fine_gen += 1
        self.fine_run_btn.setEnabled(False)
        self.fine_cancel_btn.setVisible(True)
        self.fine_progress_label.setText(f"분석 준비 중… (0 / {len(deltas)})")
        self.fine_table_label.setText("—")
        self.fine_frame_table_label.setText("—")
        self.fine_summary_label.setText("—")
        self.fine_spatial_label.setText("—")
        self._fine_result = None
        self._fine_T = (sol.R.copy(), sol.t.copy())
        self._fine_mode = mode

        p = self.project
        self.request_fine_scan.emit(
            self._fine_gen, mode, self._current_bag(), p.lidar_topic, p.camera_topic,
            p.camera, sol.R, sol.t, list(self._times), self.pert_frames_spin.value(),
            axis_key, deltas,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
            float(self.max_sync_spin.value()), self._image, self._cloud,
        )

    def _cancel_fine_scan(self):
        self._worker.cancel_fine_scan()
        self.fine_progress_label.setText(self.fine_progress_label.text() + "  (취소 중…)")

    def _on_fine_progress(self, gen: int, done: int, total: int):
        if gen != self._fine_gen:
            return
        self.fine_progress_label.setText(f"평가 중 {done} / {total}")

    def _on_fine_ready(self, gen: int, axis: "pert.AxisSensitivity"):
        if gen != self._fine_gen:
            return
        self.fine_run_btn.setEnabled(True)
        self.fine_cancel_btn.setVisible(False)
        self._fine_result = axis
        self._display_fine_scan(axis)

    def _on_fine_failed(self, gen: int, msg: str):
        if gen != self._fine_gen:
            return
        self.fine_run_btn.setEnabled(True)
        self.fine_cancel_btn.setVisible(False)
        self.fine_progress_label.setText("")
        self.fine_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _display_fine_scan(self, axis: "pert.AxisSensitivity"):
        self.fine_progress_label.setText(f"완료 — {len(axis.points)} delta 평가")
        if axis.baseline is None:
            self.fine_table_label.setText(
                "<span style='color:palette(mid)'>baseline 평가 전에 취소되었습니다.</span>"
            )
            self.fine_frame_table_label.setText("—")
            self.fine_summary_label.setText("—")
            self.fine_spatial_label.setText("—")
            return
        self.fine_table_label.setText(self._perturbation_table_html(axis))
        self.fine_frame_table_label.setText(self._perturbation_frame_table_html(axis))
        self.fine_summary_label.setText(self._perturbation_summary_html(axis))
        self.fine_spatial_label.setText(self._perturbation_spatial_html(axis))

    # ------------------------------------------------------ diagnostic evidence
    #
    # Only gathers what the sections above already computed and hands it to
    # gui.core.evaluation.diagnostics -- every rule lives there, not here.
    # Nothing in this section writes to self._sol, the project, or a file.

    def _diagnostic_inputs(self) -> "dg.DiagnosticInputs":
        sol = self._effective()
        R, t = (sol.R, sol.t) if sol is not None and sol.ok else (None, None)
        notes: list = []

        def fresh(result, T, name):
            if result is None:
                return None
            if T is None or not dg.extrinsic_matches(T[0], T[1], R, t):
                notes.append(f"{name} result excluded: it was computed for a different extrinsic -- rerun it.")
                return None
            return result

        pert_result = fresh(self._pert_result, self._pert_T, "Perturbation Sensitivity")
        fine_result = fresh(self._fine_result, self._fine_T, "Fine Scan")
        mf_result = fresh(self._mf_result, self._mf_T, "Multi-frame evaluation")
        eval_result = fresh(self._eval_result, self._eval_T, "Current-frame evaluation")
        eval_spatial = self._eval_spatial if eval_result is not None else None

        calibration = dg.CalibrationContext()
        step = self._calibrate_step()
        if self._loaded_from is not None:
            calibration.unavailable_reason = (
                "The extrinsic was loaded from a file; Step 6 leave-one-out describes a different solution."
            )
        elif step is None or getattr(step, "solution", None) is not self._sol or self._sol is None:
            calibration.unavailable_reason = "No Step 6 solution matches the extrinsic being verified."
        else:
            calibration.loo = getattr(step, "loo", None)
            calibration.rmse_m = self._sol.rmse
            calibration.n_scenes = len(self._sol.scene_ids)
            if self._flipped:
                notes.append(
                    "The shown extrinsic is the half-turn flipped version; Step 6 leave-one-out describes the "
                    "unflipped solve."
                )

        return dg.DiagnosticInputs(
            perturbation=pert_result,
            fine_scan=fine_result,
            fine_scan_mode=self._fine_mode,
            multiframe=mf_result,
            multiframe_sync_limit_ms=self._mf_sync_limit_ms or None,
            perturbation_sync_limit_ms=self._pert_sync_limit_ms or None,
            current_frame=eval_result,
            current_frame_spatial=eval_spatial,
            calibration=calibration,
            notes=notes,
        )

    def _generate_diagnostics(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        self._diag_report = dg.build_diagnostic_report(self._diagnostic_inputs())
        self.diag_label.setText(self._diagnostic_html(self._diag_report))
        self.diag_copy_btn.setEnabled(True)

    def _copy_diagnostics(self):
        if self._diag_report is None:
            return
        QtWidgets.QApplication.clipboard().setText(dg.format_report_text(self._diag_report))
        self.window().statusBar().showMessage("진단 결과를 복사했습니다.", 3000)

    @staticmethod
    def _diagnostic_html(report: "dg.DiagnosticReport") -> str:
        esc = lambda s: html.escape(s).replace("\n", "<br>")  # noqa: E731
        muted = {dg.NOT_OBSERVED, dg.INSUFFICIENT, dg.UNAVAILABLE}

        def strength(item) -> str:
            style = "color:palette(mid)" if item.strength in muted else "font-weight:bold"
            text = f"<span style='{style}'>{esc(item.strength)}</span>"
            if item.mixed:
                text += " <span style='color:#b26a00;font-weight:bold'>(Mixed)</span>"
            return text

        mode_text = {dg.MODE_MULTI: "Multi-frame", dg.MODE_SINGLE: "Single-frame diagnostic",
                     dg.MODE_NONE: "Insufficient data"}
        parts = [f"<div style='color:palette(mid)'>Mode: {mode_text.get(report.mode, report.mode)}</div>"]
        parts += [f"<div style='color:palette(mid)'>• {esc(n)}</div>" for n in report.notes]
        parts.append(f"<p><b>{esc(report.headline)}</b></p>")

        parts.append("<p><b>Root Cause Candidates</b><br><span style='color:palette(mid)'>"
                     "Ordered by evidence strength and frame consistency; order is not a probability.</span></p>")
        if report.candidates:
            items = "".join(
                f"<li>{esc(c.title)} — Evidence: {strength(c)}"
                + "".join(f"<br>{esc(t)}" for t in c.interpretations)
                + "</li>"
                for c in report.candidates
            )
            parts.append(f"<ol>{items}</ol>")
        else:
            parts.append("<div style='color:palette(mid)'>(none)</div>")

        for category in dg.CATEGORY_ORDER:
            items = [i for i in report.evidence if i.category == category]
            if not items:
                continue
            parts.append(f"<hr><b>{esc(category)}</b>")
            for item in items:
                block = [f"<p><b>{esc(item.title)}</b><br>Evidence: {strength(item)}</p>"]
                if item.metrics:
                    block.append(VerifyStep._rows_table([(esc(k), esc(v)) for k, v in item.metrics]))
                if item.observations:
                    block.append("Observed:<ul>" + "".join(f"<li>{esc(o)}</li>" for o in item.observations) + "</ul>")
                if item.interpretations:
                    block.append("Possible interpretation:<ul>"
                                 + "".join(f"<li>{esc(t)}</li>" for t in item.interpretations) + "</ul>")
                if item.caveats:
                    block.append("<ul style='color:palette(mid)'>"
                                 + "".join(f"<li>{esc(c)}</li>" for c in item.caveats) + "</ul>")
                parts.append("".join(block))
        return "".join(parts)

    # ---------------------------------------------------------------- export

    def _export_text(self) -> str:
        sol = self._effective()
        if sol is None:
            return ""
        kind = self.export_combo.currentData()
        cam = self.project.camera
        if kind == "livo2":
            h, w = (self._image.shape[:2] if self._image is not None else (0, 0))
            return to_fast_livo2(sol, cam, w, h)
        if kind == "yaml":
            note = "half-turn flipped by hand" if self._flipped else ""
            return verify.to_yaml(
                sol, cam, self.project.target, note,
                project=self.project, method=self._method(),
            )
        if kind == "rt":
            return verify.to_rt_matrix(sol, cam, project=self.project, method=self._method())
        return verify.to_static_transform(sol) + "\n"

    def _update_export(self):
        self.export_text.setPlainText(self._export_text())

    def _capture_note(self) -> str:
        """What the picture was made from.

        A projection image is evidence, and evidence is only readable with its
        conditions attached: which bag, which moment, which extrinsic, and what
        was filtered out before drawing. Written beside the image so the two
        cannot drift apart.
        """
        p = self.project
        t_ns = self._times[self.slider.value()] if self._times else 0
        rel = (t_ns - self._times[0]) / 1e9 if self._times else 0.0
        pr = self._last_projection
        cam = p.camera
        lines = [
            f"프로젝트   : {p.name}",
            f"bag        : {Path(self._current_bag()).name}",
            f"시각       : {rel:.2f} s (t_ns {t_ns})",
            f"LiDAR 토픽 : {p.lidar_topic}",
            f"카메라 토픽: {p.camera_topic}",
            f"카메라     : fx {cam.fx:.4f} fy {cam.fy:.4f} cx {cam.cx:.4f} cy {cam.cy:.4f}",
            f"왜곡       : {'rational_polynomial(8)' if cam.rational else 'plumb_bob(5)'} "
            + " ".join(f"{v:.6f}" for v in cam.dist()),
            f"extrinsic  : {self._loaded_from or '6단계 계산 결과'}"
            + (f" (RMSE {self._sol.rmse * 1000:.2f} mm)" if self._sol else ""),
            "",
            "그리기 조건",
            f"  거리 필터 : {self.near_spin.value():.2f} ~ {self.far_spin.value():.2f} m"
            + ("  (0 = 제한 없음)" if not self.far_spin.value() else ""),
            f"  색상      : {self.colour_combo.currentText()}",
            f"  점 크기   : {self.size_spin.value():.1f} px   배경 {self.dim_slider.value()}%",
        ]
        if pr is not None:
            lines += [
                "",
                "점 개수",
                f"  입력        {pr.n_input:,}",
                f"  카메라 뒤   {pr.n_behind:,}",
                f"  시야 밖(왜곡 모델 유효 범위 초과) {pr.n_folded:,}",
                f"  화면 밖     {pr.n_outside:,}",
                f"  그린 점     {pr.n_visible:,}",
            ]
        return "\n".join(lines) + "\n"

    def _save_capture(self):
        # `_pix` only exists once something has been drawn; _apply_zoom guards the
        # same way.
        pix = getattr(self, "_pix", None)
        if pix is None or pix.isNull():
            self.window().statusBar().showMessage("아직 그릴 그림이 없습니다.", 4000)
            return
        stem = self.project.name if self.project.name != "untitled" else "capture"
        t_ns = self._times[self.slider.value()] if self._times else 0
        rel = (t_ns - self._times[0]) / 1e9 if self._times else 0.0
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        default = RESULT_DIR / f"{stem}_projection_{rel:07.2f}s.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "캡처 저장", str(default), "PNG 이미지 (*.png)"
        )
        if not path:
            return
        if not pix.save(path, "PNG"):
            self.window().statusBar().showMessage("저장하지 못했습니다.", 5000)
            return
        note = Path(path).with_suffix(".txt")
        note.write_text(self._capture_note())
        self.window().statusBar().showMessage(
            f"저장했습니다: {path}  (조건은 {note.name})", 6000
        )

    def _copy_export(self):
        text = self.export_text.toPlainText()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.window().statusBar().showMessage("복사했습니다.", 3000)

    def _save_export(self):
        text = self.export_text.toPlainText()
        if not text:
            return
        kind = self.export_combo.currentData()
        stem = self.project.name if self.project.name != "untitled" else "extrinsic"
        name = {"livo2": f"{stem}_fast_livo2.txt", "yaml": f"{stem}.yaml",
                "tf": f"{stem}_static_transform.sh", "rt": f"{stem}_rt.yaml"}[kind]
        # calib_result/ holds one extrinsic per vehicle and is committed: this is
        # the file the rest of the team uses, so its history is worth keeping.
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "extrinsic 저장", str(RESULT_DIR / name))
        if path:
            with open(path, "w") as fh:
                fh.write(text)
            self.window().statusBar().showMessage(f"저장했습니다: {path}", 5000)

    # ------------------------------------------------------------------ state

    def is_complete(self) -> bool:
        return self._sol is not None and self._sol.ok

    def status_text(self) -> str:
        if self._sol is None or not self._sol.ok:
            return "6단계 필요"
        if self._loaded_from is not None:
            return f"불러온 extrinsic 검증 ({self._loaded_from})"
        return "뒤집어 확인함" if self._flipped else "투영 확인"

    def shutdown(self):
        self._worker.cancel_multiframe()  # let a long multi-frame run exit early rather than block close
        self._worker.shutdown()
        self._thread.quit()
        self._thread.wait(1500)

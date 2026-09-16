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

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gui.core import verify
from gui.core.bag_reader import BagSource, accumulate_cloud
from gui.core.decode import image_to_bgr
from gui.core.detect_camera import detect as detect_camera
from gui.core.detect_lidar import DetectParams, apply_box, detect as detect_lidar, point_spacing
from pathlib import Path

from gui.core.project import RESULT_DIR, Project
from gui.core.solve import Solution, solve_rigid, sort_centers, to_fast_livo2
from gui.ui.steps import StepPage
from gui.ui.steps.step_filter import derive_params


class _Worker(QtCore.QObject):
    opened = QtCore.Signal(int, object)  # gen, LiDAR message times
    ready = QtCore.Signal(int, object, object, object, object, object)  # gen, image, cloud, intensity, lidar c, cam c
    failed = QtCore.Signal(int, str)
    scanning = QtCore.Signal(int, int, int)  # gen, messages read, total

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None

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
             t_ns: int, frames: int, scene, camera, target, method: str):
        """One moment's cloud and image, and the hole centres when asked for.

        `scene` is None while scrubbing freely. Detection needs the scene's filter
        box to know where to look, so away from a captured moment only the
        projection is produced -- which is still enough to judge an extrinsic by
        eye, and often more telling than four points, since the whole scene has to
        line up rather than just the board.
        """
        try:
            src = self._source(bag)

            _, img_msg = src.first_after(camera_topic, t_ns)
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
    request_load = QtCore.Signal(int, str, str, str, object, int, object, object, object, str)

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
        side_box.setFixedWidth(330)

        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(self.scroll, 1)
        middle.addWidget(side_box)

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
        self.request_load.connect(self._worker.load)
        self.request_open.connect(self._worker.open_bag)
        self._thread.start()

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
        self.request_load.emit(
            self._gen, self._current_bag(), p.lidar_topic, p.camera_topic,
            t_ns, frames, self._scene, p.camera, p.target, self._method(),
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
        self._worker.shutdown()
        self._thread.quit()
        self._thread.wait(1500)

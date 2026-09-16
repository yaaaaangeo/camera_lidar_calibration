"""Step 5 — the distance filter, with the detector running behind it.

The filter box is typed in, not clicked: numbers are what end up in the saved
project, and they are what you adjust when something is slightly off.

Detection is run on demand, not on every keystroke. Six numbers have to be set
before the box means anything, so re-detecting after each one just makes the
fields sluggish for results nobody is reading yet. Adjust the box against the
live 3D view, then press Detect.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gui.core import roi_auto
from gui.core.bag_reader import BagSource, accumulate_cloud
from gui.core.decode import image_to_bgr
from gui.core.detect_camera import detect as detect_camera
from gui.core.detect_lidar import (
    DetectParams, LidarDetection, apply_region, box_mask, detect, point_spacing,
    region_mask,
)
from gui.core.project import FilterBox, PlaneRegionSpec, Project, Scene, Target
from gui.core.solve import solve
from gui.ui.cloud_view import CloudView
from gui.ui.color_controls import ColorControls
from gui.ui.steps import StepPage

# The box is edited as centre plus size, not as six bounds, because the rotation
# turns about the centre. With bounds, the centre is a derived value: moving
# x_min drags the axis of rotation halfway with it, and the whole box swings
# instead of one face moving. Measured on a real ROI at 127 degrees of yaw,
# pulling x_min in by 10 cm shifted all eight corners by 4-9 cm, in directions
# that have nothing to do with x. Separating the two makes size changes grow the
# box in place and centre changes a pure translation.
#
# Centre is in sensor coordinates, size along the box's own axes -- which is why
# they are named differently. After a rotation "x" and "가로" no longer point the
# same way, and the labels have to say so.
_CENTRE = [("cx", "중심 x"), ("cy", "중심 y"), ("cz", "중심 z")]
_SIZE = [("sx", "가로"), ("sy", "세로"), ("sz", "두께")]

# Rotating the box is what makes a tilted board separable: an axis-aligned box
# has to span the board's diagonal, and the wall behind fills what that leaves
# empty. Turned square with the panel, the box can be thin again.
_ANGLES = [("yaw", "yaw"), ("pitch", "pitch"), ("roll", "roll")]

# Smallest edge the box may have, in metres. Zero would make the region empty
# and the slider unrecoverable.
_MIN_SIZE = 0.02

# Edge length a fresh box starts at. Sized for a board rather than for the
# recording: see _reset_box.
_DEFAULT_SIZE = 5.0

# Where that box starts, in metres. Centred on the sensor in x and y, but lifted
# in z: a cube centred at z=0 spends half its height below the sensor, and on a
# ground-referenced cloud that half is under the road. Starting at 2.5 makes the
# box span 0 to 5 m up, which is where a board actually is.
_DEFAULT_CENTRE = (0.0, 0.0, 2.5)


def box_untouched(scene) -> bool:
    """Has this scene's region been left at whatever it started as?

    Copying a box over a scene someone already tuned throws that work away with
    no undo, so the copy dialog needs to tell the two apart. A scene is
    "untouched" if it has no region and its box is either empty (never opened) or
    still the default cube at the origin.
    """
    if scene.region.is_set:
        return False
    box = scene.filter
    if not box.is_set:
        return True
    if box.rotated:
        return False
    if max(abs(v - c) for v, c in zip(box.center(), _DEFAULT_CENTRE)) > 1e-6:
        return False
    sizes = (box.x_max - box.x_min, box.y_max - box.y_min, box.z_max - box.z_min)
    return all(abs(v - _DEFAULT_SIZE) < 1e-6 for v in sizes)


def box_summary(scene) -> str:
    """One short phrase describing a scene's region, for the copy dialog."""
    if scene.region.is_set:
        return "평면 영역 지정됨"
    box = scene.filter
    if not box.is_set:
        return "아직 안 열어봄"
    sizes = (box.x_max - box.x_min, box.y_max - box.y_min, box.z_max - box.z_min)
    text = "×".join(f"{v:.2f}" for v in sizes) + " m"
    if box.rotated:
        text += f", yaw {box.yaw:.0f}°"
    return text


class CopyTargetDialog(QtWidgets.QDialog):
    """Which scenes to copy the current region onto.

    The old button overwrote every other scene unconditionally -- including the
    ones already tuned, which is the opposite of what it is wanted for. What it
    is actually for is giving the scenes not yet worked on a sensible starting
    box, so those are the ones checked by default; overwriting real work stays
    possible but has to be asked for.
    """

    def __init__(self, parent, source, others):
        super().__init__(parent)
        self.setWindowTitle("필터 복사")
        self._items = []

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for sc in others:
            fresh = box_untouched(sc)
            label = f"{sc.id}    {box_summary(sc)}"
            if not fresh:
                label += "    ← 덮어씀"
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if fresh else QtCore.Qt.Unchecked)
            if not fresh:
                item.setForeground(QtGui.QColor("#b26a00"))
            self.list.addItem(item)
            self._items.append((item, sc))
        self.list.itemChanged.connect(self._update_count)

        all_btn = QtWidgets.QPushButton("전체 선택")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QtWidgets.QPushButton("전체 해제")
        none_btn.clicked.connect(lambda: self._set_all(False))

        self.count = QtWidgets.QLabel()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok.setText("복사")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        head = QtWidgets.QLabel(
            f"<b>{source.id}</b> 의 영역을 어느 scene 에 복사할까요?<br>"
            "<span style='color:palette(mid)'>아직 손대지 않은 scene 만 미리 선택했습니다. "
            "주황색은 이미 조정된 scene 이라 누르면 지금 값이 사라집니다.</span>"
        )
        head.setWordWrap(True)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        row.addWidget(self.count)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(head)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        lay.addWidget(buttons)
        self.resize(420, 320)
        self._update_count()

    def _set_all(self, on: bool):
        state = QtCore.Qt.Checked if on else QtCore.Qt.Unchecked
        for item, _ in self._items:
            item.setCheckState(state)

    def _update_count(self):
        picked = self.chosen()
        over = sum(1 for sc in picked if not box_untouched(sc))
        text = f"{len(picked)}개 선택"
        if over:
            text += f" · <span style='color:#b26a00'>{over}개 덮어씀</span>"
        self.count.setText(text)
        self.ok.setEnabled(bool(picked))

    def chosen(self):
        return [sc for item, sc in self._items if item.checkState() == QtCore.Qt.Checked]


def box_to_fields(box) -> dict[str, float]:
    """FilterBox bounds -> the centre/size/angle values the panel edits."""
    out = {}
    for (ck, _), (sk, _), axis in zip(_CENTRE, _SIZE, "xyz"):
        lo, hi = getattr(box, f"{axis}_min"), getattr(box, f"{axis}_max")
        out[ck] = (lo + hi) / 2.0
        out[sk] = max(hi - lo, 0.0)
    for key, _ in _ANGLES:
        out[key] = getattr(box, key)
    return out


def fields_to_box(box, fields: dict[str, float]):
    """Write centre/size/angle values back onto a FilterBox as bounds."""
    for (ck, _), (sk, _), axis in zip(_CENTRE, _SIZE, "xyz"):
        half = max(fields[sk], _MIN_SIZE) / 2.0
        setattr(box, f"{axis}_min", fields[ck] - half)
        setattr(box, f"{axis}_max", fields[ck] + half)
    for key, _ in _ANGLES:
        setattr(box, key, fields[key])


def derive_params(spacing: float) -> DetectParams:
    """Scale the density-dependent constants to the cloud actually in hand.

    The stock values assume the dense clouds the original tool was built on.
    The boundary test needs a neighbourhood several points across to mean
    anything: at 30 mm radius and 30 mm spacing every point looks like an edge.
    """
    p = DetectParams(sweep_spacing=spacing)
    if spacing > 0:
        p.boundary_radius = max(p.boundary_radius, spacing * 4)
        p.cluster_tolerance = max(p.cluster_tolerance, spacing * 2.5)
    return p


class _Worker(QtCore.QObject):
    accumulated = QtCore.Signal(int, object, int, object, object, object)  # gen, xyz, frames, sweep, intensity, ring
    detected = QtCore.Signal(int, object)  # gen, LidarDetection
    failed = QtCore.Signal(int, str)
    scanning = QtCore.Signal(int, int, int)  # gen, messages read, total

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None

    def accumulate(self, gen: int, bag: str, topic: str, t_ns: int, frames: int):
        try:
            # Scenes can come from different recordings, so reopen when the
            # bag changes rather than assuming one file for the project.
            if self._src is None or str(self._src.path) != bag:
                if self._src is not None:
                    self._src.close()
                self._src = BagSource(bag)
                self._src.open()
            # The first scene on a large recording spends seventeen seconds
            # indexing before anything can be drawn, and a still label through all
            # of it reads as a hung window -- it was reported as exactly that.
            xyz, n, single, intensity, ring = accumulate_cloud(
                self._src, topic, t_ns, frames,
                progress=lambda done, total: self.scanning.emit(gen, done, total),
            )
            self.accumulated.emit(gen, xyz, n, single, intensity, ring)
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def run_detect(self, gen: int, xyz, box: FilterBox, target: Target, params: DetectParams, ring):
        try:
            self.detected.emit(gen, detect(xyz, box, target, params, ring=ring))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")


class _SweepWorker(QtCore.QObject):
    """Places the box for every scene that has not had one drawn.

    Runs off the UI thread because it reads the whole recording again -- one
    cloud and one image per scene -- and then detects several times over.

    Scenes are visited seeds first. The seeds are what there is an extrinsic
    from, and until there is one there is nowhere to crop a cloud to; once there
    is, every later scene can be cut down to a ball around where its board is
    predicted to be, which is the difference between holding a hundred megabytes
    and holding six.
    """

    progress = QtCore.Signal(str)
    finished = QtCore.Signal(object, str)  # {scene id: FilterBox}, summary
    failed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None

    def _open(self, bag: str):
        if self._src is None or str(self._src.path) != bag:
            if self._src is not None:
                self._src.close()
            self._src = BagSource(bag)
            self._src.open()
        return self._src

    def _read(self, scene, bag, lidar_topic, camera_topic, camera, target):
        """(camera detection, stacked cloud, single sweep, ring) for one scene."""
        src = self._open(bag)
        _, img = src.first_after(camera_topic, scene.t_ns)
        cam_det = detect_camera(image_to_bgr(img), camera, target) if img is not None else None
        cloud, _, single, _, ring = accumulate_cloud(src, lidar_topic, scene.t_ns, scene.frames)
        return cam_det, cloud, single, ring

    def run(self, jobs, lidar_topic: str, camera_topic: str, camera, target, method: str):
        try:
            self._sweep(jobs, lidar_topic, camera_topic, camera, target, method)
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _sweep(self, jobs, lidar_topic, camera_topic, camera, target, method):
        spacing: dict[str, float] = {}
        rings: dict[str, object] = {}

        def run_detect(sc, box):
            params = derive_params(spacing.get(sc.scene_id, 0.0))
            params.method = method
            return detect(sc.cloud, box, target, params, ring=rings.get(sc.scene_id))

        seeds, rest = [job for job in jobs if job[0].roi_set], [job for job in jobs if not job[0].roi_set]
        if not seeds:
            self.finished.emit({}, "박스가 그려진 scene 이 하나도 없습니다.")
            return

        # Seeds first: a scene only counts as one if its own hand-drawn box
        # actually yields four holes, so a half-finished box does not get to
        # steer where every other box lands.
        placed_seeds, notes = [], []
        for scene, bag in seeds:
            self.progress.emit(f"{scene.id} — 그려둔 박스 확인 중")
            cam_det, cloud, single, ring = self._read(scene, bag, lidar_topic, camera_topic, camera, target)
            if cam_det is None or not cam_det.ok:
                notes.append(f"{scene.id}: 카메라가 보드를 못 찾음")
                continue
            box = scene.roi()
            inside = region_mask(single, box)
            spacing[scene.id] = point_spacing(single[inside]) if int(inside.sum()) > 20 else 0.0
            rings[scene.id] = ring
            sc = roi_auto.SweepScene(scene.id, cloud, cam_det)
            det = run_detect(sc, box)
            if not det.ok:
                notes.append(f"{scene.id}: {det.reason}")
                continue
            sc.centres, sc.box = det.centers, box
            placed_seeds.append(sc)

        if not placed_seeds:
            self.finished.emit({}, "그려둔 박스에서 원 4개가 나온 scene 이 없습니다.\n"
                                   "한 scene 만 먼저 손으로 맞춰 검출을 성공시켜 주세요.")
            return

        # Seed clouds are only needed for the detection just done; drop them
        # before reading eighteen more.
        for sc in placed_seeds:
            sc.cloud = np.zeros((0, 3), np.float32)
        seed_sol = solve([(sc.scene_id, sc.centres, sc.cam_det.centers) for sc in placed_seeds])
        if not seed_sol.ok:
            self.finished.emit({}, f"출발점 extrinsic 을 풀지 못했습니다: {seed_sol.reason}")
            return

        others = []
        for scene, bag in rest:
            self.progress.emit(f"{scene.id} — 읽는 중")
            cam_det, cloud, single, ring = self._read(scene, bag, lidar_topic, camera_topic, camera, target)
            if cam_det is None or not cam_det.ok:
                notes.append(f"{scene.id}: 카메라가 보드를 못 찾음")
                continue
            rings[scene.id] = ring
            spacing[scene.id] = point_spacing(single) if len(single) > 20 else 0.0
            others.append(roi_auto.SweepScene(
                scene.id, roi_auto.crop_near_board(cloud, seed_sol, cam_det, target), cam_det))

        if not others:
            self.finished.emit({}, "자동으로 배치할 scene 이 없습니다.")
            return

        report = roi_auto.sweep(
            placed_seeds + others, target, run_detect,
            progress=lambda rnd, sid: self.progress.emit(f"{rnd}회차 — {sid} 배치 중"),
        )
        boxes = {sc.scene_id: sc.box for sc in others if sc.box is not None}
        missed = [f"{sc.scene_id}: {sc.note or '원 4개를 못 찾음'}" for sc in others if sc.box is None]
        sol = report.solution
        summary = (
            f"{len(boxes)}개 scene 에 박스를 놓았습니다 "
            f"(손으로 그린 {len(placed_seeds)}개 + 자동 {len(boxes)}개, {report.rounds}회차).\n"
            f"이 {sol.n_pairs // 4}개로 푼 extrinsic 의 RMSE 는 {sol.rmse * 1000:.1f} mm 입니다."
        )
        if missed or notes:
            summary += "\n\n놓지 못한 scene:\n" + "\n".join(notes + missed)
        self.finished.emit(boxes, summary)


class FilterStep(StepPage):
    title = "5. 거리 필터"
    subtitle = "보드 주변만 남기기"

    request_accumulate = QtCore.Signal(int, str, str, object, int)
    request_detect = QtCore.Signal(int, object, object, object, object, object)
    request_sweep = QtCore.Signal(object, str, str, object, object, str)

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self._gen = 0
        self._cloud: np.ndarray | None = None
        self._single: np.ndarray | None = None
        self._intensity: np.ndarray | None = None
        self._ring: np.ndarray | None = None
        self._det: LidarDetection | None = None
        self._base_colors: np.ndarray | None = None
        self._colored_style = None
        # Drawing is done on a thinned copy; detection always uses the full cloud.
        self._draw_xyz: np.ndarray | None = None
        self._draw_intensity: np.ndarray | None = None
        self._spacing = 0.0
        self._scene: Scene | None = None
        self._params = DetectParams()

        # --- scene picker ----------------------------------------------------
        self.scene_combo = QtWidgets.QComboBox()
        self.scene_combo.currentIndexChanged.connect(self._select_scene)

        # How many sweeps to stack. One is the default: see Scene.frames for why
        # more is usually worse, not just slower.
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(1, 40)
        self.frames_spin.setValue(1)
        self.frames_spin.setSuffix(" 프레임")
        self.frames_spin.setToolTip("이 시점 주변에서 합칠 LiDAR 스윕 수")
        self.frames_spin.valueChanged.connect(self._on_frames)
        self.cloud_info = QtWidgets.QLabel("—")
        self.cloud_info.setStyleSheet("color: palette(mid);")
        self.cloud_info.setWordWrap(True)

        # --- filter fields ---------------------------------------------------
        self.spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.sliders: dict[str, QtWidgets.QSlider] = {}
        grid = QtWidgets.QGridLayout()
        row = 0

        def heading(text: str, hint: str):
            nonlocal row
            lbl = QtWidgets.QLabel(f"{text}  <span style='color:palette(mid)'>{hint}</span>")
            lbl.setStyleSheet("font-weight: 600; margin-top: 6px;")
            grid.addWidget(lbl, row, 0, 1, 3)
            row += 1

        def add(key, label, lo, hi, step, decimals, suffix, scale, wrap=False):
            nonlocal row
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            spin.setWrapping(wrap)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(int(lo * scale), int(hi * scale))
            spin.valueChanged.connect(lambda v, k=key, s=scale: self._on_spin(k, v, s))
            slider.valueChanged.connect(lambda v, k=key, s=scale: self._on_slider(k, v, s))
            self.spins[key], self.sliders[key] = spin, slider
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(spin, row, 1)
            grid.addWidget(slider, row, 2)
            row += 1

        heading("위치", "센서 좌표 기준 — 박스를 통째로 옮깁니다")
        for key, label in _CENTRE:
            add(key, label, -200.0, 200.0, 0.05, 2, " m", 100.0)

        heading("크기", "박스 자기 축 기준 — 중심은 그대로 둔 채 늘고 줍니다")
        for key, label in _SIZE:
            add(key, label, _MIN_SIZE, 100.0, 0.05, 2, " m", 100.0)

        heading("회전", "박스 중심을 축으로 돌립니다")
        for key, label in _ANGLES:
            add(key, label, -180.0, 180.0, 1.0, 1, " °", 10.0, wrap=True)

        grid.setColumnStretch(2, 1)

        # Ordered by provenance: the two ports of the original first, then the two
        # written here. The old ordering put a work-in-progress method at the top
        # and gave no way to tell which of "ring scan" and "original" came from
        # where -- one is a variant of the other, which the names hid.
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem("FAST-Calib (기계식, 링 점프)", "original")
        self.method_combo.addItem("FAST-Calib (솔리드, 이웃 각도)", "boundary")
        self.method_combo.addItem("링 점프 쌍 묶기 (ring: o)", "ring")
        self.method_combo.addItem("평면 격자 (ring: x)", "occupancy")
        self.method_combo.addItem("실험용 (평면 격자 사본)", "experimental")
        self.method_combo.setToolTip(
            "FAST-Calib (기계식): detect_mech_lidar 를 상수까지 그대로 옮긴 것.\n"
            "  링을 따라가다 거리가 튀는 곳을 찾고, RANSAC 으로 원을 찾아 그 점을\n"
            "  지우며 반복합니다. 실측에서 유일하게 안정적입니다 (10스윕 중 10회).\n\n"
            "FAST-Calib (솔리드): detect_solid_lidar. 각 점의 이웃이 한쪽으로 비었는지\n"
            "  봅니다. Livox 같은 솔리드스테이트용이라 기계식 데이터에는 맞지 않습니다.\n\n"
            "링 점프 쌍 묶기: 위 기계식 방식의 변형. 구멍 좌우 테두리를 한 쌍으로\n"
            "  묶어 클러스터링합니다. 같은 조건에서 10스윕 중 2회로, 원본보다 못합니다.\n\n"
            "평면 격자: 평면을 격자로 만들고 채우고 남은 빈 곳을 구멍으로 봅니다.\n"
            "  ring 이 없는 병합 클라우드에서도 동작합니다. 아직 개발 중입니다.\n\n"
            "실험용: 평면 격자의 사본. 손을 대는 쪽은 이쪽이고 평면 격자는 그대로\n"
            "  둡니다. 같은 scene 에서 둘을 번갈아 눌러 비교하면 바뀐 것이 나은지\n"
            "  바로 보입니다. 처음에는 평면 격자와 결과가 완전히 같습니다."
        )
        self.method_combo.currentIndexChanged.connect(self._mark_stale)

        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("검출 방식"))
        method_row.addWidget(self.method_combo, 1)

        # One line of numbers, ready to copy into notes or a case file. Reading
        # six spin boxes off the screen and retyping them is where transcription
        # errors come from.
        self.roi_line = QtWidgets.QLineEdit()
        self.roi_line.setReadOnly(True)
        self.roi_line.setStyleSheet("font-family: monospace;")
        self.roi_line.setToolTip("x_min x_max y_min y_max z_min z_max — 클릭하면 전체 선택됩니다")
        copy_roi = QtWidgets.QPushButton("복사")
        copy_roi.setFixedWidth(52)
        copy_roi.clicked.connect(self._copy_roi)

        roi_row = QtWidgets.QHBoxLayout()
        roi_row.addWidget(self.roi_line, 1)
        roi_row.addWidget(copy_roi)

        self.detect_btn = QtWidgets.QPushButton("원 검출  (Enter)")
        self.detect_btn.setDefault(True)
        self.detect_btn.clicked.connect(self._run_detect)

        # Clicking beats typing when the board is held at an angle: an
        # axis-aligned box has to span a much larger volume then, and the margin
        # cannot be trimmed by hand. Growing the box from the plane under the
        # cursor sidesteps the estimate entirely.
        self.align_btn = QtWidgets.QPushButton("박스 안 평면에 맞추기")
        self.align_btn.setToolTip(
            "박스 안에서 가장 큰 평면을 찾아 박스를 그 방향으로 돌리고 얇게 만듭니다.\n"
            "대략 잡아둔 뒤 누르면 됩니다. 이후 숫자로 미세조정할 수 있습니다."
        )
        self.align_btn.clicked.connect(self._align_to_board)

        reset_btn = QtWidgets.QPushButton("클라우드 전체 범위로")
        reset_btn.clicked.connect(self._reset_box)
        copy_btn = QtWidgets.QPushButton("다른 scene에 복사…")
        self.auto_btn = QtWidgets.QPushButton("나머지 scene 자동 배치…")
        self.auto_btn.setToolTip(
            "이 scene 처럼 박스가 그려진 scene 으로 extrinsic 을 풀고,\n"
            "그것으로 아직 박스가 없는 scene 의 박스를 놓습니다."
        )
        self.auto_btn.clicked.connect(self._auto_place)
        copy_btn.clicked.connect(self._copy_to_others)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.align_btn)
        btns.addWidget(reset_btn)
        btns.addWidget(copy_btn)
        btns.addWidget(self.auto_btn)

        # --- results ---------------------------------------------------------
        self.results = QtWidgets.QLabel("—")
        self.results.setTextFormat(QtCore.Qt.RichText)
        self.results.setAlignment(QtCore.Qt.AlignTop)
        self.results.setWordWrap(True)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("scene"))
        scene_row = QtWidgets.QHBoxLayout()
        scene_row.addWidget(self.scene_combo, 1)
        scene_row.addWidget(self.frames_spin)
        left.addLayout(scene_row)
        left.addWidget(self.cloud_info)
        left.addSpacing(6)
        left.addLayout(grid)
        left.addLayout(roi_row)
        left.addLayout(method_row)
        left.addWidget(self.detect_btn)
        left.addLayout(btns)
        left.addSpacing(10)
        left.addWidget(QtWidgets.QLabel("검출 결과"))
        left.addWidget(self.results, 1)

        left_box = QtWidgets.QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(400)

        # --- 3D --------------------------------------------------------------
        self.view = CloudView()
        self.colors = ColorControls()
        self.colors.changed.connect(lambda: self._draw())
        self.show_edges = QtWidgets.QCheckBox("검출 지점 표시")
        self.show_edges.setToolTip(
            "점유 격자: 찾아낸 구멍 테두리의 점\n경계점 클러스터: 경계로 판정된 점 전체"
        )
        self.show_edges.setChecked(True)
        self.show_edges.toggled.connect(lambda: self._draw())
        self.hide_outside = QtWidgets.QCheckBox("박스 밖 숨기기")
        self.hide_outside.toggled.connect(lambda: self._draw())
        preset_bar = QtWidgets.QHBoxLayout()
        for name in ("top", "front", "side", "iso"):
            b = QtWidgets.QPushButton(name.capitalize())
            b.clicked.connect(lambda _=False, n=name: self.view.apply_preset(n))
            preset_bar.addWidget(b)
        fit_box = QtWidgets.QPushButton("박스에 맞춤")
        fit_box.setToolTip("필터 박스를 화면에 채우고, 회전 중심도 박스 안으로 옮깁니다")
        fit_box.clicked.connect(self._focus_box)
        preset_bar.addWidget(fit_box)
        fit_all = QtWidgets.QPushButton("전체 보기")
        fit_all.clicked.connect(lambda: self.view.fit(self._draw_xyz))
        preset_bar.addWidget(fit_all)
        preset_bar.addWidget(self.show_edges)
        preset_bar.addWidget(self.hide_outside)
        preset_bar.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(preset_bar)
        right.addWidget(self.colors)
        right.addWidget(self.view, 1)

        row = QtWidgets.QHBoxLayout(self)
        row.addWidget(left_box)
        row.addLayout(right, 1)

        # --- worker ----------------------------------------------------------
        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._worker.accumulated.connect(self._on_accumulated)
        self._worker.detected.connect(self._on_detected)
        self._worker.failed.connect(self._on_failed)
        self._worker.scanning.connect(self._on_scanning)
        self.request_accumulate.connect(self._worker.accumulate)
        self.request_detect.connect(self._worker.run_detect)
        self._thread.start()

        self._sweep_thread = QtCore.QThread(self)
        self._sweeper = _SweepWorker()
        self._sweeper.moveToThread(self._sweep_thread)
        self._sweeper.progress.connect(self._on_sweep_progress)
        self._sweeper.finished.connect(self._on_sweep_done)
        self._sweeper.failed.connect(self._on_sweep_failed)
        self.request_sweep.connect(self._sweeper.run)
        self._sweep_thread.start()

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, activated=self._run_detect)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, activated=self._run_detect)

    # ------------------------------------------------------------------ enter

    def on_enter(self):
        p = self.project
        if not p.scenes:
            self._clear_view()
            self.cloud_info.setText("4단계에서 scene을 먼저 캡처하세요.")
            return
        if not p.lidar_topic:
            self.cloud_info.setText("1단계에서 LiDAR 토픽을 지정하세요.")
            return

        # on_enter() runs on every navigator refresh, which happens on each edit,
        # so the combo is only rebuilt when the scene list really changed --
        # rebuilding re-selects, and re-selecting discards the accumulated cloud.
        ids = [sc.id for sc in p.scenes]
        stale = [self.scene_combo.itemText(i) for i in range(self.scene_combo.count())] != ids
        if stale or self._scene not in p.scenes:
            keep = p.scenes.index(self._scene) if self._scene in p.scenes else min(
                max(self.scene_combo.currentIndex(), 0), len(ids) - 1
            )
            self.scene_combo.blockSignals(True)
            self.scene_combo.clear()
            self.scene_combo.addItems(ids)
            self.scene_combo.setCurrentIndex(keep)
            self.scene_combo.blockSignals(False)
            self._select_scene(keep)
        elif self._cloud is None:
            self._select_scene(self.scene_combo.currentIndex())

    def _clear_view(self):
        """Nothing left to show -- drop the cloud rather than leaving a stale one."""
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.blockSignals(False)
        self._scene = None
        self._cloud = self._single = self._intensity = None
        self._det = None
        self._base_colors = None
        self._colored_style = None
        self._draw_xyz = self._draw_intensity = None
        self.view.set_points(np.zeros((0, 3), np.float32), (1.0, 1.0, 1.0, 1.0))
        self.view.clear_markers()
        self.view.clear_box()
        self.results.setText("—")

    def _on_frames(self, value: int):
        if self._scene is not None and value != self._scene.frames:
            self._scene.frames = value
            self._select_scene(self.scene_combo.currentIndex())

    def _select_scene(self, index: int):
        if not (0 <= index < len(self.project.scenes)):
            return
        self._scene = self.project.scenes[index]
        self.frames_spin.blockSignals(True)
        self.frames_spin.setValue(self._scene.frames)
        self.frames_spin.blockSignals(False)
        self._cloud = None
        self._det = None
        self.cloud_info.setText("클라우드 누적 중…")
        self._gen += 1
        self.request_accumulate.emit(
            self._gen, self.project.bag_for(self._scene), self.project.lidar_topic,
            self._scene.t_ns, self._scene.frames,
        )

    def _on_scanning(self, gen: int, done: int, total: int):
        """Show that the first read is progressing, not stuck.

        Only the first scene on a bag reaches here: after that the message index
        is cached and the whole thing returns instantly.
        """
        if gen != self._gen:
            return
        if total > 0:
            self.cloud_info.setText(
                f"bag 인덱스 읽는 중… {done:,} / {total:,} 메시지 ({done / total * 100:.0f}%)\n"
                "큰 파일은 처음 한 번만 오래 걸리고, 이후 scene 전환은 즉시 됩니다."
            )
        else:
            self.cloud_info.setText(f"bag 인덱스 읽는 중… {done:,} 메시지")

    def _on_accumulated(self, gen: int, xyz, frames: int, single, intensity, ring):
        if gen != self._gen:
            return
        self._cloud = xyz
        self._single = single
        self._intensity = intensity
        self._ring = ring
        self._base_colors = None

        # A ring field is what the sensor actually measured; not offering the
        # method that uses it would throw that away. Merged clouds never carry
        # one -- ring numbers from different sensors cannot be combined.
        has_ring = ring is not None
        model = self.method_combo.model()
        for name in ("ring", "original"):
            idx = self.method_combo.findData(name)
            if idx >= 0:
                model.item(idx).setEnabled(has_ring)
        if not has_ring and self.method_combo.currentData() in ("ring", "original"):
            self.method_combo.setCurrentIndex(self.method_combo.findData("occupancy"))

        # A million points is far more than the screen can show, and re-masking
        # that many on every edit is what makes the fields feel sticky. Detection
        # still runs on all of them.
        limit = 250_000
        if len(xyz) > limit:
            step = np.random.default_rng(0).permutation(len(xyz))[:limit]
            step.sort()
            self._draw_xyz = xyz[step]
            self._draw_intensity = None if intensity is None else intensity[step]
        else:
            self._draw_xyz, self._draw_intensity = xyz, intensity
        if len(xyz) == 0:
            self.cloud_info.setText("이 시각에 LiDAR 점이 없습니다.")
            return
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        self.cloud_info.setText(
            f"{frames} 프레임, {len(xyz):,} 점\n"
            f"x [{lo[0]:.1f}, {hi[0]:.1f}]  y [{lo[1]:.1f}, {hi[1]:.1f}]  z [{lo[2]:.1f}, {hi[2]:.1f}]"
        )
        if not self._scene.filter.is_set:
            self._reset_box()
            self.view.fit(xyz)
        else:
            self._load_box()
            self._focus_box()
        self._draw()
        self._mark_stale()

    def _align_to_board(self):
        if self._scene is None or self._single is None:
            return
        from gui.core.detect_lidar import align_box_to_board

        turned, note = align_box_to_board(self._single, self._scene.filter)
        if turned is self._scene.filter:
            self.window().statusBar().showMessage(f"회전하지 못했습니다 — {note}", 6000)
            return
        self._scene.filter = turned
        self._load_box()
        self._focus_box()
        self._draw()
        self._mark_stale()
        self.changed.emit()
        self.window().statusBar().showMessage(f"보드 평면에 맞췄습니다 — {note}", 7000)

    def _focus_box(self):
        if self._scene is not None and self._scene.filter.is_set:
            self.view.focus_on(self._scene.filter.as_tuple())

    # ------------------------------------------------------------------- box

    def _fields(self) -> dict[str, float]:
        """What the panel currently shows."""
        return {key: spin.value() for key, spin in self.spins.items()}

    def _load_box(self):
        for key, v in box_to_fields(self._scene.filter).items():
            scale = 10.0 if key in ("yaw", "pitch", "roll") else 100.0
            self.spins[key].blockSignals(True)
            self.sliders[key].blockSignals(True)
            self.spins[key].setValue(v)
            self.sliders[key].setValue(int(v * scale))
            self.spins[key].blockSignals(False)
            self.sliders[key].blockSignals(False)
        self._update_roi_line()

    def _reset_box(self):
        """Start at the sensor origin with a fixed 5 m cube.

        Two earlier defaults were worse. Fitting the bounds to the cloud's extent
        put the centre on whatever the scene happened to contain -- 3.24, -1.98,
        -0.06 on one capture -- and those digits give nothing to reason from.
        Keeping the centre at zero but sizing to the cloud was better, yet on a
        merged recording that reaches tens of metres it still means the first job
        is always shrinking, from 50 m down to under a metre.

        A 5 m cube is already the right order for a board, so adjusting starts
        near the answer. The sensor sits at the origin of x and y, so those centre
        fields read directly as distance from the LiDAR: type 3 into x and the box
        is 3 m ahead. Height is the exception -- see _DEFAULT_CENTRE.
        """
        if self._scene is None:
            return
        box = self._scene.filter
        half = _DEFAULT_SIZE / 2.0
        for axis, centre in zip("xyz", _DEFAULT_CENTRE):
            setattr(box, f"{axis}_min", centre - half)
            setattr(box, f"{axis}_max", centre + half)
        box.yaw = box.pitch = box.roll = 0.0
        self._load_box()
        self._draw()
        self._mark_stale()

    def _on_spin(self, key: str, value: float, scale: float = 100.0):
        self.sliders[key].blockSignals(True)
        self.sliders[key].setValue(int(value * scale))
        self.sliders[key].blockSignals(False)
        self._box_changed()

    def _on_slider(self, key: str, value: int, scale: float = 100.0):
        v = value / scale
        self.spins[key].blockSignals(True)
        self.spins[key].setValue(v)
        self.spins[key].blockSignals(False)
        self._box_changed()

    def _box_changed(self):
        if self._scene is None:
            return
        fields_to_box(self._scene.filter, self._fields())
        self._update_roi_line()
        self._draw()  # box and colouring follow immediately; detection does not
        self._mark_stale()
        self.changed.emit()

    def _update_roi_line(self):
        if self._scene is None:
            self.roi_line.clear()
            return
        b = self._scene.filter
        text = " ".join(f"{v:.2f}" for v in b.as_tuple())
        if b.rotated:
            text += "   " + " ".join(f"{v:.1f}" for v in (b.yaw, b.pitch, b.roll))
        self.roi_line.setText(text)

    def _copy_roi(self):
        if self.roi_line.text():
            QtWidgets.QApplication.clipboard().setText(self.roi_line.text())
            self.window().statusBar().showMessage(f"복사됨: {self.roi_line.text()}", 4000)

    def _mark_stale(self):
        """Say the shown result no longer matches the box, without recomputing."""
        if self._det is None:
            self.results.setText(
                '<span style="color:palette(mid)">박스를 보드 주변으로 좁힌 뒤 '
                "<b>원 검출</b>을 누르세요.</span>"
            )
        else:
            self._det = None
            self.view.set_markers("edges", None, (0, 0, 0, 0))
            self.view.set_markers("centers", None, (0, 0, 0, 0))
            self.results.setText(
                '<span style="color:palette(mid)">박스가 바뀌었습니다 — <b>원 검출</b>을 다시 누르세요.</span>'
            )

    def _copy_to_others(self):
        if self._scene is None:
            return
        others = [s for s in self.project.scenes if s is not self._scene]
        if not others:
            QtWidgets.QMessageBox.information(
                self, "복사", "복사할 다른 scene 이 없습니다."
            )
            return
        dlg = CopyTargetDialog(self, self._scene, others)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        picked = dlg.chosen()
        if not picked:
            return
        # Both halves of the ROI travel together: `Scene.roi()` prefers `region`
        # when it is set, so copying the box alone would land on a scene that
        # then ignores it.
        src_region = self._scene.region
        for sc in picked:
            sc.filter = FilterBox(**vars(self._scene.filter))
            # A fresh list per scene: `corners` is four nested [x, y, z] lists, so
            # handing over the same object would make editing one scene's corner
            # silently move every copy's.
            sc.region = PlaneRegionSpec(
                corners=[list(c) for c in src_region.corners],
                thickness=src_region.thickness,
            )
        self.window().statusBar().showMessage(
            f"{self._scene.id} 의 영역을 {len(picked)}개 scene 에 복사했습니다: "
            + ", ".join(sc.id for sc in picked),
            6000,
        )
        self.changed.emit()

    # ---------------------------------------------------------- auto placing

    def _auto_place(self):
        p = self.project
        drawn = [sc for sc in p.scenes if sc.roi_set]
        todo = [sc for sc in p.scenes if not sc.roi_set]
        if not drawn:
            QtWidgets.QMessageBox.information(
                self, "자동 배치",
                "먼저 scene 하나에 박스를 손으로 맞추고 원 검출을 성공시켜 주세요.\n"
                "그 하나로 나머지를 놓습니다.")
            return
        if not todo:
            QtWidgets.QMessageBox.information(
                self, "자동 배치", "모든 scene 에 이미 박스가 있습니다.\n"
                "다시 놓으려면 그 scene 의 박스를 먼저 지우세요.")
            return
        if not p.camera.is_set or not p.camera_topic:
            QtWidgets.QMessageBox.warning(
                self, "자동 배치",
                "카메라 intrinsic 과 카메라 토픽이 있어야 합니다 (1·2단계).")
            return
        answer = QtWidgets.QMessageBox.question(
            self, "자동 배치",
            f"박스가 그려진 {len(drawn)}개 scene 으로 extrinsic 을 풀고,\n"
            f"박스가 없는 {len(todo)}개 scene 에 박스를 놓습니다.\n\n"
            "이미 그려둔 박스는 건드리지 않습니다. 녹화를 다시 읽으므로 "
            "scene 수만큼 시간이 걸립니다.\n\n진행할까요?")
        if answer != QtWidgets.QMessageBox.Yes:
            return

        self.auto_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.results.setText('<span style="color:palette(mid)">자동 배치 준비 중…</span>')
        jobs = [(sc, p.bag_for(sc)) for sc in p.scenes]
        self.request_sweep.emit(
            jobs, p.lidar_topic, p.camera_topic, p.camera, p.target,
            self.method_combo.currentData(),
        )

    def _on_sweep_progress(self, message: str):
        self.results.setText(f'<span style="color:palette(mid)">{message}</span>')

    def _on_sweep_failed(self, message: str):
        self.auto_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.results.setText(f'<span style="color:#d9534f">{message}</span>')

    def _on_sweep_done(self, boxes: dict, summary: str):
        self.auto_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        by_id = {sc.id: sc for sc in self.project.scenes}
        for sid, box in boxes.items():
            sc = by_id.get(sid)
            if sc is None:
                continue
            sc.filter = box
            # `Scene.roi()` prefers `region` when it is set, so a stale one from
            # an earlier attempt would quietly win over the box just placed.
            sc.region = PlaneRegionSpec()
        QtWidgets.QMessageBox.information(self, "자동 배치", summary)
        if boxes:
            self.changed.emit()
            self._load_box()
            self._draw()
        self.results.setText(
            f'<span style="color:palette(mid)">{len(boxes)}개 scene 에 박스를 놓았습니다.</span>')

    # -------------------------------------------------------------- detection

    def _run_detect(self):
        if self._cloud is None or self._scene is None:
            return
        # A very wide box is allowed but slow, so confirm rather than silently
        # grinding for several seconds.
        inside = int(box_mask(self._cloud, self._scene.filter).sum())
        if inside > 400_000:
            answer = QtWidgets.QMessageBox.question(
                self, "원 검출",
                f"박스 안에 {inside:,} 점이 있습니다. 몇 초 걸릴 수 있습니다.\n"
                "보드 주변으로 좁히면 훨씬 빠릅니다. 그래도 진행할까요?",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self.detect_btn.setEnabled(False)
        self.results.setText('<span style="color:palette(mid)">검출 중…</span>')
        roi = FilterBox(**{k: getattr(self._scene.filter, k) for k in
                           ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
                            "yaw", "pitch", "roll")})
        # Spacing from a single sweep inside the region, then scale the constants.
        if self._single is not None:
            inside = apply_region(self._single, roi)
            self._spacing = point_spacing(inside) if len(inside) > 20 else 0.0
        self._params = derive_params(self._spacing)
        self._params.method = self.method_combo.currentData()
        self._gen += 1
        self.request_detect.emit(
            self._gen, self._cloud, roi, self.project.target, self._params, self._ring
        )

    def _on_detected(self, gen: int, det: LidarDetection):
        if gen != self._gen:
            return
        self._det = det
        self.detect_btn.setEnabled(True)
        self._show_results(det)
        self._draw(det)

    def _on_failed(self, gen: int, message: str):
        self.detect_btn.setEnabled(True)
        if gen == self._gen:
            self.results.setText(f'<span style="color:#d9534f">{message}</span>')

    def _show_results(self, det: LidarDetection):
        def row(label, value, warn=""):
            tail = f' <span style="color:#d9534f">{warn}</span>' if warn else ""
            return f'<tr><td style="padding-right:12px">{label}</td><td>{value}{tail}</td></tr>'

        spacing_mm = self._spacing * 1000
        rows = [
            row("박스 안 점", f"{len(det.filtered):,}"),
            row("점 간격 (단일 스윕)", f"{spacing_mm:.0f} mm",
                "너무 성김 — 보드를 가까이" if spacing_mm > 40 else ""),
        ]
        if len(det.plane):
            rows += [
                row("평면 점", f"{len(det.plane):,}"),
                row("평면 잔차", f"{det.plane_rms * 1000:.1f} mm", "큼" if det.plane_rms > 0.02 else ""),
            ]

        # The two methods reach the circles by different routes, so the numbers
        # worth watching differ. Showing cluster counts for a grid run (or a cell
        # size for a boundary run) would just be noise.
        if det.method in ("ring", "original"):
            rows.append(row("링 갭", f"{det.ring_gap * 1000:.0f} mm",
                            "원본 고정값" if det.method == "original" else ""))
            rows += [
                row("경계점", f"{len(det.edges):,}"),
                row("원 후보" if det.method == "original" else "클러스터", f"{det.n_clusters}"),
            ]
        elif det.method in ("occupancy", "experimental"):
            if det.cell_size:
                rows.append(row("격자 셀", f"{det.cell_size * 1000:.0f} mm"))
            rows.append(row("구멍 후보", f"{det.n_clusters}",
                            "4개 미만 — 박스 안에 보드 전체가 들어왔는지 확인" if det.n_clusters < 4 else ""))
        else:
            rows += [
                row("경계 검출 반경", f"{self._params.boundary_radius * 1000:.0f} mm"),
                row("경계점", f"{len(det.edges):,}"),
                row("클러스터", f"{det.n_clusters}"),
            ]

        mark = "✓" if det.ok else "✗"
        colour = "#2e9e4f" if det.ok else "#d9534f"
        rows.append(row("원 검출", f'<span style="color:{colour};font-weight:600">{mark} {det.n_circles} / 4</span>'))

        html = f"<table>{''.join(rows)}</table>"
        if det.radii:
            expected = self.project.target.circle_radius
            unit = "테두리 점" if det.method in ("occupancy", "experimental") else "인라이어"
            lines = "".join(
                f"<tr><td>{i}</td><td>{r * 1000:.0f} mm</td>"
                f"<td>({(r - expected) * 1000:+.0f})</td><td>{n} {unit}</td></tr>"
                for i, (r, n) in enumerate(zip(det.radii, det.edge_counts))
            )
            html += (
                f"<br><b>맞춘 원</b> (도면 {expected * 1000:.0f} mm)<table>{lines}</table>"
                '<div style="color:palette(mid);font-size:11px">빔 폭 때문에 관측 반지름은 '
                "도면보다 작거나 크게 나올 수 있습니다. 중심 위치는 그 영향을 거의 받지 않습니다.</div>"
            )
        if not det.ok and det.reason:
            html += f'<br><span style="color:#d9534f">{det.reason}</span>'
        self.results.setText(html)

    # ------------------------------------------------------------------- draw

    def _draw(self, det: LidarDetection | None = None):
        if self._cloud is None or self._scene is None:
            return
        box = self._scene.filter
        # Recolouring a million points takes a few hundred milliseconds, and
        # dragging a filter value does not change any point's colour -- only
        # which side of the box it is on. So the colours are computed once per
        # style change and cached, and the box edit just reapplies the mask.
        style = self.colors.style
        if self._base_colors is None or self._colored_style != style:
            self._base_colors, used = self.view.compute_colors(
                self._draw_xyz, self._draw_intensity, style
            )
            self._colored_style = style
            self.colors.report_range(used, self._intensity is not None)
        self.view.apply_colors(
            self._draw_xyz, self._base_colors,
            keep_mask=box.mask(self._draw_xyz),
            hide_outside=self.hide_outside.isChecked(),
        )
        self.view.set_box(box.corners() if box.rotated else box.as_tuple())
        if not isinstance(det, LidarDetection):
            det = self._det
        if det is None:
            return
        self.view.set_markers(
            "edges", det.edges if self.show_edges.isChecked() else None, (1.0, 0.85, 0.2, 1.0), 4.0
        )
        self.view.set_markers("centers", det.centers, (1.0, 0.2, 0.9, 1.0), 16.0)

    # ------------------------------------------------------------------ state

    def shutdown(self):
        self._sweep_thread.quit()
        self._sweep_thread.wait(3000)
        self._thread.quit()
        self._thread.wait(3000)

    def is_complete(self) -> bool:
        return bool(self.project.scenes) and all(s.filter.is_set for s in self.project.scenes)

    def status_text(self) -> str:
        if not self.project.scenes:
            return "scene 없음"
        done = sum(1 for s in self.project.scenes if s.filter.is_set)
        return f"{done}/{len(self.project.scenes)} scene 설정됨"

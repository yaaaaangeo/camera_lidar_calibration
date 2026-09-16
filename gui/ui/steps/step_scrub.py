"""Step 4 — pick moments out of the recordings.

Bags are never cut. You scrub the timeline, and the frames you keep are recorded
as (bag, timestamp). That is the whole point: no trimming, no exporting images,
no editing a config between runs.

Both recording habits work. One long take with the board carried around means
scrubbing within a single bag; a short bag per board position means stepping
through the bag list, capturing one scene from each. They can be mixed.

Detection runs on whatever frame is showing, so you find out immediately whether
a moment is usable instead of after a full calibration pass.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pathlib import Path

from gui.core.bag_reader import BagSource
from gui.core.decode import image_to_bgr
from gui.core.detect_camera import CameraDetection, detect, draw_overlay
from gui.core.project import Project, Scene
from gui.ui.steps import StepPage


class _Worker(QtCore.QObject):
    """Loads and analyses one frame at a time, off the UI thread."""

    ready = QtCore.Signal(int, object, object)  # generation, bgr image, CameraDetection
    failed = QtCore.Signal(int, str)
    scanned_full = QtCore.Signal(object)          # [(t_ns, CameraDetection)]
    scan_progress = QtCore.Signal(int, int)       # done, total

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None
        self._topic = ""

    def setup(self, bag_path: str, topic: str):
        if self._src is not None:
            self._src.close()
        self._src = BagSource(bag_path)
        self._src.open()
        self._topic = topic

    def load(self, gen: int, t_ns: int, camera, target):
        if self._src is None:
            return
        try:
            _, msg = self._src.first_after(self._topic, t_ns)
            if msg is None:
                self.failed.emit(gen, "해당 시각에 이미지가 없습니다")
                return
            img = image_to_bgr(msg)
            det = detect(img, camera, target)
            self.ready.emit(gen, img, det)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def scan_full(self, times: list, camera, target):
        """Full detection at every given time, for choosing scenes.

        There used to be a cheaper pass ahead of this one, counting markers to
        find the stretches worth looking at properly. It was removed: the caller
        already thins the whole recording to a few hundred instants, which is
        cheap enough on its own, and a button whose only output was a coloured
        strip is a button to press before the one that does the work.
        """
        if self._src is None:
            return
        out = []
        total = len(times)
        for i, t in enumerate(times):
            try:
                _, msg = self._src.first_after(self._topic, t)
                if msg is not None:
                    out.append((t, detect(image_to_bgr(msg), camera, target)))
            except Exception:  # noqa: BLE001 - a bad frame is simply not a candidate
                pass
            if (i + 1) % 10 == 0 or i + 1 == total:
                self.scan_progress.emit(i + 1, total)
        self.scanned_full.emit(out)


class _Strip(QtWidgets.QWidget):
    """Marker-count bars under the timeline. Shows where to look; picks nothing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        self.samples: list[tuple[int, int]] = []
        self.marks: list[int] = []
        self.t0 = self.t1 = 0

    def set_marks(self, marks):
        self.marks = list(marks)
        self.update()

    def set_range(self, t0: int, t1: int):
        self.t0, self.t1 = t0, t1
        self.update()

    def set_samples(self, samples):
        self.samples = samples
        self.update()

    def paintEvent(self, _):
        if not self.samples or self.t1 <= self.t0:
            return
        p = QtGui.QPainter(self)
        span = self.t1 - self.t0
        w = max(self.width() / max(len(self.samples), 1), 1.0)
        for t, n in self.samples:
            x = (t - self.t0) / span * self.width()
            if n >= 4:
                c = QtGui.QColor(60, 190, 90)
            elif n >= 1:
                c = QtGui.QColor(210, 170, 60)
            else:
                c = QtGui.QColor(0, 0, 0, 20)
            p.fillRect(QtCore.QRectF(x, 0, w, self.height()), c)
        for t in self.marks:
            x = (t - self.t0) / span * self.width()
            p.fillRect(QtCore.QRectF(x - 1, 0, 2, self.height()), QtGui.QColor(40, 40, 40))


class SuggestDialog(QtWidgets.QDialog):
    """The moments the scan proposes, for the user to confirm.

    Nothing is added until this is accepted. Capturing is easy to undo but easy
    to lose track of, and a list of eighteen scenes that appeared on their own is
    harder to trust than one that was ticked through.
    """

    def __init__(self, parent, chosen, taken: set, t0: int, cov=None):
        super().__init__(parent)
        self.setWindowTitle("scene 자동 추천")
        self._rows: list[tuple[QtWidgets.QTableWidgetItem, object]] = []

        self.table = QtWidgets.QTableWidget(len(chosen), 8)
        self.table.setHorizontalHeaderLabels(
            ["사용", "시각", "거리", "기울기", "재투영", "짝 어긋남", "안전 누적", "비고"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for r, c in enumerate(chosen):
            dup = c.t_ns in taken
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Unchecked if dup else QtCore.Qt.Checked)
            self.table.setItem(r, 0, chk)
            cells = [
                f"{(c.t_ns - t0) / 1e9:.2f} s",
                f"{c.distance:.2f} m",
                f"{c.tilt_deg:.0f}°",
                f"{c.reproj_mm:.1f} mm",
                f"{c.pair_mm:.1f} mm",
                f"{c.safe_frames}프레임",
                "이미 캡처됨" if dup else c.note,
            ]
            for col, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                if dup:
                    item.setForeground(QtGui.QColor("#999"))
                self.table.setItem(r, col, item)
            self._rows.append((chk, c))
        self.table.resizeColumnsToContents()
        self.table.itemChanged.connect(lambda *_: self._update_count())

        head = QtWidgets.QLabel(
            "마커가 다 보이고, 이미지와 스윕이 같은 자리를 보고 있고, "
            "서로 자세가 다른 순간들입니다.<br>"
            "<span style='color:palette(mid)'>확인하고 필요 없는 것은 체크를 푸세요. "
            "누르기 전에는 아무것도 추가되지 않습니다.</span>"
        )
        head.setWordWrap(True)

        # What the set spans -- the part choosing better cannot fix. A recording
        # where the board never left the middle of the frame stays that way
        # however the moments are picked, and the only thing to do about it is
        # record again, so it has to be said out loud rather than left implied.
        self.cover = QtWidgets.QLabel()
        self.cover.setWordWrap(True)
        if cov is not None:
            warn = []
            if cov.y_frac < 0.35:
                warn.append(f"세로 {cov.y_frac*100:.0f}%")
            if cov.x_frac < 0.35:
                warn.append(f"가로 {cov.x_frac*100:.0f}%")
            tail = ("<br><span style='color:#d9534f'>보드가 화면의 "
                    + ", ".join(warn) + " 안에만 있었습니다 — 고르기로는 넓힐 수 없고, "
                    "다시 취득해야 넓어집니다.</span>") if warn else ""
            self.cover.setText(
                f"<span style='color:palette(mid)'>화면 커버리지 가로 {cov.x_frac*100:.0f}% "
                f"세로 {cov.y_frac*100:.0f}% ({cov.cells}/9칸) · "
                f"거리 {cov.distance[0]:.1f}~{cov.distance[1]:.1f} m · "
                f"기울기 폭 {cov.tilt_range:.0f}° · 조건 {cov.condition:.1f}</span>{tail}"
            )

        all_btn = QtWidgets.QPushButton("전체 선택")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QtWidgets.QPushButton("전체 해제")
        none_btn.clicked.connect(lambda: self._set_all(False))
        self.count = QtWidgets.QLabel()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok.setText("추가")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        row.addWidget(self.count)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(head)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.cover)
        lay.addLayout(row)
        lay.addWidget(buttons)
        self.resize(820, 480)
        self._update_count()

    def _set_all(self, on: bool):
        state = QtCore.Qt.Checked if on else QtCore.Qt.Unchecked
        for chk, _ in self._rows:
            chk.setCheckState(state)

    def _update_count(self):
        n = len(self.chosen())
        self.count.setText(f"{n}개 선택")
        self.ok.setEnabled(bool(n))

    def chosen(self):
        return [c for chk, c in self._rows if chk.checkState() == QtCore.Qt.Checked]


class ScrubberStep(StepPage):
    title = "4. Scene 캡처"
    subtitle = "타임라인에서 시점 고르기"

    # Signals rather than invokeMethod: Qt's int is 32-bit and cannot carry a
    # nanosecond timestamp, and Q_ARG has no meta type for plain Python objects.
    request_setup = QtCore.Signal(str, str)
    request_load = QtCore.Signal(int, object, object, object)
    request_scan_full = QtCore.Signal(object, object, object)

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self._gen = 0
        self._times: list[int] = []
        # Coarse-pass results, reused to aim the full pass at the stretches
        # where markers were actually seen.
        self._suggest_running = False
        self._image: np.ndarray | None = None
        self._sweep_cache: dict[str, float] = {}
        self._det: CameraDetection | None = None
        self._loaded_for = ("", "")
        self._bag_start: dict[str, int] = {}

        # --- bag picker ------------------------------------------------------
        self.bag_combo = QtWidgets.QComboBox()
        self.bag_combo.currentIndexChanged.connect(self._switch_bag)

        # --- preview --------------------------------------------------------
        self.view = QtWidgets.QLabel("1단계에서 bag과 토픽을 지정한 뒤 이 단계로 오세요.")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(560, 380)
        self.view.setStyleSheet("background: palette(base); color: palette(mid);")

        # --- timeline -------------------------------------------------------
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.strip = _Strip()

        self.time_label = QtWidgets.QLabel("—")
        self.time_label.setFixedWidth(150)
        self.time_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        prev_btn = QtWidgets.QToolButton()
        prev_btn.setText("◀")
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn = QtWidgets.QToolButton()
        next_btn.setText("▶")
        next_btn.clicked.connect(lambda: self._step(+1))

        self.capture_btn = QtWidgets.QPushButton("이 시점 캡처  (Space)")
        self.capture_btn.clicked.connect(self._capture)
        self.capture_btn.setEnabled(False)


        self.suggest_btn = QtWidgets.QPushButton("scene 자동 추천…")
        self.suggest_btn.setToolTip(
            "마커가 보이는 구간을 훑어 서로 자세가 다른 순간들을 골라 옵니다.\n"
            "고른 것을 표로 보여주고, 확인해야 추가됩니다"
        )
        self.suggest_btn.clicked.connect(self._suggest)
        self.suggest_btn.setEnabled(False)

        self._want_spin = QtWidgets.QSpinBox()
        self._want_spin.setRange(2, 40)
        self._want_spin.setValue(6)
        self._want_spin.setPrefix("최대 ")
        self._want_spin.setSuffix("개")
        self._want_spin.setToolTip("추천받을 scene 개수. 자세가 서로 다른 것부터 고릅니다")

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(prev_btn)
        bar.addWidget(self.slider, 1)
        bar.addWidget(next_btn)
        bar.addWidget(self.time_label)

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.capture_btn)
        actions.addWidget(self.suggest_btn)
        actions.addWidget(self._want_spin)
        actions.addStretch(1)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.bag_combo)
        left.addWidget(self.view, 1)
        left.addLayout(bar)
        left.addWidget(self.strip)
        left.addLayout(actions)

        # --- quality + captured list ----------------------------------------
        self.quality = QtWidgets.QLabel("—")
        self.quality.setWordWrap(True)
        self.quality.setAlignment(QtCore.Qt.AlignTop)
        self.quality.setTextFormat(QtCore.Qt.RichText)
        self.quality.setMinimumHeight(150)

        self.scene_list = QtWidgets.QListWidget()
        # Clicking a capture jumps back to the moment it was taken. Without this a
        # scene can be captured but never revisited: checking whether it was a
        # good pick means hunting for that instant on the timeline again, so the
        # cheaper move becomes "delete it and recapture". Looking first is cheaper
        # still.
        self.scene_list.currentRowChanged.connect(self._goto_scene)
        self.scene_list.setToolTip("클릭하면 그 scene 을 캡처한 시점으로 돌아갑니다")
        remove_btn = QtWidgets.QPushButton("선택 삭제")
        remove_btn.clicked.connect(self._remove)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("현재 프레임"))
        right.addWidget(self.quality)
        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("캡처된 scene"))
        right.addWidget(self.scene_list, 1)
        right.addWidget(remove_btn)

        right_box = QtWidgets.QWidget()
        right_box.setLayout(right)
        right_box.setFixedWidth(320)

        row = QtWidgets.QHBoxLayout(self)
        row.addLayout(left, 1)
        row.addWidget(right_box)

        # --- worker ---------------------------------------------------------
        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.scanned_full.connect(self._on_scanned_full)
        self._worker.scan_progress.connect(self._on_scan_progress)
        self.request_setup.connect(self._worker.setup)
        self.request_load.connect(self._worker.load)
        self.request_scan_full.connect(self._worker.scan_full)
        self._thread.start()

        self._debounce = QtCore.QTimer(self, singleShot=True, interval=120)
        self._debounce.timeout.connect(self._request)

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self._capture)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, activated=lambda: self._step(-1))
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=lambda: self._step(+1))

        self._refresh_scene_list()

    # ------------------------------------------------------------------ setup

    def on_enter(self):
        p = self.project
        if not p.bag_paths or not p.camera_topic:
            self.view.setText("1단계에서 bag과 카메라 토픽을 지정하세요.")
            return

        names = [Path(b).name for b in p.bag_paths]
        if [self.bag_combo.itemText(i) for i in range(self.bag_combo.count())] != names:
            keep = self.bag_combo.currentIndex()
            self.bag_combo.blockSignals(True)
            self.bag_combo.clear()
            self.bag_combo.addItems(names)
            self.bag_combo.setCurrentIndex(max(0, min(keep, len(names) - 1)))
            self.bag_combo.blockSignals(False)
        self.bag_combo.setVisible(len(names) > 1)
        self._load_current_bag()

    def _sweep_s(self) -> float:
        """Seconds between LiDAR sweeps in the bag being scrubbed."""
        return self._sweep_cache.get(self._current_bag(), 0.0)

    def _image_size(self):
        """(width, height) of the frame on screen, for reporting coverage against."""
        if self._image is None:
            return None
        h, w = self._image.shape[:2]
        return (w, h)

    def _current_bag(self) -> str:
        i = self.bag_combo.currentIndex()
        paths = self.project.bag_paths
        return paths[i] if 0 <= i < len(paths) else (paths[0] if paths else "")

    def _switch_bag(self):
        self._load_current_bag()

    def _load_current_bag(self):
        p = self.project
        bag = self._current_bag()
        key = (bag, p.camera_topic)
        if key == self._loaded_for:
            self._request()
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            with BagSource(bag) as src:
                self._times = sorted(src.timestamps(p.camera_topic))
                # Read while the bag is already open. The picker needs the sweep
                # interval, not the sweeps: how far apart they are is what says
                # how stale the cloud paired with an image can be.
                if p.lidar_topic and bag not in self._sweep_cache:
                    lt = sorted(src.timestamps(p.lidar_topic))
                    self._sweep_cache[bag] = (
                        float(np.median(np.diff(lt)) / 1e9) if len(lt) > 2 else 0.0
                    )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self.view.setText(f"{Path(bag).name} 을 읽지 못했습니다.\n{type(exc).__name__}: {exc}")
            self.slider.setEnabled(False)
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        if not self._times:
            self.view.setText(f"{Path(bag).name} 에 '{p.camera_topic}' 이미지가 없습니다.")
            self.slider.setEnabled(False)
            return

        self._loaded_for = key
        self._bag_start[bag] = self._times[0]
        self.request_setup.emit(bag, p.camera_topic)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self._times) - 1)
        self.slider.setValue(len(self._times) // 2)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        self.suggest_btn.setEnabled(True)
        self.strip.set_range(self._times[0], self._times[-1])
        self.strip.set_samples([])
        self._refresh_scene_list()
        self._request()

    # --------------------------------------------------------------- browsing

    def _on_slider(self):
        self._update_time_label()
        self._debounce.start()

    def _step(self, delta: int):
        if self._times:
            self.slider.setValue(self.slider.value() + delta)

    def _update_time_label(self):
        if not self._times:
            return
        i = self.slider.value()
        t = (self._times[i] - self._times[0]) / 1e9
        self.time_label.setText(f"{t:7.2f} s   [{i + 1}/{len(self._times)}]")

    def _request(self):
        if not self._times:
            return
        self._gen += 1
        self._update_time_label()
        self.request_load.emit(
            self._gen, self._times[self.slider.value()], self.project.camera, self.project.target
        )

    def _on_ready(self, gen: int, image, det: CameraDetection):
        if gen != self._gen:
            return  # a newer request already went out
        self._image, self._det = image, det
        self.capture_btn.setEnabled(det.ok)
        self._render()
        self._show_quality(det)

    def _on_failed(self, gen: int, message: str):
        if gen == self._gen:
            self.view.setText(message)

    def _render(self):
        if self._image is None:
            return
        img = draw_overlay(self._image, self._det, self.project.camera) if self._det else self._image
        h, w = img.shape[:2]
        qimg = QtGui.QImage(
            np.ascontiguousarray(img[:, :, ::-1]).data, w, h, 3 * w, QtGui.QImage.Format_RGB888
        )
        self.view.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # --------------------------------------------------------------- quality

    def _show_quality(self, det: CameraDetection):
        def line(label, value, warn=""):
            colour = "#d9534f" if warn else "palette(text)"
            tail = f' <span style="color:#d9534f">{warn}</span>' if warn else ""
            return f'<tr><td style="padding-right:10px">{label}</td><td style="color:{colour}">{value}{tail}</td></tr>'

        rows = [line("마커", f"{det.n_markers} / 4", "" if det.n_markers == 4 else "부족")]
        if det.marker_px:
            rows.append(line("마커 크기", f"{det.marker_px:.0f} px", "작음" if det.marker_px < 40 else ""))
        if det.ok:
            rows += [
                line("거리", f"{det.distance:.2f} m"),
                line("기울기", f"{det.tilt_deg:.0f}°", "정면에 가까움" if det.tilt_deg < 12 else ""),
                line("재투영", f"{det.reproj_rms:.2f} px", "큼" if det.reproj_rms > 1.5 else ""),
            ]
        else:
            rows.append(line("상태", det.reason))
        self.quality.setText(f"<table>{''.join(rows)}</table>")

    # --------------------------------------------------------------- capture

    def _capture(self):
        if not self._det or not self._det.ok or not self._times:
            return
        t_ns = self._times[self.slider.value()]
        bag = self._current_bag()
        if any(s.t_ns == t_ns and self.project.bag_for(s) == bag for s in self.project.scenes):
            return
        self.project.scenes.append(Scene(id=self._new_scene_id(), t_ns=t_ns, bag=bag))
        self._refresh_scene_list()
        self.changed.emit()

    def _new_scene_id(self) -> str:
        """Ids are never reused. Later steps key their results on the id, so
        renumbering after a delete would silently attach one scene's detection
        to another."""
        used = {s.id for s in self.project.scenes}
        n = 1
        while f"s{n:02d}" in used:
            n += 1
        return f"s{n:02d}"

    def _goto_scene(self, row: int):
        """Move the timeline to the moment a captured scene came from.

        Only meaningful for scenes from the bag currently loaded -- a project can
        mix recordings, and jumping to a timestamp that belongs to a different
        file would land somewhere arbitrary.
        """
        if not (0 <= row < len(self.project.scenes)) or not self._times:
            return
        scene = self.project.scenes[row]
        if self.project.bag_for(scene) != self._current_bag():
            self.window().statusBar().showMessage(
                f"{scene.id} 은 다른 bag 의 scene 입니다 — 위에서 그 bag 을 선택하세요.", 5000
            )
            return
        # Nearest frame rather than exact match: the scene was captured on the
        # camera's clock, and the slider steps through camera frames.
        idx = min(range(len(self._times)), key=lambda i: abs(self._times[i] - scene.t_ns))
        if idx != self.slider.value():
            self.slider.setValue(idx)  # triggers the debounced reload
        else:
            self._request()

    def _remove(self):
        row = self.scene_list.currentRow()
        if 0 <= row < len(self.project.scenes):
            del self.project.scenes[row]
            self._refresh_scene_list()
            self.changed.emit()

    def _refresh_scene_list(self):
        from pathlib import Path

        self.scene_list.clear()
        multi = len(self.project.bag_paths) > 1
        for s in self.project.scenes:
            bag = self.project.bag_for(s)
            base = self._bag_start.get(bag, s.t_ns)
            label = f"{s.id}    {(s.t_ns - base) / 1e9:6.2f} s"
            if multi:
                label += f"    {Path(bag).name}"
            self.scene_list.addItem(label)

    # ------------------------------------------------------------------ scan

    def _suggest(self):
        """Scan the marker-visible stretches, then offer what they contain."""
        if not self._times:
            return
        cam = self.project.camera
        if not cam.is_set:
            QtWidgets.QMessageBox.information(
                self, "scene 자동 추천", "2단계에서 카메라 내부 파라미터를 먼저 넣으세요."
            )
            return
        # Thin the whole recording rather than aim at part of it. Four hundred
        # instants is a few seconds of detection, and a 30 Hz recording of four
        # minutes is 6600 frames -- running the pose fit on every one of them
        # would take minutes for candidates that sit milliseconds apart.
        times = list(self._times)
        step = max(len(times) // 400, 1)
        times = times[::step]

        self._suggest_running = True
        self.suggest_btn.setEnabled(False)
        self.suggest_btn.setText("훑는 중…")
        self.request_scan_full.emit(times, cam, self.project.target)

    def _on_scan_progress(self, done: int, total: int):
        if getattr(self, "_suggest_running", False):
            self.suggest_btn.setText(f"훑는 중… {done}/{total}")

    def _on_scanned_full(self, samples):
        from gui.core import scene_pick, verify

        self._suggest_running = False
        self.suggest_btn.setEnabled(True)
        self.suggest_btn.setText("scene 자동 추천…")
        if not samples:
            QtWidgets.QMessageBox.information(self, "scene 자동 추천", "쓸 만한 프레임이 없습니다.")
            return

        # The strip's bars come from this pass now, since it is the only one that
        # looks at the whole recording. Free -- the detections are already in
        # hand -- and set before anything can return early, because a run that
        # chose nothing is exactly when seeing where the markers were helps.
        self.strip.set_samples([(t, d.n_markers if d.ok else 0) for t, d in samples])

        limit = verify.radial_limit(self.project.camera)
        res = scene_pick.pick(
            samples, radial_limit=limit, want=self._want_spin.value(),
            camera=self.project.camera, sweep_s=self._sweep_s(), size=self._image_size(),
        )
        if not res.chosen:
            why = ", ".join(f"{k} {v}" for k, v in sorted(res.rejected.items()))
            QtWidgets.QMessageBox.information(
                self, "scene 자동 추천",
                f"{res.n_scanned}프레임을 훑었지만 쓸 수 있는 것이 없습니다.\n{why}",
            )
            return

        bag = self._current_bag()
        taken = {s.t_ns for s in self.project.scenes if self.project.bag_for(s) == bag}
        self.strip.set_marks([c.t_ns for c in res.chosen])
        dlg = SuggestDialog(self, res.chosen, taken, self._times[0], res.coverage)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        added = 0
        for c in dlg.chosen():
            if c.t_ns in taken:
                continue
            self.project.scenes.append(Scene(id=self._new_scene_id(), t_ns=c.t_ns, bag=bag))
            taken.add(c.t_ns)
            added += 1
        self._refresh_scene_list()
        self.changed.emit()
        self.window().statusBar().showMessage(
            f"scene {added}개 추가 · 훑은 프레임 {res.n_scanned}, 쓸 수 있는 것 {res.n_usable}, "
            f"정지 구간 {len(res.holds)}, 조건 {res.coverage.condition:.1f}", 8000
        )

    # ----------------------------------------------------------------- state

    def shutdown(self):
        self._debounce.stop()
        self._thread.quit()
        self._thread.wait(3000)

    def is_complete(self) -> bool:
        return len(self.project.scenes) >= 1

    def status_text(self) -> str:
        n = len(self.project.scenes)
        return f"scene {n}개" if n else "캡처 없음"

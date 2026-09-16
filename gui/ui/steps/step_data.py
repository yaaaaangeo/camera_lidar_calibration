"""Step 1 — choose the recordings and say which topics to use.

Two recording habits are both common: one long take with the board carried
around, or a short bag per board position. So this takes a *list* of bags, and
the scrubber lets you move between them.

Topics are never picked automatically. A recording can easily hold ten
PointCloud2 topics (raw per-sensor, merged, ground/no-ground, debug), and
guessing would be wrong more often than right. The table is there so you do not
have to remember the exact string, but the choice stays with you.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from gui.core.bag_reader import BagInfo, inspect
from gui.core.project import Project
from gui.ui.steps import StepPage


class _InspectWorker(QtCore.QThread):
    done = QtCore.Signal(object, object)  # list[BagInfo], list[str] errors

    def __init__(self, paths: list[str]):
        super().__init__()
        self.paths = paths

    def run(self):
        infos, errors = [], []
        for path in self.paths:
            try:
                infos.append(inspect(path))
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                errors.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
        self.done.emit(infos, errors)


class DataStep(StepPage):
    title = "1. 데이터"
    subtitle = "bag 파일과 사용할 토픽을 지정합니다"

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self.infos: list[BagInfo] = []
        self._worker: _InspectWorker | None = None

        # --- bag list -------------------------------------------------------
        self.bag_list = QtWidgets.QListWidget()
        self.bag_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.bag_list.setMaximumHeight(130)
        self.bag_list.currentRowChanged.connect(self._show_topics_for_current)

        add_btn = QtWidgets.QPushButton("bag 추가…")
        add_btn.setToolTip("ROS1 bag 파일(.bag)을 고릅니다")
        add_btn.clicked.connect(self._add_bags)
        # A ROS 2 recording is a folder holding metadata.yaml and the .db3/.mcap,
        # so a file dialog cannot reach it. Qt has no dialog that accepts both a
        # file and a folder, hence the second button.
        add_dir_btn = QtWidgets.QPushButton("ROS2 bag 폴더 추가…")
        add_dir_btn.setToolTip("metadata.yaml 이 들어 있는 폴더를 고릅니다")
        add_dir_btn.clicked.connect(self._add_bag_dir)
        remove_btn = QtWidgets.QPushButton("선택 제거")
        remove_btn.clicked.connect(self._remove_selected)

        bag_buttons = QtWidgets.QVBoxLayout()
        bag_buttons.addWidget(add_btn)
        bag_buttons.addWidget(add_dir_btn)
        bag_buttons.addWidget(remove_btn)
        bag_buttons.addStretch(1)

        bag_row = QtWidgets.QHBoxLayout()
        bag_row.addWidget(self.bag_list, 1)
        bag_row.addLayout(bag_buttons)

        self.summary = QtWidgets.QLabel("bag을 추가하세요. 여러 개를 한 번에 골라도 됩니다.")
        self.summary.setStyleSheet("color: palette(mid);")
        self.summary.setWordWrap(True)

        # --- topic table ----------------------------------------------------
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["토픽", "타입", "개수", "Hz", "종류"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setToolTip("줄을 더블클릭하면 타입에 맞는 칸으로 들어갑니다")
        self.table.itemSelectionChanged.connect(self._sync_assign_buttons)
        self.table.itemDoubleClicked.connect(self._assign_double_clicked)

        self.hide_noise = QtWidgets.QCheckBox("보조 토픽 숨기기 (status, parameter, rosout …)")
        self.hide_noise.setChecked(True)
        self.hide_noise.toggled.connect(self._show_topics_for_current)

        # --- project name ---------------------------------------------------
        # Names the vehicle, and ends up inside the extrinsic file. That file is
        # committed and read by other people, so it has to say what it belongs to
        # -- a filename alone gets renamed, copied, and separated from its meaning.
        self.name_edit = QtWidgets.QLineEdit(project.name if project.name != "untitled" else "")
        self.name_edit.setPlaceholderText("차량 이름 — 예: CA_County_01")
        self.name_edit.setToolTip(
            "프로젝트와 extrinsic 파일에 기록됩니다. 차량별로 구분되는 이름을 쓰세요."
        )
        self.name_edit.textChanged.connect(self._on_name_changed)

        # --- topic assignment ----------------------------------------------
        self.lidar_edit = QtWidgets.QLineEdit(project.lidar_topic)
        self.camera_edit = QtWidgets.QLineEdit(project.camera_topic)
        for e in (self.lidar_edit, self.camera_edit):
            e.setPlaceholderText("표에서 고르거나 직접 입력")
            e.textChanged.connect(self._on_topics_changed)

        self.assign_lidar = QtWidgets.QPushButton("← LiDAR로 지정")
        self.assign_camera = QtWidgets.QPushButton("← 카메라로 지정")
        self.assign_lidar.clicked.connect(lambda: self._assign(self.lidar_edit))
        self.assign_camera.clicked.connect(lambda: self._assign(self.camera_edit))

        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("차량 / 프로젝트 이름"), 0, 0)
        form.addWidget(self.name_edit, 0, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("LiDAR 토픽"), 1, 0)
        form.addWidget(self.lidar_edit, 1, 1)
        form.addWidget(self.assign_lidar, 1, 2)
        form.addWidget(QtWidgets.QLabel("카메라 토픽"), 2, 0)
        form.addWidget(self.camera_edit, 2, 1)
        form.addWidget(self.assign_camera, 2, 2)
        form.setColumnStretch(1, 1)

        self.topic_warning = QtWidgets.QLabel()
        self.topic_warning.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bag_row)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.hide_noise)
        layout.addLayout(form)
        layout.addWidget(self.topic_warning)

        self._sync_assign_buttons()
        if project.bag_paths:
            self._inspect(project.bag_paths)

    def _on_name_changed(self, text: str):
        self.project.name = text.strip() or "untitled"
        self.changed.emit()

    def on_enter(self):
        """Reread the project, so an opened file shows up here.

        This page only ever wrote outwards before: there was no way to load a
        project, so nothing could change its state from underneath. Opening one
        now leaves the bag list and topic fields showing whatever the previous
        session had, while the project itself holds something else.

        Inspecting reads the bag index, which is quick -- unlike listing every
        message time, it does not walk the file.
        """
        p = self.project
        listed = [str(i.path) for i in self.infos]
        if listed != list(p.bag_paths):
            self._inspect(list(p.bag_paths))
        shown = "" if p.name == "untitled" else p.name
        if self.name_edit.text() != shown:
            self.name_edit.blockSignals(True)
            self.name_edit.setText(shown)
            self.name_edit.blockSignals(False)
        for edit, value in ((self.lidar_edit, p.lidar_topic),
                            (self.camera_edit, p.camera_topic)):
            if edit.text() != value:
                edit.blockSignals(True)
                edit.setText(value)
                edit.blockSignals(False)
        self._sync_assign_buttons()

    # ------------------------------------------------------------------ load

    def _add_bags(self):
        start = str(Path(self.project.bag_path).parent) if self.project.bag_path else ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "bag 선택", start, "ROS bag (*.bag)")
        if not paths:
            return
        merged = list(self.project.bag_paths)
        merged += [p for p in paths if p not in merged]
        self._inspect(merged)

    def _add_bag_dir(self):
        start = str(Path(self.project.bag_path).parent) if self.project.bag_path else ""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "ROS2 bag 폴더 선택", start)
        if not path:
            return
        if not (Path(path) / "metadata.yaml").exists():
            QtWidgets.QMessageBox.warning(
                self, "ROS2 bag",
                f"{Path(path).name} 안에 metadata.yaml 이 없습니다.\n"
                "ROS2 bag 은 metadata.yaml 과 .db3/.mcap 이 함께 든 폴더입니다.",
            )
            return
        merged = list(self.project.bag_paths)
        if path not in merged:
            merged.append(path)
        self._inspect(merged)

    def _remove_selected(self):
        doomed = {self.bag_list.item(i.row()).data(QtCore.Qt.UserRole) for i in self.bag_list.selectedIndexes()}
        if not doomed:
            return
        used = {s.bag for s in self.project.scenes if s.bag}
        blocked = doomed & used
        if blocked:
            QtWidgets.QMessageBox.warning(
                self, "제거", "이 bag에서 캡처한 scene이 있습니다:\n" + "\n".join(Path(b).name for b in blocked)
            )
            return
        self._inspect([p for p in self.project.bag_paths if p not in doomed])

    def _inspect(self, paths: list[str]):
        if not paths:
            self.project.bag_paths = []
            self.infos = []
            self.bag_list.clear()
            self.table.setRowCount(0)
            self.summary.setText("bag을 추가하세요.")
            self.changed.emit()
            return
        # Replacing self._worker while the previous QThread is still running drops
        # its last reference, and Qt aborts the process rather than collecting a
        # live thread. Opening a project used to reach here twice in a row, which
        # is exactly that.
        running = getattr(self, "_worker", None)
        if running is not None and running.isRunning():
            if [str(x) for x in running.paths] == [str(x) for x in paths]:
                return  # same request already in flight
            running.wait(5000)

        self.summary.setText(f"{len(paths)}개 읽는 중…")
        self._worker = _InspectWorker(paths)
        self._worker.done.connect(self._loaded)
        self._worker.start()

    def _loaded(self, infos: list[BagInfo], errors: list[str]):
        self.infos = infos
        self.project.bag_paths = [str(i.path) for i in infos]

        self.bag_list.blockSignals(True)
        self.bag_list.clear()
        for info in infos:
            size_gb = info.path.stat().st_size / 1e9
            item = QtWidgets.QListWidgetItem(
                f"{info.path.name}      {size_gb:.2f} GB   {info.duration:.1f} s"
            )
            item.setData(QtCore.Qt.UserRole, str(info.path))
            item.setToolTip(str(info.path))
            self.bag_list.addItem(item)
        self.bag_list.blockSignals(False)
        if infos:
            self.bag_list.setCurrentRow(0)

        total_gb = sum(i.path.stat().st_size for i in infos) / 1e9
        total_s = sum(i.duration for i in infos)
        text = f"bag {len(infos)}개   |   {total_gb:.2f} GB   |   합계 {total_s:.1f} 초"
        if errors:
            text += "\n읽기 실패 — " + " / ".join(errors)
        self.summary.setText(text)

        self._show_topics_for_current()
        self._check_topics()
        self.changed.emit()

    # ---------------------------------------------------------------- topics

    def _current_info(self) -> BagInfo | None:
        row = self.bag_list.currentRow()
        return self.infos[row] if 0 <= row < len(self.infos) else None

    def _show_topics_for_current(self):
        info = self._current_info()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if info is None:
            self.table.setSortingEnabled(True)
            return

        rows = [t for t in info.topics if not self.hide_noise.isChecked() or t.kind != "-"]
        self.table.setRowCount(len(rows))
        for r, t in enumerate(rows):
            for c, text in enumerate([t.name, t.msgtype, f"{t.count:,}", f"{t.hz:.1f}", t.kind]):
                item = QtWidgets.QTableWidgetItem(text)
                if c in (2, 3):
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

    def _selected_topic(self) -> tuple[str, str] | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        r = rows[0].row()
        return self.table.item(r, 0).text(), self.table.item(r, 4).text()

    def _assign(self, edit: QtWidgets.QLineEdit):
        sel = self._selected_topic()
        if sel:
            edit.setText(sel[0])

    def _assign_double_clicked(self, item: QtWidgets.QTableWidgetItem):
        """Double-click sends a topic to whichever field its type belongs in."""
        kind = self.table.item(item.row(), 4).text()
        name = self.table.item(item.row(), 0).text()
        if kind == "lidar":
            self.lidar_edit.setText(name)
        elif kind == "image":
            self.camera_edit.setText(name)

    def _sync_assign_buttons(self):
        sel = self._selected_topic()
        self.assign_lidar.setEnabled(sel is not None and sel[1] == "lidar")
        self.assign_camera.setEnabled(sel is not None and sel[1] == "image")

    def _on_topics_changed(self):
        self.project.lidar_topic = self.lidar_edit.text().strip()
        self.project.camera_topic = self.camera_edit.text().strip()
        self._check_topics()
        self.changed.emit()

    def _check_topics(self):
        """Warn if a chosen topic is missing from some bags -- scenes there would fail."""
        if not self.infos:
            self.topic_warning.clear()
            return
        problems = []
        for label, topic in (("LiDAR", self.project.lidar_topic), ("카메라", self.project.camera_topic)):
            if not topic:
                continue
            missing = [i.path.name for i in self.infos if not any(t.name == topic for t in i.topics)]
            if missing:
                problems.append(f"{label} 토픽 '{topic}' 없음: {', '.join(missing)}")
        if problems:
            self.topic_warning.setText("⚠ " + "\n⚠ ".join(problems))
            self.topic_warning.setStyleSheet("color: #d9534f;")
        else:
            self.topic_warning.clear()

    # ----------------------------------------------------------------- state

    def is_complete(self) -> bool:
        return bool(self.infos and self.project.lidar_topic and self.project.camera_topic)

    def status_text(self) -> str:
        if not self.infos:
            return "bag 미선택"
        missing = [n for n, v in (("LiDAR", self.project.lidar_topic), ("카메라", self.project.camera_topic)) if not v]
        if missing:
            return f"{', '.join(missing)} 토픽 미지정"
        return f"bag {len(self.infos)}개" if len(self.infos) > 1 else Path(self.project.bag_path).name

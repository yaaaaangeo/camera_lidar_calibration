"""Main window: a step navigator on the left, the active step on the right.

The point of the whole tool is that every stage is inspectable before you move
on, so the navigator always shows where each step stands rather than hiding it
behind a Next button.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from gui.core.project import PROJECT_DIR, Project
from gui.ui.steps import StepPage
from gui.ui.steps.step_calibrate import CalibrateStep
from gui.ui.steps.step_camera import CameraStep
from gui.ui.steps.step_data import DataStep
from gui.ui.steps.step_filter import FilterStep
from gui.ui.steps.step_scrub import ScrubberStep
from gui.ui.steps.step_target import TargetStep
from gui.ui.steps.step_verify import VerifyStep



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, project: Project | None = None):
        super().__init__()
        self.project = project or Project()
        self.project_path: Path | None = None
        self.setWindowTitle("Camera–LiDAR Calibration")
        self.resize(1280, 860)

        self.pages: list[StepPage] = [
            DataStep(self.project),
            CameraStep(self.project),
            TargetStep(self.project),
            ScrubberStep(self.project),
            FilterStep(self.project),
            CalibrateStep(self.project),
            VerifyStep(self.project),
        ]

        self.header = QtWidgets.QLabel()
        self.header.setStyleSheet("font-size: 16px; font-weight: 600; padding: 4px 2px;")

        self.nav = QtWidgets.QListWidget()
        # Resizable, not fixed. A fixed 260 px added straight onto every page's
        # own minimum, and the widest page (step 5) forced a 1422 px floor on the
        # whole window -- wider than the 1280 it opens at, so the window could be
        # grown but never shrunk.
        self.nav.setMinimumWidth(150)
        self.nav.setMaximumWidth(340)
        self.nav.setSpacing(2)
        self.stack = QtWidgets.QStackedWidget()

        for page in self.pages:
            page.changed.connect(self._refresh_nav)
            # Each page keeps its own natural size and scrolls when the window is
            # smaller, so a laptop screen can still reach every control instead of
            # having the window refuse to fit.
            area = QtWidgets.QScrollArea()
            area.setWidget(page)
            area.setWidgetResizable(True)
            area.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.stack.addWidget(area)
            self.nav.addItem(QtWidgets.QListWidgetItem())

        self.nav.currentRowChanged.connect(self._goto)
        self.nav.setCurrentRow(0)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.header)
        right.addWidget(self.stack, 1)
        right_box = QtWidgets.QWidget()
        right_box.setLayout(right)

        # A splitter lets the divider be dragged, which is the other half of
        # "resizable": narrowing the window should be able to take space from the
        # step list rather than only from the work area.
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.nav)
        split.addWidget(right_box)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 1020])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self._build_menu()
        self.statusBar().showMessage("bag 을 선택하면 시작합니다.")
        self._refresh_nav()

    def closeEvent(self, event):
        for page in self.pages:
            page.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------ menu

    def _build_menu(self):
        m = self.menuBar().addMenu("프로젝트")
        act_open = m.addAction("열기…")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open)
        act_save = m.addAction("저장…")
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save)

    def _open(self):
        """Load a saved project into this window.

        Saving worked from the start but opening only printed an apology, so every
        session began by re-picking scenes and redrawing filter boxes that were
        already on disk. The state is all in Project, so the fix is to swap the
        object every page holds a reference to and let each page reread it.
        """
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        start = str(self.project_path or PROJECT_DIR)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "프로젝트 열기", start, "캘리브레이션 프로젝트 (*.calib.yaml *.yaml)"
        )
        if not path:
            return
        try:
            loaded = Project.load(path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.critical(
                self, "프로젝트 열기 실패", f"{type(exc).__name__}: {exc}"
            )
            return

        missing = [b for b in loaded.bag_paths if not Path(b).exists()]

        # Pages keep `self.project`, so rebinding the attribute on each one is what
        # makes them see the new state. Replacing the window's project alone would
        # leave every page pointing at the old object.
        self.project = loaded
        for page in self.pages:
            page.project = loaded
            for attr in ("_scene", "_cloud", "_det", "_times", "_loaded_for", "solution"):
                if hasattr(page, attr):
                    setattr(page, attr, None if attr != "_times" else [])
        self.project_path = Path(path)

        # Steps 1-3 hold their state in widgets, so they have to be told to
        # reread. Going to step 1 alone would only refresh that one, and the
        # camera and target pages would still show the previous session.
        # Move first, then refresh. setCurrentRow emits currentRowChanged, which
        # already calls _goto -> on_enter for step 1; calling on_enter here as well
        # started its bag inspection twice, and the second QThread assignment
        # destroyed the first one mid-run.
        self.nav.blockSignals(True)
        self.nav.setCurrentRow(0)
        self.nav.blockSignals(False)
        self._goto(0)
        for page in self.pages[1:3]:
            page.on_enter()
        self._refresh_nav()

        msg = f"불러왔습니다: {Path(path).name} — scene {len(loaded.scenes)}개, bag {len(loaded.bag_paths)}개"
        if missing:
            msg += f"  ·  찾을 수 없는 bag {len(missing)}개 (1단계에서 다시 지정하세요)"
        self.statusBar().showMessage(msg, 12000)
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "bag 을 찾을 수 없음",
                "이 경로의 bag 이 없습니다. 다른 머신에서 만든 프로젝트라면 정상입니다 —\n"
                "1단계에서 같은 파일을 다시 지정하면 scene 과 필터는 그대로 쓰입니다.\n\n"
                + "\n".join(missing[:5]),
            )

    def _save(self):
        # calib_data holds working state: which moments were captured, how the
        # filter box was drawn, where the bag sits on this machine. All in one
        # file, and the whole folder stays out of the repository -- what the team
        # consumes is the extrinsic in calib_result, not the route to it.
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        default = str(self.project_path or PROJECT_DIR / f"{self.project.name}.calib.yaml")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "프로젝트 저장", default, "캘리브레이션 프로젝트 (*.yaml)"
        )
        if not path:
            return
        self.project.save(path)
        self.project_path = Path(path)
        self.statusBar().showMessage(
            f"저장됨: {Path(path).name}  —  scene {len(self.project.scenes)}개, bag {len(self.project.bag_paths)}개", 8000
        )

    # ------------------------------------------------------------------- nav

    def _goto(self, row: int):
        if row < 0:
            return
        page = self.pages[row]
        self.stack.setCurrentIndex(row)
        self.header.setText(f"{page.title} — {page.subtitle}" if page.subtitle else page.title)
        page.on_enter()

    def _refresh_nav(self):
        for i, page in enumerate(self.pages):
            mark = "✓" if page.is_complete() else "○"
            status = page.status_text()
            item = self.nav.item(i)
            item.setText(f" {mark}  {page.title}\n      {status}" if status else f" {mark}  {page.title}")
            # Steps stay readable whether or not they are done -- the ✓/○ marker
            # carries the state, greying the text just made the list hard to scan.
            item.setForeground(QtWidgets.QApplication.palette().text())
        # Labels only. This used to end by calling _goto on the current row, which
        # re-entered the visible page's on_enter -- harmless while the early steps
        # had no on_enter, but once they gained one it became a loop: editing a
        # field emitted `changed`, which refreshed the nav, which re-entered step 1
        # and set it reading a bag from inside step 3's field update.

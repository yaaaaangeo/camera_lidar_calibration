"""Step 3 — calibration target geometry.

Six numbers decide where the tool thinks the markers and holes are. Nothing
downstream can detect that they are wrong: a board model that is consistently
off just biases the pose while the reprojection error stays small. So this page
draws the board to scale from whatever is entered, to be checked against the
physical thing with a tape measure.
"""

from __future__ import annotations

from dataclasses import asdict

from PySide6 import QtCore, QtGui, QtWidgets

from gui.core import presets
from gui.core.project import Project, Target
from gui.ui.steps import StepPage

STOCK = Target()  # the upstream CAD
_CUSTOM = "— 직접 입력 —"

_FIELDS = [
    ("marker_size", "마커 한 변"),
    ("delta_width_qr_center", "마커 중심 간 가로 ÷ 2"),
    ("delta_height_qr_center", "마커 중심 간 세로 ÷ 2"),
    ("delta_width_circles", "구멍 중심 간 가로"),
    ("delta_height_circles", "구멍 중심 간 세로"),
    ("circle_radius", "구멍 반지름"),
]

# Marker index -> (x sign, y sign, ArUco id), matching qr_detect.hpp
_MARKERS = [(-1, +1, 1), (+1, +1, 2), (+1, -1, 4), (-1, -1, 3)]


def check(t: Target) -> list[str]:
    """Geometry problems worth warning about, in plain language."""
    out = []
    if min(asdict(t).values()) <= 0:
        out.append("0 이하인 값이 있습니다.")
        return out
    if 2 * t.delta_width_qr_center - t.marker_size <= 0:
        out.append("좌우 마커가 서로 겹칩니다.")
    if 2 * t.delta_height_qr_center - t.marker_size <= 0:
        out.append("위아래 마커가 서로 겹칩니다.")
    if t.delta_width_circles <= 2 * t.circle_radius:
        out.append("좌우 구멍이 서로 겹칩니다.")
    if t.delta_height_circles <= 2 * t.circle_radius:
        out.append("위아래 구멍이 서로 겹칩니다.")

    # A hole must not eat into a marker.
    gap_x = t.delta_width_qr_center - t.marker_size / 2 - (t.delta_width_circles / 2 + t.circle_radius)
    gap_y = t.delta_height_qr_center - t.marker_size / 2 - (t.delta_height_circles / 2 + t.circle_radius)
    if gap_x < 0 and gap_y < 0:
        out.append("구멍이 마커 영역과 겹칩니다.")
    return out


class BoardDiagram(QtWidgets.QWidget):
    """Scale drawing of the board from the current numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target = Target()
        self.setMinimumSize(420, 320)

    def set_target(self, t: Target):
        self.target = t
        self.update()

    def paintEvent(self, _):
        t = self.target
        if min(asdict(t).values()) <= 0:
            return

        half_w = t.delta_width_qr_center + t.marker_size / 2
        half_h = t.delta_height_qr_center + t.marker_size / 2
        board_w, board_h = 2 * half_w * 1.12, 2 * half_h * 1.18

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette()

        margin = 46
        s = min((self.width() - 2 * margin) / board_w, (self.height() - 2 * margin) / board_h)
        cx, cy = self.width() / 2, self.height() / 2

        def X(x):  # board metres -> widget pixels (+y is up on the board)
            return cx + x * s

        def Y(y):
            return cy - y * s

        # board outline
        p.setPen(QtGui.QPen(pal.mid().color(), 1.5))
        p.setBrush(pal.base())
        p.drawRect(QtCore.QRectF(X(-board_w / 2), Y(board_h / 2), board_w * s, board_h * s))

        # markers
        p.setBrush(pal.text())
        p.setPen(QtCore.Qt.NoPen)
        for sx, sy, mid in _MARKERS:
            x = sx * t.delta_width_qr_center
            y = sy * t.delta_height_qr_center
            m = t.marker_size
            p.drawRect(QtCore.QRectF(X(x - m / 2), Y(y + m / 2), m * s, m * s))
            p.setPen(pal.highlight().color())
            p.drawText(
                QtCore.QRectF(X(x - m / 2), Y(y + m / 2), m * s, m * s),
                QtCore.Qt.AlignCenter,
                str(mid),
            )
            p.setPen(QtCore.Qt.NoPen)

        # holes
        p.setBrush(pal.window())
        p.setPen(QtGui.QPen(pal.highlight().color(), 1.8))
        for sx in (-1, 1):
            for sy in (-1, 1):
                x = sx * t.delta_width_circles / 2
                y = sy * t.delta_height_circles / 2
                r = t.circle_radius * s
                p.drawEllipse(QtCore.QPointF(X(x), Y(y)), r, r)

        # dimensions
        p.setPen(QtGui.QPen(pal.mid().color(), 1))
        font = p.font()
        font.setPointSizeF(max(font.pointSizeF() - 1, 7))
        p.setFont(font)

        y_dim = board_h / 2 + 0.035 * board_h
        p.drawLine(
            QtCore.QPointF(X(-t.delta_width_circles / 2), Y(y_dim)),
            QtCore.QPointF(X(t.delta_width_circles / 2), Y(y_dim)),
        )
        p.drawText(
            QtCore.QRectF(X(-board_w / 2), Y(y_dim) - 20, board_w * s, 16),
            QtCore.Qt.AlignCenter,
            f"구멍 가로 {t.delta_width_circles * 1000:.1f} mm",
        )

        x_dim = -board_w / 2 - 0.02 * board_w
        p.drawLine(
            QtCore.QPointF(X(x_dim), Y(-t.delta_height_circles / 2)),
            QtCore.QPointF(X(x_dim), Y(t.delta_height_circles / 2)),
        )
        p.save()
        p.translate(X(x_dim) - 6, cy)
        p.rotate(-90)
        p.drawText(QtCore.QRectF(-80, -16, 160, 14), QtCore.Qt.AlignCenter, f"세로 {t.delta_height_circles * 1000:.1f} mm")
        p.restore()

        p.drawText(
            QtCore.QRectF(0, self.height() - 24, self.width(), 18),
            QtCore.Qt.AlignCenter,
            f"보드 최소 {2 * half_w * 1000:.0f} × {2 * half_h * 1000:.0f} mm    "
            f"구멍 지름 {2 * t.circle_radius * 1000:.1f} mm    마커 {t.marker_size * 1000:.1f} mm",
        )


class TargetStep(StepPage):
    title = "3. 타겟"
    subtitle = "캘리브레이션 보드 규격"

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self.presets = presets.load_targets()

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem(_CUSTOM)
        for t in self.presets:
            self.preset_combo.addItem(t.name)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)

        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        self.preset_note.setStyleSheet("color: palette(mid);")

        self.edits: dict[str, QtWidgets.QLineEdit] = {}
        form = QtWidgets.QGridLayout()
        for row, (key, label) in enumerate(_FIELDS):
            e = QtWidgets.QLineEdit()
            e.setValidator(QtGui.QDoubleValidator(0.0, 100.0, 6))
            e.textChanged.connect(self._on_edit)
            self.edits[key] = e
            form.addWidget(QtWidgets.QLabel(label), row, 0)
            form.addWidget(e, row, 1)
            form.addWidget(QtWidgets.QLabel("m"), row, 2)
        form.setColumnStretch(1, 1)

        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setRange(0.05, 5.0)
        self.scale.setSingleStep(0.01)
        self.scale.setDecimals(4)
        self.scale.setValue(1.0)
        apply_scale = QtWidgets.QPushButton("원본 CAD × 배율 적용")
        apply_scale.setToolTip("원본 도면을 축소·확대 출력했다면 배율만 넣으면 여섯 값이 한꺼번에 맞춰집니다")
        apply_scale.clicked.connect(self._apply_scale)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(self.scale)
        scale_row.addWidget(apply_scale, 1)

        self.warn = QtWidgets.QLabel()
        self.warn.setWordWrap(True)

        save_btn = QtWidgets.QPushButton("현재 값을 저장…")
        save_btn.setToolTip("이 보드를 목록에 추가해 다음에 바로 고를 수 있게 합니다")
        save_btn.clicked.connect(self._save_preset)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("저장된 보드"))
        left.addWidget(self.preset_combo)
        left.addWidget(self.preset_note)
        left.addSpacing(8)
        left.addLayout(form)
        left.addLayout(scale_row)
        left.addWidget(save_btn)
        left.addWidget(self.warn)
        left.addStretch(1)
        left.addWidget(
            QtWidgets.QLabel("도면을 실물과 줄자로 대조하세요.\n여기가 틀리면 뒤에서 잡아낼 방법이 없습니다.")
        )

        left_box = QtWidgets.QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(340)

        self.diagram = BoardDiagram()

        row = QtWidgets.QHBoxLayout(self)
        row.addWidget(left_box)
        row.addWidget(self.diagram, 1)

        self._load_from_project()

    def on_enter(self):
        """Reread the project -- see the note in step 2."""
        self._load_from_project()

    def _load_from_project(self):
        for key, e in self.edits.items():
            e.blockSignals(True)
            e.setText(f"{getattr(self.project.target, key):g}")
            e.blockSignals(False)
        self._on_edit()

    def _apply_preset(self, index: int):
        if index <= 0:
            self.preset_note.clear()
            return
        preset = self.presets[index - 1]
        self.preset_note.setText(preset.note)
        for key, e in self.edits.items():
            e.setText(f"{getattr(preset.target, key):.12g}")

    def _sync_preset_combo(self, t: Target):
        """Follow the fields: show the matching saved board, or 직접 입력."""
        match = next((i for i, p in enumerate(self.presets) if p.matches(t)), None)
        target_index = 0 if match is None else match + 1
        if self.preset_combo.currentIndex() != target_index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(target_index)
            self.preset_combo.blockSignals(False)
        self.preset_note.setText("" if match is None else self.presets[match].note)

    def _save_preset(self):
        t = self.project.target
        if min(asdict(t).values()) <= 0:
            QtWidgets.QMessageBox.information(self, "저장", "치수를 먼저 입력하세요.")
            return
        current = self.preset_combo.currentText()
        name, ok = QtWidgets.QInputDialog.getText(
            self, "보드 저장", "이름 (기존 이름을 쓰면 덮어씁니다)",
            text="" if current == _CUSTOM else current,
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        existing = next((p for p in self.presets if p.name == name), None)
        if existing is not None:
            confirm = QtWidgets.QMessageBox.question(
                self, "덮어쓰기", f"'{name}' 이(가) 이미 있습니다. 값을 덮어쓸까요?"
            )
            if confirm != QtWidgets.QMessageBox.Yes:
                return
        note, _ = QtWidgets.QInputDialog.getText(
            self, "보드 저장", "설명 (선택)", text=existing.note if existing else ""
        )

        presets.add_target(name, note.strip(), t)
        self.presets = presets.load_targets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(_CUSTOM)
        for p in self.presets:
            self.preset_combo.addItem(p.name)
        self.preset_combo.blockSignals(False)
        self._sync_preset_combo(t)
        self.window().statusBar().showMessage(
            f"'{name}' 저장됨 — {presets.TARGET_CONFIG_PATH}", 6000
        )

    def _apply_scale(self):
        scaled = STOCK.scaled(self.scale.value())
        for key, e in self.edits.items():
            e.setText(f"{getattr(scaled, key):g}")

    def _on_edit(self):
        t = Target()
        for key, e in self.edits.items():
            try:
                setattr(t, key, float(e.text()))
            except ValueError:
                setattr(t, key, 0.0)
        self.project.target = t
        self._sync_preset_combo(t)
        self.diagram.set_target(t)

        problems = check(t)
        if problems:
            self.warn.setText("⚠ " + "\n⚠ ".join(problems))
            self.warn.setStyleSheet("color: #d9534f;")
        else:
            ratio = t.marker_size / STOCK.marker_size if STOCK.marker_size else 0
            same = all(
                abs(getattr(t, k) / getattr(STOCK, k) - ratio) < 1e-3 for k, _ in _FIELDS if getattr(STOCK, k)
            )
            note = f"원본 CAD의 {ratio:.4g}배 — 전 항목 배율이 같습니다." if same else "원본 CAD와 배율이 항목마다 다릅니다."
            self.warn.setText("문제 없음.\n" + note)
            self.warn.setStyleSheet("color: palette(mid);")
        self.changed.emit()

    def is_complete(self) -> bool:
        return min(asdict(self.project.target).values()) > 0 and not check(self.project.target)

    def status_text(self) -> str:
        t = self.project.target
        if min(asdict(t).values()) <= 0:
            return "미입력"
        return f"마커 {t.marker_size * 1000:.0f}mm  구멍 ⌀{t.circle_radius * 2000:.0f}mm"

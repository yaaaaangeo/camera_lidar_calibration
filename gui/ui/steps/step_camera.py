"""Step 2 — camera intrinsics.

Includes k3. The original tool hard-codes the fifth distortion coefficient to
zero, which quietly biases pose estimation on wide lenses, and both cameras on
this rig have a non-zero k3.

The undistort preview is the point of this page: straight edges in the scene
(door frames, ceiling beams, floor joints) only come out straight when the
numbers are actually the ones for this camera.
"""

from __future__ import annotations

import re

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gui.core import presets
from gui.core.project import Camera, Project
from gui.ui.steps import StepPage

_CUSTOM = "— 직접 입력 —"


def _fmt(v: float) -> str:
    """Enough digits that a value survives a round trip through the field.

    Plain %g keeps six significant digits, which silently rounds 909.02985013
    to 909.03 and makes a preset stop matching itself.
    """
    return f"{v:.12g}"

_FIELDS = [
    ("fx", "초점거리 x"),
    ("fy", "초점거리 y"),
    ("cx", "주점 x"),
    ("cy", "주점 y"),
    ("k1", "방사 왜곡 k1"),
    ("k2", "방사 왜곡 k2"),
    ("p1", "접선 왜곡 p1"),
    ("p2", "접선 왜곡 p2"),
    ("k3", "방사 왜곡 k3"),
    ("k4", "유리함수 k4"),
    ("k5", "유리함수 k5"),
    ("k6", "유리함수 k6"),
]

# k4~k6 은 rational_polynomial 모델의 분모 항이다. 셋 다 비어 있으면 plumb_bob 과
# 같으므로 5계수 카메라는 그대로 5계수로 남는다.
_RATIONAL = ("k4", "k5", "k6")

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_intrinsics(text: str) -> dict | None:
    """Pull intrinsics out of pasted text.

    Understands the `cameraMatrix: "..."` / `distCoeffs: "..."` form the rig's
    calibration files use, and falls back to "9 numbers then the rest".

    Distortion is read by position for as many coefficients as are there, so a
    plumb_bob file (5) and a rational_polynomial one (8) both land correctly. Any
    of k4..k6 the text does not mention are left out, which zeroes them and keeps
    the camera on the 5-coefficient path.
    """
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    out = {}
    for key, want in (("cameramatrix", 9), ("distcoeffs", 4)):
        m = re.search(rf"{key}\s*:?\s*\"?([^\"\n]*)\"?", text, re.IGNORECASE)
        if not m:
            continue
        nums = [float(v) for v in _NUM.findall(m.group(1))]
        if len(nums) < want:
            continue
        if key == "cameramatrix":
            out.update(fx=nums[0], cx=nums[2], fy=nums[4], cy=nums[5])
        else:
            out.update(dict(zip(names, nums[:8])))

    if not out:
        nums = [float(v) for v in _NUM.findall(text)]
        if len(nums) >= 14:  # K(9) + at least plumb_bob(5); 17 with rational
            out = dict(fx=nums[0], cx=nums[2], fy=nums[4], cy=nums[5])
            out.update(dict(zip(names, nums[9:17])))
    return out or None


class CameraStep(StepPage):
    title = "2. 카메라"
    subtitle = "카메라 내부 파라미터"

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self._frame: np.ndarray | None = None
        self.presets = presets.load()

        # --- saved cameras ---------------------------------------------------
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem(_CUSTOM)
        for p in self.presets:
            self.preset_combo.addItem(p.name)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)

        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        self.preset_note.setStyleSheet("color: palette(mid);")

        # --- entry fields ---------------------------------------------------
        self.edits: dict[str, QtWidgets.QLineEdit] = {}
        form = QtWidgets.QGridLayout()
        # The row counter has to advance past the separator, not reuse its row:
        # spanning both columns there put the next field's label and edit box into
        # the same cell as the rule, and the rule drew straight through the label
        # like a strikethrough.
        row = 0
        for key, label in _FIELDS:
            e = QtWidgets.QLineEdit()
            e.setValidator(QtGui.QDoubleValidator())
            e.textChanged.connect(self._on_edit)
            self.edits[key] = e
            form.addWidget(QtWidgets.QLabel(label), row, 0)
            form.addWidget(e, row, 1)
            row += 1
            if key in ("cy", "k3"):  # intrinsics / plumb_bob / rational
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.HLine)
                form.addWidget(sep, row, 0, 1, 2)
                row += 1
        form.setColumnStretch(1, 1)

        paste_btn = QtWidgets.QPushButton("붙여넣기로 입력…")
        paste_btn.setToolTip("cameraMatrix / distCoeffs 형식 텍스트를 그대로 붙여넣으면 채워집니다")
        paste_btn.clicked.connect(self._paste)

        self.save_preset_btn = QtWidgets.QPushButton("현재 값을 저장…")
        self.save_preset_btn.setToolTip("이 카메라를 목록에 추가해 다음에 바로 고를 수 있게 합니다")
        self.save_preset_btn.clicked.connect(self._save_preset)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(paste_btn)
        buttons.addWidget(self.save_preset_btn)

        self.derived = QtWidgets.QLabel()
        self.derived.setWordWrap(True)
        self.derived.setStyleSheet("color: palette(mid);")

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("저장된 카메라"))
        left.addWidget(self.preset_combo)
        left.addWidget(self.preset_note)
        left.addSpacing(8)
        left.addLayout(form)
        left.addLayout(buttons)
        left.addWidget(self.derived)
        left.addStretch(1)

        left_box = QtWidgets.QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(340)

        # --- preview --------------------------------------------------------
        self.load_btn = QtWidgets.QPushButton("bag에서 프레임 불러오기")
        self.load_btn.clicked.connect(self._load_frame)
        self.undistort_cb = QtWidgets.QCheckBox("왜곡 보정 적용")
        self.undistort_cb.toggled.connect(self._render)
        self.undistort_cb.setEnabled(False)

        self.view = QtWidgets.QLabel("프레임을 불러오면 왜곡 보정 결과를 볼 수 있습니다.")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(480, 360)
        self.view.setStyleSheet("background: palette(base); color: palette(mid);")

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.load_btn)
        bar.addWidget(self.undistort_cb)
        bar.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(bar)
        right.addWidget(self.view, 1)
        right.addWidget(
            QtWidgets.QLabel("직선이 직선으로 펴지는지 보세요 — 문틀, 천장 빔, 바닥 이음선이 판단하기 좋습니다.")
        )

        row = QtWidgets.QHBoxLayout(self)
        row.addWidget(left_box)
        row.addLayout(right, 1)

        self._load_from_project()

    # ------------------------------------------------------------------ data

    def on_enter(self):
        """Reread the project, so an opened file shows up here.

        These fields only ever wrote outwards before -- there was no way to load a
        project, so nothing could change them from underneath. Now that opening
        works, a page that never rereads shows the previous session's values while
        the project holds different ones.
        """
        self._load_from_project()  # _on_edit inside also refreshes the preset combo

    def _load_from_project(self):
        cam = self.project.camera
        for key, e in self.edits.items():
            v = getattr(cam, key)
            e.blockSignals(True)
            e.setText("" if v == 0.0 and key in ("fx", "fy", "cx", "cy") else _fmt(v))
            e.blockSignals(False)
        self._on_edit()

    def _apply_preset(self, index: int):
        if index <= 0:
            self.preset_note.clear()
            return
        preset = self.presets[index - 1]
        self.preset_note.setText(preset.note)
        for key, e in self.edits.items():
            e.setText(_fmt(getattr(preset.camera, key)))

    def _sync_preset_combo(self, cam: Camera):
        """Follow the fields: show the matching saved camera, or 직접 입력."""
        match = next((i for i, p in enumerate(self.presets) if p.matches(cam)), None)
        target = 0 if match is None else match + 1
        if self.preset_combo.currentIndex() != target:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(target)
            self.preset_combo.blockSignals(False)
        self.preset_note.setText("" if match is None else self.presets[match].note)

    def _save_preset(self):
        cam = self.project.camera
        if not cam.is_set:
            QtWidgets.QMessageBox.information(self, "저장", "fx, fy 를 먼저 입력하세요.")
            return
        current = self.preset_combo.currentText()
        suggested = "" if current == _CUSTOM else current
        name, ok = QtWidgets.QInputDialog.getText(
            self, "카메라 저장", "이름 (기존 이름을 쓰면 덮어씁니다)", text=suggested
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
            self, "카메라 저장", "설명 (선택)", text=existing.note if existing else ""
        )

        presets.add(name, note.strip(), cam)
        self.presets = presets.load()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(_CUSTOM)
        for p in self.presets:
            self.preset_combo.addItem(p.name)
        self.preset_combo.blockSignals(False)
        self._sync_preset_combo(cam)
        self.window().statusBar().showMessage(f"'{name}' 저장됨 — {presets.CONFIG_PATH}", 6000)

    def _on_edit(self):
        cam = Camera()
        for key, e in self.edits.items():
            try:
                setattr(cam, key, float(e.text()))
            except ValueError:
                setattr(cam, key, 0.0)
        self.project.camera = cam
        self._sync_preset_combo(cam)
        self._update_derived()
        self._render()
        self.changed.emit()

    def _update_derived(self):
        cam = self.project.camera
        if not cam.is_set:
            self.derived.setText("fx, fy 를 입력하세요.")
            return
        parts = []
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            hfov = 2 * np.degrees(np.arctan(w / 2 / cam.fx))
            vfov = 2 * np.degrees(np.arctan(h / 2 / cam.fy))
            parts.append(f"{w}×{h} 기준 화각  가로 {hfov:.1f}°  세로 {vfov:.1f}°")
            off_x, off_y = cam.cx - w / 2, cam.cy - h / 2
            parts.append(f"주점이 화면 중심에서 ({off_x:+.0f}, {off_y:+.0f}) px")
        else:
            parts.append("프레임을 불러오면 화각을 계산합니다.")
        if cam.rational:
            parts.append(
                "왜곡 모델: rational_polynomial (8계수) — k4~k6 이 분모에 들어갑니다."
            )
        else:
            parts.append("왜곡 모델: plumb_bob (5계수)")
            if cam.k3 == 0.0:
                parts.append("k3 가 0 입니다 — 광각 렌즈라면 가장자리에 오차가 남습니다.")
        self.derived.setText("\n".join(parts))

    def _paste(self):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "붙여넣기로 입력",
            'cameraMatrix / distCoeffs 를 포함한 텍스트를 붙여넣으세요.',
            "",
        )
        if not ok or not text.strip():
            return
        vals = parse_intrinsics(text)
        if not vals:
            QtWidgets.QMessageBox.warning(self, "붙여넣기", "숫자를 찾지 못했습니다.")
            return
        for key, v in vals.items():
            self.edits[key].setText(_fmt(v))

    # --------------------------------------------------------------- preview

    def _load_frame(self):
        p = self.project
        if not p.bag_path or not p.camera_topic:
            QtWidgets.QMessageBox.information(self, "프레임", "1단계에서 bag과 카메라 토픽을 먼저 지정하세요.")
            return
        from gui.core.bag_reader import BagSource
        from gui.core.decode import image_to_bgr

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            with BagSource(p.bag_path) as src:
                _, msg = src.first_after(p.camera_topic, src.at_fraction(0.5))
            if msg is None:
                raise LookupError("해당 토픽에 메시지가 없습니다")
            self._frame = image_to_bgr(msg)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self, "프레임", f"불러오지 못했습니다.\n{type(exc).__name__}: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.undistort_cb.setEnabled(True)
        self._update_derived()
        self._render()

    def _render(self):
        if self._frame is None:
            return
        import cv2

        img = self._frame
        if self.undistort_cb.isChecked() and self.project.camera.is_set:
            img = cv2.undistort(img, self.project.camera.matrix(), self.project.camera.dist())

        h, w = img.shape[:2]
        qimg = QtGui.QImage(np.ascontiguousarray(img[:, :, ::-1]).data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ----------------------------------------------------------------- state

    def is_complete(self) -> bool:
        return self.project.camera.is_set

    def status_text(self) -> str:
        cam = self.project.camera
        return f"fx {cam.fx:.1f}  k3 {cam.k3:g}" if cam.is_set else "미입력"

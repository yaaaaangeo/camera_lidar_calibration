"""Color Transformer controls, laid out like rviz's.

Sits above a CloudView and decides how its points get painted. Kept separate
from the view so the verification step can reuse it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from gui.core.colorize import AXES, COLORMAPS, ColorStyle

_MODE_LABELS = [("flat", "단색"), ("intensity", "Intensity"), ("axis", "축 (AxisColor)"), ("distance", "거리")]


class ColorControls(QtWidgets.QWidget):
    """Emits `changed` whenever the style is edited."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.style = ColorStyle()

        self.mode = QtWidgets.QComboBox()
        for key, label in _MODE_LABELS:
            self.mode.addItem(label, key)
        self.mode.setCurrentIndex(1)  # intensity
        self.mode.currentIndexChanged.connect(self._emit)

        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(list(COLORMAPS))
        self.colormap.setCurrentText("turbo")
        self.colormap.currentTextChanged.connect(self._emit)

        self.axis = QtWidgets.QComboBox()
        self.axis.addItems([a.upper() for a in AXES])
        self.axis.setCurrentText("Z")
        self.axis.currentTextChanged.connect(self._emit)

        self.invert = QtWidgets.QCheckBox("반전")
        self.invert.toggled.connect(self._emit)

        self.auto = QtWidgets.QCheckBox("자동 범위")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._on_auto)

        self.vmin = QtWidgets.QDoubleSpinBox()
        self.vmax = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.vmin, 0.0), (self.vmax, 255.0)):
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._emit)

        self.range_label = QtWidgets.QLabel("—")
        self.range_label.setStyleSheet("color: palette(mid);")

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QtWidgets.QLabel("색상 기준"), 0, 0)
        grid.addWidget(self.mode, 0, 1)
        grid.addWidget(self.axis, 0, 2)
        grid.addWidget(QtWidgets.QLabel("컬러맵"), 1, 0)
        grid.addWidget(self.colormap, 1, 1)
        grid.addWidget(self.invert, 1, 2)
        grid.addWidget(self.auto, 2, 0)
        grid.addWidget(self.vmin, 2, 1)
        grid.addWidget(self.vmax, 2, 2)
        grid.addWidget(self.range_label, 3, 0, 1, 3)
        grid.setColumnStretch(1, 1)

        self._sync_enabled()

    # ------------------------------------------------------------------ state

    def _on_auto(self, checked: bool):
        self.vmin.setEnabled(not checked)
        self.vmax.setEnabled(not checked)
        self._emit()

    def _sync_enabled(self):
        mode = self.mode.currentData()
        scalar = mode != "flat"
        self.colormap.setEnabled(scalar)
        self.invert.setEnabled(scalar)
        self.auto.setEnabled(scalar)
        self.axis.setVisible(mode == "axis")
        self.vmin.setEnabled(scalar and not self.auto.isChecked())
        self.vmax.setEnabled(scalar and not self.auto.isChecked())

    def _emit(self):
        self._sync_enabled()
        self.style = ColorStyle(
            mode=self.mode.currentData(),
            colormap=self.colormap.currentText(),
            axis=self.axis.currentText().lower(),
            auto_bounds=self.auto.isChecked(),
            min_value=self.vmin.value(),
            max_value=self.vmax.value(),
            invert=self.invert.isChecked(),
        )
        self.changed.emit()

    def report_range(self, used: tuple[float, float] | None, has_intensity: bool = True):
        """Show the range actually applied, and fill the boxes when auto is on."""
        if self.mode.currentData() == "intensity" and not has_intensity:
            self.range_label.setText("이 토픽에 intensity 필드가 없습니다.")
            return
        if used is None:
            self.range_label.clear()
            return
        lo, hi = used
        self.range_label.setText(f"적용 범위  {lo:.2f} ~ {hi:.2f}")
        if self.auto.isChecked():
            for spin, value in ((self.vmin, lo), (self.vmax, hi)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

"""Smoke test for the 3D view.

Builds a synthetic scene shaped like a real capture -- ground, a back wall, and
a calibration board with four holes -- then renders it in CloudView. Run this
before building anything else: it tells us whether OpenGL is hardware
accelerated on this machine, which decides how many points the real views can
afford to draw.

    python3.10 gui/check_gl.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
from PySide6 import QtCore, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui.ui.cloud_view import CloudView  # noqa: E402

# Board geometry, taken from the resized target actually in use (the stock CAD
# scaled by 0.594). Only used to make the synthetic scene representative.
BOARD_HALF_W = 0.44
BOARD_HALF_H = 0.30
CIRCLE_DX = 0.2970 / 2.0
CIRCLE_DY = 0.2376 / 2.0
CIRCLE_R = 0.07128
POINT_PITCH = 0.008  # ~8 mm, the spacing quoted for this LiDAR at a few metres

# Roughly where the board sat in one of the recorded scenes. Note it is nowhere
# near the +x axis -- forward is about -y here.
BOARD_CENTER = np.array([2.2, -3.25, -0.05])
FILTER_BOX = (1.7, 2.7, -3.8, -2.7, -0.5, 0.4)


def _board_points() -> np.ndarray:
    """A planar board facing the sensor, with four circular holes punched out."""
    n = -BOARD_CENTER / np.linalg.norm(BOARD_CENTER)
    up = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(up, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    us = np.arange(-BOARD_HALF_W, BOARD_HALF_W, POINT_PITCH)
    vs = np.arange(-BOARD_HALF_H, BOARD_HALF_H, POINT_PITCH)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()

    keep = np.ones(uu.shape, dtype=bool)
    for cu in (-CIRCLE_DX, CIRCLE_DX):
        for cv in (-CIRCLE_DY, CIRCLE_DY):
            keep &= (uu - cu) ** 2 + (vv - cv) ** 2 > CIRCLE_R**2
    uu, vv = uu[keep], vv[keep]

    pts = BOARD_CENTER + uu[:, None] * e1 + vv[:, None] * e2
    return pts + np.random.normal(0.0, 0.002, pts.shape)


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    ground = np.column_stack([
        rng.uniform(-6, 6, 260_000),
        rng.uniform(-8, 2, 260_000),
        np.full(260_000, -1.2) + rng.normal(0, 0.01, 260_000),
    ])
    wall = np.column_stack([
        rng.uniform(-6, 6, 140_000),
        np.full(140_000, -6.0) + rng.normal(0, 0.01, 140_000),
        rng.uniform(-1.2, 2.0, 140_000),
    ])
    return np.vstack([ground, wall, _board_points()]).astype(np.float32)


class Window(QtWidgets.QWidget):
    def __init__(self, points: np.ndarray):
        super().__init__()
        self.setWindowTitle("CloudView smoke test")
        self.resize(1100, 780)
        self.points = points

        self.view = CloudView()
        self.frames = 0
        self._wrap_paint()

        bar = QtWidgets.QHBoxLayout()
        for name in ("top", "front", "side", "iso"):
            b = QtWidgets.QPushButton(name.capitalize())
            b.clicked.connect(lambda _=False, n=name: self.view.apply_preset(n))
            bar.addWidget(b)
        fit = QtWidgets.QPushButton("Fit")
        fit.clicked.connect(lambda: self.view.fit(self.points))
        bar.addWidget(fit)

        self.split = QtWidgets.QCheckBox("필터 하이라이트")
        self.split.setChecked(True)
        self.split.toggled.connect(self._redraw)
        bar.addWidget(self.split)
        bar.addStretch(1)

        self.status = QtWidgets.QLabel("...")
        bar.addWidget(self.status)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)

        self._redraw()
        self.view.set_box(FILTER_BOX)
        self.view.fit(self.points)

        # Redraw as fast as Qt allows so the frame counter means something.
        self._spin = QtCore.QTimer(self, interval=0, timeout=self.view.update)
        self._spin.start()
        self._report = QtCore.QTimer(self, interval=1000, timeout=self._tick)
        self._report.start()
        self._t0 = time.perf_counter()

    def _wrap_paint(self):
        original = self.view.paintGL

        def counted(*a, **kw):
            self.frames += 1
            return original(*a, **kw)

        self.view.paintGL = counted

    def _redraw(self):
        if self.split.isChecked():
            x0, x1, y0, y1, z0, z1 = FILTER_BOX
            p = self.points
            mask = (
                (p[:, 0] >= x0) & (p[:, 0] <= x1)
                & (p[:, 1] >= y0) & (p[:, 1] <= y1)
                & (p[:, 2] >= z0) & (p[:, 2] <= z1)
            )
            self.view.set_points_split(self.points, mask)
        else:
            self.view.set_points(self.points, (0.75, 0.78, 0.82, 0.9))

    def _tick(self):
        now = time.perf_counter()
        fps = self.frames / max(now - self._t0, 1e-6)
        self.frames, self._t0 = 0, now
        mode = "소프트웨어" if self.view.is_software_rendering() else "하드웨어"
        self.status.setText(
            f"{len(self.points):,} pts | {fps:5.1f} FPS | {mode} | "
            f"{self.view.gl_renderer} | GL {self.view.gl_version}"
        )


def main():
    pts = _scene()
    print(f"synthetic scene: {len(pts):,} points")
    app = QtWidgets.QApplication(sys.argv)
    win = Window(pts)
    win.show()
    app.processEvents()  # force the GL context to come up so the probe is filled in
    print(f"GL renderer : {win.view.gl_renderer}")
    print(f"GL version  : {win.view.gl_version}")
    print(f"software    : {win.view.is_software_rendering()}")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

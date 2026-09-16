"""Reusable 3D point cloud view.

Used by the distance-filter step (to show what a filter box keeps and cuts)
and by the verification step (to show the colored cloud). Navigation matches
rviz: left-drag orbits, wheel zooms, middle-drag pans.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui
from pyqtgraph import Vector

# Camera presets. Azimuth is the angle in the XY plane, elevation the angle
# above it, both in degrees. The LiDAR frame is assumed to be x forward,
# y left, z up.
VIEW_PRESETS = {
    "top": dict(elevation=89.9, azimuth=-90.0),
    "front": dict(elevation=0.0, azimuth=180.0),
    "side": dict(elevation=0.0, azimuth=-90.0),
    "iso": dict(elevation=25.0, azimuth=-135.0),
}

def _as_xyz(value) -> np.ndarray:
    """The orbit centre as three floats.

    pyqtgraph hands this back as its own Vector on some paths and as a Qt
    QVector3D on others, and only the first is iterable -- so read the
    components by name rather than trusting the container.
    """
    if hasattr(value, "x") and callable(value.x):
        return np.array([value.x(), value.y(), value.z()], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).ravel()[:3]


CUT_COLOR = (0.45, 0.45, 0.48, 0.35)
KEEP_COLOR = (0.25, 0.75, 1.00, 1.00)
BOX_COLOR = (1.00, 0.72, 0.10, 1.00)


class CloudView(gl.GLViewWidget):
    """A GLViewWidget holding one point cloud plus an optional wireframe box."""

    # Emitted with the 3D position of the drawn point under the cursor.
    point_picked = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=12.0, elevation=25.0, azimuth=-135.0)

        self._grid = gl.GLGridItem()
        self._grid.setSize(20, 20)
        self._grid.setSpacing(1, 1)
        self.addItem(self._grid)

        # Drawn as lines rather than with GLAxisItem, which fixes its width at one
        # pixel and chooses its own colours. One pixel vanishes against a dense
        # cloud. Red/green/blue for x/y/z matches RViz, so a direction can be read
        # off without first working out which convention this view uses.
        #
        # Width is kept at 2 to match the filter box, which is known to draw on
        # this hardware -- glLineWidth above the driver's maximum is rejected
        # outright rather than clamped, and would take the rest of the frame with
        # it.
        axis_len = 2.0
        self._axis = gl.GLLinePlotItem(
            pos=np.array([
                [0, 0, 0], [axis_len, 0, 0],
                [0, 0, 0], [0, axis_len, 0],
                [0, 0, 0], [0, 0, axis_len],
            ], dtype=np.float32),
            color=np.array([
                [1.0, 0.30, 0.30, 1.0], [1.0, 0.30, 0.30, 1.0],
                [0.30, 1.0, 0.40, 1.0], [0.30, 1.0, 0.40, 1.0],
                [0.40, 0.60, 1.0, 1.0], [0.40, 0.60, 1.0, 1.0],
            ], dtype=np.float32),
            width=2.0,
            mode="lines",
            antialias=True,
        )
        self.addItem(self._axis)

        self._scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32))
        self._scatter.setGLOptions("translucent")
        self.addItem(self._scatter)

        self._box: gl.GLLinePlotItem | None = None
        self._quad_item: gl.GLLinePlotItem | None = None
        self._markers: dict[str, gl.GLScatterPlotItem] = {}
        self._center = np.zeros(3, dtype=np.float32)

        # Filled in once the GL context exists; used to detect software rendering.
        self.gl_renderer = "?"
        self.gl_version = "?"
        self.pick_mode = False

    def set_pick_mode(self, on: bool):
        """While on, a left click reports the point under the cursor instead of orbiting."""
        self.pick_mode = on
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, ev):
        if self.pick_mode and ev.button() == QtCore.Qt.LeftButton:
            pos = ev.position() if hasattr(ev, "position") else ev.posF()
            target = self._cursor_target(pos)
            if target is not None:
                self.point_picked.emit(target)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.pick_mode:
            ev.accept()  # no orbiting while picking
            return
        super().mouseMoveEvent(ev)

    # ------------------------------------------------------------------ points

    def set_points(self, xyz: np.ndarray, colors: np.ndarray | tuple, size: float = 2.0):
        """Show a cloud. `colors` is either one RGBA tuple or one RGBA row per point."""
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        if isinstance(colors, tuple):
            colors = np.tile(np.array(colors, dtype=np.float32), (len(xyz), 1))
        else:
            colors = np.ascontiguousarray(colors, dtype=np.float32)
        self._scatter.setData(pos=xyz, color=colors, size=size, pxMode=True)
        if len(xyz):
            self._center = xyz.mean(axis=0)

    def compute_colors(self, xyz: np.ndarray, intensity, style):
        """RGBA for every point, plus the value range used. Cache this."""
        from gui.core.colorize import colorize

        return colorize(xyz, intensity, style)

    def apply_colors(
        self,
        xyz: np.ndarray,
        colors: np.ndarray,
        keep_mask: np.ndarray | None = None,
        hide_outside: bool = False,
        size: float = 2.0,
    ):
        """Draw pre-computed colours, dimming or hiding what the box cuts."""
        if hide_outside and keep_mask is not None:
            self.set_points(xyz[keep_mask], colors[keep_mask], size=size)
            return
        if keep_mask is not None:
            colors = colors.copy()
            outside = ~keep_mask
            colors[outside, :3] = colors[outside, :3] * 0.35 + 0.30
            colors[outside, 3] *= 0.35
        self.set_points(xyz, colors, size=size)

    def set_points_styled(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        style,
        keep_mask: np.ndarray | None = None,
        hide_outside: bool = False,
        size: float = 2.0,
    ):
        """Paint by a ColorStyle, optionally dimming or hiding what the box cuts.

        Returns the value range the colour map ended up using, so the controls
        can show it.
        """
        from gui.core.colorize import colorize

        if hide_outside and keep_mask is not None:
            xyz = xyz[keep_mask]
            intensity = None if intensity is None else intensity[keep_mask]
            keep_mask = None

        colors, used = colorize(xyz, intensity, style)
        if keep_mask is not None:
            # Outside points stay visible but recede: half-faded and pulled
            # towards grey, so the box boundary reads without hiding data.
            outside = ~keep_mask
            colors[outside, :3] = colors[outside, :3] * 0.35 + 0.30
            colors[outside, 3] *= 0.35
        self.set_points(xyz, colors, size=size)
        return used

    def set_points_split(self, xyz: np.ndarray, keep_mask: np.ndarray, size: float = 2.0):
        """Flat two-tone version: inside the box blue, outside grey."""
        colors = np.tile(np.array(CUT_COLOR, dtype=np.float32), (len(xyz), 1))
        colors[keep_mask] = KEEP_COLOR
        self.set_points(xyz, colors, size=size)

    def set_markers(self, name: str, xyz: np.ndarray | None, color: tuple, size: float = 8.0):
        """A named overlay layer -- detected edges, hole centres, and so on."""
        item = self._markers.get(name)
        if xyz is None or len(xyz) == 0:
            if item is not None:
                self.removeItem(item)
                del self._markers[name]
            return
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        if item is None:
            item = gl.GLScatterPlotItem(pos=xyz, color=color, size=size, pxMode=True)
            item.setGLOptions("translucent")
            self.addItem(item)
            self._markers[name] = item
        else:
            item.setData(pos=xyz, color=color, size=size)

    def clear_markers(self):
        for item in self._markers.values():
            self.removeItem(item)
        self._markers.clear()

    # --------------------------------------------------------------------- box

    def set_box(self, bounds):
        """Draw the filter box as a wireframe.

        Takes either six bounds or the eight corners already placed, so a box
        that has been turned to sit square with the board draws correctly.
        """
        pts = np.asarray(bounds, dtype=np.float64)
        if pts.shape == (8, 3):
            # corners() order: x outer, y middle, z inner
            c = pts[[0, 4, 6, 2, 1, 5, 7, 3]].astype(np.float32)
        else:
            x0, x1, y0, y1, z0, z1 = bounds
            c = np.array(
                [
                    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                ],
                dtype=np.float32,
            )
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        pts = np.vstack([c[list(e)] for e in edges]).astype(np.float32)

        if self._box is None:
            self._box = gl.GLLinePlotItem(pos=pts, color=BOX_COLOR, width=2.0, mode="lines", antialias=True)
            self.addItem(self._box)
        else:
            self._box.setData(pos=pts, color=BOX_COLOR, width=2.0, mode="lines")

    def set_quad(self, corners, thickness: float = 0.0):
        """Outline four board corners, and the slab in front of and behind them."""
        if corners is None or len(corners) != 4:
            for name in ("quad", "quad_front", "quad_back"):
                if name in self._markers:
                    self.removeItem(self._markers.pop(name))
            if hasattr(self, "_quad_item") and self._quad_item is not None:
                self.removeItem(self._quad_item)
                self._quad_item = None
            return

        pts = np.asarray(corners, dtype=np.float64)
        origin = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
        u, v, n = vt[0], vt[1], vt[2] / np.linalg.norm(vt[2])
        order = np.argsort(np.arctan2((pts - origin) @ v, (pts - origin) @ u))
        ring = pts[order]

        segs = []
        for offset in ((0,) if thickness <= 0 else (-thickness / 2, thickness / 2)):
            face = ring + offset * n
            segs.extend([face[i], face[(i + 1) % 4]] for i in range(4))
        if thickness > 0:
            for i in range(4):
                segs.append([ring[i] - thickness / 2 * n, ring[i] + thickness / 2 * n])
        line = np.vstack([np.array(s_) for s_ in segs]).astype(np.float32)

        if getattr(self, "_quad_item", None) is None:
            self._quad_item = gl.GLLinePlotItem(
                pos=line, color=BOX_COLOR, width=2.0, mode="lines", antialias=True
            )
            self.addItem(self._quad_item)
        else:
            self._quad_item.setData(pos=line, color=BOX_COLOR, width=2.0, mode="lines")

    def clear_box(self):
        if self._box is not None:
            self.removeItem(self._box)
            self._box = None

    # -------------------------------------------------------------------- view

    def apply_preset(self, name: str):
        preset = VIEW_PRESETS.get(name)
        if preset is None:
            return
        self.setCameraPosition(**preset)  # keep the current centre and distance

    def fit(self, xyz: np.ndarray | None = None, margin: float = 1.4):
        """Frame the data. Falls back to the whole cloud when nothing is focused."""
        if xyz is not None and len(xyz):
            xyz = np.asarray(xyz, dtype=np.float32)
            self._center = xyz.mean(axis=0)
            extent = float(np.linalg.norm(np.ptp(xyz, axis=0)))
        else:
            extent = 10.0
        self.setCameraPosition(pos=Vector(*self._center), distance=max(extent * margin, 0.5))

    def focus_on(self, bounds: tuple[float, float, float, float, float, float], margin: float = 1.8):
        """Put the orbit centre inside a region of interest.

        Framing the whole cloud instead leaves the centre tens of metres away
        from the board, and orbiting about a point that far off makes the area
        you actually care about nearly unreachable.
        """
        x0, x1, y0, y1, z0, z1 = bounds
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], dtype=np.float32)
        diagonal = float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))
        self._center = centre
        self.setCameraPosition(pos=Vector(*centre), distance=max(diagonal * margin, 0.5))

    # ------------------------------------------------------------ navigation

    def wheelEvent(self, ev):
        """Zoom towards the cursor, and keep zooming after the orbit centre.

        The stock behaviour only shrinks the orbit radius, so the camera stalls
        as it approaches its focal point and panning near it barely moves --
        the same wall rviz's orbit camera has. Dragging the focal point towards
        whatever is under the cursor lets the view keep going in.
        """
        delta = ev.angleDelta().y() or ev.angleDelta().x()
        if not delta:
            return
        if ev.modifiers() & QtCore.Qt.ControlModifier:  # keep fov on Ctrl
            self.opts["fov"] *= 0.999**delta
            self.update()
            return

        factor = 0.999**delta
        target = self._cursor_target(ev.position() if hasattr(ev, "position") else ev.posF())
        if target is not None:
            centre = _as_xyz(self.opts["center"])
            # Move the centre a fraction of the way to the target, matching how
            # much closer we just got, so the point under the cursor stays put.
            self.opts["center"] = pg.Vector(*(centre + (target - centre) * (1.0 - factor)))
        self.opts["distance"] *= factor
        self.update()

    def _cursor_target(self, pos) -> np.ndarray | None:
        """The drawn point nearest the cursor, if the cursor is near one."""
        pts = self._scatter.pos
        if pts is None or len(pts) == 0:
            return None
        pts = np.asarray(pts, dtype=np.float64)

        # Project every point once. Cheap at these sizes, and it avoids reading
        # back a depth buffer, which is slow and driver-dependent.
        view = np.array(self.viewMatrix().data(), dtype=np.float64).reshape(4, 4)
        w, h = max(self.width(), 1), max(self.height(), 1)
        f = 1.0 / np.tan(np.radians(self.opts["fov"]) / 2.0)
        cam = np.column_stack([pts, np.ones(len(pts))]) @ view
        depth = -cam[:, 2]
        ahead = depth > 1e-6
        if not ahead.any():
            return None

        sx = (cam[ahead, 0] / depth[ahead] * f * h / w * 0.5 + 0.5) * w
        sy = (0.5 - cam[ahead, 1] / depth[ahead] * f * 0.5) * h
        d2 = (sx - pos.x()) ** 2 + (sy - pos.y()) ** 2
        i = int(np.argmin(d2))
        if d2[i] > 80.0**2:  # nothing under the cursor
            return None
        return pts[ahead][i]

    # ---------------------------------------------------------------- GL probe

    def initializeGL(self):
        super().initializeGL()
        from OpenGL.GL import GL_RENDERER, GL_VERSION, glGetString

        def _s(enum):
            raw = glGetString(enum)
            return raw.decode("utf-8", "replace") if raw else "?"

        self.gl_renderer = _s(GL_RENDERER)
        self.gl_version = _s(GL_VERSION)

    def is_software_rendering(self) -> bool:
        low = self.gl_renderer.lower()
        return any(k in low for k in ("llvmpipe", "softpipe", "swrast", "software"))

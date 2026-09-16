"""Colouring a point cloud by one of its properties, rvizy-style.

A cloud drawn in one flat colour hides the thing you are looking for. The board
is a white panel with black markers, so intensity separates it from its
surroundings immediately -- and that channel is already in the messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Colour maps as control points, interpolated per point. Kept here rather than
# pulled from a plotting library so the view has no extra dependency.
COLORMAPS: dict[str, np.ndarray] = {
    "turbo": np.array([
        [0.19, 0.07, 0.23], [0.28, 0.40, 0.91], [0.11, 0.75, 0.83],
        [0.40, 0.95, 0.44], [0.87, 0.90, 0.20], [0.99, 0.55, 0.11],
        [0.83, 0.20, 0.03], [0.48, 0.02, 0.01],
    ]),
    "viridis": np.array([
        [0.27, 0.00, 0.33], [0.28, 0.17, 0.48], [0.23, 0.32, 0.55],
        [0.17, 0.45, 0.56], [0.13, 0.57, 0.55], [0.21, 0.72, 0.47],
        [0.57, 0.85, 0.27], [0.99, 0.91, 0.15],
    ]),
    "grey": np.array([[0.05, 0.05, 0.05], [1.0, 1.0, 1.0]]),
    "hot": np.array([
        [0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.0, 0.45, 0.0], [1.0, 1.0, 1.0],
    ]),
    "rainbow": np.array([
        [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 0.0],
    ]),
}

MODES = ("flat", "intensity", "axis", "distance")
AXES = ("x", "y", "z")


@dataclass
class ColorStyle:
    """How to paint a cloud. Mirrors rviz's Color Transformer settings."""

    mode: str = "intensity"
    colormap: str = "turbo"
    axis: str = "z"
    auto_bounds: bool = True
    min_value: float = 0.0
    max_value: float = 255.0
    flat_color: tuple = (0.25, 0.75, 1.00, 1.00)
    invert: bool = False


def _ramp(t: np.ndarray, name: str) -> np.ndarray:
    stops = COLORMAPS.get(name, COLORMAPS["turbo"])
    pos = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    lo = np.clip(pos.astype(int), 0, len(stops) - 2)
    frac = (pos - lo)[:, None]
    return stops[lo] * (1 - frac) + stops[lo + 1] * frac


def scalar_field(xyz: np.ndarray, intensity: np.ndarray | None, style: ColorStyle):
    """The per-point value a style paints by, or None for flat colouring."""
    if style.mode == "intensity":
        return intensity.astype(np.float64) if intensity is not None else None
    if style.mode == "axis":
        return xyz[:, AXES.index(style.axis)].astype(np.float64)
    if style.mode == "distance":
        return np.linalg.norm(xyz, axis=1)
    return None


def auto_range(values: np.ndarray) -> tuple[float, float]:
    """Robust bounds: percentiles, so a few stray returns do not flatten the map."""
    if len(values) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi - lo < 1e-9:
        lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize(
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    style: ColorStyle,
    alpha: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """RGBA per point, plus the value range used (None when flat)."""
    n = len(xyz)
    values = scalar_field(xyz, intensity, style)
    if values is None:
        rgba = np.tile(np.array(style.flat_color, dtype=np.float32), (n, 1))
        rgba[:, 3] *= alpha
        return rgba, None

    lo, hi = auto_range(values) if style.auto_bounds else (style.min_value, style.max_value)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    t = (values - lo) / (hi - lo)
    if style.invert:
        t = 1.0 - t

    rgba = np.ones((n, 4), dtype=np.float32)
    rgba[:, :3] = _ramp(t, style.colormap)
    rgba[:, 3] = alpha
    return rgba, (lo, hi)

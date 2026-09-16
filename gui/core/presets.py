"""Saved camera intrinsics and board specs, loaded from `gui/config/`.

Kept in data files rather than in code so a new camera or board can be added
without touching Python, and so the lists travel with the repo to other
machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from gui.core.project import Camera, Target

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_PATH = _CONFIG_DIR / "camera_intrinsics.yaml"
TARGET_CONFIG_PATH = _CONFIG_DIR / "target_specs.yaml"

_KEYS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
_TARGET_KEYS = (
    "marker_size",
    "delta_width_qr_center",
    "delta_height_qr_center",
    "delta_width_circles",
    "delta_height_circles",
    "circle_radius",
)


@dataclass
class CameraPreset:
    name: str
    note: str
    camera: Camera

    def matches(self, other: Camera, tol: float = 1e-6) -> bool:
        return all(abs(getattr(self.camera, k) - getattr(other, k)) <= tol for k in _KEYS)


def load(path: Path | None = None) -> list[CameraPreset]:
    path = path or CONFIG_PATH
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return []

    out = []
    for entry in data.get("cameras") or []:
        try:
            out.append(
                CameraPreset(
                    name=str(entry["name"]),
                    note=str(entry.get("note", "")),
                    camera=Camera(**{k: float(entry.get(k, 0.0)) for k in _KEYS}),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries rather than failing to start
    return out


def add(name: str, note: str, camera: Camera, path: Path | None = None):
    """Append a preset, replacing any existing one with the same name."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            data = {}

    entries = [e for e in (data.get("cameras") or []) if e.get("name") != name]
    entries.append({"name": name, "note": note, **{k: getattr(camera, k) for k in _KEYS}})
    data["cameras"] = entries
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


@dataclass
class TargetPreset:
    name: str
    note: str
    target: Target

    def matches(self, other: Target, tol: float = 1e-9) -> bool:
        return all(abs(getattr(self.target, k) - getattr(other, k)) <= tol for k in _TARGET_KEYS)


def load_targets(path: Path | None = None) -> list[TargetPreset]:
    path = path or TARGET_CONFIG_PATH
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return []

    out = []
    for entry in data.get("targets") or []:
        try:
            out.append(
                TargetPreset(
                    name=str(entry["name"]),
                    note=str(entry.get("note", "")),
                    target=Target(**{k: float(entry[k]) for k in _TARGET_KEYS}),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries rather than failing to start
    return out


def add_target(name: str, note: str, target: Target, path: Path | None = None):
    """Append a board spec, replacing any existing one with the same name."""
    path = path or TARGET_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            data = {}

    entries = [e for e in (data.get("targets") or []) if e.get("name") != name]
    entries.append({"name": name, "note": note, **{k: getattr(target, k) for k in _TARGET_KEYS}})
    data["targets"] = entries
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def names(presets) -> list[str]:
    return [p.name for p in presets]

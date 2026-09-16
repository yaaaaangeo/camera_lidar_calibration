"""Project state — everything the user builds up across the steps.

Split in two files on purpose:

  <name>.calib.yaml    camera, target, scene picks, filters. Small, portable,
                       meant to be committed and synced between machines.
  <name>.local.yaml    where the bag actually lives on *this* machine.

Scenes are identified by timestamp, so opening the project on another machine
only needs the bag to be located again -- everything else carries over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


# The package's own folders, resolved from this file so they hold wherever the
# repository is checked out.
#
#   calib_data    project files: which moments were captured and how the filter
#                 box was drawn. Working state on the way to an answer, useful to
#                 resume from but not something the team needs, so it stays out of
#                 the repository.
#   calib_result  the extrinsic itself, one file per vehicle. This is what the
#                 team actually consumes, so it is committed -- and then git log
#                 says when a vehicle's calibration changed and why.
PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = PACKAGE_DIR / "calib_data"
RESULT_DIR = PACKAGE_DIR / "calib_result"


@dataclass
class Camera:
    """Pinhole intrinsics: plumb_bob (5 coefficients) or rational_polynomial (8).

    k3 is included; the original tool forced it to zero. k4..k6 are the rational
    model's denominator terms, and OpenCV picks the model from how many
    coefficients it is handed -- so `dist()` returning eight is all solvePnP,
    projectPoints and undistort need to use them.

    Truncating a rational calibration to five is not a small approximation: the
    denominator disappears, so the curve changes shape rather than losing a
    high-order term, and the error runs to hundreds of pixels near the edges.
    """

    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    # Rational-model denominator. All three zero means plumb_bob, and then the
    # rational form is identical to it -- so a 5-coefficient camera stays
    # 5-coefficient everywhere, byte for byte.
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0

    @property
    def rational(self) -> bool:
        """Does this camera need the 8-coefficient form?"""
        return bool(self.k4 or self.k5 or self.k6)

    @property
    def is_set(self) -> bool:
        return self.fx > 0 and self.fy > 0

    def matrix(self):
        import numpy as np

        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def dist(self):
        import numpy as np

        coeffs = [self.k1, self.k2, self.p1, self.p2, self.k3]
        if self.rational:
            coeffs += [self.k4, self.k5, self.k6]
        return np.array(coeffs, dtype=np.float64)

    def dist_names(self) -> tuple[str, ...]:
        """Coefficient names in the order `dist()` returns them."""
        base = ("k1", "k2", "p1", "p2", "k3")
        return (base + ("k4", "k5", "k6")) if self.rational else base


@dataclass
class Target:
    """Calibration board geometry, in metres. Defaults are the stock CAD."""

    marker_size: float = 0.20
    delta_width_qr_center: float = 0.55
    delta_height_qr_center: float = 0.35
    delta_width_circles: float = 0.50
    delta_height_circles: float = 0.40
    circle_radius: float = 0.12

    def scaled(self, factor: float) -> "Target":
        return Target(**{k: v * factor for k, v in asdict(self).items()})


@dataclass
class FilterBox:
    """The region kept for detection: six bounds, plus an optional rotation.

    An axis-aligned box cannot isolate a board held at an angle. It has to span
    the board's full diagonal, and the wall behind fills the corners that leaves
    empty -- which is exactly what stops the holes being found. Turning the box
    to sit square with the board lets it be thin again.

    Rotation is about the box centre so the board does not slide off screen
    while being adjusted, and the bounds stay in the box's own frame.
    """

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    yaw: float = 0.0  # degrees, about the sensor's z
    pitch: float = 0.0  # about the box's y after yaw
    roll: float = 0.0  # about the box's x after pitch

    @property
    def is_set(self) -> bool:
        return self.x_max > self.x_min and self.y_max > self.y_min and self.z_max > self.z_min

    @property
    def rotated(self) -> bool:
        return bool(self.yaw or self.pitch or self.roll)

    def as_tuple(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def center(self):
        import numpy as np

        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        ])

    def rotation(self):
        """Box axes as columns: sensor frame = R @ box frame."""
        import numpy as np

        cy, sy = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        cp, sp = np.cos(np.radians(self.pitch)), np.sin(np.radians(self.pitch))
        cr, sr = np.cos(np.radians(self.roll)), np.sin(np.radians(self.roll))
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    def mask(self, xyz):
        """Which points fall inside, taking the rotation into account."""
        import numpy as np

        pts = np.asarray(xyz, np.float64)
        if self.rotated:
            pts = (pts - self.center()) @ self.rotation() + self.center()
        return (
            (pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max)
            & (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max)
            & (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        )

    def corners(self):
        """The eight corners in sensor coordinates, for drawing."""
        import numpy as np

        lo = np.array([self.x_min, self.y_min, self.z_min])
        hi = np.array([self.x_max, self.y_max, self.z_max])
        pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        if self.rotated:
            c = self.center()
            pts = (pts - c) @ self.rotation().T + c
        return pts


@dataclass
class PlaneRegionSpec:
    """Four board corners plus a slab thickness, saved with the project.

    An axis-aligned box cannot isolate a board held at an angle -- the volume it
    spans grows with the tilt and the wall behind fills the corners. Four corners
    give the plane and its outline together, so what is kept is what lies on the
    panel.
    """

    corners: list = field(default_factory=list)  # four [x, y, z]
    thickness: float = 0.10

    @property
    def is_set(self) -> bool:
        return len(self.corners) == 4


@dataclass
class Scene:
    """One captured moment: a timestamp, and which recording it came from.

    Recordings get made two ways. Sometimes it is one long take with the board
    carried around; sometimes it is a short bag per board position. `bag` says
    which file this scene belongs to, so both work and can even be mixed.
    Empty means the project's first bag.
    """

    id: str
    t_ns: int
    bag: str = ""
    # LiDAR sweeps to stack, centred on t_ns. Counted rather than timed: sensors
    # differ enough that a fixed window means very different things. A merged
    # 19 Hz cloud gives 63k points per message, while a raw 128-channel unit at
    # 10 Hz gives 920k -- the same half-second is 1M points on one and 9M on the
    # other.
    # One sweep by default. Stacking was meant to fill a hole's rim, but the two
    # ring-based detectors read points in acquisition order and extra sweeps blur
    # the crossings rather than sharpen them: on ten consecutive sweeps of a
    # stationary board the original method found 4/4 every time at one frame.
    # Raise it when a detector actually benefits, not by habit.
    frames: int = 1
    filter: FilterBox = field(default_factory=FilterBox)
    # Preferred over `filter` when set; see PlaneRegionSpec.
    region: PlaneRegionSpec = field(default_factory=PlaneRegionSpec)
    enabled: bool = True

    def roi(self):
        """Whatever should actually be used to cut the cloud."""
        if self.region.is_set:
            from gui.core.detect_lidar import PlaneRegion
            import numpy as np

            return PlaneRegion(
                corners=np.asarray(self.region.corners, float), thickness=self.region.thickness
            )
        return self.filter

    @property
    def roi_set(self) -> bool:
        return self.region.is_set or self.filter.is_set


@dataclass
class Project:
    name: str = "untitled"
    lidar_topic: str = ""
    camera_topic: str = ""
    camera: Camera = field(default_factory=Camera)
    target: Target = field(default_factory=Target)
    scenes: list[Scene] = field(default_factory=list)

    # Machine-local, kept out of the shared file. Scenes refer to entries here
    # by name, so moving the project to another machine only means pointing at
    # the recordings again.
    bag_paths: list[str] = field(default_factory=list)

    @property
    def bag_path(self) -> str:
        """The first recording -- what single-bag screens work against."""
        return self.bag_paths[0] if self.bag_paths else ""

    @bag_path.setter
    def bag_path(self, value: str):
        if not value:
            self.bag_paths = []
        elif value not in self.bag_paths:
            self.bag_paths.insert(0, value)

    def bag_for(self, scene: "Scene") -> str:
        """Resolve a scene's recording, tolerating a project saved elsewhere."""
        if not scene.bag:
            return self.bag_path
        if scene.bag in self.bag_paths:
            return scene.bag
        # Saved on another machine: match on file name.
        name = Path(scene.bag).name
        for candidate in self.bag_paths:
            if Path(candidate).name == name:
                return candidate
        return self.bag_path

    # ------------------------------------------------------------------- io

    def save(self, path: str | Path):
        """Write the whole project to one file.

        Bag paths used to live in a sibling `.local.yaml`, on the reasoning that
        they differ per machine and would otherwise be overwritten every time
        someone else committed. That holds when the project file is shared -- but
        it is not: `calib_data/` is ignored wholesale, and what the team consumes
        is the extrinsic in `calib_result/`. With nothing shared there is nothing
        to overwrite, and two files that must travel together are one more thing
        to get wrong.
        """
        path = Path(path)
        shared = {
            "name": self.name,
            "bag_paths": list(self.bag_paths),
            "lidar_topic": self.lidar_topic,
            "camera_topic": self.camera_topic,
            "camera": asdict(self.camera),
            "target": asdict(self.target),
            "scenes": [
                {
                    "id": s.id,
                    "t_ns": s.t_ns,
                    "bag": Path(s.bag).name if s.bag else "",
                    "frames": s.frames,
                    "filter": asdict(s.filter),
                    "region": asdict(s.region),
                    "enabled": s.enabled,
                }
                for s in self.scenes
            ],
        }
        path.write_text(yaml.safe_dump(shared, sort_keys=False, allow_unicode=True))

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        d = yaml.safe_load(path.read_text()) or {}
        p = cls(
            name=d.get("name", path.stem),
            lidar_topic=d.get("lidar_topic", ""),
            camera_topic=d.get("camera_topic", ""),
            camera=Camera(**(d.get("camera") or {})),
            target=Target(**(d.get("target") or {})),
            scenes=[
                Scene(
                    id=s["id"],
                    t_ns=s["t_ns"],
                    bag=s.get("bag", ""),
                    frames=int(s.get("frames", 1)),
                    filter=FilterBox(**(s.get("filter") or {})),
                    region=PlaneRegionSpec(**(s.get("region") or {})),
                    enabled=s.get("enabled", True),
                )
                for s in (d.get("scenes") or [])
            ],
        )
        p.bag_paths = list(d.get("bag_paths") or [])

        # Projects written before bag paths moved into this file kept them in a
        # sibling .local.yaml. Read it when it is there so those still open.
        if not p.bag_paths:
            legacy = path.with_suffix(".local.yaml")
            if legacy.exists():
                d_legacy = yaml.safe_load(legacy.read_text()) or {}
                p.bag_paths = list(d_legacy.get("bag_paths") or [])
                if not p.bag_paths and d_legacy.get("bag_path"):
                    p.bag_paths = [d_legacy["bag_path"]]
        return p

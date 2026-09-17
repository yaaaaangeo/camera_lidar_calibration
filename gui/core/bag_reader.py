"""Reading ROS1 bags without ROS.

Uses `rosbags`, a pure-Python implementation, so the GUI never needs a sourced
ROS environment or a running master. Bag paths are chosen by the user at
runtime; nothing here assumes any directory layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Message types we can currently turn into points or images. Anything else is
# listed for the user but not offered as a source.
LIDAR_TYPES = ("sensor_msgs/msg/PointCloud2",)
IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


# Older rosbag2 recordings carry no message definitions of their own, and the
# reader then refuses to open them. A stock ROS 2 typestore fills that gap. It is
# only a fallback: a bag that does embed its definitions -- every ROS 1 bag, and
# newer rosbag2 -- still uses its own, so this cannot silently override anything.
_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def _reader_for(paths) -> AnyReader:
    return AnyReader(list(paths), default_typestore=_TYPESTORE)


@dataclass
class TopicInfo:
    name: str
    msgtype: str
    count: int
    hz: float

    @property
    def kind(self) -> str:
        if self.msgtype in LIDAR_TYPES:
            return "lidar"
        if self.msgtype in IMAGE_TYPES:
            return "image"
        return "-"


@dataclass
class BagInfo:
    path: Path
    duration: float  # seconds
    start: float  # unix seconds
    end: float
    message_count: int
    topics: list[TopicInfo]

    def by_kind(self, kind: str) -> list[TopicInfo]:
        return [t for t in self.topics if t.kind == kind]


class BagSource:
    """A bag held open for repeated reads.

    Opening a large bag costs about a second, so the scrubber keeps one of
    these around rather than reopening per frame. Seeks go through the bag
    index, so jumping to an arbitrary time stays cheap.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader: AnyReader | None = None
        self.start_ns = 0
        self.end_ns = 0
        self._stamps: dict[str, list[int]] = {}

    def __enter__(self) -> "BagSource":
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self):
        if self._reader is not None:
            return
        self._reader = _reader_for([self.path])
        self._reader.open()
        self.start_ns, self.end_ns = self._reader.start_time, self._reader.end_time

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._stamps.clear()

    @property
    def duration(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def at_fraction(self, f: float) -> int:
        """Bag timestamp `f` of the way through the recording (0..1)."""
        return self.start_ns + int((self.end_ns - self.start_ns) * min(max(f, 0.0), 1.0))

    def _connections(self, topic: str):
        conns = [c for c in self._reader.connections if c.topic == topic]
        if not conns:
            raise KeyError(f"topic not in bag: {topic}")
        return conns

    def message_count(self, topic: str) -> int:
        """How many messages a topic holds, from the index rather than by reading."""
        self.open()
        info = self._reader.topics.get(topic)
        return int(getattr(info, "msgcount", 0) or 0) if info else 0

    def timestamps(self, topic: str, progress=None) -> list[int]:
        """Every message time on a topic, read once and kept.

        This walks the whole bag, so its cost scales with the recording rather
        than with what is being asked for: 17 seconds on a 36 GB file. Every scene
        switch used to call it again for the same topic and get the same answer,
        which made opening a scene take 43 seconds and read as a hung window.
        Held per source, so it is dropped when the bag is closed.

        `progress(done, total)` is called every few hundred messages when given.
        The first read is slow enough that silence is indistinguishable from a
        crash -- which is exactly how it was reported.
        """
        cached = self._stamps.get(topic)
        if cached is not None:
            return cached
        self.open()
        total = self.message_count(topic)
        stamps = []
        # Report often enough to look alive, rarely enough not to pay for it: a
        # signal per message would cost more than the read itself.
        step = max(total // 100, 200)
        for _, stamp, _ in self._reader.messages(connections=self._connections(topic)):
            stamps.append(stamp)
            if progress is not None and len(stamps) % step == 0:
                progress(len(stamps), total)
        if progress is not None:
            progress(len(stamps), total)
        self._stamps[topic] = stamps
        return stamps

    def first_after(self, topic: str, t_ns: int):
        """(timestamp, deserialised message) for the first message at or after t_ns."""
        self.open()
        for conn, stamp, raw in self._reader.messages(connections=self._connections(topic), start=t_ns):
            return stamp, self._reader.deserialize(raw, conn.msgtype)
        return None, None

    def nearest(self, topic: str, t_ns: int, window_ns: int = 200_000_000):
        """The message on `topic` closest to t_ns, searching a window either side."""
        self.open()
        best = (None, None)
        best_dt = None
        conns = self._connections(topic)
        for conn, stamp, raw in self._reader.messages(
            connections=conns, start=t_ns - window_ns, stop=t_ns + window_ns
        ):
            dt = abs(stamp - t_ns)
            if best_dt is None or dt < best_dt:
                best_dt, best = dt, (stamp, self._reader.deserialize(raw, conn.msgtype))
        return best

    def range(self, topic: str, start_ns: int, stop_ns: int):
        """Yield (timestamp, message) over a time window -- used for accumulation."""
        self.open()
        for conn, stamp, raw in self._reader.messages(
            connections=self._connections(topic), start=start_ns, stop=stop_ns
        ):
            yield stamp, self._reader.deserialize(raw, conn.msgtype)


# Only the pin lookup below needs to be exact rather than merely close, so its
# window just has to be comfortably tighter than real camera message spacing
# (tens of milliseconds at any normal frame rate) -- 1 ms leaves no room for a
# neighbouring message to be mistaken for the one actually pinned.
_PIN_WINDOW_NS = 1_000_000


def pick_camera_for_frame(src, camera_topic: str, lidar_t_ns: int, pinned_camera_t_ns: "int | None" = None):
    """Which camera message a LiDAR moment should be shown with.

    Ordinary navigation asks for the first camera message at or after the
    LiDAR timestamp (`first_after`) -- reproducible from `lidar_t_ns` alone,
    which is all normal timeline scrubbing has to work with.

    `pinned_camera_t_ns` exists for exactly one case: jumping to a frame a
    *different* pairing rule already scored -- Multi-frame Consistency pairs
    each sampled LiDAR moment with its temporally *nearest* camera image, not
    the first one after it, because on a moving vehicle a loose pairing shows
    up in the pixel error just like a rotation error would. Left to
    `first_after`, jumping to that frame's timestamp could display a
    different image than the one actually scored -- looking directly for the
    exact camera timestamp the evaluation used is what closes that gap,
    without needing a second rendering path: this is still the same
    `(stamp, message)` pair `first_after`/`nearest` always return, just with
    the search anchored on the known camera stamp instead of the LiDAR one.
    """
    if pinned_camera_t_ns is not None:
        return src.nearest(camera_topic, pinned_camera_t_ns, window_ns=_PIN_WINDOW_NS)
    return src.first_after(camera_topic, lidar_t_ns)


def sweeps_around(src: "BagSource", topic: str, t_ns: int, frames: int = 5, progress=None):
    """The individual sweeps nearest t_ns, newest-in-the-middle order kept.

    Split out from `accumulate_cloud` so a caller can try one sweep, then three,
    then five without reading the bag again. The window grows symmetrically about
    the centre sweep, so the smaller counts are exactly the middle of the larger
    ones -- trying several depths costs one read, not one read per depth.

    Returns (chunks, intensities, rings); the lists line up and any of the last
    two may hold None where the topic does not carry that field.
    """
    import numpy as np

    from gui.core.decode import cloud_extra, cloud_to_struct

    stamps = sorted(src.timestamps(topic, progress=progress))
    if not stamps:
        return [], [], []

    want = max(int(frames), 1)
    centre = min(range(len(stamps)), key=lambda i: abs(stamps[i] - t_ns))
    lo = max(0, centre - want // 2)
    hi = min(len(stamps), lo + want)
    lo = max(0, hi - want)

    chunks, intensities, rings = [], [], []
    for _, msg in src.range(topic, stamps[lo], stamps[hi - 1] + 1):
        rec = cloud_to_struct(msg)
        xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        chunks.append(xyz[finite])
        inten = cloud_extra(msg, "intensity")
        intensities.append(None if inten is None else np.asarray(inten)[finite].astype(np.float32))
        rg = cloud_extra(msg, "ring")
        rings.append(None if rg is None else np.asarray(rg)[finite].astype(np.int32))
    return chunks, intensities, rings


def stack_sweeps(chunks, intensities, rings):
    """Glue sweeps into one cloud, as `accumulate_cloud` returns them."""
    import numpy as np

    if not chunks:
        empty = np.empty((0, 3), np.float32)
        return empty, 0, empty, None, None
    mid = len(chunks) // 2
    intensity = np.concatenate(intensities) if all(i is not None for i in intensities) else None
    # Sweeps are stacked, so ring numbers repeat. Offset each sweep so a walk
    # along "one ring" does not jump between sweeps.
    if all(r is not None for r in rings):
        span = max(int(r.max()) + 1 for r in rings)
        ring = np.concatenate([r + i * span for i, r in enumerate(rings)])
    else:
        ring = None
    return np.vstack(chunks), len(chunks), chunks[mid], intensity, ring


def middle_slice(items, want: int):
    """The middle `want` entries -- how a smaller sweep count sits inside a larger."""
    if want >= len(items):
        return list(items)
    lo = (len(items) - want) // 2
    return list(items[lo:lo + want])


def accumulate_cloud(src: "BagSource", topic: str, t_ns: int, frames: int = 5, progress=None):
    """Stack `frames` sweeps centred on t_ns.

    Counted, not timed. Sensors differ too much for a fixed window to mean the
    same thing: a merged 19 Hz cloud carries 63k points per message where a raw
    128-channel unit at 10 Hz carries 920k, so half a second is 1M points on one
    and 9M on the other.

    Returns (stacked, count, middle sweep, intensity, ring). `ring` is None when
    the topic does not carry one -- a merged cloud never does, since ring numbers
    from different sensors cannot be combined.

    The middle sweep comes back separately because point spacing must be judged
    on a single sweep; the stack's nearest-neighbour distance reports the voxel
    size instead.
    """
    return stack_sweeps(*sweeps_around(src, topic, t_ns, frames, progress))


def inspect(path: str | Path) -> BagInfo:
    """Read a bag's index and report what is inside. Does not decode messages."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with _reader_for([path]) as reader:
        start_ns, end_ns = reader.start_time, reader.end_time
        duration = (end_ns - start_ns) / 1e9

        topics = []
        for name, info in reader.topics.items():
            count = getattr(info, "msgcount", 0) or 0
            topics.append(
                TopicInfo(
                    name=name,
                    msgtype=getattr(info, "msgtype", "?"),
                    count=count,
                    hz=count / duration if duration > 0 else 0.0,
                )
            )

        return BagInfo(
            path=path,
            duration=duration,
            start=start_ns / 1e9,
            end=end_ns / 1e9,
            message_count=getattr(reader, "message_count", sum(t.count for t in topics)),
            topics=sorted(topics, key=lambda t: t.name),
        )

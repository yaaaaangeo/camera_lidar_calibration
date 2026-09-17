"""Unit tests for gui.core.bag_reader.pick_camera_for_frame.

No real bag file needed: a tiny stand-in exposing the two methods
pick_camera_for_frame actually calls is enough to pin down the pairing rule,
and keeps this test free of PySide6/rosbags entirely.
"""

from __future__ import annotations

from gui.core.bag_reader import BagSource, pick_camera_for_frame


def test_bagsource_keeps_its_full_public_interface():
    """A regression guard, not a behaviour test: pick_camera_for_frame was
    briefly mis-indented during development so that it landed *inside*
    BagSource's body, silently turning `range` from a method into a dead
    nested function and breaking accumulate_cloud/sweeps_around (and so every
    step that reads a LiDAR sweep) without a single existing test noticing --
    none of them construct a real BagSource. This just checks the class
    still has every method it is supposed to."""
    expected = {
        "open", "close", "duration", "at_fraction", "message_count",
        "timestamps", "first_after", "nearest", "range",
    }
    missing = expected - set(dir(BagSource))
    assert not missing, f"BagSource is missing: {missing}"


class _FakeSource:
    """A fixed set of "camera messages" (timestamp, label), searched the same
    way BagSource.first_after / .nearest do."""

    def __init__(self, camera_stamps_ns):
        self._stamps = sorted(camera_stamps_ns)

    def first_after(self, topic: str, t_ns: int):
        for stamp in self._stamps:
            if stamp >= t_ns:
                return stamp, f"frame@{stamp}"
        return None, None

    def nearest(self, topic: str, t_ns: int, window_ns: int = 200_000_000):
        candidates = [s for s in self._stamps if abs(s - t_ns) <= window_ns]
        if not candidates:
            return None, None
        best = min(candidates, key=lambda s: abs(s - t_ns))
        return best, f"frame@{best}"


def test_pick_camera_for_frame_normal_navigation_matches_first_after():
    """Ordinary scrubbing (no pin) must behave exactly as Step 7 always has."""
    src = _FakeSource([9_997_000_000, 10_018_000_000])
    stamp, msg = pick_camera_for_frame(src, "cam", lidar_t_ns=10_000_000_000)
    assert stamp == 10_018_000_000  # 9.997s is before the LiDAR moment, first_after skips it
    assert msg == "frame@10018000000"


def test_pick_camera_for_frame_pin_reproduces_multiframe_nearest_pairing():
    """The scenario from the bug report: LiDAR at 10.000s, a camera frame at
    9.997s that `nearest` would score (3ms away) against one at 10.018s that
    plain `first_after` lands on instead (18ms away, but the first one *at or
    after* the LiDAR moment). Jumping to the evaluated frame must land on
    9.997s, not 10.018s."""
    src = _FakeSource([9_997_000_000, 10_018_000_000])

    # What Multi-frame evaluation actually used.
    evaluated_stamp, _ = src.nearest("cam", 10_000_000_000, window_ns=200_000_000)
    assert evaluated_stamp == 9_997_000_000

    # Sanity check: unpinned navigation really would have shown a different
    # frame -- otherwise this test would not be exercising the fix at all.
    unpinned_stamp, _ = pick_camera_for_frame(src, "cam", lidar_t_ns=10_000_000_000)
    assert unpinned_stamp == 10_018_000_000

    pinned_stamp, pinned_msg = pick_camera_for_frame(
        src, "cam", lidar_t_ns=10_000_000_000, pinned_camera_t_ns=evaluated_stamp,
    )
    assert pinned_stamp == evaluated_stamp
    assert pinned_msg == "frame@9997000000"


def test_pick_camera_for_frame_pin_is_precise_even_with_a_close_neighbour():
    """The pin's search window must be tight enough not to grab a
    neighbouring message a few milliseconds away instead of the exact one."""
    src = _FakeSource([10_000_000_000, 10_004_000_000])  # 4ms apart
    stamp, _ = pick_camera_for_frame(src, "cam", lidar_t_ns=0, pinned_camera_t_ns=10_004_000_000)
    assert stamp == 10_004_000_000


def test_pick_camera_for_frame_pin_missing_returns_none():
    src = _FakeSource([10_003_000_000])
    stamp, msg = pick_camera_for_frame(src, "cam", lidar_t_ns=0, pinned_camera_t_ns=99_000_000_000)
    assert stamp is None and msg is None

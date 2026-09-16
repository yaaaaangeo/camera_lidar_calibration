"""Print what is inside a bag.

    python3.10 gui/check_bag.py /path/to/one.bag [more.bag ...]

Reads only the index, so it is fast even on very large bags.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui.core.bag_reader import inspect  # noqa: E402


def show(path: str):
    try:
        info = inspect(path)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        print(f"\n{path}\n  읽기 실패: {type(exc).__name__}: {exc}")
        return

    size_gb = info.path.stat().st_size / 1e9
    started = datetime.fromtimestamp(info.start).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{info.path}")
    print(f"  {size_gb:.1f} GB | {info.duration:.1f} s | {started} | {info.message_count:,} msgs")
    print(f"  {'topic':<40} {'type':<34} {'count':>9} {'Hz':>7}  kind")
    print(f"  {'-' * 40} {'-' * 34} {'-' * 9} {'-' * 7}  ----")
    for t in info.topics:
        print(f"  {t.name:<40} {t.msgtype:<34} {t.count:>9,} {t.hz:>7.1f}  {t.kind}")

    lidar = [t.name for t in info.by_kind("lidar")]
    image = [t.name for t in info.by_kind("image")]
    print(f"  => LiDAR 후보: {lidar or '없음'}")
    print(f"  => 이미지 후보: {image or '없음'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for p in sys.argv[1:]:
        show(p)

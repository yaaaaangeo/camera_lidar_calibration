"""Pull one LiDAR frame and one image out of a bag and report what came back.

Answers the questions the GUI design depends on: what fields the cloud has,
whether the board is in it, and how long a single frame costs.

    python3.10 gui/check_frame.py BAG --lidar TOPIC --camera TOPIC [--at 0.5] [--show]

--at is a fraction of the recording (0.5 = halfway). --show opens the 3D view.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui.core.decode import cloud_extra, cloud_fields, cloud_xyz, image_to_bgr  # noqa: E402
from pathlib import Path  # noqa: E402


def first_message_after(reader, topic: str, t_ns: int):
    """Read the first message on `topic` at or after `t_ns`, using the bag index."""
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        raise KeyError(f"topic not in bag: {topic}")
    for conn, stamp, raw in reader.messages(connections=conns, start=t_ns):
        return stamp, reader.deserialize(raw, conn.msgtype)
    raise LookupError(f"no message on {topic} after t={t_ns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--at", type=float, default=0.5, help="fraction into the recording")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outdir", default="/tmp")
    args = ap.parse_args()

    t_open = time.perf_counter()
    with AnyReader([Path(args.bag)]) as reader:
        open_s = time.perf_counter() - t_open
        span = reader.end_time - reader.start_time
        target = reader.start_time + int(span * args.at)
        print(f"bag 열기        {open_s:6.2f} s   (길이 {span / 1e9:.1f} s)")

        t0 = time.perf_counter()
        l_stamp, l_msg = first_message_after(reader, args.lidar, target)
        t_read_l = time.perf_counter() - t0

        t0 = time.perf_counter()
        xyz = cloud_xyz(l_msg)
        t_dec_l = time.perf_counter() - t0

        t0 = time.perf_counter()
        c_stamp, c_msg = first_message_after(reader, args.camera, target)
        t_read_c = time.perf_counter() - t0

        t0 = time.perf_counter()
        img = image_to_bgr(c_msg)
        t_dec_c = time.perf_counter() - t0

        ring = cloud_extra(l_msg, "ring")
        inten = cloud_extra(l_msg, "intensity")

    print("\n=== 소요 시간 ===")
    print(f"  LiDAR  탐색 {t_read_l * 1e3:7.1f} ms   디코딩 {t_dec_l * 1e3:7.1f} ms")
    print(f"  이미지 탐색 {t_read_c * 1e3:7.1f} ms   디코딩 {t_dec_c * 1e3:7.1f} ms")
    print(f"  한 프레임 합계  {(t_read_l + t_dec_l + t_read_c + t_dec_c) * 1e3:.0f} ms")

    print("\n=== 포인트클라우드 ===")
    print(f"  필드: {[f'{n}:{d}@{o}' for n, d, o in cloud_fields(l_msg)]}")
    print(f"  point_step={l_msg.point_step}  width={l_msg.width}  height={l_msg.height}")
    print(f"  유효 점수: {len(xyz):,}")
    if len(xyz):
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        print(f"  범위  x [{lo[0]:7.2f}, {hi[0]:7.2f}]  y [{lo[1]:7.2f}, {hi[1]:7.2f}]  z [{lo[2]:7.2f}, {hi[2]:7.2f}]")
        r = np.linalg.norm(xyz, axis=1)
        print(f"  거리  중앙값 {np.median(r):.2f} m   최대 {r.max():.2f} m")
    if ring is not None:
        print(f"  ring 필드 있음: {len(np.unique(ring))}개 링 (0~{ring.max()})")
    else:
        print("  ring 필드 없음  <- 기계식 경로는 ring이 필요합니다")
    if inten is not None:
        print(f"  intensity 있음: {inten.min()}~{inten.max()}")

    print("\n=== 이미지 ===")
    print(f"  {img.shape[1]}x{img.shape[0]}  dt={(c_stamp - l_stamp) / 1e6:+.1f} ms (LiDAR 대비)")

    os.makedirs(args.outdir, exist_ok=True)
    img_path = os.path.join(args.outdir, "frame.png")
    import cv2

    cv2.imwrite(img_path, img)
    print(f"  저장: {img_path}")

    # Quick ArUco probe -- tells us straight away whether the board is in frame.
    import cv2.aruco as aruco

    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_6X6_250))
    corners, ids, _ = det.detectMarkers(img)
    print(f"  ArUco 검출: {0 if ids is None else len(ids)}개  ids={None if ids is None else sorted(ids.ravel().tolist())}")

    if args.show:
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        from PySide6 import QtWidgets

        from gui.ui.cloud_view import CloudView

        app = QtWidgets.QApplication(sys.argv)
        v = CloudView()
        v.setWindowTitle(f"{args.lidar}  ({len(xyz):,} pts)")
        v.resize(1100, 800)
        v.set_points(xyz, (0.35, 0.8, 1.0, 0.9), size=2.0)
        v.fit(xyz)
        v.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()

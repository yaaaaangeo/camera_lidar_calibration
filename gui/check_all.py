"""Run detection across a folder of bags and report what worked.

    python3.10 gui/check_all.py BAG [BAG ...] --camera NAME --lidar TOPIC --camera-topic TOPIC

Regression harness: run every recording through the detector and print what came
out, so a change can be checked against all of them at once rather than the one
that happened to be open.

Filter boxes come from a YAML file, not from the cloud. Finding the board
automatically is its own hard problem -- the camera's range cannot be reused
because the two sensors sit metres apart (7 m from the camera reads 12 m from
the LiDAR on this rig), and picking the flattest board-sized cluster lands on
walls and pillars instead. Since the point here is to test the *detector*, the
boxes are given, exactly as a user would draw them.

    python3.10 gui/check_all.py cases.yaml

    cases:
      - bag: /path/to/one.bag
        lidar: /sensor/lidar/merge/no_ground_points
        camera_topic: /sensor/camera/top_center/compressed
        camera: 31CAM_HFOV54          # preset name
        box: [11.50, 12.50, -3.10, -2.55, 1.20, 1.70]
        at: 0.5                        # optional, fraction through the bag
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.core import detect_lidar as dl  # noqa: E402
from gui.core import presets  # noqa: E402
from gui.core.bag_reader import BagSource, accumulate_cloud  # noqa: E402
from gui.core.decode import image_to_bgr  # noqa: E402
from gui.core.detect_camera import detect as detect_camera  # noqa: E402
from gui.core.project import FilterBox, Target  # noqa: E402


def best_frame(src: BagSource, topic: str, camera, target, probes: int = 12):
    """The sampled frame whose ArUco pose looks strongest."""
    stamps = sorted(src.timestamps(topic))
    if not stamps:
        return None, None
    step = max(len(stamps) // probes, 1)
    best = (None, None, -1.0)
    for t in stamps[::step]:
        _, msg = src.first_after(topic, t)
        if msg is None:
            continue
        det = detect_camera(image_to_bgr(msg), camera, target)
        if not det.ok:
            continue
        # Prefer big markers and a tilted board; both make the pose better
        # conditioned, which is what step 4 tells the user to aim for.
        score = det.marker_px * (1.0 + det.tilt_deg / 45.0) / max(det.reproj_rms, 0.3)
        if score > best[2]:
            best = (t, det, score)
    return best[0], best[1]


def check(case: dict, target: Target, camera) -> dict:
    out = {"bag": Path(case["bag"]).name, "note": ""}
    box = FilterBox(*case["box"])
    with BagSource(case["bag"]) as src:
        t = src.at_fraction(float(case.get("at", 0.5)))
        _, img = src.first_after(case["camera_topic"], t)
        cam = detect_camera(image_to_bgr(img), camera, target) if img is not None else None
        cloud, n, single, _ = accumulate_cloud(src, case["lidar"], t, float(case.get("window", 0.5)))

    out["cam"] = cam.n_markers if cam else 0
    out["cam_ok"] = bool(cam and cam.ok)
    if not len(cloud):
        out["note"] = "LiDAR 없음"
        return out

    inside = dl.apply_box(single, box)
    spacing = dl.point_spacing(inside) if len(inside) > 20 else 0.0
    out.update(spacing=spacing, sweeps=n, box_pts=int(dl.box_mask(cloud, box).sum()))

    for method in ("occupancy", "boundary"):
        params = dl.DetectParams(sweep_spacing=spacing, method=method)
        if method == "boundary" and spacing > 0:
            params.boundary_radius = max(params.boundary_radius, spacing * 4)
            params.cluster_tolerance = max(params.cluster_tolerance, spacing * 2.5)
        start_t = time.perf_counter()
        det = dl.detect(cloud, box, target, params)
        out[method] = {
            "n": det.n_circles,
            "ms": (time.perf_counter() - start_t) * 1000,
            "diag_err": diagonal_error(det.centers, target),
            "reason": det.reason,
        }
    return out


def diagonal_error(centers, target: Target):
    """Worst deviation of the found quad's diagonals from the board, in mm."""
    if centers is None or len(centers) != 4:
        return float("nan")
    rel = centers - centers.mean(axis=0)
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    ring = centers[np.argsort(np.arctan2(rel @ vt[1], rel @ vt[0]))]
    want = np.hypot(target.delta_width_circles, target.delta_height_circles)
    got = [np.linalg.norm(ring[2] - ring[0]), np.linalg.norm(ring[3] - ring[1])]
    return max(abs(g - want) for g in got) * 1000


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("cases", help="YAML file listing bags, topics and filter boxes")
    ap.add_argument("--scale", type=float, default=0.594, help="board scale vs the stock CAD")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.cases).read_text()) or {}
    cases = spec.get("cases") or []
    target = Target().scaled(spec.get("scale", args.scale))
    cams = {p.name: p.camera for p in presets.load()}

    print(f"{'bag':<24} {'ArUco':>6} {'간격':>6} {'박스점':>8} {'격자':>13} {'경계':>13}")
    rows = []
    for case in cases:
        try:
            r = check(case, target, cams[case["camera"]])
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the sweep
            print(f"{Path(case['bag']).name:<24} 오류: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        if r["note"]:
            print(f"{r['bag']:<24} {r['note']}")
            continue

        def cell(m):
            d = r[m]
            return f"{d['n']}/4 {d['diag_err']:5.0f}mm" if d["n"] == 4 else f"{d['n']}/4       -"

        print(
            f"{r['bag']:<24} {r['cam']:>3}/4{'✓' if r['cam_ok'] else ' '}"
            f" {r['spacing'] * 1000:5.0f}mm {r['box_pts']:>8,}"
            f" {cell('occupancy'):>13} {cell('boundary'):>13}"
        )

    good = [r for r in rows if not r["note"]]
    if not good:
        return
    print()
    for method, label in (("occupancy", "점유 격자"), ("boundary", "경계점  ")):
        ok = [r for r in good if r[method]["n"] == 4]
        errs = [r[method]["diag_err"] for r in ok if np.isfinite(r[method]["diag_err"])]
        ms = np.mean([r[method]["ms"] for r in good])
        line = f"{label}: {len(ok)}/{len(good)} 검출"
        if errs:
            line += f", 대각오차 중앙 {np.median(errs):.0f}mm 최대 {np.max(errs):.0f}mm"
        print(line + f", 평균 {ms:.0f}ms")
    cam_ok = sum(1 for r in good if r["cam_ok"])
    print(f"카메라   : {cam_ok}/{len(good)} 검출")


if __name__ == "__main__":
    main()

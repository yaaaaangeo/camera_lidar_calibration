"""Measure detection against synthetic ground truth, over a grid of conditions.

    python3.10 gui/check_synth.py                       # regression: current defaults
    python3.10 gui/check_synth.py --sweep ring_gap_frac=0.1,0.2,0.3,0.4,0.6
    python3.10 gui/check_synth.py --method ring --detail

Two jobs, and the difference matters.

*Regression* runs the current defaults across the grid and reports what fraction
of conditions each method handles. Run it after any change to `detect_lidar`.

*Sweep* runs one threshold at several values across the same grid. What it is for
is finding the plateau -- the span of values that all work -- not the best value.
A best value is fitted to whatever conditions produced it and stops being true
when they change; the middle of a wide plateau does not. If a sweep shows no
plateau, or a narrow one, that is evidence the threshold has the wrong *form*
and should be derived from something that scales, not nudged to a better number.

Why a grid and not one sensor: a threshold measured at one resolution says
nothing about another. The grid spans denser and sparser than the demo rig,
which was measured at 0.133 deg between rings and 0.118 deg along one, with a
5.3 mm plane residual at 3.8 m -- that row is included so real captures can be
placed against the table.

Ground truth here is exact: the generator places the holes, so the centre error
is the real one. What the generator cannot vouch for is whether it resembles the
sensor in hand -- check that separately, with `check_bag.py`, which reports the
same input statistics (ring spacing, along-ring spacing, plane residual) from a
recording without reference to whether detection succeeded.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.core import detect_lidar as dl  # noqa: E402
from gui.core import synth  # noqa: E402
from gui.core.project import Target  # noqa: E402

METHODS = ("original", "ring", "occupancy", "boundary")

# (label, degrees between rings, degrees along a ring). Spans either side of the
# demo rig so a threshold that only works there shows up as one row passing.
SENSORS = [
    ("dense   ", 0.090, 0.090),
    ("demo rig", 0.133, 0.118),
    ("wide-v  ", 0.267, 0.118),
    ("sparse  ", 0.400, 0.200),
]
NOISES = (0.006, 0.015)
RANGES = (2.0, 4.0, 6.0, 9.0)
POSES = 3

_scans: dict = {}


def scan_cached(target, dist, v_res, h_res, noise, sweeps, seed):
    """A scan plus its single-sweep spacing, kept so a sweep does not re-trace.

    Ray tracing dominates the runtime and does not depend on any detection
    threshold, so every value in a sweep sees the identical cloud -- which also
    makes the comparison exact rather than merely similar.
    """
    key = (dist, v_res, h_res, noise, sweeps, seed, target.circle_radius)
    if key in _scans:
        return _scans[key]

    sensor = synth.Sensor(v_res_deg=v_res, h_res_deg=h_res, range_noise_m=noise)
    pose = synth.BoardPose(
        distance=dist,
        yaw_deg=5 + 12 * seed,
        pitch_deg=-8 + 7 * seed,
        azimuth_deg=-6 + 4 * seed,
    )
    pts, truth, ring = synth.scan(target, pose, sensor, sweeps=sweeps, seed=seed)
    box = synth.bounding_box(truth, pad=0.15)
    if sweeps == 1:
        spacing = dl.point_spacing(dl.apply_region(pts, box))
    else:
        one, _, _ = synth.scan(target, pose, sensor, sweeps=1, seed=seed)
        spacing = dl.point_spacing(dl.apply_region(one, box))

    _scans[key] = (pts, truth, ring, box, spacing)
    return _scans[key]


def one_case(target, method, params, dist, v_res, h_res, noise, sweeps, seed):
    """(detected, centre error, fitted radius) for a single condition."""
    pts, truth, ring, box, spacing = scan_cached(
        target, dist, v_res, h_res, noise, sweeps, seed
    )
    p = replace(params, method=method, sweep_spacing=spacing)
    if method == "boundary" and spacing > 0:
        # The C++ constants assume the stock board; scale the two that are
        # distances so this path is judged at its best rather than at 0/N.
        p.boundary_radius = max(p.boundary_radius, spacing * 4)
        p.cluster_tolerance = max(p.cluster_tolerance, spacing * 2.5)
    det = dl.detect(pts, box, target, p, ring=ring)
    return (
        det.ok,
        synth.centre_error(det.centers, truth) if det.ok else np.nan,
        float(np.mean(det.radii)) if det.radii else np.nan,
    )


def grid(target, method, params, sweeps, ranges, poses):
    """Every condition in the grid, keyed by (sensor label, noise, range)."""
    out = {}
    for label, v_res, h_res in SENSORS:
        for noise in NOISES:
            for dist in ranges:
                out[(label, noise, dist)] = [
                    one_case(target, method, params, dist, v_res, h_res, noise, sweeps, s)
                    for s in range(poses)
                ]
    return out


def summarise(results):
    """(detected, total, mean error mm, worst error mm) over a set of cases."""
    flat = [c for cases in results.values() for c in cases]
    ok = sum(1 for o, _, _ in flat if o)
    errs = [e for o, e, _ in flat if o and np.isfinite(e)]
    return (
        ok,
        len(flat),
        np.mean(errs) * 1000 if errs else float("nan"),
        np.max(errs) * 1000 if errs else float("nan"),
    )


def print_detail(target, method, params, sweeps, ranges, poses):
    """Per-condition table, so a failure can be attributed to a condition."""
    print(f"\n[{method}] 조건별 (자세 {poses}종, 스윕 {sweeps})")
    header = "센서       잡음 | " + " ".join(f"{d:>13.0f}m" for d in ranges)
    print(header)
    print("-" * len(header))
    for label, v_res, h_res in SENSORS:
        for noise in NOISES:
            cells = []
            for dist in ranges:
                cases = [
                    one_case(target, method, params, dist, v_res, h_res, noise, sweeps, s)
                    for s in range(poses)
                ]
                ok = sum(1 for o, _, _ in cases if o)
                errs = [e for o, e, _ in cases if o and np.isfinite(e)]
                cells.append(
                    f"{ok}/{poses} {np.mean(errs) * 1000:5.1f}mm" if errs else f"{ok}/{poses}       -"
                )
            print(f"{label} {noise * 1000:3.0f}mm | " + " ".join(f"{c:>14}" for c in cells))


def sweep(target, name, values, methods, params, sweeps, ranges, poses):
    """One threshold at several values, across the whole grid.

    Printed as coverage rather than as a winner, because the point is the shape
    of the curve: a wide flat span means the value is not doing the work, which
    is what a derived threshold should look like.
    """
    print(f"\n{name} 스윕 — 격자 전체 {len(SENSORS) * len(NOISES) * len(ranges) * poses}개 조건")
    print(f"{'값':>10} | " + " | ".join(f"{m:^26}" for m in methods))
    rows = []
    for value in values:
        trial = replace(params, **{name: value})
        cells, row = [], {"value": value}
        for method in methods:
            ok, total, mean_e, worst = summarise(
                grid(target, method, trial, sweeps, ranges, poses)
            )
            row[method] = (ok, total, mean_e, worst)
            frac = ok / total if total else 0.0
            cells.append(
                f"{ok:3}/{total:3} ({frac * 100:3.0f}%) 평균{mean_e:5.1f} 최대{worst:5.1f}mm"
                if np.isfinite(mean_e) else f"{ok:3}/{total:3} ({frac * 100:3.0f}%)"
            )
        rows.append(row)
        print(f"{value:>10.4g} | " + " | ".join(f"{c:^26}" for c in cells))

    # Name the plateau explicitly: the run of values within a whisker of the best
    # coverage. A single peak and a long flat run mean different things and the
    # table alone makes them easy to confuse.
    print()
    for method in methods:
        counts = [r[method][0] for r in rows]
        best = max(counts)
        if best == 0:
            print(f"  {method:<10} 전 구간 실패")
            continue
        near = [r["value"] for r, c in zip(rows, counts) if c >= best - max(1, best // 20)]
        span = f"{min(near):.4g} ~ {max(near):.4g}" if len(near) > 1 else f"{near[0]:.4g}"
        mid = float(np.median(near))
        note = "고원" if len(near) >= 3 else ("좁음 — 형태 재검토" if len(near) == 1 else "좁음")
        print(f"  {method:<10} 최고 {best}, {note}: {span} (중앙 {mid:.4g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.594, help="board scale vs the stock CAD")
    ap.add_argument("--sweeps", type=int, default=1, help="LiDAR sweeps accumulated per case")
    ap.add_argument("--poses", type=int, default=POSES)
    ap.add_argument("--ranges", type=float, nargs="+", default=list(RANGES))
    ap.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--detail", action="store_true", help="print the per-condition table")
    ap.add_argument("--sweep", metavar="NAME=V1,V2,...",
                    help="one DetectParams field at several values")
    args = ap.parse_args()

    target = Target().scaled(args.scale)
    params = dl.DetectParams()
    half_w, half_h = synth.board_extent(target)
    print(f"보드 {args.scale:.3f}배 · 구멍 반지름 {target.circle_radius * 1000:.1f}mm "
          f"· 패널 {half_w * 2000:.0f}x{half_h * 2000:.0f}mm")
    print(f"격자: 센서 {len(SENSORS)}종 x 잡음 {len(NOISES)}종 x 거리 {len(args.ranges)}종 "
          f"x 자세 {args.poses}종 = {len(SENSORS) * len(NOISES) * len(args.ranges) * args.poses}조건 "
          f"(스윕 {args.sweeps})")

    start = time.perf_counter()
    if args.sweep:
        name, _, raw = args.sweep.partition("=")
        name = name.strip()
        if not hasattr(params, name):
            sys.exit(f"DetectParams 에 없는 항목: {name}")
        current = getattr(params, name)
        values = [type(current)(v) for v in raw.split(",")]
        sweep(target, name, values, args.method, params, args.sweeps, args.ranges, args.poses)
        print(f"\n현재 기본값: {name} = {current}")
    else:
        for method in args.method:
            res = grid(target, method, params, args.sweeps, args.ranges, args.poses)
            ok, total, mean_e, worst = summarise(res)
            line = f"{method:<10} {ok:3}/{total:3} ({ok / total * 100:3.0f}%) 검출"
            if np.isfinite(mean_e):
                line += f", 중심오차 평균 {mean_e:.1f}mm 최대 {worst:.1f}mm"
            print(line)
        if args.detail:
            for method in args.method:
                print_detail(target, method, params, args.sweeps, args.ranges, args.poses)

    print(f"\n{time.perf_counter() - start:.0f}초, 광선추적 {len(_scans)}회")


if __name__ == "__main__":
    main()

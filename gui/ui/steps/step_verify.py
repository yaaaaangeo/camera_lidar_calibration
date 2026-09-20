"""Step 7 — put the cloud back on the image and decide whether to believe it.

Every number before this page can look right while the answer is wrong. Four
hole centres form a rectangle, a rectangle is unchanged by a half turn, and so
an extrinsic rotated 180 degrees about the board normal produces *the same
residual to the decimal* -- measured at 2.78 mm against 2.78 mm on a real scene.
With several scenes disagreeing about it, `solve()` resolves the ambiguity. With
one scene there is nothing to disagree with, and only the overlay separates them.

So the page shows the overlay, and offers the swap as a button. Deciding by eye
is not a weakness here: it is the only thing that can decide.
"""

from __future__ import annotations

import html
import threading
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gui.core import bag_reader, roi_auto, verify
from gui.core.bag_reader import BagSource, accumulate_cloud, pick_camera_for_frame
from gui.core.decode import cloud_to_struct, image_to_bgr
from gui.core.detect_camera import detect as detect_camera
from gui.core.detect_lidar import DetectParams, apply_box, detect as detect_lidar, point_spacing
from pathlib import Path

from gui.core.evaluation import ab_export as abx
from gui.core.evaluation import board_motion as bm
from gui.core.evaluation import board_timestamp as bt
from gui.core.evaluation import diagnostics as dg
from gui.core.evaluation import edge_alignment as ea
from gui.core.evaluation import holdout as ho
from gui.core.evaluation import multiframe_consistency as mc
from gui.core.evaluation import perturbation as pert
from gui.core.evaluation import spatial_analysis as sa
from gui.core.evaluation import timestamp_diagnostic as td
from gui.core.project import RESULT_DIR, Project
from gui.core.solve import Solution, solve_rigid, sort_centers, to_fast_livo2
from gui.ui.steps import StepPage
from gui.ui.steps.step_filter import derive_params


@dataclass
class _HoldoutRequest:
    """Everything one Hold-out run needs, bundled into a single object so the
    Qt signal carrying it to the worker thread does not have to grow another
    positional parameter every time this feature gains one more piece of
    provenance -- see gui.core.evaluation.holdout's module docstring for what
    most of these fields mean. Built once, in `VerifyStep._run_holdout`, at
    the exact moment "Hold-out 평가 실행" is pressed: this *is* the candidate
    lock -- nothing downstream re-reads the UI's combo boxes.
    """

    gen: int
    bag: str
    lidar_topic: str
    camera_topic: str
    camera: object
    baseline_R: np.ndarray
    baseline_t: np.ndarray
    axis: str
    delta: float
    source_label: str
    times: list          # usable (already margin-filtered) timestamps to actually load
    min_range: float
    max_range: float
    max_sync_offset_ms: float
    discovery_bag: str
    discovery_timestamps: tuple
    exclusion_margin_s: float
    requested_timestamps: tuple    # sampled before exclusion
    excluded_near_discovery: tuple
    bootstrap_iterations: int
    bootstrap_block_s: float
    # The value of VerifyStep._discovery_revision (project-persistent, see
    # gui.core.evaluation.holdout.CandidateProvenance's docstring) at the
    # moment this candidate was locked -- folded into holdout.CandidateProvenance
    # so exposure tracking can tell "the same candidate" from "read off a
    # Discovery result superseded since this dataset was first shown", even
    # across an app restart -- see gui.core.evaluation.holdout.classify_exposure.
    discovery_revision: int


class _Worker(QtCore.QObject):
    opened = QtCore.Signal(int, object)  # gen, LiDAR message times
    ready = QtCore.Signal(int, object, object, object, object, object)  # gen, image, cloud, intensity, lidar c, cam c
    failed = QtCore.Signal(int, str)
    scanning = QtCore.Signal(int, int, int)  # gen, messages read, total

    # Target-independent validation (Multi-frame Consistency) -- separate from
    # the single-frame load/ready pair above so a long multi-frame run cannot
    # be mistaken for, or interrupted by, ordinary timeline scrubbing.
    mf_progress = QtCore.Signal(int, int, int)  # gen, done, total
    mf_ready = QtCore.Signal(int, object)       # gen, MultiFrameConsistencyResult
    mf_failed = QtCore.Signal(int, str)

    # Current-frame quantitative evaluation -- measured at ~0.4-1.0s on a 4K
    # frame with a realistic point count (see bench notes in the PR), which is
    # long enough to visibly stall the GUI thread on a button click, so this
    # also runs here rather than inline in the main thread.
    eval_ready = QtCore.Signal(int, object)     # gen, EdgeAlignmentResult
    eval_failed = QtCore.Signal(int, str)

    # Perturbation Sensitivity -- diagnostic-only, never writes back to a
    # Solution or a project file (see gui.core.evaluation.perturbation).
    pert_progress = QtCore.Signal(int, int, int)  # gen, done, total
    pert_ready = QtCore.Signal(int, object)       # gen, PerturbationResult
    pert_failed = QtCore.Signal(int, str)
    # Fine Scan -- a single-axis, custom-range re-run of the same machinery.
    fine_progress = QtCore.Signal(int, int, int)  # gen, done, total
    fine_ready = QtCore.Signal(int, object)       # gen, AxisSensitivity
    fine_failed = QtCore.Signal(int, str)

    # Independent Hold-out Validation -- a second bag (or a disjoint time
    # range of the current one), opened purely to check whether a candidate
    # already locked from Perturbation/Fine Scan still looks better than
    # baseline on data that candidate was never searched against. See
    # gui.core.evaluation.holdout's module docstring: no fitting, no search,
    # ever happens against hold-out data.
    holdout_bag_progress = QtCore.Signal(int, int, int)  # gen, done, total
    holdout_bag_ready = QtCore.Signal(int, object)       # gen, sorted lidar timestamps
    holdout_bag_failed = QtCore.Signal(int, str)
    holdout_progress = QtCore.Signal(int, str, int, int)  # gen, phase ("prepare"|"evaluate"), done, total
    holdout_ready = QtCore.Signal(int, object)            # gen, holdout.HoldoutResult
    holdout_failed = QtCore.Signal(int, str)
    board_motion_progress = QtCore.Signal(int, int, int)  # gen, done, total
    board_motion_ready = QtCore.Signal(int, object)       # gen, BoardMotionResult
    board_motion_failed = QtCore.Signal(int, str)
    tsdiag_progress = QtCore.Signal(int, str, int, int)   # gen, phase, done, total
    tsdiag_ready = QtCore.Signal(int, object)             # gen, list[TimestampDiagnosticMetrics]
    tsdiag_failed = QtCore.Signal(int, str)
    board_ts_progress = QtCore.Signal(int, int, int)      # gen, done, total
    board_ts_ready = QtCore.Signal(int, object)           # gen, BoardTimestampResult
    board_ts_failed = QtCore.Signal(int, str)

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None
        self._holdout_src: BagSource | None = None  # kept separate so a hold-out bag never evicts the main view's cache
        self._mf_cancel = threading.Event()
        self._pert_cancel = threading.Event()
        self._fine_cancel = threading.Event()
        self._holdout_cancel = threading.Event()
        self._board_motion_cancel = threading.Event()
        self._tsdiag_cancel = threading.Event()
        self._board_ts_cancel = threading.Event()

    def _source(self, bag: str) -> BagSource:
        if self._src is None or str(self._src.path) != bag:
            if self._src is not None:
                self._src.close()
            self._src = BagSource(bag)
            self._src.open()
        return self._src

    def _holdout_source(self, bag: str) -> BagSource:
        """The reader a hold-out run should read `bag` from.

        The "Current Bag Time Range" hold-out mode reads the exact same file
        the main view already has open (`_src`), so it is served from there
        rather than opened a second time. "Separate Bag" reads a different
        file, kept in its own `_holdout_src` so opening it never evicts the
        main view's cached reader -- switching back to scrub the main
        timeline would otherwise have to reopen a bag that never stopped
        being relevant.
        """
        if self._src is not None and str(self._src.path) == bag:
            return self._source(bag)
        if self._holdout_src is None or str(self._holdout_src.path) != bag:
            if self._holdout_src is not None:
                self._holdout_src.close()
            self._holdout_src = BagSource(bag)
            self._holdout_src.open()
        return self._holdout_src

    def open_bag(self, gen: int, bag: str, lidar_topic: str):
        """Read the LiDAR message times so the timeline has something to span.

        Separate from `load` because it is the slow part -- seventeen seconds on a
        36 GB recording -- and it only has to happen once per bag. The reader keeps
        the result, so switching back to a bag already visited is instant.
        """
        try:
            src = self._source(bag)
            times = src.timestamps(
                lidar_topic,
                progress=lambda done, total: self.scanning.emit(gen, done, total),
            )
            self.opened.emit(gen, sorted(times))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def load(self, gen: int, bag: str, lidar_topic: str, camera_topic: str,
             t_ns: int, frames: int, scene, camera, target, method: str,
             pinned_camera_t_ns: "int | None" = None):
        """One moment's cloud and image, and the hole centres when asked for.

        `scene` is None while scrubbing freely. Detection needs the scene's filter
        box to know where to look, so away from a captured moment only the
        projection is produced -- which is still enough to judge an extrinsic by
        eye, and often more telling than four points, since the whole scene has to
        line up rather than just the board.

        `pinned_camera_t_ns` is set for exactly one caller: a Worst-Frame jump
        from Multi-frame Consistency, which paired this LiDAR moment with its
        *nearest* camera image rather than the first one after it (see
        `gui.core.bag_reader.pick_camera_for_frame`). Left at None -- every
        other caller -- this is byte-for-byte the pairing Step 7 has always used.
        """
        try:
            src = self._source(bag)

            _, img_msg = pick_camera_for_frame(src, camera_topic, t_ns, pinned_camera_t_ns)
            if img_msg is None:
                self.failed.emit(gen, "이 시점에 카메라 이미지가 없습니다")
                return
            image = image_to_bgr(img_msg)

            cloud, _, single, intensity, ring = accumulate_cloud(
                src, lidar_topic, t_ns, frames,
                progress=lambda done, total: self.scanning.emit(gen, done, total),
            )

            lidar_c = cam_c = None
            if scene is not None:
                inside = apply_box(single, scene.filter)
                params = derive_params(point_spacing(inside) if len(inside) > 20 else 0.0)
                params.method = method
                lid = detect_lidar(cloud, scene.filter, target, params, ring=ring)
                cam = detect_camera(image, camera, target)
                lidar_c = lid.centers if lid.ok else None
                cam_c = cam.centers if cam.ok else None

            self.ready.emit(gen, image, cloud, intensity, lidar_c, cam_c)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def run_multiframe(self, gen: int, bag: str, lidar_topic: str, camera_topic: str,
                        camera, sol, times: list[int], n_samples: int, min_range: float, max_range: float,
                        max_sync_offset_ms: float):
        """Sample `n_samples` moments evenly across `times` and run Edge
        Alignment on each -- an ordinary frame, not a captured scene, so this
        is the only place that answers "does the extrinsic hold generally"
        rather than "did it fit the four holes it was solved from".

        Pairs each sampled LiDAR moment with its *nearest* camera image, not
        the first one at or after it: on a moving vehicle, a loose pairing
        shows up in the pixel error exactly like a rotation error would, so
        the evaluation path needs the closest timestamp match it can get, not
        whatever the display path's `first_after` happens to land on -- and it
        reports how far off that pairing was so a wrong-looking frame can be
        told apart from a badly-synced one instead of blaming the extrinsic.

        One sweep per frame (not the scene's own `frames` count, since these
        moments were never captured scenes) -- accumulating several sweeps per
        sampled frame would multiply the read cost for no benefit here.
        """
        self._mf_cancel.clear()
        try:
            src = self._source(bag)
            frame_loader = self._nearest_frame_loader(src, lidar_topic, camera_topic, times, max_sync_offset_ms)

            result = mc.evaluate_multiframe_consistency(
                frame_loader=frame_loader,
                n_total_timeline=len(times),
                n_samples=n_samples,
                sol=sol,
                camera=camera,
                min_range=min_range,
                max_range=max_range,
                max_sync_offset_ms=max_sync_offset_ms if max_sync_offset_ms > 0 else None,
                progress=lambda done, total: self.mf_progress.emit(gen, done, total),
                should_cancel=self._mf_cancel.is_set,
            )
            self.mf_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.mf_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def cancel_multiframe(self):
        self._mf_cancel.set()

    def cancel_board_motion(self):
        self._board_motion_cancel.set()

    def cancel_timestamp_diagnostic(self):
        self._tsdiag_cancel.set()

    def cancel_board_timestamp_diagnostic(self):
        self._board_ts_cancel.set()

    def run_timestamp_diagnostic(self, gen: int, bag: str, lidar_topic: str, camera_topic: str,
                                 camera, sol, times: list[int], n_samples: int,
                                 min_range: float, max_range: float,
                                 stable_only: bool, stable_times: list[int]):
        """Compare timestamp pairing modes without changing the normal pairing path."""
        from gui.core import verify
        from gui.core.evaluation.edge_alignment import aggregate_point_stats

        self._tsdiag_cancel.clear()
        try:
            src = self._source(bag)
            source_times = list(stable_times) if stable_only and stable_times else list(times)
            indices = mc.sample_frame_indices(len(source_times), n_samples)
            if not indices:
                self.tsdiag_failed.emit(gen, "No frames available for timestamp diagnostic.")
                return

            camera_record_times = src.timestamps(camera_topic)
            camera_entries = []
            for rec_ns, msg in src.range(camera_topic, camera_record_times[0], camera_record_times[-1] + 1):
                h_ns = td.header_stamp_ns(msg)
                if h_ns is None:
                    continue
                camera_entries.append((td.CameraStamp(record_ns=rec_ns, header_ns=h_ns), msg))
            if not camera_entries:
                self.tsdiag_failed.emit(gen, "Camera messages have no usable header.stamp.")
                return
            camera_stamps = [c[0] for c in camera_entries]

            lidar_stamps, lidar_payloads = [], []
            total = len(indices)
            base_t = source_times[0]
            for done, idx in enumerate(indices, start=1):
                if self._tsdiag_cancel.is_set():
                    break
                record_ns = source_times[idx]
                got_ns, msg = src.nearest(lidar_topic, record_ns, window_ns=1_000_000)
                if msg is None:
                    self.tsdiag_progress.emit(gen, "load", done, total)
                    continue
                rec = cloud_to_struct(msg)
                xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
                finite = np.isfinite(xyz).all(axis=1)
                xyz = xyz[finite]
                point_ts = np.asarray(rec["timestamp"])[finite] if "timestamp" in (rec.dtype.names or ()) else None
                stamp = td.make_lidar_stamp(got_ns, msg, point_ts)
                lidar_stamps.append(stamp)
                lidar_payloads.append((idx, got_ns, (got_ns - base_t) / 1e9, xyz))
                self.tsdiag_progress.emit(gen, "load", done, total)

            if len(lidar_stamps) < 2:
                self.tsdiag_failed.emit(gen, "Need at least two LiDAR frames with usable timestamps.")
                return

            results = []
            modes = [
                td.PairingMode.RECORD_TO_RECORD,
                td.PairingMode.HEADER_TO_SWEEP_START,
                td.PairingMode.HEADER_TO_SWEEP_MID,
                td.PairingMode.HEADER_TO_SWEEP_END,
            ]
            for mode_i, mode in enumerate(modes, start=1):
                if self._tsdiag_cancel.is_set():
                    break
                pairs, pair_summary = td.pair_by_mode(
                    camera_stamps, lidar_stamps, mode, list(range(len(lidar_stamps)))
                )
                frame_results = []
                edge_results = []
                frames = []
                for done, (li, ci, lid_ref, cam_ref) in enumerate(pairs, start=1):
                    if self._tsdiag_cancel.is_set():
                        break
                    timeline_idx, lidar_record_ns, offset_s, cloud = lidar_payloads[li]
                    image = image_to_bgr(camera_entries[ci][1])
                    h, w = image.shape[:2]
                    pr = verify.project_cloud(cloud, sol, camera, w, h, min_range=min_range, max_range=max_range)
                    frames.append(td.TimestampDiagnosticFrame(
                        timeline_index=timeline_idx,
                        lidar_record_ns=lidar_record_ns,
                        lidar_reference_ns=lid_ref,
                        camera_record_ns=camera_entries[ci][0].record_ns,
                        camera_reference_ns=cam_ref,
                        signed_sync_offset_ms=(cam_ref - lid_ref) / 1e6,
                    ))
                    if pr.n_visible:
                        er = ea.evaluate_edge_alignment(image, pr.uv, pr.depth, ea.EdgeAlignmentParams())
                        if er.ok:
                            edge_results.append((er, w, h))
                    self.tsdiag_progress.emit(gen, td.PAIRING_LABELS[mode], done, len(pairs))

                metrics = self._timestamp_metrics_from_edges(mode, pair_summary, frames, edge_results, aggregate_point_stats)
                results.append(metrics)
                self.tsdiag_progress.emit(gen, "mode", mode_i, len(modes))
            self.tsdiag_ready.emit(gen, results)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.tsdiag_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def run_board_timestamp_diagnostic(self, gen: int, bag: str, lidar_topic: str, camera_topic: str,
                                       camera, target, sol, times: list[int], scenes: list,
                                       n_samples: int, stable_times: list[int], method: str):
        """Board-specific LiDAR timestamp diagnostic.

        Uses existing camera board detection, existing ROI/auto-ROI placement,
        and the existing LiDAR board detector. The board point set is the
        detector's board-plane inlier set (`LidarDetection.plane_indices`).
        """
        self._board_ts_cancel.clear()
        try:
            src = self._source(bag)
            indices = mc.sample_frame_indices(len(times), n_samples)
            if not indices:
                self.board_ts_failed.emit(gen, "No frames available.")
                return

            camera_record_times = src.timestamps(camera_topic)
            camera_record_index = {int(t): i for i, t in enumerate(camera_record_times)}
            camera_headers = []
            if camera_record_times:
                for rec_ns, msg in src.range(camera_topic, camera_record_times[0], camera_record_times[-1] + 1):
                    h_ns = td.header_stamp_ns(msg)
                    if h_ns is None:
                        continue
                    camera_headers.append((camera_record_index.get(int(rec_ns), -1), int(rec_ns), int(h_ns)))
            stable_set = set(int(t) for t in stable_times or [])
            scene_rois = [(s.t_ns, s.roi()) for s in scenes if getattr(s, "roi_set", False)]
            frames = []
            failed = 0
            processed = 0
            plane_success = 0
            enough_plane_points = 0
            circle_candidates = 0
            four_or_more_circles = 0
            geometry_passed = 0
            full_target_success = 0
            board_timestamp_valid = 0
            board_pose_valid = 0
            lidar_failure_reasons: dict[str, int] = {}
            for done, idx in enumerate(indices, start=1):
                if self._board_ts_cancel.is_set():
                    break
                processed += 1
                lidar_record_ns = times[idx]
                got_ns, lidar_msg = src.nearest(lidar_topic, lidar_record_ns, window_ns=1_000_000)
                cam_record_ns, cam_msg = src.nearest(camera_topic, lidar_record_ns, window_ns=200_000_000)
                if lidar_msg is None or cam_msg is None:
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue

                rec = cloud_to_struct(lidar_msg)
                xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
                finite = np.isfinite(xyz).all(axis=1)
                xyz = xyz[finite]
                if "timestamp" not in (rec.dtype.names or ()):
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue
                point_ts = np.asarray(rec["timestamp"])[finite]
                valid_ts = point_ts[np.isfinite(point_ts)]
                if valid_ts.size == 0:
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue
                sweep_start_ns = int(float(np.min(valid_ts)) * 1e9)
                sweep_end_ns = int(float(np.max(valid_ts)) * 1e9)

                image = image_to_bgr(cam_msg)
                cam_det = detect_camera(image, camera, target)
                if not cam_det.ok:
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue

                roi = self._nearest_scene_roi(scene_rois, lidar_record_ns)
                if roi is None:
                    roi = self._board_timestamp_auto_roi(xyz, sol, cam_det, target, idx)
                    if roi is None:
                        failed += 1
                        self.board_ts_progress.emit(gen, done, len(indices))
                        continue

                inside = apply_box(xyz, roi)
                params = derive_params(point_spacing(inside) if len(inside) > 20 else 0.0)
                params.method = method
                ring = np.asarray(rec["ring"])[finite].astype(np.int32) if "ring" in (rec.dtype.names or ()) else None
                det = detect_lidar(xyz, roi, target, params, ring=ring)
                if det.plane_indices is None or len(det.plane_indices) == 0:
                    key = det.reason or "no plane indices"
                    lidar_failure_reasons[key] = lidar_failure_reasons.get(key, 0) + 1
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue
                plane_success += 1
                if len(det.plane) >= 50:
                    enough_plane_points += 1
                if det.n_clusters > 0:
                    circle_candidates += 1
                if det.n_circles >= 4:
                    four_or_more_circles += 1
                if det.ok:
                    geometry_passed += 1
                    full_target_success += 1
                    board_pose_valid += 1
                else:
                    key = det.reason or "full target rejected"
                    lidar_failure_reasons[key] = lidar_failure_reasons.get(key, 0) + 1

                board_ts = point_ts[np.asarray(det.plane_indices, dtype=np.int64)]
                board_ts = board_ts[np.isfinite(board_ts)]
                if len(board_ts) == 0:
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue
                board_timestamp_valid += 1
                cam_header_ns = td.header_stamp_ns(cam_msg)
                if cam_header_ns is None:
                    failed += 1
                    self.board_ts_progress.emit(gen, done, len(indices))
                    continue
                current_camera_index = camera_record_index.get(int(cam_record_ns), -1)
                status = "STABLE" if int(got_ns) in stable_set else "ALL"
                frames.append(bt.frame_from_timestamps(
                    frame_index=idx,
                    lidar_record_ns=got_ns,
                    camera_header_ns=cam_header_ns,
                    sweep_start_ns=sweep_start_ns,
                    sweep_end_ns=sweep_end_ns,
                    board_point_timestamps_s=board_ts,
                    status=status,
                    current_camera_index=current_camera_index,
                    current_camera_record_ns=int(cam_record_ns),
                    camera_headers=camera_headers,
                    plane_detected=True,
                    full_target_detected=bool(det.ok),
                    board_pose_valid=bool(det.ok),
                    lidar_plane_point_count=len(det.plane),
                    lidar_circle_candidate_count=det.n_clusters,
                    lidar_circle_count=det.n_circles,
                    lidar_radii=list(det.radii),
                    lidar_detection_reason=det.reason,
                ))
                self.board_ts_progress.emit(gen, done, len(indices))
            self.board_ts_ready.emit(gen, bt.BoardTimestampResult(
                frames=frames,
                failed_frames=failed,
                processed_frames=processed,
                plane_detection_success=plane_success,
                enough_plane_points_success=enough_plane_points,
                circle_candidate_success=circle_candidates,
                four_or_more_circles_success=four_or_more_circles,
                geometry_validation_success=geometry_passed,
                full_target_detection_success=full_target_success,
                board_timestamp_valid_count=board_timestamp_valid,
                board_pose_valid_count=board_pose_valid,
                lidar_failure_reasons=lidar_failure_reasons,
            ))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.board_ts_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _nearest_scene_roi(scene_rois: list, t_ns: int, max_dt_ns: int = 500_000_000):
        if not scene_rois:
            return None
        t, roi = min(scene_rois, key=lambda item: abs(int(item[0]) - int(t_ns)))
        return roi if abs(int(t) - int(t_ns)) <= max_dt_ns else None

    @staticmethod
    def _board_timestamp_auto_roi(xyz, sol, cam_det, target, idx: int):
        for margin in (roi_auto.MARGIN_M, 1.5, 3.0, 5.0, 8.0):
            placement = roi_auto.place(xyz, sol, cam_det, target, scene_id=str(idx), margin=margin)
            if placement.ok and placement.box is not None:
                return placement.box
        return None

    @staticmethod
    def _timestamp_metrics_from_edges(mode, pair_summary, frames, edge_results, aggregate_point_stats):
        if not edge_results:
            return td.TimestampDiagnosticMetrics(
                mode=mode, n_total=len(frames), n_failed=len(frames),
                reason=pair_summary.warning or "No valid edge-alignment frames.",
                sync_stats=pair_summary.sync_stats, frames=frames,
            )
        errors = np.concatenate([r.edge_errors_px for r, _, _ in edge_results if r.edge_errors_px is not None])
        matched = np.concatenate([r.edge_matched for r, _, _ in edge_results if r.edge_matched is not None])
        du = np.concatenate([r.edge_delta_u_px for r, _, _ in edge_results if r.edge_delta_u_px is not None])
        dv = np.concatenate([r.edge_delta_v_px for r, _, _ in edge_results if r.edge_delta_v_px is not None])
        agg = aggregate_point_stats(errors, matched, du, dv)
        frame_matched_means = np.asarray([r.matched_mean_px for r, _, _ in edge_results], dtype=float)
        frame_match_rates = np.asarray([r.match_rate for r, _, _ in edge_results], dtype=float)

        # Build one synthetic EdgeAlignmentResult so existing spatial/depth
        # grouping code can reduce all points in the same way as the rest of
        # the evaluation UI.
        pooled = ea.EdgeAlignmentResult(
            ok=True,
            edge_pixels=np.vstack([r.edge_pixels for r, _, _ in edge_results if r.edge_pixels is not None]),
            edge_depths=np.concatenate([r.edge_depths for r, _, _ in edge_results if r.edge_depths is not None]),
            edge_errors_px=errors,
            edge_matched=matched,
            edge_delta_u_px=du,
            edge_delta_v_px=dv,
        )
        width, height = edge_results[-1][1], edge_results[-1][2]
        spatial = sa.analyze_spatial(pooled, width, height)
        horizontal = {}
        depth = {}
        if spatial is not None:
            horizontal = {
                label: {
                    "matched_mean_px": b.matched_mean_px,
                    "match_rate": b.match_rate,
                    "median_delta_u_px": b.median_delta_u_px,
                    "median_delta_v_px": b.median_delta_v_px,
                }
                for label, b in spatial.horizontal.items()
            }
            depth = {
                label: {
                    "matched_mean_px": b.matched_mean_px,
                    "match_rate": b.match_rate,
                    "median_delta_u_px": b.median_delta_u_px,
                    "median_delta_v_px": b.median_delta_v_px,
                }
                for label, b in spatial.depth_bins.items()
            }
        return td.TimestampDiagnosticMetrics(
            mode=mode,
            n_total=len(frames),
            n_valid=len(edge_results),
            n_failed=max(0, len(frames) - len(edge_results)),
            reason=pair_summary.warning,
            mean_px=float(np.mean(errors)),
            median_px=float(np.median(errors)),
            p95_px=float(np.percentile(errors, 95)),
            matched_mean_px=agg["matched_mean_px"],
            matched_median_px=agg["matched_median_px"],
            matched_p95_px=agg["matched_p95_px"],
            matched_p99_px=agg["matched_p99_px"],
            matched_max_px=agg["matched_max_px"],
            match_rate_mean=float(np.nanmean(frame_match_rates)),
            match_rate_median=float(np.nanmedian(frame_match_rates)),
            matched_std_px=float(np.nanstd(frame_matched_means, ddof=1)) if np.isfinite(frame_matched_means).sum() > 1 else 0.0,
            within_2px=agg["match_rate_by_threshold_px"].get(2.0, float("nan")),
            within_3px=agg["match_rate_by_threshold_px"].get(3.0, float("nan")),
            within_5px=agg["match_rate_by_threshold_px"].get(5.0, float("nan")),
            within_10px=agg["match_rate_by_threshold_px"].get(10.0, float("nan")),
            within_15px=agg["match_rate_by_threshold_px"].get(15.0, float("nan")),
            median_delta_u_px=agg["median_delta_u_px"],
            median_delta_v_px=agg["median_delta_v_px"],
            sync_stats=pair_summary.sync_stats,
            frames=frames,
            horizontal=horizontal,
            depth=depth,
        )

    def run_board_motion(self, gen: int, bag: str, camera_topic: str, camera, target,
                         n_samples: int, translation_threshold_mm: float,
                         rotation_threshold_deg: float):
        """Detect the calibration board in camera frames and classify motion.

        Camera timestamps are the bag message timestamps exposed by BagSource.
        The camera detector uses ChArUco/Aruco-style marker PnP
        (`gui.core.detect_camera.detect`), so this phase depends only on camera
        board pose and does not change or re-fit the extrinsic.
        """
        import cv2

        self._board_motion_cancel.clear()
        try:
            src = self._source(bag)
            camera_times = src.timestamps(
                camera_topic,
                progress=lambda done, total: self.scanning.emit(gen, done, total),
            )
            indices = mc.sample_frame_indices(len(camera_times), n_samples)
            total = len(indices)
            samples = []
            invalid = []
            for done, idx in enumerate(indices, start=1):
                if self._board_motion_cancel.is_set():
                    break
                stamp = camera_times[idx]
                _, img_msg = src.nearest(camera_topic, stamp, window_ns=1_000_000)
                if img_msg is not None:
                    det = detect_camera(image_to_bgr(img_msg), camera, target)
                    reason = self._camera_pose_invalid_reason(det)
                    if reason is None and det.rvec is not None and det.tvec is not None:
                        R, _ = cv2.Rodrigues(det.rvec)
                        samples.append(
                            bm.BoardPoseSample(
                                timestamp_ns=stamp,
                                frame_index=idx,
                                R=R,
                                t=np.asarray(det.tvec, dtype=float).reshape(3),
                                quality=1.0 / max(det.reproj_rms, 1e-6),
                                reproj_rms=det.reproj_rms,
                                reproj_median=det.reproj_median,
                                reproj_p95=det.reproj_p95,
                                reproj_max=det.reproj_max,
                            )
                        )
                    else:
                        invalid.append(bm.BoardMotionFrame(
                            timestamp_ns=stamp,
                            frame_index=idx,
                            status=bm.INVALID,
                            valid_pose=False,
                            invalid_reason=reason or "pose unavailable",
                            reproj_rms=getattr(det, "reproj_rms", float("nan")),
                            reproj_median=getattr(det, "reproj_median", float("nan")),
                            reproj_p95=getattr(det, "reproj_p95", float("nan")),
                            reproj_max=getattr(det, "reproj_max", float("nan")),
                        ))
                else:
                    invalid.append(bm.BoardMotionFrame(
                        timestamp_ns=stamp,
                        frame_index=idx,
                        status=bm.INVALID,
                        valid_pose=False,
                        invalid_reason="missing camera message",
                    ))
                self.board_motion_progress.emit(gen, done, total)
            result = bm.analyze_board_motion(
                samples,
                translation_threshold_mm=translation_threshold_mm,
                rotation_threshold_deg=rotation_threshold_deg,
                attempted_count=total,
                invalid_frames=invalid,
            )
            self.board_motion_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.board_motion_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _camera_pose_invalid_reason(det) -> str | None:
        if det is None:
            return "camera detector returned no result"
        if not getattr(det, "ok", False):
            return getattr(det, "reason", "board detection failed") or "board detection failed"
        if getattr(det, "rvec", None) is None or getattr(det, "tvec", None) is None:
            return "solvePnP pose unavailable"
        tvec = np.asarray(det.tvec, dtype=float).reshape(-1)
        rvec = np.asarray(det.rvec, dtype=float).reshape(-1)
        if not np.isfinite(tvec).all() or not np.isfinite(rvec).all():
            return "pose contains NaN/Inf"
        distance = float(np.linalg.norm(tvec))
        if not np.isfinite(distance) or distance < 0.2 or distance > 80.0:
            return f"unrealistic board distance {distance:.2f} m"
        if getattr(det, "n_markers", 0) < 3:
            return f"too few board markers ({getattr(det, 'n_markers', 0)})"
        if getattr(det, "reproj_rms", 0.0) > 5.0 or getattr(det, "reproj_p95", 0.0) > 8.0:
            return (
                f"high reprojection error rms={getattr(det, 'reproj_rms', float('nan')):.2f}px "
                f"p95={getattr(det, 'reproj_p95', float('nan')):.2f}px"
            )
        return None

    def _nearest_frame_loader(self, src, lidar_topic: str, camera_topic: str,
                               times: list[int], max_sync_offset_ms: float):
        """Shared by run_multiframe and run_perturbation: pairs each sampled
        LiDAR moment with its *nearest* camera image, not the first one at or
        after it (Step 7's display path) -- on a moving vehicle, a loose
        pairing shows up in the pixel error exactly like a rotation error
        would, so both evaluation paths need the closest timestamp match they
        can get. `img_stamp` is handed back raw so the caller can derive
        signed/absolute sync offset itself (see `pick_camera_for_frame` for
        how a Worst-Frame jump later pins the display to this same timestamp).
        """
        base_t = times[0] if times else 0
        # Wide enough to still find a pairing at whatever sync tolerance the
        # user configured, without turning into an unbounded bag scan.
        window_ns = max(500_000_000, int((max_sync_offset_ms or 0) * 2_000_000))

        def frame_loader(idx: int):
            # A single unreadable frame (a truncated message, a decode error)
            # must count as one failed frame, not abort a run that may cover
            # hundreds of others -- so failures are swallowed here rather
            # than left to escape and cancel the whole batch.
            try:
                t_ns = times[idx]
                img_stamp, img_msg = src.nearest(camera_topic, t_ns, window_ns=window_ns)
                if img_msg is None:
                    return None
                image = image_to_bgr(img_msg)
                cloud, _, _, _, _ = accumulate_cloud(src, lidar_topic, t_ns, frames=1)
                if len(cloud) == 0:
                    return None
                return image, cloud, t_ns, (t_ns - base_t) / 1e9, img_stamp
            except Exception:  # noqa: BLE001 - counted as a failed frame, see above
                return None

        return frame_loader

    def run_eval_current(self, gen: int, image, cloud, sol, camera, min_range: float, max_range: float):
        """The current-frame "정량 평가" button's work, off the GUI thread.

        Projects `cloud` fresh with the caller-supplied evaluation range --
        deliberately not reusing whatever Projection the display last drew,
        so this metric cannot silently move when someone nudges the display's
        range sliders (see the Evaluation Range control this is paired with).
        """
        try:
            h, w = image.shape[:2]
            pr = verify.project_cloud(cloud, sol, camera, w, h, min_range=min_range, max_range=max_range)
            if pr.n_visible == 0:
                self.eval_failed.emit(gen, "평가 범위 안에 투영된 점이 없습니다.")
                return
            result = ea.evaluate_edge_alignment(image, pr.uv, pr.depth, ea.EdgeAlignmentParams())
            self.eval_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.eval_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def _build_perturbation_frames(self, mode: str, bag: str, lidar_topic: str, camera_topic: str,
                                    camera, times: list[int], n_frames: int,
                                    max_sync_offset_ms: float, edge_params, pinned_image, pinned_cloud,
                                    should_cancel):
        """Shared by run_perturbation and run_fine_scan: build the fixed frame
        set a run scores every trial against.

        `mode == "current_frame"` reuses `pinned_image`/`pinned_cloud` exactly
        (no bag read at all -- the same identity-based pairing guarantee
        `run_eval_current` already relies on). `mode == "multi_frame"` samples
        `n_frames` moments from `times` with the same nearest-pairing policy
        Multi-frame Consistency uses, loading them once up front (see
        `perturbation.prepare_frames`'s docstring for why: the same frames get
        re-scored dozens of times here, not loaded-scored-discarded once each).
        A Fine Scan run with the same `n_frames` as the preceding Quick/Full
        run samples the identical timeline indices again (`sample_frame_indices`
        is a deterministic linspace), so the two stay comparable without
        needing to explicitly carry the frame list between UI actions.
        """
        if mode == "current_frame":
            return [pert.frame_from_current(pinned_image, pinned_cloud, edge_params)]
        src = self._source(bag)
        frame_loader = self._nearest_frame_loader(src, lidar_topic, camera_topic, times, max_sync_offset_ms)
        return pert.prepare_frames(
            frame_loader, n_total_timeline=len(times), n_samples=n_frames,
            max_sync_offset_ms=max_sync_offset_ms if max_sync_offset_ms > 0 else None,
            edge_params=edge_params, should_cancel=should_cancel,
        )

    def run_perturbation(self, gen: int, mode: str, bag: str, lidar_topic: str, camera_topic: str,
                          camera, R, t, times: list[int], n_frames: int,
                          rotation_deltas: tuple, translation_deltas: tuple,
                          min_range: float, max_range: float, max_sync_offset_ms: float,
                          pinned_image, pinned_cloud):
        """Score the current T, then every requested small rotation/translation
        nudge around it, purely as a diagnostic -- see
        `gui.core.evaluation.perturbation`'s module docstring. `R`/`t` are read
        here, never written to; the caller's `Solution`/project state is
        untouched regardless of what this finds.
        """
        self._pert_cancel.clear()
        edge_params = ea.EdgeAlignmentParams()
        try:
            frames = self._build_perturbation_frames(
                mode, bag, lidar_topic, camera_topic, camera, times, n_frames,
                max_sync_offset_ms, edge_params, pinned_image, pinned_cloud, self._pert_cancel.is_set,
            )
            result = pert.evaluate_perturbation_grid(
                frames, R, t, camera, min_range=min_range, max_range=max_range, mode=mode,
                rotation_deltas_deg=rotation_deltas, translation_deltas_mm=translation_deltas,
                edge_params=edge_params,
                progress=lambda done, total: self.pert_progress.emit(gen, done, total),
                should_cancel=self._pert_cancel.is_set,
            )
            self.pert_ready.emit(gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.pert_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def run_fine_scan(self, gen: int, mode: str, bag: str, lidar_topic: str, camera_topic: str,
                       camera, R, t, times: list[int], n_frames: int, axis: str, deltas: tuple,
                       min_range: float, max_range: float, max_sync_offset_ms: float,
                       pinned_image, pinned_cloud):
        """Fine Scan: the same trial-scoring machinery as run_perturbation, but
        for one caller-chosen axis and delta range (see
        `gui.core.evaluation.perturbation.evaluate_single_axis`)."""
        self._fine_cancel.clear()
        edge_params = ea.EdgeAlignmentParams()
        try:
            frames = self._build_perturbation_frames(
                mode, bag, lidar_topic, camera_topic, camera, times, n_frames,
                max_sync_offset_ms, edge_params, pinned_image, pinned_cloud, self._fine_cancel.is_set,
            )
            axis_result = pert.evaluate_single_axis(
                frames, R, t, camera, axis, deltas,
                min_range=min_range, max_range=max_range, mode=mode, edge_params=edge_params,
                progress=lambda done, total: self.fine_progress.emit(gen, done, total),
                should_cancel=self._fine_cancel.is_set,
            )
            self.fine_ready.emit(gen, axis_result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.fine_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def cancel_fine_scan(self):
        self._fine_cancel.set()

    def cancel_perturbation(self):
        self._pert_cancel.set()

    def open_holdout_bag(self, gen: int, bag: str, lidar_topic: str):
        """Read a candidate hold-out bag's LiDAR message times -- the same
        slow, once-per-bag read `open_bag` does for the main view, kept on
        `_holdout_src` instead so it never evicts the main bag's reader (see
        `_holdout_source`)."""
        try:
            src = self._holdout_source(bag)
            times = src.timestamps(
                lidar_topic,
                progress=lambda done, total: self.holdout_bag_progress.emit(gen, done, total),
            )
            self.holdout_bag_ready.emit(gen, sorted(times))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.holdout_bag_failed.emit(gen, f"{type(exc).__name__}: {exc}")

    def _cached_source_for(self, bag: str) -> "BagSource | None":
        """An already-open reader for `bag`, if one happens to be sitting on
        `_src`/`_holdout_src` -- used only for the best-effort bag-signature
        comparison below, which must never itself trigger opening (and so
        possibly evicting, and possibly costing a slow re-scan of) a bag nobody
        asked to open."""
        if self._src is not None and str(self._src.path) == bag:
            return self._src
        if self._holdout_src is not None and str(self._holdout_src.path) == bag:
            return self._holdout_src
        return None

    @staticmethod
    def _bag_signature(src: BagSource, lidar_topic: str) -> "bag_reader.BagSignature":
        """Built from an already-open `src` -- `.timestamps()` reuses its
        per-topic cache when the caller already read it (true for both the
        main view's and Hold-out's own readers by the time this runs), so
        this costs only `inspect`'s index-only scan, not a second full read."""
        stamps = src.timestamps(lidar_topic)
        info = bag_reader.inspect(src.path)
        return bag_reader.BagSignature(
            file_size_bytes=bag_reader.path_size_bytes(src.path), duration_s=src.duration,
            n_topics=len(info.topics), total_message_count=info.message_count,
            first_lidar_timestamp_ns=stamps[0] if stamps else 0,
            last_lidar_timestamp_ns=stamps[-1] if stamps else 0,
        )

    def _holdout_bag_signature_best_effort(self, src: BagSource, lidar_topic: str) -> "bag_reader.BagSignature | None":
        """This Hold-out bag's own `BagSignature`, best-effort -- used both
        to fold a copied bag into the same exposure Source
        (`ho.resolve_source_identity`) and, when Discovery used a different
        (already-cached) bag, to compare against it below. Never raises: a
        signature that could not be computed just means neither of those
        gets one, not that the run fails."""
        try:
            return self._bag_signature(src, lidar_topic)
        except Exception:  # noqa: BLE001 - informational/identity only, never fails the run
            return None

    def _discovery_bag_signature_best_effort(self, req: "_HoldoutRequest",
                                              holdout_sig: "bag_reader.BagSignature | None") -> "bag_reader.BagSignature | None":
        """This Discovery bag's own `BagSignature`, best-effort -- see
        `ho.DiscoveryProvenance.signature`'s docstring for why Candidate A/B
        Test Package Export needs it. `holdout_sig` is reused outright when
        Discovery and Hold-out are the same bag (no second read needed);
        otherwise only attempted when Discovery's bag is already open under
        a cached reader (see `_cached_source_for`), exactly like
        `_discovery_comparison_notes` below -- never opens a bag freshly
        just for this, and never fails the run.
        """
        if not req.discovery_bag:
            return None
        if ho.same_bag_identity(req.discovery_bag, req.bag):
            return holdout_sig
        discovery_src = self._cached_source_for(req.discovery_bag)
        if discovery_src is None:
            return None
        try:
            return self._bag_signature(discovery_src, req.lidar_topic)
        except Exception:  # noqa: BLE001 - informational/identity only, never fails the run
            return None

    def _discovery_comparison_notes(self, req: "_HoldoutRequest",
                                     holdout_sig: "bag_reader.BagSignature | None",
                                     discovery_sig: "bag_reader.BagSignature | None") -> "tuple[str, str]":
        """(signature_warning, cross_bag_time_overlap_note) -- both best-
        effort and both purely informational (see
        gui.core.evaluation.holdout's module docstring). `discovery_sig` is
        whatever `_discovery_bag_signature_best_effort` already produced
        (never recomputed here, so Discovery's bag is never read twice).
        """
        if holdout_sig is None or discovery_sig is None or ho.same_bag_identity(req.discovery_bag, req.bag):
            return "", ""
        warning = (
            "Hold-out bag appears identical to the Discovery bag. Independent validation may be invalid."
            if ho.signatures_suspiciously_similar(discovery_sig, holdout_sig) else ""
        )
        note = (
            "Discovery and Hold-out bags have overlapping sensor time ranges. If they share a clock domain, "
            "this may indicate overlapping recordings (informational only -- not confirmed)."
            if ho.cross_bag_time_ranges_overlap(discovery_sig, holdout_sig) else ""
        )
        return warning, note

    def run_holdout(self, req: "_HoldoutRequest"):
        """Independent Hold-out Validation: prepare the hold-out frame set
        once (image, cloud, timestamps, Canny/orientation cache -- see
        `perturbation.prepare_frames`), then score baseline and the one
        already-locked candidate against it with
        `gui.core.evaluation.holdout.evaluate_holdout`. `req.axis`/`req.delta`/
        `req.source_label` were fixed by the caller before this call was ever
        made -- nothing here searches for, or adjusts, a candidate; nothing
        here writes to a Solution, a project, or a file.
        """
        self._holdout_cancel.clear()
        edge_params = ea.EdgeAlignmentParams()
        try:
            src = self._holdout_source(req.bag)
            frame_loader = self._nearest_frame_loader(src, req.lidar_topic, req.camera_topic, req.times,
                                                        req.max_sync_offset_ms)
            frames = pert.prepare_frames(
                frame_loader, n_total_timeline=len(req.times), n_samples=len(req.times),
                max_sync_offset_ms=req.max_sync_offset_ms if req.max_sync_offset_ms > 0 else None,
                edge_params=edge_params, should_cancel=self._holdout_cancel.is_set,
                progress=lambda done, total: self.holdout_progress.emit(req.gen, "prepare", done, total),
            )
            candidate = ho.CandidateSpec(axis=req.axis, delta=req.delta, source=req.source_label)
            # Computed once, always (not just when Discovery used a different
            # bag): needed for exposure-registry Source identity regardless,
            # and reused below for the Discovery-comparison notes and for
            # ho.DiscoveryProvenance.signature (Candidate A/B Test Package
            # Export provenance) too.
            holdout_sig = self._holdout_bag_signature_best_effort(src, req.lidar_topic)
            discovery_sig = self._discovery_bag_signature_best_effort(req, holdout_sig)
            signature_warning, cross_bag_note = self._discovery_comparison_notes(req, holdout_sig, discovery_sig)
            discovery = ho.DiscoveryProvenance(
                bag=req.discovery_bag, timestamps=tuple(req.discovery_timestamps),
                exclusion_margin_s=req.exclusion_margin_s, signature=discovery_sig,
            ) if req.discovery_bag else None

            result = ho.evaluate_holdout(
                frames, req.baseline_R, req.baseline_t, candidate, req.camera,
                min_range=req.min_range, max_range=req.max_range, edge_params=edge_params,
                should_cancel=self._holdout_cancel.is_set,
                progress=lambda done, total: self.holdout_progress.emit(req.gen, "evaluate", done, total),
                discovery=discovery, holdout_bag=req.bag, holdout_signature=holdout_sig,
                requested_timestamps=req.requested_timestamps, excluded_near_discovery=req.excluded_near_discovery,
                signature_warning=signature_warning, cross_bag_time_overlap_note=cross_bag_note,
                discovery_revision=req.discovery_revision,
                bootstrap_iterations=req.bootstrap_iterations, bootstrap_block_s=req.bootstrap_block_s,
            )
            self.holdout_ready.emit(req.gen, result)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.holdout_failed.emit(req.gen, f"{type(exc).__name__}: {exc}")

    def cancel_holdout(self):
        self._holdout_cancel.set()

    def shutdown(self):
        if self._src is not None:
            self._src.close()
            self._src = None
        if self._holdout_src is not None:
            self._holdout_src.close()
            self._holdout_src = None


class _FrameSlider(QtWidgets.QSlider):
    """The timeline: one wheel notch is one frame.

    A plain QSlider moves by the desktop's scroll-lines setting -- three frames a
    notch here -- which overshoots whatever you were trying to land on. Stepping
    one at a time makes the wheel the natural way to walk a recording.
    """

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        if not steps or not self.isEnabled():
            super().wheelEvent(ev)
            return
        # Wheel-up reads as "forward in time", the direction the frames advance.
        self.setValue(self.value() + int(round(steps)) * self.singleStep())
        ev.accept()


class _ZoomLabel(QtWidgets.QLabel):
    """The overlay: wheel to zoom about the cursor, left-drag to pan.

    Buttons alone meant losing your place on every step -- the view grows about
    its own centre, so a hole being inspected at the edge slides off and has to be
    found again. Zooming about the pointer keeps whatever is under it still, and
    dragging is how you move once zoomed in past the window.
    """

    zoomed = QtCore.Signal(float, QtCore.QPointF)
    panned = QtCore.Signal(QtCore.QPoint)  # movement in pixels, to subtract from scroll
    stepped = QtCore.Signal(int)           # Shift+wheel: frames to advance
    sized = QtCore.Signal(int)             # Ctrl+wheel: notches of point size
    picked = QtCore.Signal(QtCore.QPointF)  # a click that did not drag, in label coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("6단계 캘리브레이션을 완료해주세요.")
        self._drag_from: QtCore.QPoint | None = None
        # Where the button went down, so a click can be told from a drag: panning
        # and picking share the left button, and only movement separates them.
        self._press_at: QtCore.QPointF | None = None

    def _has_image(self) -> bool:
        pm = self.pixmap()
        return pm is not None and not pm.isNull()

    def wheelEvent(self, ev):
        if not self._has_image():
            super().wheelEvent(ev)
            return
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            super().wheelEvent(ev)
            return
        # Three things one wants to sweep without leaving the image: the zoom, the
        # frame, and how fat the points are. Bare wheel is zoom because it is the
        # one you reach for most; the other two take a modifier.
        mods = ev.modifiers()
        if mods & QtCore.Qt.ShiftModifier:
            self.stepped.emit(int(round(steps)))
        elif mods & QtCore.Qt.ControlModifier:
            self.sized.emit(int(round(steps)))
        else:
            self.zoomed.emit(1.25 ** steps, ev.position())
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._has_image():
            self._press_at = ev.position()
            self._drag_from = ev.position().toPoint()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag_from is not None:
            now = ev.position().toPoint()
            self.panned.emit(now - self._drag_from)
            # The scroll offset moves under the widget, so the grab point stays
            # where it is in widget coordinates -- do not advance it.
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self._drag_from is not None:
            start = self._press_at
            self._press_at = None
            self._drag_from = None
            if start is not None:
                moved = (ev.position() - start).manhattanLength()
                if moved <= 3:
                    self.picked.emit(ev.position())
            self.unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class VerifyStep(StepPage):
    title = "7. 검증"
    subtitle = "이미지에 투영해 확인"

    request_open = QtCore.Signal(int, str, str)
    # ... method, pinned camera t_ns (None for ordinary navigation)
    request_load = QtCore.Signal(int, str, str, str, object, int, object, object, object, str, object)
    # gen, bag, lidar topic, camera topic, camera, sol, times, n_samples, eval min, eval max, max sync offset ms
    request_multiframe = QtCore.Signal(int, str, str, str, object, object, object, int, float, float, float)
    # gen, image, cloud, sol, camera, eval min, eval max
    request_eval_current = QtCore.Signal(int, object, object, object, object, float, float)
    # gen, mode, bag, lidar topic, camera topic, camera, R, t, times, n_frames,
    # eval min, eval max, max sync offset ms, pinned image, pinned cloud
    # gen, mode, bag, lidar topic, camera topic, camera, R, t, times, n_frames,
    # rotation deltas, translation deltas, eval min, eval max, max sync offset ms, pinned image, pinned cloud
    request_perturbation = QtCore.Signal(
        int, str, str, str, str, object, object, object, object, int,
        object, object, float, float, float, object, object,
    )
    # gen, mode, bag, lidar topic, camera topic, camera, R, t, times, n_frames,
    # axis, deltas, eval min, eval max, max sync offset ms, pinned image, pinned cloud
    request_fine_scan = QtCore.Signal(
        int, str, str, str, str, object, object, object, object, int, str, object,
        float, float, float, object, object,
    )
    # gen, bag path, lidar topic -- Independent Hold-out Validation's own bag open
    request_holdout_open = QtCore.Signal(int, str, str)
    # a single _HoldoutRequest -- see its docstring for why this is bundled
    # rather than yet another long positional-argument list
    request_holdout = QtCore.Signal(object)
    # gen, bag, camera topic, camera, target, samples, trans threshold mm, rot threshold deg
    request_board_motion = QtCore.Signal(int, str, str, object, object, int, float, float)
    # gen, bag, lidar topic, camera topic, camera, sol, times, samples, eval min/max, stable_only, stable_times
    request_timestamp_diagnostic = QtCore.Signal(
        int, str, str, str, object, object, object, int, float, float, bool, object
    )
    request_board_timestamp_diagnostic = QtCore.Signal(
        int, str, str, str, object, object, object, object, object, int, object, str
    )
    # Emitted right after self.project.validation_history is updated in
    # memory, following a Hold-out result actually being revealed --
    # MainWindow listens for this to persist the project to its existing
    # file path immediately (see main_window.py's
    # _autosave_validation_history), since exposure history is provenance
    # that should survive even if the user never presses "저장…". Carries no
    # payload; the project object itself already holds the new state.
    validation_history_changed = QtCore.Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self._gen = 0
        self._image: np.ndarray | None = None
        self._cloud: np.ndarray | None = None
        self._last_projection = None
        self._picked: int | None = None
        self._intensity: np.ndarray | None = None
        self._lidar_c: np.ndarray | None = None
        self._cam_c: np.ndarray | None = None
        self._sol: Solution | None = None
        self._flipped = False
        self._zoom = 1.0
        self._fit = True
        self._times: list[int] = []
        self._loaded_for: tuple | None = None
        self._scene = None  # the captured scene at the current time, if any
        self._loaded_from: str | None = None  # extrinsic opened from a file
        self._loaded_meta: dict = {}
        self._mf_gen = 0
        self._mf_result: mc.MultiFrameConsistencyResult | None = None
        self._eval_gen = 0
        self._eval_frame_shape: tuple[int, int] = (0, 0)
        # One-shot: consumed and cleared by the very next _request(), so only
        # the jump that set it is affected -- normal scrubbing right after a
        # Worst-Frame jump goes straight back to ordinary first_after pairing.
        self._pinned_camera_t_ns: "int | None" = None
        self._pert_gen = 0
        self._pert_result: pert.PerturbationResult | None = None
        self._fine_gen = 0
        self._fine_result: pert.AxisSensitivity | None = None
        # What each stored result was computed against -- (R, t) copies plus
        # the run settings the diagnostics need to describe it. Diagnostics
        # compare these to the extrinsic shown *now* and leave a stale result
        # out rather than interpret numbers that belong to a different T.
        self._eval_result: ea.EdgeAlignmentResult | None = None
        self._eval_spatial: sa.SpatialAnalysisResult | None = None
        self._eval_T: tuple | None = None
        self._mf_T: tuple | None = None
        self._mf_sync_limit_ms: float = 0.0
        self._pert_T: tuple | None = None
        self._pert_sync_limit_ms: float = 0.0
        self._fine_T: tuple | None = None
        self._fine_mode: str = "multi_frame"
        self._diag_report: dg.DiagnosticReport | None = None
        self._board_motion_gen = 0
        self._board_motion_result: bm.BoardMotionResult | None = None
        self._board_motion_stable_lidar_times: list[int] = []
        self._tsdiag_gen = 0
        self._tsdiag_results: list[td.TimestampDiagnosticMetrics] = []
        self._board_ts_gen = 0
        self._board_ts_result: bt.BoardTimestampResult | None = None

        # Independent Hold-out Validation (see gui.core.evaluation.holdout).
        # Discovery frame-timestamp sets, captured at the moment each run is
        # launched (sample_frame_indices is a deterministic function of
        # len(self._times) alone, so this needs no bag I/O) -- purely to let
        # a later hold-out run check contamination; never influences what
        # Perturbation/Fine Scan themselves score.
        self._pert_discovery_bag: str | None = None
        self._pert_discovery_timestamps: frozenset = frozenset()
        self._pert_search_source: str = "full"  # "quick" | "full" -- which grid search produced self._pert_result
        self._fine_discovery_bag: str | None = None
        self._fine_discovery_timestamps: frozenset = frozenset()
        self._holdout_bag_gen = 0
        self._holdout_bag_path: str | None = None
        self._holdout_times: list[int] = []
        self._holdout_gen = 0
        self._holdout_result: ho.HoldoutResult | None = None
        self._holdout_T: tuple | None = None  # baseline (R, t) the locked candidate was compared against
        # Project-persistent, monotonically increasing -- bumped by exactly 1
        # only when a Perturbation Grid or Fine Scan run *successfully*
        # produces a result (never on a failed/cancelled run, never on a
        # Hold-out run itself, never on a mere combo-box selection -- see
        # _note_discovery_success, called from _on_pert_ready/_on_fine_ready).
        # Restored from self.project.validation_history's top-level
        # "discovery_revision" on project sync (_sync_holdout_history_from_project),
        # so it keeps counting up across an app restart instead of resetting
        # to 0 -- see gui.core.evaluation.holdout.CandidateProvenance's
        # docstring for exactly why a session-local counter here used to
        # make the "Discovery moved on" check unreliable across a restart.
        # Part of every locked candidate's provenance (see
        # gui.core.evaluation.holdout.CandidateProvenance) and compared
        # against the value a hold-out run locked in, to catch a result that
        # has gone stale because Discovery moved on after it was shown. See
        # _holdout_status below.
        self._discovery_revision = 0
        self._holdout_locked_discovery_revision = 0
        # Session-level exposure history, keyed by content-based Source
        # identity -- dict[source_identity, ho.HoldoutSourceExposure], each
        # holding every slice (usable timestamp set) of that source ever
        # revealed. A copied bag folds into the same Source
        # (ho.resolve_source_identity); a heavily-overlapping resample folds
        # into the same Slice (ho.classify_exposure) instead of looking like
        # fresh, unseen data. Recorded only once a result is actually shown
        # to a person (see _on_holdout_ready), never merely because a run
        # started or finished. See gui.core.evaluation.holdout's module
        # docstring and determine_and_record_exposure.
        #
        # Persisted, not just session-local: kept in sync with
        # self.project.validation_history on every write (see
        # _on_holdout_ready), so it survives Project.save/load and an app
        # restart -- otherwise a Hold-out already peeked at in an earlier
        # session could look Independent again simply because the process
        # restarted. `_holdout_history_project` tracks which Project object
        # this registry was last restored from, so switching to a different
        # opened project (see main_window.py's _open, which rebinds
        # self.project without recreating this page) is picked up the next
        # time Step 7 is entered -- see _sync_holdout_history_from_project.
        self._holdout_exposure_registry: dict = {}
        self._holdout_history_project: Project | None = None
        # This result's (B) classification against the registry above,
        # frozen at the moment it was first shown -- combined with the live
        # (A) discovery-staleness check in _holdout_status.
        self._holdout_reveal_status: str = ho.STATUS_INDEPENDENT
        # Kept in sync by main_window.py (mirroring how it already rebinds
        # `self.project` on every page -- see `_open`/`_save`), purely so
        # this page can tell whether `validation_history_changed` will
        # actually reach disk yet. None until the project has been saved (or
        # opened from) a file at least once.
        self.project_path: Path | None = None

        # Candidate A/B Test Package Export (see gui.core.evaluation.ab_export).
        # Read-only repackaging of the Hold-out result above -- never a new
        # evaluation, never a new calibration, never a new candidate search,
        # never a write to the validated baseline. Remembers only the last
        # export directory, purely so "폴더 열기" has somewhere to point;
        # never influences what a later export produces.
        self._last_ab_export_dir: Path | None = None

        # --- controls --------------------------------------------------------
        # Checking the extrinsic only at the scenes it was fitted to is marking
        # your own work: those four points are what the fit minimised, so they
        # agree by construction. A timeline over the whole recording -- and over
        # every bag in the project, including one added purely to check against --
        # is what actually tests it.
        self.bag_combo = QtWidgets.QComboBox()
        self.bag_combo.currentIndexChanged.connect(self._switch_bag)

        self.slider = _FrameSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimumHeight(24)  # a wheel target you can hit without aiming
        self.slider.setToolTip("드래그하거나 마우스 휠로 한 프레임씩 이동합니다")
        self.slider.valueChanged.connect(self._on_slider)
        self.time_label = QtWidgets.QLabel("—")
        self.time_label.setStyleSheet("font-family: monospace;")
        self.time_label.setFixedWidth(150)

        # Reloading on every slider tick would queue a decode per pixel dragged.
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._request)

        self.scene_combo = QtWidgets.QComboBox()
        self.scene_combo.setToolTip("캡처한 scene 으로 이동합니다")
        self.scene_combo.currentIndexChanged.connect(self._goto_scene)

        self.colour_combo = QtWidgets.QComboBox()
        self.colour_combo.addItem("거리", "depth")
        self.colour_combo.addItem("Intensity", "intensity")
        # RViz's Axis Color. Height separates road from wall, which distance
        # cannot: on a forward view both sit at the same depth. It is the default
        # because that separation is what one is usually looking for.
        self.colour_combo.addItem("높이 z", "axis_z")
        self.colour_combo.addItem("전후 x", "axis_x")
        self.colour_combo.addItem("좌우 y", "axis_y")
        # One flat colour, for when the only question is whether lines fall where
        # they should and a ramp just adds a second pattern to read past.
        self.colour_combo.addItem("단색", "solid")
        self.colour_combo.setCurrentIndex(self.colour_combo.findData("axis_z"))
        self.colour_combo.currentIndexChanged.connect(self._redraw)

        self.size_spin = QtWidgets.QDoubleSpinBox()
        self.size_spin.setRange(0.5, 7.0)
        self.size_spin.setSingleStep(0.1)
        self.size_spin.setDecimals(1)
        self.size_spin.setValue(1.5)
        self.size_spin.setSuffix(" px")
        self.size_spin.setToolTip(
            "점 하나를 몇 픽셀로 그릴지. 1.5 미만은 한 픽셀로 찍습니다.\n"
            "실제로 굵어지는 지점은 1.5, 3.0, 5.1 이고 그 사이는 같게 보입니다"
        )
        self.size_spin.valueChanged.connect(self._redraw)

        self.dim_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dim_slider.setRange(0, 100)
        self.dim_slider.setValue(40)
        self.dim_slider.setFixedWidth(90)
        self.dim_slider.valueChanged.connect(self._redraw)

        # Both ends, not just the far one. Points a few centimetres from the lens
        # -- vehicle body, a mount, the sensor's own housing -- project across the
        # whole frame and bury the board, so a near cut matters as much as a far one.
        self.near_spin = QtWidgets.QDoubleSpinBox()
        self.near_spin.setRange(0.0, 200.0)
        self.near_spin.setValue(0.0)
        self.near_spin.setSingleStep(0.5)
        self.near_spin.setSuffix(" m")
        self.near_spin.setSpecialValueText("제한 없음")
        self.near_spin.setToolTip("이 거리보다 가까운 점은 그리지 않습니다")
        self.near_spin.valueChanged.connect(self._on_range)

        self.far_spin = QtWidgets.QDoubleSpinBox()
        self.far_spin.setRange(0.0, 200.0)
        # 10 m showed only the road right in front. Most of what tells you an
        # extrinsic is right -- walls, poles, the far edge of the carriageway --
        # sits beyond that.
        self.far_spin.setValue(50.0)
        self.far_spin.setSingleStep(0.5)
        self.far_spin.setSuffix(" m")
        self.far_spin.setSpecialValueText("제한 없음")
        self.far_spin.setToolTip("이 거리보다 먼 점은 그리지 않습니다")
        self.far_spin.valueChanged.connect(self._on_range)

        # Sliders alongside the boxes: sweeping a cut through the scene to see what
        # each depth contains is a different action from typing a known number, and
        # typing cannot do it. Tenths of a metre, which is the useful resolution
        # here -- 0.5 m steps jump straight past a board.
        self.near_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # Matched to the spin box's 0-200 m at 1/100 m. They were 0-2000 (20 m)
        # while the boxes went to 200, so any value past 20 m pinned the slider
        # at its end and the two disagreed.
        self.near_slider.setRange(0, 20000)
        self.near_slider.setFixedWidth(110)
        self.near_slider.setToolTip("가까운 쪽 자르기")
        self.near_slider.valueChanged.connect(
            lambda v: self._sync_range(self.near_spin, v / 100.0)
        )
        self.far_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.far_slider.setRange(0, 20000)
        self.far_slider.setValue(5000)
        self.far_slider.setFixedWidth(110)
        self.far_slider.setToolTip("먼 쪽 자르기")
        self.far_slider.valueChanged.connect(
            lambda v: self._sync_range(self.far_spin, v / 100.0)
        )

        self.markers_check = QtWidgets.QCheckBox("원 중심 표시")
        self.markers_check.setChecked(True)
        self.markers_check.setToolTip(
            "원 = 카메라가 계산한 구멍 중심\n십자 = LiDAR 가 찾은 중심을 extrinsic 으로 옮긴 것\n"
            "둘이 겹치면 맞고, 대각선으로 어긋나면 반바퀴 뒤집힌 것입니다."
        )
        self.markers_check.toggled.connect(self._redraw)

        self.flip_btn = QtWidgets.QPushButton("반바퀴 뒤집기")
        self.flip_btn.setToolTip(
            "구멍 사각형은 반 바퀴 돌려도 같아 보여서, 180도 뒤집힌 extrinsic 도 잔차가 똑같이 나옵니다.\n"
            "scene 이 하나뿐이면 계산으로는 구분할 수 없으니 화면을 보고 고르세요."
        )
        self.flip_btn.clicked.connect(self._flip)

        self.fit_btn = QtWidgets.QPushButton("전체 보기")
        self.fit_btn.clicked.connect(self._fit_view)
        self.zoom_in = QtWidgets.QPushButton("＋")
        self.zoom_out = QtWidgets.QPushButton("－")
        for b in (self.zoom_in, self.zoom_out):
            b.setFixedWidth(32)
        self.zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.4))
        self.zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.4))

        # --- image -----------------------------------------------------------
        self.canvas = _ZoomLabel()
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setStyleSheet("background: #101216; color: palette(mid);")
        self.canvas.setToolTip(
            "휠: 확대·축소 (커서 기준)\n"
            "Shift+휠: 한 프레임씩 이동\n"
            "Ctrl+휠: 점 크기\n"
            "왼쪽 드래그: 화면 이동"
        )
        self.canvas.zoomed.connect(self._on_wheel)
        self.canvas.stepped.connect(self._step_frames)
        self.canvas.sized.connect(self._step_size)
        self.canvas.panned.connect(self._on_pan)
        self.canvas.picked.connect(self._pick_point)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(360)

        # --- readouts --------------------------------------------------------
        self.verdict = QtWidgets.QLabel("—")
        self.verdict.setTextFormat(QtCore.Qt.RichText)
        self.verdict.setWordWrap(True)

        self.numbers = QtWidgets.QLabel("—")
        self.numbers.setTextFormat(QtCore.Qt.RichText)
        self.numbers.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.export_combo = QtWidgets.QComboBox()
        self.export_combo.addItem("FAST-LIVO2 형식", "livo2")
        self.export_combo.addItem("YAML (4x4 변환)", "yaml")
        self.export_combo.addItem("ROS static_transform_publisher", "tf")
        self.export_combo.addItem("RT_Matrix (평문 3x4 + K + D)", "rt")
        self.export_combo.currentIndexChanged.connect(self._update_export)
        self.export_text = QtWidgets.QPlainTextEdit()
        self.export_text.setReadOnly(True)
        self.export_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.export_text.setMaximumHeight(150)
        # Clicking a projected dot answers "what was this before it became a
        # pixel" -- the question that comes up every time something looks wrong in
        # the overlay and cannot be settled from the picture alone.
        self.pick_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>점을 클릭하면 원래 3D 좌표를 봅니다.</span>"
        )
        self.pick_label.setWordWrap(True)
        self.pick_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.capture_btn = QtWidgets.QPushButton("캡처 저장…")
        self.capture_btn.setToolTip(
            "지금 보고 있는 겹침 그림을 원본 해상도로 저장합니다.\n"
            "같은 이름의 .txt 에 어떤 조건으로 만든 그림인지 함께 기록합니다"
        )
        self.capture_btn.clicked.connect(self._save_capture)

        copy_btn = QtWidgets.QPushButton("복사")
        copy_btn.clicked.connect(self._copy_export)
        save_btn = QtWidgets.QPushButton("파일로 저장…")
        save_btn.clicked.connect(self._save_export)

        # Checking someone else's answer is the common case on a team, and until
        # now this page only opened after step 6 had run here. A saved extrinsic
        # carries the vehicle, the bag and the topics with it, so the file says
        # what it should be checked against.
        self.load_btn = QtWidgets.QPushButton("extrinsic 불러와 검증…")
        self.load_btn.setToolTip(
            "다른 사람이 계산한 extrinsic 파일(YAML)을 열어 이 bag 에 투영해 봅니다.\n"
            "6단계를 돌리지 않아도 되고, 파일에 적힌 bag·토픽·카메라를 그대로 씁니다."
        )
        self.load_btn.clicked.connect(self._load_extrinsic)

        # --- quantitative validation (target-independent) --------------------
        # Step 6's RMSE/LOO/coverage judge fit to the four scenes an extrinsic
        # was solved from. These judge the same extrinsic against ordinary
        # frames it never saw -- a different question, so they live on their
        # own tab rather than mixed into "판정"/"수치" above.
        self.eval_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>Target 없이, 일반 장면에서의 정렬 오차(px)를 봅니다. "
            "6단계 RMSE와는 별개의 지표입니다.</span>"
        )
        self.eval_info_label.setWordWrap(True)

        self.board_motion_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>Phase 1: detects camera board pose per frame, "
            "then classifies board motion. This is analysis only; extrinsic files are not changed.</span>"
        )
        self.board_motion_info_label.setWordWrap(True)
        self.board_motion_samples_spin = QtWidgets.QSpinBox()
        self.board_motion_samples_spin.setRange(5, 5000)
        self.board_motion_samples_spin.setValue(200)
        self.board_motion_samples_spin.setToolTip("Camera frames sampled across the current bag")
        self.board_motion_trans_spin = QtWidgets.QDoubleSpinBox()
        self.board_motion_trans_spin.setRange(0.1, 100.0)
        self.board_motion_trans_spin.setDecimals(1)
        self.board_motion_trans_spin.setValue(5.0)
        self.board_motion_trans_spin.setSuffix(" mm")
        self.board_motion_rot_spin = QtWidgets.QDoubleSpinBox()
        self.board_motion_rot_spin.setRange(0.001, 10.0)
        self.board_motion_rot_spin.setDecimals(3)
        self.board_motion_rot_spin.setValue(0.1)
        self.board_motion_rot_spin.setSuffix(" deg")
        self.board_motion_run_btn = QtWidgets.QPushButton("Run Board Motion Analysis")
        self.board_motion_run_btn.clicked.connect(self._run_board_motion)
        self.board_motion_cancel_btn = QtWidgets.QPushButton("Cancel")
        self.board_motion_cancel_btn.setVisible(False)
        self.board_motion_cancel_btn.clicked.connect(self._cancel_board_motion)
        self.board_motion_export_btn = QtWidgets.QPushButton("Export stable CSV")
        self.board_motion_export_btn.setEnabled(False)
        self.board_motion_export_btn.clicked.connect(self._export_board_motion_csv)
        self.board_motion_stable_only_check = QtWidgets.QCheckBox("Use only stable board frames for Multi-frame")
        self.board_motion_stable_only_check.setEnabled(False)
        self.board_motion_progress_label = QtWidgets.QLabel("")
        self.board_motion_progress_label.setStyleSheet("color: palette(mid);")
        self.board_motion_summary_label = QtWidgets.QLabel("—")
        self.board_motion_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.board_motion_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.tsdiag_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>Diagnostic only: compares Record->Record, "
            "Header->Sweep Start/Mid/End pairing. It does not apply a correction or overwrite calibration.</span>"
        )
        self.tsdiag_info_label.setWordWrap(True)
        self.tsdiag_samples_spin = QtWidgets.QSpinBox()
        self.tsdiag_samples_spin.setRange(5, 500)
        self.tsdiag_samples_spin.setValue(100)
        self.tsdiag_samples_spin.setToolTip("Frames sampled for each timestamp pairing mode")
        self.tsdiag_stable_only_check = QtWidgets.QCheckBox("Stable-only")
        self.tsdiag_run_btn = QtWidgets.QPushButton("Run Timestamp Diagnostic")
        self.tsdiag_run_btn.clicked.connect(self._run_timestamp_diagnostic)
        self.tsdiag_cancel_btn = QtWidgets.QPushButton("Cancel")
        self.tsdiag_cancel_btn.setVisible(False)
        self.tsdiag_cancel_btn.clicked.connect(self._cancel_timestamp_diagnostic)
        self.tsdiag_progress_label = QtWidgets.QLabel("")
        self.tsdiag_progress_label.setStyleSheet("color: palette(mid);")
        self.tsdiag_summary_label = QtWidgets.QLabel("—")
        self.tsdiag_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.tsdiag_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.tsdiag_detail_label = QtWidgets.QLabel("—")
        self.tsdiag_detail_label.setTextFormat(QtCore.Qt.RichText)
        self.tsdiag_detail_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.tsdiag_detail_label.setWordWrap(True)

        self.board_ts_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>Uses existing ROI/auto-ROI and LiDAR board detector; "
            "summarizes per-point timestamps of the detected board plane only. Analysis only.</span>"
        )
        self.board_ts_info_label.setWordWrap(True)
        self.board_ts_samples_spin = QtWidgets.QSpinBox()
        self.board_ts_samples_spin.setRange(5, 500)
        self.board_ts_samples_spin.setValue(100)
        self.board_ts_run_btn = QtWidgets.QPushButton("Run Board Timestamp Diagnostic")
        self.board_ts_run_btn.clicked.connect(self._run_board_timestamp_diagnostic)
        self.board_ts_cancel_btn = QtWidgets.QPushButton("Cancel")
        self.board_ts_cancel_btn.setVisible(False)
        self.board_ts_cancel_btn.clicked.connect(self._cancel_board_timestamp_diagnostic)
        self.board_ts_export_btn = QtWidgets.QPushButton("Export board timestamp CSV")
        self.board_ts_export_btn.setEnabled(False)
        self.board_ts_export_btn.clicked.connect(self._export_board_timestamp_csv)
        self.board_ts_progress_label = QtWidgets.QLabel("")
        self.board_ts_progress_label.setStyleSheet("color: palette(mid);")
        self.board_ts_summary_label = QtWidgets.QLabel("—")
        self.board_ts_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.board_ts_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.board_ts_summary_label.setWordWrap(True)

        # Deliberately separate from near_spin/far_spin above (the display
        # range): reusing the display's projection would mean nudging that
        # slider silently changes the metric, and two calibrations checked at
        # different display ranges are not a fair comparison. This range
        # applies to both the current-frame button and every Multi-frame run.
        self.eval_range_note = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>표시(Display) range와 별개입니다 — "
            "평가에는 항상 이 범위만 적용됩니다.</span>"
        )
        self.eval_range_note.setWordWrap(True)

        self.eval_min_range_spin = QtWidgets.QDoubleSpinBox()
        self.eval_min_range_spin.setRange(0.0, 200.0)
        self.eval_min_range_spin.setValue(0.0)
        self.eval_min_range_spin.setSingleStep(0.5)
        self.eval_min_range_spin.setSuffix(" m")
        self.eval_min_range_spin.setSpecialValueText("제한 없음")
        self.eval_min_range_spin.setToolTip("평가 범위: 이 거리보다 가까운 점은 제외합니다")

        self.eval_max_range_spin = QtWidgets.QDoubleSpinBox()
        self.eval_max_range_spin.setRange(0.0, 200.0)
        self.eval_max_range_spin.setValue(0.0)
        self.eval_max_range_spin.setSingleStep(0.5)
        self.eval_max_range_spin.setSuffix(" m")
        self.eval_max_range_spin.setSpecialValueText("제한 없음")
        self.eval_max_range_spin.setToolTip("평가 범위: 이 거리보다 먼 점은 제외합니다")

        self.eval_run_btn = QtWidgets.QPushButton("현재 프레임 평가")
        self.eval_run_btn.setToolTip(
            "지금 화면의 LiDAR depth-edge 와 카메라 Canny edge 사이 pixel 거리를 계산합니다.\n"
            "위 평가 범위로 새로 투영합니다 (화면 표시용 range와는 다를 수 있습니다).\n"
            "4K 해상도에서 최대 1초 가까이 걸릴 수 있어 백그라운드에서 실행합니다."
        )
        self.eval_run_btn.clicked.connect(self._eval_current_frame)

        self.eval_edge_label = QtWidgets.QLabel("—")
        self.eval_edge_label.setTextFormat(QtCore.Qt.RichText)
        self.eval_edge_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.eval_spatial_label = QtWidgets.QLabel("—")
        self.eval_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.eval_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.eval_depth_label = QtWidgets.QLabel("—")
        self.eval_depth_label.setTextFormat(QtCore.Qt.RichText)
        self.eval_depth_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.mf_count_spin = QtWidgets.QSpinBox()
        self.mf_count_spin.setRange(5, 5000)
        self.mf_count_spin.setValue(100)
        self.mf_count_spin.setToolTip("전체 timeline 에서 균등하게 뽑아 평가할 frame 개수")

        # On a moving vehicle, a camera/LiDAR pairing looser than this reads
        # as a rotation error in the pixel statistics -- such a frame is kept
        # out of the geometric aggregate entirely (see sync_rejected below)
        # rather than quietly counted as evidence about the extrinsic.
        self.max_sync_spin = QtWidgets.QSpinBox()
        self.max_sync_spin.setRange(0, 1000)
        self.max_sync_spin.setValue(50)
        self.max_sync_spin.setSuffix(" ms")
        self.max_sync_spin.setSpecialValueText("제한 없음")
        self.max_sync_spin.setToolTip(
            "camera/LiDAR 페어링 시간 차이가 이 값을 넘는 frame은 geometric 평가에서 제외하고\n"
            "sync_rejected 로 따로 집계합니다. 0 = 제한 없음(그래도 오차는 계속 보고합니다)."
        )

        self.mf_run_btn = QtWidgets.QPushButton("Multi-frame 평가 실행")
        self.mf_run_btn.setToolTip(
            "scene 이 아닌 일반 frame을 timeline 전체에서 균등 추출해, camera/LiDAR nearest timestamp 로\n"
            "짝지은 뒤 각각 Edge Alignment 를 계산합니다. 시간이 걸릴 수 있어 백그라운드에서 실행합니다."
        )
        self.mf_run_btn.clicked.connect(self._run_multiframe)

        self.mf_cancel_btn = QtWidgets.QPushButton("취소")
        self.mf_cancel_btn.setVisible(False)
        self.mf_cancel_btn.clicked.connect(self._cancel_multiframe)

        self.mf_progress_label = QtWidgets.QLabel("")
        self.mf_progress_label.setStyleSheet("color: palette(mid);")

        self.mf_summary_label = QtWidgets.QLabel("—")
        self.mf_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.mf_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.mf_worst_table = QtWidgets.QTableWidget(0, 5)
        self.mf_worst_table.setHorizontalHeaderLabels(["시간", "Error", "Match", "Sync Δt", "이동"])
        self.mf_worst_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.mf_worst_table.verticalHeader().setVisible(False)

        # --- perturbation sensitivity (diagnostic only) -----------------------
        # Answers "is the current T sitting at a local minimum of Edge
        # Alignment error, or would a tiny nudge do better" -- strictly by
        # re-evaluating the existing metric at small, temporary copies of T.
        # Nothing here ever writes to Solution/Project; see
        # gui.core.evaluation.perturbation's module docstring.
        self.pert_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>진단 전용 — T_cam_lidar를 자동으로 수정하거나 저장하지 않습니다.</span>"
        )
        self.pert_info_label.setWordWrap(True)

        self.pert_mode_current_radio = QtWidgets.QRadioButton("Current Frame")
        self.pert_mode_multi_radio = QtWidgets.QRadioButton("Multi-frame")
        self.pert_mode_multi_radio.setChecked(True)
        self.pert_mode_group = QtWidgets.QButtonGroup(self)
        self.pert_mode_group.addButton(self.pert_mode_current_radio)
        self.pert_mode_group.addButton(self.pert_mode_multi_radio)
        self.pert_mode_current_radio.toggled.connect(self._update_pert_trial_count)

        # Quick (13 trials) is the default so a first look stays fast; Full
        # (43 trials, the original grid) is unchanged and still available.
        self.pert_search_quick_radio = QtWidgets.QRadioButton("Quick")
        self.pert_search_full_radio = QtWidgets.QRadioButton("Full")
        self.pert_search_quick_radio.setChecked(True)
        self.pert_search_group = QtWidgets.QButtonGroup(self)
        self.pert_search_group.addButton(self.pert_search_quick_radio)
        self.pert_search_group.addButton(self.pert_search_full_radio)
        self.pert_search_quick_radio.setToolTip(
            f"Roll/Pitch/Yaw {pert.QUICK_ROTATION_DELTAS_DEG}, Tx/Ty/Tz {pert.QUICK_TRANSLATION_DELTAS_MM} -- 13 trial"
        )
        self.pert_search_full_radio.setToolTip(
            f"Roll/Pitch/Yaw {pert.DEFAULT_ROTATION_DELTAS_DEG}, Tx/Ty/Tz {pert.DEFAULT_TRANSLATION_DELTAS_MM} -- 43 trial"
        )
        self.pert_search_quick_radio.toggled.connect(self._update_pert_trial_count)

        self.pert_frames_spin = QtWidgets.QSpinBox()
        self.pert_frames_spin.setRange(5, 500)
        self.pert_frames_spin.setValue(30)
        self.pert_frames_spin.setToolTip(
            "Multi-frame 모드에서 사용할 frame 수.\n"
            "축(6개) x perturbation 값마다 다시 평가하므로 큰 값은 비용이 매우 커집니다."
        )
        self.pert_frames_spin.valueChanged.connect(self._update_pert_trial_count)

        self.pert_trial_count_label = QtWidgets.QLabel("")
        self.pert_trial_count_label.setStyleSheet("color: palette(mid);")

        self.pert_run_btn = QtWidgets.QPushButton("분석 실행")
        self.pert_run_btn.setToolTip(
            "현재 T 주변에서 Roll/Pitch/Yaw/Tx/Ty/Tz를 각각 조금씩 바꿔가며\n"
            "Edge Alignment가 어떻게 변하는지 봅니다. T는 절대 변경되지 않습니다."
        )
        self.pert_run_btn.clicked.connect(self._run_perturbation)

        self.pert_cancel_btn = QtWidgets.QPushButton("취소")
        self.pert_cancel_btn.setVisible(False)
        self.pert_cancel_btn.clicked.connect(self._cancel_perturbation)

        self.pert_progress_label = QtWidgets.QLabel("")
        self.pert_progress_label.setStyleSheet("color: palette(mid);")

        self.pert_axis_combo = QtWidgets.QComboBox()
        for key in (*pert.ROTATION_AXES, *pert.TRANSLATION_AXES):
            self.pert_axis_combo.addItem(pert.AXIS_LABELS[key], key)
        self.pert_axis_combo.currentIndexChanged.connect(self._display_perturbation_axis)
        self.pert_axis_combo.currentIndexChanged.connect(self._on_pert_axis_changed_for_fine_scan)

        self.pert_table_label = QtWidgets.QLabel("—")
        self.pert_table_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.pert_frame_table_label = QtWidgets.QLabel("—")
        self.pert_frame_table_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_frame_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.pert_frame_table_label.setToolTip(
            "Multi-frame 모드의 기본 판단 기준: pooled(합산) 통계는 edge 점이 많은 frame이\n"
            "결과를 지배할 수 있어, frame마다 동일 가중치로 비교한 값을 별도로 보여줍니다."
        )

        self.pert_summary_label = QtWidgets.QLabel("—")
        self.pert_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_summary_label.setWordWrap(True)
        self.pert_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.pert_spatial_label = QtWidgets.QLabel("—")
        self.pert_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.pert_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- fine scan (one axis, custom range) -------------------------------
        self.fine_axis_label = QtWidgets.QLabel("—")
        self.fine_axis_label.setStyleSheet("color: palette(mid);")

        self.fine_min_spin = QtWidgets.QDoubleSpinBox()
        self.fine_min_spin.setRange(-100.0, 100.0)
        self.fine_min_spin.setDecimals(3)
        self.fine_min_spin.setValue(-0.2)

        self.fine_max_spin = QtWidgets.QDoubleSpinBox()
        self.fine_max_spin.setRange(-100.0, 100.0)
        self.fine_max_spin.setDecimals(3)
        self.fine_max_spin.setValue(0.2)

        self.fine_step_spin = QtWidgets.QDoubleSpinBox()
        self.fine_step_spin.setRange(0.001, 100.0)
        self.fine_step_spin.setDecimals(3)
        self.fine_step_spin.setValue(0.05)

        self.fine_run_btn = QtWidgets.QPushButton("Fine Scan 실행")
        self.fine_run_btn.setToolTip("선택한 축(위 콤보박스)만, 지정한 범위/간격으로 다시 촘촘히 평가합니다.")
        self.fine_run_btn.clicked.connect(self._run_fine_scan)

        self.fine_cancel_btn = QtWidgets.QPushButton("취소")
        self.fine_cancel_btn.setVisible(False)
        self.fine_cancel_btn.clicked.connect(self._cancel_fine_scan)

        self.fine_progress_label = QtWidgets.QLabel("")
        self.fine_progress_label.setStyleSheet("color: palette(mid);")

        self.fine_table_label = QtWidgets.QLabel("—")
        self.fine_table_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.fine_frame_table_label = QtWidgets.QLabel("—")
        self.fine_frame_table_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_frame_table_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.fine_summary_label = QtWidgets.QLabel("—")
        self.fine_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_summary_label.setWordWrap(True)
        self.fine_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.fine_spatial_label = QtWidgets.QLabel("—")
        self.fine_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.fine_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- independent hold-out validation -----------------------------------
        # See gui.core.evaluation.holdout's module docstring: a candidate is
        # chosen from Discovery data (Perturbation Sensitivity / Fine Scan,
        # via the axis combo above); this section only checks whether that
        # *same, locked* candidate still looks better than baseline on data
        # it was never searched against. Never fits T, never re-searches,
        # never writes to self._sol, the project, or a file.
        self.holdout_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>Candidate was selected using Discovery data.<br>"
            "Hold-out data is used only for validation, not for candidate search.<br>"
            "Re-running Fine Scan after seeing a hold-out result and locking a new candidate from it means "
            "this hold-out no longer validates an untouched candidate.</span>"
        )
        self.holdout_info_label.setWordWrap(True)

        self.holdout_persistence_note_label = QtWidgets.QLabel("")
        self.holdout_persistence_note_label.setWordWrap(True)
        self.holdout_persistence_note_label.setStyleSheet("color:#b26a00")

        self.holdout_source_bag_radio = QtWidgets.QRadioButton("Separate Bag")
        self.holdout_source_bag_radio.setChecked(True)
        self.holdout_source_range_radio = QtWidgets.QRadioButton("Current Bag Time Range")
        self._holdout_source_group = QtWidgets.QButtonGroup(self)
        self._holdout_source_group.addButton(self.holdout_source_bag_radio)
        self._holdout_source_group.addButton(self.holdout_source_range_radio)
        self.holdout_source_bag_radio.toggled.connect(self._on_holdout_source_changed)

        self.holdout_bag_combo = QtWidgets.QComboBox()
        self.holdout_bag_combo.setToolTip(
            "프로젝트에 추가된 bag 중에서 고르거나, '찾아보기…'로 다른 .bag 파일을 선택하세요.\n"
            "가장 독립적인 검증은 calibration/Discovery 에 쓰이지 않은 별도 recording 입니다."
        )
        self.holdout_bag_combo.currentIndexChanged.connect(self._on_holdout_bag_combo_changed)
        self.holdout_bag_status_label = QtWidgets.QLabel("—")
        self.holdout_bag_status_label.setStyleSheet("color:palette(mid)")
        self.holdout_bag_status_label.setWordWrap(True)

        self.holdout_range_start_spin = QtWidgets.QDoubleSpinBox()
        self.holdout_range_start_spin.setRange(0.0, 0.0)
        self.holdout_range_start_spin.setDecimals(2)
        self.holdout_range_start_spin.setSuffix(" s")
        self.holdout_range_start_spin.setEnabled(False)
        self.holdout_range_start_spin.valueChanged.connect(self._update_holdout_preview)

        self.holdout_range_end_spin = QtWidgets.QDoubleSpinBox()
        self.holdout_range_end_spin.setRange(0.0, 0.0)
        self.holdout_range_end_spin.setDecimals(2)
        self.holdout_range_end_spin.setSuffix(" s")
        self.holdout_range_end_spin.setEnabled(False)
        self.holdout_range_end_spin.valueChanged.connect(self._update_holdout_preview)

        self.holdout_range_note = QtWidgets.QLabel("")
        self.holdout_range_note.setStyleSheet("color:palette(mid)")
        self.holdout_range_note.setWordWrap(True)
        self.holdout_range_note.setVisible(False)

        # Candidate lock -- axis is the same combo Perturbation/Fine Scan
        # already use above (self.pert_axis_combo); only which *source*
        # (Quick/Full grid vs Fine Scan) and which of its tested, non-zero
        # deltas to lock are chosen here. The candidate actually used is
        # frozen into the request at the moment "Hold-out 평가 실행" is
        # pressed -- see _run_holdout -- so changing these combos afterward
        # can never affect a run already in flight.
        self.holdout_candidate_source_combo = QtWidgets.QComboBox()
        self.holdout_candidate_source_combo.addItem("Perturbation Grid (Quick/Full)", "grid")
        self.holdout_candidate_source_combo.addItem("Fine Scan", "fine")
        self.holdout_candidate_source_combo.currentIndexChanged.connect(self._refresh_holdout_candidate_deltas)
        self.pert_axis_combo.currentIndexChanged.connect(self._refresh_holdout_candidate_deltas)

        self.holdout_delta_combo = QtWidgets.QComboBox()
        self.holdout_delta_combo.setToolTip("Perturbation Sensitivity 또는 Fine Scan이 실제로 평가한 delta 중에서만 고릅니다.")
        self.holdout_delta_combo.currentIndexChanged.connect(self._update_holdout_preview)

        self.holdout_candidate_label = QtWidgets.QLabel("—")
        self.holdout_candidate_label.setWordWrap(True)

        self.holdout_frames_spin = QtWidgets.QSpinBox()
        self.holdout_frames_spin.setRange(1, 2000)
        self.holdout_frames_spin.setValue(50)
        self.holdout_frames_spin.valueChanged.connect(self._update_holdout_preview)

        # Temporal guard band -- see gui.core.evaluation.holdout.check_overlap.
        # An exact-timestamp check alone is not a reliable independence test:
        # adjacent frames from a moving vehicle can be nearly the same scene
        # at different exact timestamps.
        self.holdout_margin_spin = QtWidgets.QDoubleSpinBox()
        self.holdout_margin_spin.setRange(0.0, 3600.0)
        self.holdout_margin_spin.setDecimals(1)
        self.holdout_margin_spin.setSuffix(" sec")
        self.holdout_margin_spin.setValue(ho.DEFAULT_EXCLUSION_MARGIN_S)
        self.holdout_margin_spin.setToolTip(
            "Discovery frame 주변 이 시간 안의 Hold-out frame은 제외합니다 (같은 bag/시간 구간일 때만).\n"
            "목적: 데이터 누수 방지. Bootstrap Block Duration(시간 상관 구조 반영)과는 별개의 설정이며\n"
            "하나를 바꿔도 다른 하나는 바뀌지 않습니다.\n"
            "Configurable heuristic default일 뿐, 검증된 절대 기준이 아닙니다."
        )
        self.holdout_margin_spin.valueChanged.connect(self._update_holdout_preview)

        self.holdout_overlap_label = QtWidgets.QLabel("")
        self.holdout_overlap_label.setWordWrap(True)

        # Block bootstrap -- see gui.core.evaluation.holdout._run_block_bootstrap.
        self.holdout_bootstrap_iter_spin = QtWidgets.QSpinBox()
        self.holdout_bootstrap_iter_spin.setRange(100, 20000)
        self.holdout_bootstrap_iter_spin.setSingleStep(100)
        self.holdout_bootstrap_iter_spin.setValue(ho.DEFAULT_BOOTSTRAP_ITERATIONS)

        self.holdout_bootstrap_block_spin = QtWidgets.QDoubleSpinBox()
        self.holdout_bootstrap_block_spin.setRange(0.1, 300.0)
        self.holdout_bootstrap_block_spin.setDecimals(1)
        self.holdout_bootstrap_block_spin.setSuffix(" sec")
        self.holdout_bootstrap_block_spin.setValue(ho.DEFAULT_BOOTSTRAP_BLOCK_S)
        self.holdout_bootstrap_block_spin.setToolTip(
            "Method: Non-overlapping Block Bootstrap.\n"
            "연속된 주행 frame은 서로 상관되어 있어, 개별 frame을 독립 표본처럼 resampling하면\n"
            "불확실성을 과소평가합니다. 이 시간 폭의 겹치지 않는 block 단위로만 resampling합니다.\n"
            "Discovery Exclusion Margin(데이터 누수 방지)과는 별개의 설정입니다 -- 하나를 바꿔도\n"
            "다른 하나는 바뀌지 않습니다.\n"
            "Configurable heuristic default일 뿐, 검증된 절대 기준이 아닙니다."
        )

        self.holdout_run_btn = QtWidgets.QPushButton("Hold-out 평가 실행")
        self.holdout_run_btn.setEnabled(False)
        self.holdout_run_btn.clicked.connect(self._run_holdout)

        self.holdout_cancel_btn = QtWidgets.QPushButton("취소")
        self.holdout_cancel_btn.setVisible(False)
        self.holdout_cancel_btn.clicked.connect(self._cancel_holdout)

        self.holdout_progress_label = QtWidgets.QLabel("")
        self.holdout_progress_label.setStyleSheet("color:palette(mid)")

        self.holdout_status_label = QtWidgets.QLabel("")
        self.holdout_status_label.setWordWrap(True)
        self.holdout_status_label.setToolTip(
            "Independent = 이 결과를 계산한 뒤 Perturbation/Fine Scan을 다시 실행하지 않았습니다.\n"
            "Exploratory = 이 결과를 본 뒤 Discovery를 다시 실행했습니다 -- 더 이상 손대지 않은 후보에 대한\n"
            "독립 검증으로 볼 수 없습니다."
        )

        self.holdout_summary_label = QtWidgets.QLabel("—")
        self.holdout_summary_label.setTextFormat(QtCore.Qt.RichText)
        self.holdout_summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.holdout_spatial_label = QtWidgets.QLabel("—")
        self.holdout_spatial_label.setTextFormat(QtCore.Qt.RichText)
        self.holdout_spatial_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.holdout_depth_label = QtWidgets.QLabel("—")
        self.holdout_depth_label.setTextFormat(QtCore.Qt.RichText)
        self.holdout_depth_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- candidate A/B test package export ----------------------------------
        # See gui.core.evaluation.ab_export's module docstring: packages the
        # Hold-out comparison above into a reproducible directory for a real
        # vehicle/projection/E2E pipeline to A/B baseline against candidate --
        # never a new evaluation, never a new calibration, never a new
        # candidate search, and never a write to the validated baseline. The
        # actual precondition checks (current extrinsic still matches the
        # locked baseline, recomputed candidate still matches its locked
        # fingerprint) run inside _export_ab_package itself
        # (ab_export.build_snapshot), not here -- this button is enabled
        # whenever there is something worth trying to export.
        self.ab_export_btn = QtWidgets.QPushButton("Candidate A/B Package 내보내기")
        self.ab_export_btn.setEnabled(False)
        self.ab_export_btn.setToolTip(
            "현재 Baseline, 잠긴 Candidate, 이번 Hold-out 결과를 하나의 재현 가능한 디렉터리로 내보냅니다.\n"
            "새로운 평가/calibration을 실행하지 않으며, 기존 baseline calibration을 절대 수정하지 않습니다."
        )
        self.ab_export_btn.clicked.connect(self._export_ab_package)

        self.ab_export_open_btn = QtWidgets.QPushButton("폴더 열기")
        self.ab_export_open_btn.setEnabled(False)
        self.ab_export_open_btn.clicked.connect(self._open_ab_export_folder)

        self.ab_export_status_label = QtWidgets.QLabel("")
        self.ab_export_status_label.setWordWrap(True)
        self.ab_export_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- diagnostic evidence (interpretation only) -------------------------
        # Reads the results above as they already are; never re-projects,
        # re-scores, re-perturbs, or re-solves, and never offers to change T --
        # see gui.core.evaluation.diagnostics.
        self.diag_info_label = QtWidgets.QLabel(
            "<span style='color:palette(mid)'>이미 계산된 결과만 해석합니다 — 원인을 확정하거나 "
            "T_cam_lidar를 수정하지 않습니다.<br>"
            "<b>Evidence strength, not a calibration correctness score.</b><br>"
            "Strength thresholds (0.05 px, 70%, ...) are configurable heuristic defaults for this project, "
            "not sensor-validated correctness cut-offs — see DiagnosticThresholds.</span>"
        )
        self.diag_info_label.setWordWrap(True)

        self.diag_run_btn = QtWidgets.QPushButton("진단 요약 생성")
        self.diag_run_btn.setToolTip(
            "Multi-frame / Perturbation / Fine Scan / 현재 프레임 평가 / 6단계 Leave-One-Out 결과를\n"
            "evidence category 별로 묶어 보여줍니다. 새로운 평가는 실행하지 않습니다."
        )
        self.diag_run_btn.clicked.connect(self._generate_diagnostics)

        self.diag_copy_btn = QtWidgets.QPushButton("진단 결과 복사")
        self.diag_copy_btn.setEnabled(False)
        self.diag_copy_btn.clicked.connect(self._copy_diagnostics)

        self.diag_label = QtWidgets.QLabel("—")
        self.diag_label.setTextFormat(QtCore.Qt.RichText)
        self.diag_label.setWordWrap(True)
        self.diag_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # --- layout ----------------------------------------------------------
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("bag"))
        top.addWidget(self.bag_combo, 1)
        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("scene 이동"))
        top.addWidget(self.scene_combo)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("색상"))
        top.addWidget(self.colour_combo)
        top.addWidget(QtWidgets.QLabel("점"))
        top.addWidget(self.size_spin)
        top.addWidget(QtWidgets.QLabel("배경"))
        top.addWidget(self.dim_slider)
        top.addWidget(QtWidgets.QLabel("거리"))
        top.addWidget(self.near_spin)
        top.addWidget(self.near_slider)
        top.addWidget(QtWidgets.QLabel("~"))
        top.addWidget(self.far_spin)
        top.addWidget(self.far_slider)
        top.addWidget(self.markers_check)
        top.addStretch(1)
        top.addWidget(self.zoom_out)
        top.addWidget(self.zoom_in)
        top.addWidget(self.fit_btn)

        timeline = QtWidgets.QHBoxLayout()
        timeline.addWidget(self.slider, 1)
        timeline.addWidget(self.time_label)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("<b>판정</b>"))
        side.addWidget(self.verdict)
        side.addWidget(self.flip_btn)
        side.addSpacing(10)
        side.addWidget(QtWidgets.QLabel("<b>수치</b>"))
        side.addWidget(self.numbers)
        side.addStretch(1)
        side.addWidget(QtWidgets.QLabel("<b>클릭한 점</b>"))
        side.addWidget(self.pick_label)
        side.addSpacing(10)
        side.addWidget(self.capture_btn)
        side.addWidget(self.load_btn)
        side.addSpacing(10)
        side.addWidget(QtWidgets.QLabel("<b>내보내기</b>"))
        side.addWidget(self.export_combo)
        side.addWidget(self.export_text)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(copy_btn)
        row.addWidget(save_btn)
        side.addLayout(row)
        side_box = QtWidgets.QWidget()
        side_box.setLayout(side)

        eval_layout = QtWidgets.QVBoxLayout()
        eval_layout.addWidget(self.eval_info_label)
        eval_layout.addSpacing(6)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Board Motion Analysis</b>"))
        eval_layout.addWidget(self.board_motion_info_label)
        bm_samples_row = QtWidgets.QHBoxLayout()
        bm_samples_row.addWidget(QtWidgets.QLabel("Samples"))
        bm_samples_row.addWidget(self.board_motion_samples_spin)
        eval_layout.addLayout(bm_samples_row)
        bm_threshold_row = QtWidgets.QHBoxLayout()
        bm_threshold_row.addWidget(QtWidgets.QLabel("Translation"))
        bm_threshold_row.addWidget(self.board_motion_trans_spin)
        bm_threshold_row.addWidget(QtWidgets.QLabel("Rotation"))
        bm_threshold_row.addWidget(self.board_motion_rot_spin)
        eval_layout.addLayout(bm_threshold_row)
        bm_run_row = QtWidgets.QHBoxLayout()
        bm_run_row.addWidget(self.board_motion_run_btn, 1)
        bm_run_row.addWidget(self.board_motion_cancel_btn)
        bm_run_row.addWidget(self.board_motion_export_btn)
        eval_layout.addLayout(bm_run_row)
        eval_layout.addWidget(self.board_motion_stable_only_check)
        eval_layout.addWidget(self.board_motion_progress_label)
        eval_layout.addWidget(self.board_motion_summary_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Timestamp Diagnostic</b>"))
        eval_layout.addWidget(self.tsdiag_info_label)
        tsdiag_options_row = QtWidgets.QHBoxLayout()
        tsdiag_options_row.addWidget(QtWidgets.QLabel("Samples"))
        tsdiag_options_row.addWidget(self.tsdiag_samples_spin)
        tsdiag_options_row.addWidget(self.tsdiag_stable_only_check)
        eval_layout.addLayout(tsdiag_options_row)
        tsdiag_run_row = QtWidgets.QHBoxLayout()
        tsdiag_run_row.addWidget(self.tsdiag_run_btn, 1)
        tsdiag_run_row.addWidget(self.tsdiag_cancel_btn)
        eval_layout.addLayout(tsdiag_run_row)
        eval_layout.addWidget(self.tsdiag_progress_label)
        eval_layout.addWidget(self.tsdiag_summary_label)
        eval_layout.addWidget(self.tsdiag_detail_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Board Timestamp Diagnostic</b>"))
        eval_layout.addWidget(self.board_ts_info_label)
        board_ts_options_row = QtWidgets.QHBoxLayout()
        board_ts_options_row.addWidget(QtWidgets.QLabel("Samples"))
        board_ts_options_row.addWidget(self.board_ts_samples_spin)
        eval_layout.addLayout(board_ts_options_row)
        board_ts_run_row = QtWidgets.QHBoxLayout()
        board_ts_run_row.addWidget(self.board_ts_run_btn, 1)
        board_ts_run_row.addWidget(self.board_ts_cancel_btn)
        board_ts_run_row.addWidget(self.board_ts_export_btn)
        eval_layout.addLayout(board_ts_run_row)
        eval_layout.addWidget(self.board_ts_progress_label)
        eval_layout.addWidget(self.board_ts_summary_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>평가 범위 (Evaluation Range)</b>"))
        eval_layout.addWidget(self.eval_range_note)
        eval_range_row = QtWidgets.QHBoxLayout()
        eval_range_row.addWidget(QtWidgets.QLabel("Min"))
        eval_range_row.addWidget(self.eval_min_range_spin)
        eval_range_row.addWidget(QtWidgets.QLabel("Max"))
        eval_range_row.addWidget(self.eval_max_range_spin)
        eval_layout.addLayout(eval_range_row)
        eval_layout.addSpacing(10)
        eval_layout.addWidget(self.eval_run_btn)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Edge Alignment</b>"))
        eval_layout.addWidget(self.eval_edge_label)
        eval_layout.addSpacing(8)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Spatial</b>"))
        eval_layout.addWidget(self.eval_spatial_label)
        eval_layout.addSpacing(8)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Depth</b>"))
        eval_layout.addWidget(self.eval_depth_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Multi-frame 평가</b>"))
        mf_row = QtWidgets.QHBoxLayout()
        mf_row.addWidget(QtWidgets.QLabel("평가 프레임 수"))
        mf_row.addWidget(self.mf_count_spin)
        eval_layout.addLayout(mf_row)
        sync_row = QtWidgets.QHBoxLayout()
        sync_row.addWidget(QtWidgets.QLabel("Max Sync Offset"))
        sync_row.addWidget(self.max_sync_spin)
        eval_layout.addLayout(sync_row)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.mf_run_btn, 1)
        run_row.addWidget(self.mf_cancel_btn)
        eval_layout.addLayout(run_row)
        eval_layout.addWidget(self.mf_progress_label)
        eval_layout.addWidget(self.mf_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Worst Frames"))
        eval_layout.addWidget(self.mf_worst_table)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Perturbation Sensitivity</b>"))
        eval_layout.addWidget(self.pert_info_label)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self.pert_mode_current_radio)
        mode_row.addWidget(self.pert_mode_multi_radio)
        eval_layout.addLayout(mode_row)
        search_row = QtWidgets.QHBoxLayout()
        search_row.addWidget(QtWidgets.QLabel("Search Mode"))
        search_row.addWidget(self.pert_search_quick_radio)
        search_row.addWidget(self.pert_search_full_radio)
        eval_layout.addLayout(search_row)
        pert_frames_row = QtWidgets.QHBoxLayout()
        pert_frames_row.addWidget(QtWidgets.QLabel("Frames"))
        pert_frames_row.addWidget(self.pert_frames_spin)
        eval_layout.addLayout(pert_frames_row)
        eval_layout.addWidget(self.pert_trial_count_label)
        pert_run_row = QtWidgets.QHBoxLayout()
        pert_run_row.addWidget(self.pert_run_btn, 1)
        pert_run_row.addWidget(self.pert_cancel_btn)
        eval_layout.addLayout(pert_run_row)
        eval_layout.addWidget(self.pert_progress_label)
        pert_axis_row = QtWidgets.QHBoxLayout()
        pert_axis_row.addWidget(QtWidgets.QLabel("축"))
        pert_axis_row.addWidget(self.pert_axis_combo)
        eval_layout.addLayout(pert_axis_row)
        eval_layout.addWidget(QtWidgets.QLabel("Pooled (diagnostic)"))
        eval_layout.addWidget(self.pert_table_label)
        eval_layout.addWidget(QtWidgets.QLabel("Frame-balanced (Multi-frame 판단 기준)"))
        eval_layout.addWidget(self.pert_frame_table_label)
        eval_layout.addWidget(self.pert_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Spatial (Baseline vs Lowest tested)"))
        eval_layout.addWidget(self.pert_spatial_label)
        eval_layout.addSpacing(10)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Fine Scan</b>"))
        eval_layout.addWidget(self.fine_axis_label)
        fine_range_row = QtWidgets.QHBoxLayout()
        fine_range_row.addWidget(QtWidgets.QLabel("Min"))
        fine_range_row.addWidget(self.fine_min_spin)
        fine_range_row.addWidget(QtWidgets.QLabel("Max"))
        fine_range_row.addWidget(self.fine_max_spin)
        fine_range_row.addWidget(QtWidgets.QLabel("Step"))
        fine_range_row.addWidget(self.fine_step_spin)
        eval_layout.addLayout(fine_range_row)
        fine_run_row = QtWidgets.QHBoxLayout()
        fine_run_row.addWidget(self.fine_run_btn, 1)
        fine_run_row.addWidget(self.fine_cancel_btn)
        eval_layout.addLayout(fine_run_row)
        eval_layout.addWidget(self.fine_progress_label)
        eval_layout.addWidget(QtWidgets.QLabel("Pooled (diagnostic)"))
        eval_layout.addWidget(self.fine_table_label)
        eval_layout.addWidget(QtWidgets.QLabel("Frame-balanced"))
        eval_layout.addWidget(self.fine_frame_table_label)
        eval_layout.addWidget(self.fine_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Spatial (Baseline vs Lowest tested)"))
        eval_layout.addWidget(self.fine_spatial_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Independent Hold-out Validation</b>"))
        eval_layout.addWidget(self.holdout_info_label)
        eval_layout.addWidget(self.holdout_persistence_note_label)
        eval_layout.addWidget(QtWidgets.QLabel("Source"))
        holdout_source_row = QtWidgets.QHBoxLayout()
        holdout_source_row.addWidget(self.holdout_source_bag_radio)
        holdout_source_row.addWidget(self.holdout_source_range_radio)
        eval_layout.addLayout(holdout_source_row)
        eval_layout.addWidget(self.holdout_bag_combo)
        eval_layout.addWidget(self.holdout_bag_status_label)
        holdout_range_row = QtWidgets.QHBoxLayout()
        holdout_range_row.addWidget(QtWidgets.QLabel("Start"))
        holdout_range_row.addWidget(self.holdout_range_start_spin)
        holdout_range_row.addWidget(QtWidgets.QLabel("End"))
        holdout_range_row.addWidget(self.holdout_range_end_spin)
        eval_layout.addLayout(holdout_range_row)
        eval_layout.addWidget(self.holdout_range_note)
        eval_layout.addWidget(QtWidgets.QLabel("Candidate"))
        holdout_cand_row = QtWidgets.QHBoxLayout()
        holdout_cand_row.addWidget(self.holdout_candidate_source_combo, 1)
        holdout_cand_row.addWidget(self.holdout_delta_combo)
        eval_layout.addLayout(holdout_cand_row)
        eval_layout.addWidget(self.holdout_candidate_label)
        holdout_frames_row = QtWidgets.QHBoxLayout()
        holdout_frames_row.addWidget(QtWidgets.QLabel("Frames"))
        holdout_frames_row.addWidget(self.holdout_frames_spin)
        holdout_frames_row.addWidget(QtWidgets.QLabel("Discovery Exclusion Margin"))
        holdout_frames_row.addWidget(self.holdout_margin_spin)
        eval_layout.addLayout(holdout_frames_row)
        eval_layout.addWidget(self.holdout_overlap_label)
        eval_layout.addWidget(QtWidgets.QLabel("Bootstrap"))
        holdout_bootstrap_row = QtWidgets.QHBoxLayout()
        holdout_bootstrap_row.addWidget(QtWidgets.QLabel("Iterations"))
        holdout_bootstrap_row.addWidget(self.holdout_bootstrap_iter_spin)
        holdout_bootstrap_row.addWidget(QtWidgets.QLabel("Block Size"))
        holdout_bootstrap_row.addWidget(self.holdout_bootstrap_block_spin)
        eval_layout.addLayout(holdout_bootstrap_row)
        holdout_run_row = QtWidgets.QHBoxLayout()
        holdout_run_row.addWidget(self.holdout_run_btn, 1)
        holdout_run_row.addWidget(self.holdout_cancel_btn)
        eval_layout.addLayout(holdout_run_row)
        eval_layout.addWidget(self.holdout_progress_label)
        eval_layout.addWidget(self.holdout_status_label)
        eval_layout.addWidget(QtWidgets.QLabel("Baseline vs Candidate"))
        eval_layout.addWidget(self.holdout_summary_label)
        eval_layout.addWidget(QtWidgets.QLabel("Spatial (Baseline vs Candidate)"))
        eval_layout.addWidget(self.holdout_spatial_label)
        eval_layout.addWidget(QtWidgets.QLabel("Depth (Baseline vs Candidate)"))
        eval_layout.addWidget(self.holdout_depth_label)
        eval_layout.addSpacing(10)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Candidate A/B Test Package</b>"))
        ab_export_row = QtWidgets.QHBoxLayout()
        ab_export_row.addWidget(self.ab_export_btn, 1)
        ab_export_row.addWidget(self.ab_export_open_btn)
        eval_layout.addLayout(ab_export_row)
        eval_layout.addWidget(self.ab_export_status_label)
        eval_layout.addSpacing(14)
        eval_layout.addWidget(QtWidgets.QLabel("<b>Diagnostic Evidence</b>"))
        eval_layout.addWidget(self.diag_info_label)
        diag_row = QtWidgets.QHBoxLayout()
        diag_row.addWidget(self.diag_run_btn, 1)
        diag_row.addWidget(self.diag_copy_btn)
        eval_layout.addLayout(diag_row)
        eval_layout.addWidget(self.diag_label)
        eval_layout.addStretch(1)
        eval_box = QtWidgets.QWidget()
        eval_box.setLayout(eval_layout)
        eval_scroll = QtWidgets.QScrollArea()
        eval_scroll.setWidget(eval_box)
        eval_scroll.setWidgetResizable(True)

        self.side_tabs = QtWidgets.QTabWidget()
        self.side_tabs.addTab(side_box, "검증")
        self.side_tabs.addTab(eval_scroll, "정량 평가")
        # A fixed width meant neither side could be dragged wider -- the
        # projection canvas and the 검증/정량 평가 panel want different room
        # depending on what's being read at the moment (a wide overlay vs a
        # wide result table), so this is a QSplitter instead, the same
        # draggable-divider pattern main_window.py already uses for its own
        # nav/content split.
        self.side_tabs.setMinimumWidth(300)

        middle = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        middle.addWidget(self.scroll)
        middle.addWidget(self.side_tabs)
        middle.setStretchFactor(0, 1)
        middle.setSizes([900, 360])
        middle.setChildrenCollapsible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(timeline)
        layout.addWidget(middle, 1)

        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.opened.connect(self._on_opened)
        self._worker.scanning.connect(self._on_scanning)
        self._worker.mf_progress.connect(self._on_mf_progress)
        self._worker.mf_ready.connect(self._on_mf_ready)
        self._worker.mf_failed.connect(self._on_mf_failed)
        self._worker.eval_ready.connect(self._on_eval_ready)
        self._worker.eval_failed.connect(self._on_eval_failed)
        self._worker.pert_progress.connect(self._on_pert_progress)
        self._worker.pert_ready.connect(self._on_pert_ready)
        self._worker.pert_failed.connect(self._on_pert_failed)
        self._worker.fine_progress.connect(self._on_fine_progress)
        self._worker.fine_ready.connect(self._on_fine_ready)
        self._worker.fine_failed.connect(self._on_fine_failed)
        self._worker.holdout_bag_progress.connect(self._on_holdout_bag_progress)
        self._worker.holdout_bag_ready.connect(self._on_holdout_bag_ready)
        self._worker.holdout_bag_failed.connect(self._on_holdout_bag_failed)
        self._worker.holdout_progress.connect(self._on_holdout_progress)
        self._worker.holdout_ready.connect(self._on_holdout_ready)
        self._worker.holdout_failed.connect(self._on_holdout_failed)
        self._worker.board_motion_progress.connect(self._on_board_motion_progress)
        self._worker.board_motion_ready.connect(self._on_board_motion_ready)
        self._worker.board_motion_failed.connect(self._on_board_motion_failed)
        self._worker.tsdiag_progress.connect(self._on_timestamp_diagnostic_progress)
        self._worker.tsdiag_ready.connect(self._on_timestamp_diagnostic_ready)
        self._worker.tsdiag_failed.connect(self._on_timestamp_diagnostic_failed)
        self._worker.board_ts_progress.connect(self._on_board_timestamp_progress)
        self._worker.board_ts_ready.connect(self._on_board_timestamp_ready)
        self._worker.board_ts_failed.connect(self._on_board_timestamp_failed)
        self.request_load.connect(self._worker.load)
        self.request_open.connect(self._worker.open_bag)
        self.request_multiframe.connect(self._worker.run_multiframe)
        self.request_eval_current.connect(self._worker.run_eval_current)
        self.request_perturbation.connect(self._worker.run_perturbation)
        self.request_fine_scan.connect(self._worker.run_fine_scan)
        self.request_holdout_open.connect(self._worker.open_holdout_bag)
        self.request_holdout.connect(self._worker.run_holdout)
        self.request_board_motion.connect(self._worker.run_board_motion)
        self.request_timestamp_diagnostic.connect(self._worker.run_timestamp_diagnostic)
        self.request_board_timestamp_diagnostic.connect(self._worker.run_board_timestamp_diagnostic)
        self._thread.start()
        self._update_pert_trial_count()
        self._on_holdout_source_changed()
        self._refresh_holdout_candidate_deltas()
        self._refresh_holdout_persistence_note()

    # ------------------------------------------------------------------ enter

    def _calibrate_step(self):
        for page in getattr(self.window(), "pages", []):
            if hasattr(page, "solution"):
                return page
        return None

    def _method(self) -> str:
        for page in getattr(self.window(), "pages", []):
            combo = getattr(page, "method_combo", None)
            if combo is not None:
                return combo.currentData()
        return DetectParams.method

    def _symmetry_export_block_reason(self) -> str | None:
        if self._loaded_from is not None:
            return None
        step = self._calibrate_step()
        if step is None or getattr(step, "solution", None) is not self._sol:
            return None
        gate = getattr(step, "symmetry_gate", {}) or {}
        decision = gate.get("decision")
        if decision is None:
            return None
        if getattr(decision, "status", "") == "CALIBRATION_ACCEPTABLE":
            return None
        return (
            "LiDAR target symmetry could not be uniquely resolved. "
            f"Status: {decision.status}; source: {decision.selection_source}; reason: {decision.reason}"
        )

    def _load_extrinsic(self):
        """Open a saved extrinsic and verify it against a recording.

        The loaded transform replaces whatever step 6 produced, and the camera and
        topics come from the file too -- an extrinsic and the intrinsics it was
        solved with are a pair, and projecting with a different focal length gives
        a wrong answer that still looks reasonable.

        Hole markers cannot be drawn: those need a scene's filter box, which the
        file does not carry. The overlay is what does the work here anyway.
        """
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "extrinsic 불러오기", str(RESULT_DIR), "extrinsic (*.yaml *.yml);;모든 파일 (*)"
        )
        if not path:
            return
        try:
            data = verify.from_yaml(Path(path).read_text())
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.critical(self, "불러오기 실패", f"{type(exc).__name__}: {exc}")
            return
        if "R" not in data:
            QtWidgets.QMessageBox.warning(
                self, "불러오기 실패",
                "이 파일에서 변환 행렬을 찾지 못했습니다.\n"
                "T_cam_lidar 또는 rotation_matrix + translation_m 이 있어야 합니다.",
            )
            return

        self._sol = Solution(
            ok=True, R=data["R"], t=data["t"],
            rmse=data.get("rmse_mm", 0.0) / 1000.0,
            scene_ids=[], n_pairs=0,
        )
        self._flipped = False
        self._loaded_from = Path(path).name
        self._loaded_meta = data

        p = self.project
        if data.get("camera"):
            for key, value in data["camera"].items():
                setattr(p.camera, key, value)
        # Follow the file's topics when the project has none, or when they differ
        # -- verifying against topics other than the ones it was solved on is a
        # different measurement.
        for key in ("lidar_topic", "camera_topic"):
            if data.get(key):
                setattr(p, key, data[key])

        # Offer the bag the file names, if the project does not already hold it.
        missing = []
        for name in data.get("bags") or []:
            if not any(Path(b).name == name for b in p.bag_paths):
                missing.append(name)
        if missing:
            QtWidgets.QMessageBox.information(
                self, "bag 을 지정하세요",
                "이 extrinsic 은 아래 bag 으로 계산됐습니다. 1단계에서 추가하면 그 구간으로 검증할 수 있습니다.\n\n"
                + "\n".join(missing),
            )

        self._fill_bags()
        self._fill_scenes()
        self._open_current_bag()
        self.window().statusBar().showMessage(
            f"extrinsic 불러옴: {Path(path).name}"
            + (f"  ({data['vehicle']})" if data.get("vehicle") else ""), 10000
        )

    def on_enter(self):
        self._sync_holdout_history_from_project()
        # A loaded extrinsic stays until step 6 is run again -- it was opened on
        # purpose, and recomputing behind the user's back would discard it.
        if self._loaded_from is not None:
            if self._times:
                self._fill_bags()
                self._fill_scenes()
            return

        step = self._calibrate_step()
        sol = getattr(step, "solution", None) if step else None
        if sol is None or not sol.ok:
            self._sol = None
            self.canvas.setText(
                "6단계 캘리브레이션을 완료하거나, 오른쪽에서 저장된 extrinsic 을 불러오세요."
            )
            self.verdict.setText("—")
            self.numbers.setText("—")
            self.export_text.clear()
            return

        if sol is not self._sol:
            self._sol = sol
            self._flipped = False
        self._fill_bags()
        self._fill_scenes()
        self._open_current_bag()

    def _sync_holdout_history_from_project(self):
        """Restore `self._holdout_exposure_registry` and
        `self._discovery_revision` from `self.project.validation_history`
        whenever `self.project` has changed identity since the last restore
        -- a no-op otherwise, since the two are kept in sync on every write
        (see `_persist_validation_history`), so re-parsing them again here
        would be pure waste, not a correctness issue.

        This is what makes a Hold-out already peeked at in an *earlier app
        session* still register as exposed now: opening a project (see
        main_window.py's `_open`, which rebinds `self.project` on every page
        rather than recreating them) changes `self.project`'s identity, so
        the next time Step 7 is entered its saved history -- and the
        persistent Discovery revision counter -- are read back in, picking
        up exactly where the previous session left off (e.g. revision 7 at
        save time -> restart -> still 7 on restore -> the next successful
        Discovery run makes it 8, never resetting to 0). A project that
        predates this feature, or predates the revision counter
        (`validation_history` schema v1/v2), restores to an empty registry
        and `discovery_revision=0` -- ordinary, not an error (see
        `ho.VALIDATION_HISTORY_SCHEMA_VERSION`'s migration-policy comment).
        """
        if self.project is self._holdout_history_project:
            return
        registry, discovery_revision, warnings = ho.deserialize_exposure_registry(
            getattr(self.project, "validation_history", None)
        )
        self._holdout_exposure_registry = registry
        self._discovery_revision = discovery_revision
        self._holdout_history_project = self.project
        for w in warnings:
            self.window().statusBar().showMessage(w, 10000)
        self._refresh_holdout_persistence_note()

    def _refresh_holdout_persistence_note(self):
        """`self.project_path` is kept in sync by MainWindow (mirroring how
        it already rebinds `self.project` -- see `_open`/`_save`), so this is
        just reading it, not guessing at anything this page cannot know.
        Shown once a project exists on disk to *not* say anything (the
        common case), rather than clutter the section with a permanent
        "history is being saved" confirmation."""
        if self.project_path is None:
            self.holdout_persistence_note_label.setText(
                "Validation history is session-only until this project is saved (\"저장…\")."
            )
        else:
            self.holdout_persistence_note_label.setText("")

    def _current_bag(self) -> str:
        i = self.bag_combo.currentIndex()
        paths = self.project.bag_paths
        return paths[i] if 0 <= i < len(paths) else (paths[0] if paths else "")

    def _fill_bags(self):
        """Every bag in the project, whether or not it holds a captured scene.

        A recording added only to check the result has no scenes at all, and that
        is the most useful kind to look at -- it had no say in the answer.
        """
        from pathlib import Path

        names = [Path(b).name for b in self.project.bag_paths]
        if [self.bag_combo.itemText(i) for i in range(self.bag_combo.count())] != names:
            keep = self.bag_combo.currentIndex()
            self.bag_combo.blockSignals(True)
            self.bag_combo.clear()
            self.bag_combo.addItems(names)
            self.bag_combo.setCurrentIndex(max(0, min(keep, len(names) - 1)))
            self.bag_combo.blockSignals(False)
        self._fill_holdout_bags()

    def _fill_holdout_bags(self):
        """Hold-out's own bag picker: every bag already in the project (the
        recommended source -- see the module docstring's "Calibration bag /
        Diagnostic bag / Hold-out bag" note), plus a "찾아보기…" entry for a
        `.bag` not yet added to this project. Kept separate from
        `self.bag_combo` so switching the main view's bag never changes what
        hold-out is configured to read.
        """
        names = [Path(b).name for b in self.project.bag_paths] + ["찾아보기…"]
        if [self.holdout_bag_combo.itemText(i) for i in range(self.holdout_bag_combo.count())] == names:
            return
        keep_path = self.holdout_bag_combo.currentData()
        self.holdout_bag_combo.blockSignals(True)
        self.holdout_bag_combo.clear()
        for b in self.project.bag_paths:
            self.holdout_bag_combo.addItem(Path(b).name, b)
        self.holdout_bag_combo.addItem("찾아보기…", None)
        idx = self.holdout_bag_combo.findData(keep_path) if keep_path else -1
        self.holdout_bag_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.holdout_bag_combo.blockSignals(False)
        # setCurrentIndex above never emits (it was blocked, or the index was
        # already 0) -- open explicitly so a fresh project's first bag is
        # ready without the user having to reselect it. Never for "찾아보기…"
        # (data=None): that would pop a file dialog on page load.
        current = self.holdout_bag_combo.currentData()
        if current is not None and current != self._holdout_bag_path:
            self._open_holdout_bag(current)

    def _fill_scenes(self):
        """Jump targets: the scenes captured from the bag currently shown."""
        bag = self._current_bag()
        current = self.scene_combo.currentData()
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItem("— 직접 이동 —", None)
        # Relative to the start of the recording. Raw t_ns is a Unix timestamp,
        # which prints as 1785224859.29 and matches nothing else on screen.
        base = self._times[0] if self._times else 0
        for sc in self.project.scenes:
            if self.project.bag_for(sc) == bag:
                offset = (sc.t_ns - base) / 1e9 if base else 0.0
                self.scene_combo.addItem(f"{sc.id}  ({offset:.2f}s)", sc.id)
        idx = self.scene_combo.findData(current)
        self.scene_combo.setCurrentIndex(max(idx, 0))
        self.scene_combo.blockSignals(False)

    def _switch_bag(self):
        self._fill_scenes()
        self._open_current_bag()

    def _open_current_bag(self):
        p = self.project
        bag = self._current_bag()
        if not bag or not p.lidar_topic or not p.camera_topic:
            self.canvas.setText("1단계에서 bag 과 토픽을 지정하세요.")
            return
        key = (bag, p.lidar_topic)
        if key == self._loaded_for and self._times:
            self._request()
            return
        self._loaded_for = key
        self._times = []
        self.slider.setEnabled(False)
        self._gen += 1
        self.canvas.setText("bag 을 읽는 중…")
        self.request_open.emit(self._gen, bag, p.lidar_topic)

    def _on_opened(self, gen: int, times):
        if gen != self._gen:
            return
        self._times = list(times)
        self._update_holdout_range_bounds()
        if not self._times:
            self.canvas.setText("이 bag 에 LiDAR 메시지가 없습니다.")
            return
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self._times) - 1)
        # Start where a scene was captured when there is one, since that is the
        # moment with hole centres to compare against.
        first = next(
            (i for sc in self.project.scenes if self.project.bag_for(sc) == self._current_bag()
             for i in [min(range(len(self._times)), key=lambda k: abs(self._times[k] - sc.t_ns))]),
            len(self._times) // 2,
        )
        self.slider.setValue(first)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        self._fill_scenes()  # labels need self._times to show relative seconds
        self._request()

    def _on_slider(self):
        self._update_time_label()
        self._debounce.start()

    def _update_time_label(self):
        if not self._times:
            self.time_label.setText("—")
            return
        i = self.slider.value()
        t = (self._times[i] - self._times[0]) / 1e9
        tag = f"  {self._scene.id}" if self._scene is not None else ""
        self.time_label.setText(f"{t:7.2f} s  [{i + 1}/{len(self._times)}]{tag}")

    def _scene_at(self, t_ns: int):
        """The captured scene at this moment, if the slider is sitting on one.

        Detection needs a scene's filter box, so the hole markers and the numbers
        only appear here. Within half a sweep counts as the same moment.
        """
        bag = self._current_bag()
        tol = 60_000_000  # 60 ms, comfortably inside one sweep at 19 Hz
        for sc in self.project.scenes:
            if self.project.bag_for(sc) == bag and abs(sc.t_ns - t_ns) <= tol:
                return sc
        return None

    def _request(self):
        p = self.project
        if not self._times:
            return
        t_ns = self._times[self.slider.value()]
        self._scene = self._scene_at(t_ns)
        frames = self._scene.frames if self._scene is not None else 1
        self._gen += 1
        self._update_time_label()
        self.canvas.setText("불러오는 중…")
        # One-shot: only the request a Worst-Frame jump itself triggers should
        # use it, so it is cleared immediately -- any further scrubbing (even
        # before this request's result comes back) must fall through to
        # ordinary first_after pairing again.
        pinned = self._pinned_camera_t_ns
        self._pinned_camera_t_ns = None
        self.request_load.emit(
            self._gen, self._current_bag(), p.lidar_topic, p.camera_topic,
            t_ns, frames, self._scene, p.camera, p.target, self._method(), pinned,
        )

    def _goto_scene(self):
        sid = self.scene_combo.currentData()
        if sid is None or not self._times:
            return
        scene = next((sc for sc in self.project.scenes if sc.id == sid), None)
        if scene is None:
            return
        idx = min(range(len(self._times)), key=lambda i: abs(self._times[i] - scene.t_ns))
        if idx == self.slider.value():
            self._request()
        else:
            self.slider.setValue(idx)

    def _on_scanning(self, gen: int, done: int, total: int):
        if gen != self._gen:
            return
        pct = f" ({done / total * 100:.0f}%)" if total else ""
        self.canvas.setText(f"bag 인덱스 읽는 중…  {done:,} / {total:,}{pct}")

    def _on_failed(self, gen: int, msg: str):
        if gen == self._gen:
            self.canvas.setText(msg)

    def _on_ready(self, gen, image, cloud, intensity, lidar_c, cam_c):
        if gen != self._gen:
            return
        self._image, self._cloud, self._intensity = image, cloud, intensity
        self._lidar_c, self._cam_c = lidar_c, cam_c
        # Point indices mean nothing once a different frame is projected.
        self._picked = None
        self.pick_label.setText(
            "<span style='color:palette(mid)'>점을 클릭하면 원래 3D 좌표를 봅니다.</span>"
        )
        self._update_time_label()
        self._redraw()
        self._update_numbers()
        self._update_export()

    # ------------------------------------------------------------------- draw

    def _effective(self):
        """The solution as currently shown -- flipped or not.

        Which of the two half-turn choices `solve()` settled on is not recorded,
        and assuming one produces a "flip" that changes nothing half the time.
        So both are built here and the one furthest from the current rotation is
        taken as the alternative. Distance between rotation matrices is a fine
        test for this: the two differ by 180 degrees, so there is no near miss.
        """
        if self._sol is None:
            return None
        if not self._flipped or self._lidar_c is None or self._cam_c is None:
            return self._sol

        L, C = sort_centers(self._lidar_c), sort_centers(self._cam_c)
        options = [solve_rigid(np.roll(L, sh, axis=0), C) for sh in (0, 2)]
        far = max(options, key=lambda Rt: np.linalg.norm(Rt[0] - self._sol.R))
        R, t = far
        return Solution(
            ok=True, R=R, t=t, rmse=self._sol.rmse,
            scene_ids=list(self._sol.scene_ids), n_pairs=self._sol.n_pairs,
        )

    def _redraw(self):
        sol = self._effective()
        if self._image is None or self._cloud is None or sol is None:
            return
        h, w = self._image.shape[:2]
        pr = verify.project_cloud(
            self._cloud, sol, self.project.camera, w, h,
            intensity=self._intensity,
            min_range=self.near_spin.value(), max_range=self.far_spin.value(),
        )
        self._last_projection = pr
        img = verify.make_overlay(
            self._image, pr,
            colour_by=self.colour_combo.currentData(),
            point_size=self.size_spin.value(),
            dim=self.dim_slider.value() / 100.0,
        )
        if self._picked is not None and self._picked < pr.n_visible:
            import cv2 as _cv2

            pu, pv = pr.uv[self._picked]
            _cv2.circle(img, (int(pu), int(pv)), 14, (255, 255, 255), 2)
            _cv2.circle(img, (int(pu), int(pv)), 15, (0, 0, 0), 1)
        if self.markers_check.isChecked() and self._lidar_c is not None and self._cam_c is not None:
            img = verify.draw_centres(img, sol, self.project.camera, self._lidar_c, self._cam_c)

        rgb = np.ascontiguousarray(img[:, :, ::-1])
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self._apply_zoom()
        self._visible = pr.n_visible

    def _img_origin(self) -> QtCore.QPointF:
        """Where the image's top-left sits in the label, which centres its pixmap."""
        pm = self.canvas.pixmap()
        if pm is None or pm.isNull():
            return QtCore.QPointF(0.0, 0.0)
        return QtCore.QPointF(
            max((self.canvas.width() - pm.width()) / 2.0, 0.0),
            max((self.canvas.height() - pm.height()) / 2.0, 0.0),
        )

    def _pick_point(self, pos: QtCore.QPointF):
        """Report the 3D point behind the dot nearest the click."""
        def nothing(msg: str):
            # Saying why beats leaving the previous point's numbers on screen,
            # which reads as if the click had found that point again.
            self._picked = None
            self.pick_label.setText(f"<span style='color:palette(mid)'>{msg}</span>")

        pr = self._last_projection
        if pr is None or pr.n_visible == 0 or pr.axis is None:
            nothing("아직 그려진 점이 없습니다.")
            return
        scale = self._effective_scale()
        if scale <= 0:
            nothing("화면 배율을 읽을 수 없습니다.")
            return
        origin = self._img_origin()
        x = (pos.x() - origin.x()) / scale
        y = (pos.y() - origin.y()) / scale

        d = np.hypot(pr.uv[:, 0] - x, pr.uv[:, 1] - y)
        i = int(np.argmin(d))
        # Generous in image pixels but tighter the further in you are zoomed, so
        # the pick follows what is actually under the cursor on screen.
        if d[i] > max(12.0 / max(scale, 1e-6), 3.0):
            nothing(f"그 자리에 점이 없습니다 (가장 가까운 점이 {d[i]:.0f} px 떨어져 있습니다).")
            self._redraw()
            return

        self._picked = i
        sol = self._effective()
        p3 = pr.axis[i]
        cam_pt = sol.transform(p3.reshape(1, 3))[0]
        rng = float(np.linalg.norm(cam_pt))
        off = float(np.hypot(cam_pt[0], cam_pt[1]) / max(cam_pt[2], 1e-9))
        limit = verify.radial_limit(self.project.camera)
        inten = "—" if pr.intensity is None else f"{pr.intensity[i]:.1f}"
        rows = [
            ("화면", f"({pr.uv[i, 0]:.1f}, {pr.uv[i, 1]:.1f}) px"),
            ("LiDAR 좌표", f"앞 {p3[0]:+.3f}  좌 {p3[1]:+.3f}  위 {p3[2]:+.3f} m"),
            ("카메라 거리", f"{rng:.2f} m"),
            ("입사각", f"{np.degrees(np.arctan(off)):.2f}°"
                      f"  <span style='color:palette(mid)'>(한계 {np.degrees(np.arctan(limit)):.1f}°)</span>"),
            ("intensity", inten),
        ]
        self.pick_label.setText(
            "<table cellspacing='3'>"
            + "".join(
                f"<tr><td style='color:palette(mid)'>{k}</td><td>{v}</td></tr>"
                for k, v in rows
            )
            + "</table>"
        )
        self._redraw()

    def _apply_zoom(self):
        if not hasattr(self, "_pix"):
            return
        if self._fit:
            area = self.scroll.viewport().size()
            pm = self._pix.scaled(area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        else:
            pm = self._pix.scaled(
                self._pix.size() * self._zoom, QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        self.canvas.setPixmap(pm)
        self.canvas.setMinimumSize(pm.size() if not self._fit else QtCore.QSize(0, 0))

    def _sync_range(self, spin, value: float):
        """Slider moved: push the value into its spin box, which redraws."""
        if abs(spin.value() - value) < 1e-9:
            return
        spin.setValue(value)  # triggers _on_range

    def _push_range_sliders(self):
        """Mirror the spin boxes back onto the sliders without a signal loop."""
        for spin, slider in ((self.near_spin, self.near_slider),
                             (self.far_spin, self.far_slider)):
            want = int(round(spin.value() * 100))
            if slider.value() != want:
                slider.blockSignals(True)
                slider.setValue(want)
                slider.blockSignals(False)

    def _on_range(self):
        """Keep the near cut below the far one before redrawing.

        Crossed values would silently show nothing, which looks like a detection
        failure rather than a filter set inside out.
        """
        near, far = self.near_spin.value(), self.far_spin.value()
        if far > 0 and near > 0 and near >= far:
            sender = self.sender()
            blocked = self.far_spin if sender is self.near_spin else self.near_spin
            blocked.blockSignals(True)
            if sender is self.near_spin:
                self.far_spin.setValue(near + self.far_spin.singleStep())
            else:
                self.near_spin.setValue(max(0.0, far - self.near_spin.singleStep()))
            blocked.blockSignals(False)
        self._push_range_sliders()
        self._redraw()

    def _on_pan(self, delta: QtCore.QPoint):
        """Drag the view: move the scroll offset opposite to the pointer."""
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        h.setValue(h.value() - delta.x())
        v.setValue(v.value() - delta.y())

    def _step_frames(self, notches: int):
        """Shift+wheel over the image: walk the timeline without leaving it."""
        if not self._times:
            return
        self.slider.setValue(self.slider.value() + notches * self.slider.singleStep())

    def _step_size(self, notches: int):
        """Ctrl+wheel over the image: thicken the cloud until the shape reads."""
        self.size_spin.setValue(self.size_spin.value() + notches * self.size_spin.singleStep())

    def _on_wheel(self, factor: float, pos: QtCore.QPointF):
        """Zoom about the pointer, keeping what is under it in place.

        The scroll offset has to move with the scale: a point at `pos` sits at
        (offset + pos) in the scaled image, and after scaling by `factor` that
        lands at factor * (offset + pos). Subtracting `pos` again gives the offset
        that puts it back under the cursor.
        """
        h, v = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        before = self._effective_scale()
        # Wheeling out of "fit" has to carry on from what is on screen. `_zoom` is
        # still 1.0 while fitted, so using it as-is would jump to full size on the
        # first notch.
        if self._fit and before > 0:
            self._zoom = before
        self._fit = False
        self._zoom = max(0.1, min(self._zoom * factor, 12.0))
        self._apply_zoom()
        after = self._effective_scale()
        if before > 0:
            ratio = after / before
            h.setValue(int(ratio * (h.value() + pos.x()) - pos.x()))
            v.setValue(int(ratio * (v.value() + pos.y()) - pos.y()))

    def _effective_scale(self) -> float:
        """Pixels drawn per source pixel, whichever sizing mode is active."""
        if not hasattr(self, "_pix") or self._pix.isNull():
            return 0.0
        shown = self.canvas.pixmap()
        if shown is None or shown.isNull():
            return 0.0
        return shown.width() / self._pix.width()

    def _set_zoom(self, z: float):
        self._fit = False
        self._zoom = max(0.1, min(z, 12.0))
        self._apply_zoom()

    def _fit_view(self):
        self._fit = True
        self._apply_zoom()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit:
            self._apply_zoom()

    # ---------------------------------------------------------------- numbers

    def _flip(self):
        if self._lidar_c is None or self._cam_c is None:
            self.window().statusBar().showMessage(
                "뒤집기는 구멍 중심이 있어야 계산됩니다 — 캡처한 scene 시점으로 이동하세요.", 6000
            )
            return
        self._flipped = not self._flipped
        self._redraw()
        self._update_numbers()
        self._update_export()

    def _update_numbers(self):
        sol = self._effective()
        if sol is None:
            return
        if self._lidar_c is None or self._cam_c is None:
            if self._loaded_from is not None:
                meta = self._loaded_meta
                bits = [f"<b>{self._loaded_from}</b> 의 extrinsic 으로 투영하고 있습니다."]
                detail = " · ".join(
                    x for x in (
                        meta.get("vehicle"),
                        f"계산 {meta['calibrated_on']}" if meta.get("calibrated_on") else "",
                        f"검출 {meta['detector']}" if meta.get("detector") else "",
                        f"RMSE {meta['rmse_mm']:.2f}mm" if meta.get("rmse_mm") else "",
                    ) if x
                )
                if detail:
                    bits.append(f"<span style='color:palette(mid)'>{detail}</span>")
                bits.append(
                    "구멍 중심 비교는 이 scene 의 박스가 필요해 불가능합니다. "
                    "<b>클라우드가 이미지 위에 맞게 얹히는지</b>로 판단하세요 — "
                    "벽·바닥·기둥은 extrinsic 을 구할 때 쓰이지 않았으니 독립된 근거입니다."
                )
                self.verdict.setText("<br>".join(bits))
                self.numbers.setText(
                    "<table cellspacing='3'>"
                    f"<tr><td style='color:palette(mid)'>출처</td><td><b>{self._loaded_from}</b></td></tr>"
                    f"<tr><td style='color:palette(mid)'>화면에 그린 점</td>"
                    f"<td><b>{getattr(self, '_visible', 0):,}</b></td></tr>"
                    "</table>"
                )
                return
            if self._scene is None:
                self.verdict.setText(
                    "캡처한 scene 이 아닌 시점입니다. <b>구멍 중심 비교는 scene 에서만</b> "
                    "가능하니, 여기서는 클라우드가 이미지 위에 제대로 얹히는지 보세요.<br>"
                    "보드 구멍뿐 아니라 벽·바닥·기둥의 선이 맞는지가 오히려 더 확실한 근거입니다."
                )
            else:
                self.verdict.setText(
                    f"{self._scene.id} 에서 검출에 실패했습니다. 5단계에서 이 scene 의 "
                    "박스와 검출 방식을 확인하세요."
                )
            self.numbers.setText(
                "<span style='color:palette(mid)'>이 시점에는 비교할 구멍 중심이 없습니다.</span>"
            )
            return

        ag = verify.centre_agreement(self._lidar_c, self._cam_c, sol)
        ht = verify.half_turn_check(self._lidar_c, self._cam_c, sol)
        n_scenes = len(self._sol.scene_ids)

        if ht["separable"]:
            verdict = (
                "<span style='color:#2e7d32'><b>잔차로 구분됩니다.</b></span> "
                f"현재 {ht['current']:.2f} mm, 뒤집으면 {ht['half_turn']:.2f} mm."
            )
        elif n_scenes >= 2:
            verdict = (
                f"이 scene 만으로는 구분되지 않지만 scene 이 {n_scenes}개라 "
                "6단계에서 함께 결정했습니다. 그래도 아래 겹침을 한 번 확인하세요."
            )
        else:
            verdict = (
                "<span style='color:#b26a00'><b>계산으로는 구분할 수 없습니다.</b></span> "
                f"현재 {ht['current']:.2f} mm, 뒤집어도 {ht['half_turn']:.2f} mm 로 같습니다.<br>"
                "화면에서 <b>십자와 원이 겹치는지</b> 보고 판단하세요. "
                "대각선으로 어긋나 보이면 뒤집기를 누르세요."
            )
        if self._flipped:
            verdict += "<br><span style='color:#b26a00'>현재 <b>뒤집은</b> 해를 보고 있습니다.</span>"
        self.verdict.setText(verdict)

        rows = [
            ("원 중심 일치", f"평균 {ag['mean_mm']:.1f} mm · 최대 {ag['max_mm']:.1f} mm"),
            ("구멍별", " / ".join(f"{v:.1f}" for v in ag["per_hole_mm"]) + " mm"),
            ("일대일 대응", "예" if ag["bijective"] else "아니오 — 두 중심이 겹칩니다"),
            ("화면에 그린 점", f"{getattr(self, '_visible', 0):,}"),
            ("scene 수", f"{n_scenes}"),
            ("6단계 RMSE", f"{self._sol.rmse * 1000:.2f} mm"),
        ]
        self.numbers.setText(
            "<table cellspacing='3'>"
            + "".join(
                f"<tr><td style='color:palette(mid)'>{k}</td><td><b>{v}</b></td></tr>"
                for k, v in rows
            )
            + "</table>"
        )

    # ------------------------------------------------------ target-independent
    #
    # Step 6's numbers above answer "how well did this fit the four scenes it
    # was solved from". Everything below answers a different question -- "does
    # the resulting T_cam_lidar still hold on an ordinary frame it never saw"
    # -- using LiDAR depth-edges vs. camera Canny edges instead of the target's
    # holes, so it works on any frame, captured scene or not.

    @staticmethod
    def _rows_table(rows) -> str:
        return (
            "<table cellspacing='3'>"
            + "".join(
                f"<tr><td style='color:palette(mid)'>{k}</td><td><b>{v}</b></td></tr>"
                for k, v in rows
            )
            + "</table>"
        )

    @staticmethod
    def _px_or_dash(v: float, signed: bool = False) -> str:
        if not np.isfinite(v):
            return "—"
        return f"{v:+.2f} px" if signed else f"{v:.2f} px"

    def _run_board_motion(self):
        if not self._current_bag() or not self.project.camera_topic:
            self.window().statusBar().showMessage("Load a bag and camera topic first.", 4000)
            return
        if not self.project.camera.is_set:
            self.window().statusBar().showMessage("Camera intrinsics are required for board pose detection.", 5000)
            return
        self._board_motion_gen += 1
        self._board_motion_result = None
        self._board_motion_stable_lidar_times = []
        self.board_motion_run_btn.setEnabled(False)
        self.board_motion_cancel_btn.setVisible(True)
        self.board_motion_export_btn.setEnabled(False)
        self.board_motion_stable_only_check.setEnabled(False)
        self.board_motion_stable_only_check.setChecked(False)
        self.board_motion_progress_label.setText("Board motion analysis: 0 / ?")
        self.board_motion_summary_label.setText("—")
        self.request_board_motion.emit(
            self._board_motion_gen,
            self._current_bag(),
            self.project.camera_topic,
            self.project.camera,
            self.project.target,
            self.board_motion_samples_spin.value(),
            self.board_motion_trans_spin.value(),
            self.board_motion_rot_spin.value(),
        )

    def _cancel_board_motion(self):
        self._worker.cancel_board_motion()
        self.board_motion_progress_label.setText(self.board_motion_progress_label.text() + "  (cancel requested)")

    def _on_board_motion_progress(self, gen: int, done: int, total: int):
        if gen != self._board_motion_gen:
            return
        self.board_motion_progress_label.setText(f"Board motion analysis: {done} / {total}")

    def _on_board_motion_ready(self, gen: int, result: "bm.BoardMotionResult"):
        if gen != self._board_motion_gen:
            return
        self.board_motion_run_btn.setEnabled(True)
        self.board_motion_cancel_btn.setVisible(False)
        self._board_motion_result = result
        self._board_motion_stable_lidar_times = self._stable_camera_to_lidar_times(result.stable_timestamps_ns)
        self.board_motion_export_btn.setEnabled(bool(result.frames))
        self.board_motion_stable_only_check.setEnabled(bool(self._board_motion_stable_lidar_times))
        self.board_motion_summary_label.setText(self._board_motion_summary_html(result))
        self.board_motion_progress_label.setText(f"Board motion complete: {result.total_detected} detected frames")

    def _on_board_motion_failed(self, gen: int, msg: str):
        if gen != self._board_motion_gen:
            return
        self.board_motion_run_btn.setEnabled(True)
        self.board_motion_cancel_btn.setVisible(False)
        self.board_motion_progress_label.setText("")
        self.board_motion_summary_label.setText(f"<span style='color:#d9534f'>{html.escape(msg)}</span>")

    def _stable_camera_to_lidar_times(self, camera_timestamps_ns: list[int]) -> list[int]:
        if not camera_timestamps_ns or not self._times:
            return []
        lidar = np.asarray(self._times, dtype=np.int64)
        picked = []
        for cam_t in camera_timestamps_ns:
            pos = int(np.searchsorted(lidar, cam_t))
            candidates = []
            if pos < len(lidar):
                candidates.append(lidar[pos])
            if pos > 0:
                candidates.append(lidar[pos - 1])
            if candidates:
                picked.append(int(min(candidates, key=lambda t: abs(int(t) - cam_t))))
        return sorted(set(picked))

    def _export_board_motion_csv(self):
        result = self._board_motion_result
        if result is None or not result.frames:
            return
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.project.name if self.project.name != "untitled" else "board_motion"
        path = RESULT_DIR / f"{stem}_stable_board_frames.csv"
        try:
            bm.export_stable_frames_csv(result, path)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.window().statusBar().showMessage(f"CSV export failed: {type(exc).__name__}: {exc}", 7000)
            return
        self.window().statusBar().showMessage(f"Stable board frames exported: {path}", 7000)

    def _board_motion_summary_html(self, result: "bm.BoardMotionResult") -> str:
        if result.reason:
            return f"<span style='color:palette(mid)'>{html.escape(result.reason)}</span>"
        rows = [
            ("Detected frames", f"{result.total_detected}"),
            ("Stable / Marginal / Moving / Invalid",
             f"{result.stable_frames} / {result.marginal_frames} / {result.moving_frames} / {result.invalid_frames}"),
            ("Stable ratio", f"{result.stable_ratio * 100:.1f} %" if np.isfinite(result.stable_ratio) else "??"),
            ("Valid / Invalid pose ratio",
             f"{self._pct_text(result.valid_pose_ratio)} / {self._pct_text(result.invalid_pose_ratio)}"),
            ("Stable LiDAR frames for Multi-frame", f"{len(self._board_motion_stable_lidar_times)}"),
            (
                "PnP reprojection RMS median / P95",
                f"{self._px_or_dash(result.reproj_rms_median_px)} / {self._px_or_dash(result.reproj_rms_p95_px)}",
            ),
            (
                "PnP reprojection median / P95-residual median / max P95",
                f"{self._px_or_dash(result.reproj_median_px_median)} / "
                f"{self._px_or_dash(result.reproj_p95_px_median)} / "
                f"{self._px_or_dash(result.reproj_max_px_p95)}",
            ),
            (
                "Translation median / P95 / max",
                f"{result.translation_median_mm:.2f} / {result.translation_p95_mm:.2f} / {result.translation_max_mm:.2f} mm",
            ),
            (
                "Translation P10 / P25 / P50 / P75 / P90 / P95 / max",
                f"{result.translation_p10_mm:.2f} / {result.translation_p25_mm:.2f} / "
                f"{result.translation_median_mm:.2f} / {result.translation_p75_mm:.2f} / "
                f"{result.translation_p90_mm:.2f} / {result.translation_p95_mm:.2f} / "
                f"{result.translation_max_mm:.2f} mm",
            ),
            (
                "Rotation median / P95 / max",
                f"{result.rotation_median_deg:.3f} / {result.rotation_p95_deg:.3f} / {result.rotation_max_deg:.3f} deg",
            ),
            (
                "Rotation P10 / P25 / P50 / P75 / P90 / P95 / max",
                f"{result.rotation_p10_deg:.3f} / {result.rotation_p25_deg:.3f} / "
                f"{result.rotation_median_deg:.3f} / {result.rotation_p75_deg:.3f} / "
                f"{result.rotation_p90_deg:.3f} / {result.rotation_p95_deg:.3f} / "
                f"{result.rotation_max_deg:.3f} deg",
            ),
            (
                "Translation velocity median / P95 / max",
                f"{result.translation_velocity_median_mm_s:.1f} / {result.translation_velocity_p95_mm_s:.1f} / "
                f"{result.translation_velocity_max_mm_s:.1f} mm/s",
            ),
            (
                "Rotation velocity median / P95 / max",
                f"{result.rotation_velocity_median_deg_s:.2f} / {result.rotation_velocity_p95_deg_s:.2f} / "
                f"{result.rotation_velocity_max_deg_s:.2f} deg/s",
            ),
        ]
        if result.stable_segments:
            segs = ", ".join(
                f"#{i + 1} {s.start_frame_index}-{s.end_frame_index} ({s.duration_s:.2f}s)"
                for i, s in enumerate(result.stable_segments[:5])
            )
            rows.append(("Stable segments", html.escape(segs)))
        else:
            rows.append(("Stable segments", "none"))
        return self._rows_table(rows)

    def _run_timestamp_diagnostic(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "An extrinsic is required for timestamp diagnostic.", 5000
            )
            return
        if not self._times:
            self.window().statusBar().showMessage("Load a bag first.", 4000)
            return
        stable_only = self.tsdiag_stable_only_check.isChecked()
        if stable_only and not self._board_motion_stable_lidar_times:
            self.window().statusBar().showMessage(
                "Run Board Motion Analysis first; no stable frames are available.", 5000
            )
            return
        self._tsdiag_gen += 1
        self._tsdiag_results = []
        self.tsdiag_run_btn.setEnabled(False)
        self.tsdiag_cancel_btn.setVisible(True)
        self.tsdiag_progress_label.setText("Timestamp diagnostic: preparing...")
        self.tsdiag_summary_label.setText("—")
        self.tsdiag_detail_label.setText("—")
        p = self.project
        self.request_timestamp_diagnostic.emit(
            self._tsdiag_gen,
            self._current_bag(),
            p.lidar_topic,
            p.camera_topic,
            p.camera,
            sol,
            list(self._times),
            self.tsdiag_samples_spin.value(),
            self.eval_min_range_spin.value(),
            self.eval_max_range_spin.value(),
            stable_only,
            list(self._board_motion_stable_lidar_times),
        )

    def _cancel_timestamp_diagnostic(self):
        self._worker.cancel_timestamp_diagnostic()
        self.tsdiag_progress_label.setText(self.tsdiag_progress_label.text() + "  (cancel requested)")

    def _on_timestamp_diagnostic_progress(self, gen: int, phase: str, done: int, total: int):
        if gen != self._tsdiag_gen:
            return
        self.tsdiag_progress_label.setText(f"Timestamp diagnostic {phase}: {done} / {total}")

    def _on_timestamp_diagnostic_ready(self, gen: int, results: list):
        if gen != self._tsdiag_gen:
            return
        self.tsdiag_run_btn.setEnabled(True)
        self.tsdiag_cancel_btn.setVisible(False)
        self._tsdiag_results = results
        self.tsdiag_progress_label.setText(f"Timestamp diagnostic complete: {len(results)} modes")
        self.tsdiag_summary_label.setText(self._timestamp_diagnostic_summary_html(results))
        self.tsdiag_detail_label.setText(self._timestamp_diagnostic_detail_html(results))

    def _on_timestamp_diagnostic_failed(self, gen: int, msg: str):
        if gen != self._tsdiag_gen:
            return
        self.tsdiag_run_btn.setEnabled(True)
        self.tsdiag_cancel_btn.setVisible(False)
        self.tsdiag_progress_label.setText("")
        self.tsdiag_summary_label.setText(f"<span style='color:#d9534f'>{html.escape(msg)}</span>")

    def _run_board_timestamp_diagnostic(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "An extrinsic is required for board timestamp diagnostic.", 5000
            )
            return
        if not self._times:
            self.window().statusBar().showMessage("Load a bag first.", 4000)
            return
        self._board_ts_gen += 1
        self._board_ts_result = None
        self.board_ts_export_btn.setEnabled(False)
        self.board_ts_run_btn.setEnabled(False)
        self.board_ts_cancel_btn.setVisible(True)
        self.board_ts_progress_label.setText("Board timestamp diagnostic: preparing...")
        self.board_ts_summary_label.setText("—")
        p = self.project
        self.request_board_timestamp_diagnostic.emit(
            self._board_ts_gen,
            self._current_bag(),
            p.lidar_topic,
            p.camera_topic,
            p.camera,
            p.target,
            sol,
            list(self._times),
            list(p.scenes),
            self.board_ts_samples_spin.value(),
            list(self._board_motion_stable_lidar_times),
            self._method(),
        )

    def _cancel_board_timestamp_diagnostic(self):
        self._worker.cancel_board_timestamp_diagnostic()
        self.board_ts_progress_label.setText(
            self.board_ts_progress_label.text() + "  (cancel requested)"
        )

    def _on_board_timestamp_progress(self, gen: int, done: int, total: int):
        if gen != self._board_ts_gen:
            return
        self.board_ts_progress_label.setText(f"Board timestamp diagnostic: {done} / {total}")

    def _on_board_timestamp_ready(self, gen: int, result: "bt.BoardTimestampResult"):
        if gen != self._board_ts_gen:
            return
        self.board_ts_run_btn.setEnabled(True)
        self.board_ts_cancel_btn.setVisible(False)
        self._board_ts_result = result
        self.board_ts_export_btn.setEnabled(bool(result.frames))
        self.board_ts_progress_label.setText(
            f"Board timestamp diagnostic complete: {len(result.frames)} valid, {result.failed_frames} failed"
        )
        self.board_ts_summary_label.setText(self._board_timestamp_summary_html(result))

    def _on_board_timestamp_failed(self, gen: int, msg: str):
        if gen != self._board_ts_gen:
            return
        self.board_ts_run_btn.setEnabled(True)
        self.board_ts_cancel_btn.setVisible(False)
        self.board_ts_export_btn.setEnabled(False)
        self.board_ts_progress_label.setText("")
        self.board_ts_summary_label.setText(f"<span style='color:#d9534f'>{html.escape(msg)}</span>")

    def _export_board_timestamp_csv(self):
        result = self._board_ts_result
        if result is None or not result.frames:
            return
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.project.name if self.project.name != "untitled" else "board_timestamp"
        path = RESULT_DIR / f"{stem}_board_timestamps.csv"
        try:
            bt.export_frames_csv(result, path)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.window().statusBar().showMessage(
                f"Board timestamp CSV export failed: {type(exc).__name__}: {exc}", 7000
            )
            return
        self.window().statusBar().showMessage(f"Board timestamps exported: {path}", 7000)

    def _board_timestamp_summary_html(self, result: "bt.BoardTimestampResult") -> str:
        if not result.frames:
            detail = html.escape(result.reason or "No valid board timestamp frames.")
            return f"<span style='color:palette(mid)'>{detail}</span>"
        parts = [
            "<span style='color:palette(mid)'>camera_header - lidar_reference. "
            "Positive means the camera timestamp is later than the LiDAR board acquisition time.</span>"
        ]
        parts.append("<b>ALL frames</b>")
        parts.append(self._board_timestamp_stats_table(result.all_stats))
        parts.append("<b>Board-Time Camera Pairing Diagnostic</b>")
        parts.append(self._board_timestamp_pairing_table(result.all_stats))
        parts.append("<b>LiDAR target detection stages</b>")
        parts.append(self._board_timestamp_detection_table(result))
        stable = result.stable_stats
        parts.append("<b>Stable-only</b>")
        if stable.get("n", 0):
            parts.append(self._board_timestamp_stats_table(stable))
        else:
            parts.append(
                "<span style='color:palette(mid)'>No stable frames in this run. "
                "Run Phase 1 Board Motion Analysis first to populate stable-only status.</span>"
            )
        low_motion = self._board_timestamp_low_motion_html(result)
        if low_motion:
            parts.append("<b>Low-motion subsets</b>")
            parts.append(low_motion)
        if result.failed_frames:
            parts.append(
                f"<span style='color:palette(mid)'>Skipped frames without reusable board point timestamps: "
                f"{result.failed_frames}</span>"
            )
        return "<br>".join(parts)

    def _board_timestamp_stats_table(self, stats: dict) -> str:
        def ts(name: str):
            return stats.get(name, td.TimestampStats())

        start = ts("camera_to_sweep_start")
        mid = ts("camera_to_sweep_mid")
        end = ts("camera_to_sweep_end")
        board = ts("camera_to_board_median")
        nearest = ts("nearest_to_board")
        pos = ts("board_offset_from_start")
        span = ts("board_span")
        rows = [
            ("Frames", f"{stats.get('n', 0)}"),
            (
                "Camera Header -> Sweep Start mean / median / std / P95(abs)",
                f"{self._signed_ms_text(start.mean_ms)} / {self._signed_ms_text(start.median_ms)} / "
                f"{self._ms_text(start.std_ms)} / {self._ms_text(start.p95_abs_ms)}",
            ),
            (
                "Camera Header -> Sweep Mid mean / median / std / P95(abs)",
                f"{self._signed_ms_text(mid.mean_ms)} / {self._signed_ms_text(mid.median_ms)} / "
                f"{self._ms_text(mid.std_ms)} / {self._ms_text(mid.p95_abs_ms)}",
            ),
            (
                "Camera Header -> Sweep End mean / median / std / P95(abs)",
                f"{self._signed_ms_text(end.mean_ms)} / {self._signed_ms_text(end.median_ms)} / "
                f"{self._ms_text(end.std_ms)} / {self._ms_text(end.p95_abs_ms)}",
            ),
            (
                "Camera Header -> Board Median mean / median / std / P95(abs)",
                f"{self._signed_ms_text(board.mean_ms)} / {self._signed_ms_text(board.median_ms)} / "
                f"{self._ms_text(board.std_ms)} / {self._ms_text(board.p95_abs_ms)}",
            ),
            (
                "Header-nearest Camera -> Board Median mean / median / std / P95(abs)",
                f"{self._signed_ms_text(nearest.mean_ms)} / {self._signed_ms_text(nearest.median_ms)} / "
                f"{self._ms_text(nearest.std_ms)} / {self._ms_text(nearest.p95_abs_ms)}",
            ),
            (
                "Camera Header -> Board Median min / max",
                f"{self._signed_ms_text(board.min_ms)} / {self._signed_ms_text(board.max_ms)}",
            ),
            (
                "Board Median offset from Sweep Start mean / median / std",
                f"{self._ms_text(pos.mean_ms)} / {self._ms_text(pos.median_ms)} / {self._ms_text(pos.std_ms)}",
            ),
            (
                "Board point span mean / median / P95",
                f"{self._ms_text(span.mean_ms)} / {self._ms_text(span.median_ms)} / {self._ms_text(span.p95_abs_ms)}",
            ),
            (
                "Board point count mean / median",
                f"{stats.get('board_point_count_mean', float('nan')):.1f} / "
                f"{stats.get('board_point_count_median', float('nan')):.1f}",
            ),
        ]
        return self._rows_table(rows)

    def _board_timestamp_pairing_table(self, stats: dict) -> str:
        current = stats.get("camera_to_board_median", td.TimestampStats())
        nearest = stats.get("nearest_to_board", td.TimestampStats())
        rows = [
            ("Current Record Pair mean / median / std / P95(abs) / min / max",
             f"{self._signed_ms_text(current.mean_ms)} / {self._signed_ms_text(current.median_ms)} / "
             f"{self._ms_text(current.std_ms)} / {self._ms_text(current.p95_abs_ms)} / "
             f"{self._signed_ms_text(current.min_ms)} / {self._signed_ms_text(current.max_ms)}"),
            ("Header-Nearest Pair mean / median / std / P95(abs) / min / max",
             f"{self._signed_ms_text(nearest.mean_ms)} / {self._signed_ms_text(nearest.median_ms)} / "
             f"{self._ms_text(nearest.std_ms)} / {self._ms_text(nearest.p95_abs_ms)} / "
             f"{self._signed_ms_text(nearest.min_ms)} / {self._signed_ms_text(nearest.max_ms)}"),
            ("Same / Previous / Next camera frame selected",
             f"{self._pct_text(stats.get('same_camera_frame_ratio', float('nan')))} / "
             f"{self._pct_text(stats.get('previous_camera_frame_ratio', float('nan')))} / "
             f"{self._pct_text(stats.get('next_camera_frame_ratio', float('nan')))}"),
            ("Camera index shift histogram (-2/-1/0/+1/+2)",
             self._shift_hist_text(stats.get("shift_histogram", {}))),
        ]
        return self._rows_table(rows)

    def _board_timestamp_detection_table(self, result: "bt.BoardTimestampResult") -> str:
        n = max(1, int(result.processed_frames or (len(result.frames) + result.failed_frames)))
        rows = [
            ("Processed frames", str(result.processed_frames or (len(result.frames) + result.failed_frames))),
            ("Plane detection success", f"{result.plane_detection_success} ({result.plane_detection_success / n * 100:.1f} %)"),
            ("Enough plane points", f"{result.enough_plane_points_success} ({result.enough_plane_points_success / n * 100:.1f} %)"),
            ("Circle/hole candidate stage", f"{result.circle_candidate_success} ({result.circle_candidate_success / n * 100:.1f} %)"),
            (">=4 circle/hole detections", f"{result.four_or_more_circles_success} ({result.four_or_more_circles_success / n * 100:.1f} %)"),
            ("Geometry validation passed", f"{result.geometry_validation_success} ({result.geometry_validation_success / n * 100:.1f} %)"),
            ("Four-hole/full target detection success", f"{result.full_target_detection_success} ({result.full_target_detection_success / n * 100:.1f} %)"),
            ("Board timestamp valid", f"{result.board_timestamp_valid_count} ({result.board_timestamp_valid_count / n * 100:.1f} %)"),
            ("Board pose valid", f"{result.board_pose_valid_count} ({result.board_pose_valid_count / n * 100:.1f} %)"),
        ]
        if result.lidar_failure_reasons:
            reasons = sorted(result.lidar_failure_reasons.items(), key=lambda kv: kv[1], reverse=True)
            rows.append(("Top LiDAR failure reasons", "; ".join(f"{html.escape(k)}: {v}" for k, v in reasons[:5])))
        return self._rows_table(rows)

    def _board_timestamp_low_motion_html(self, result: "bt.BoardTimestampResult") -> str:
        motion = self._board_motion_result
        if motion is None or not motion.frames:
            return ""
        rows = []
        for label, frac in (("LOW MOTION 10%", 0.10), ("LOW MOTION 20%", 0.20), ("LOW MOTION 30%", 0.30)):
            idx = bm.low_motion_frame_indices(motion, frac)
            frames = [f for f in result.frames if f.current_camera_index in idx]
            stats = bt.summarize(frames)
            current = stats["camera_to_board_median"]
            nearest = stats["nearest_to_board"]
            rows.append((
                label,
                f"n={stats['n']} | current median/P95 {self._signed_ms_text(current.median_ms)} / "
                f"{self._ms_text(current.p95_abs_ms)} | header-nearest median/P95 "
                f"{self._signed_ms_text(nearest.median_ms)} / {self._ms_text(nearest.p95_abs_ms)}",
            ))
        return self._rows_table(rows)

    @staticmethod
    def _shift_hist_text(hist: dict) -> str:
        return " / ".join(str(int(hist.get(k, 0))) for k in (-2, -1, 0, 1, 2))

    def _timestamp_diagnostic_summary_html(self, results: list) -> str:
        if not results:
            return "<span style='color:palette(mid)'>No result.</span>"
        rows = [
            "<table cellspacing='4'>"
            "<tr><th>Mode</th><th>Matched Mean</th><th>Matched Median</th><th>Match Rate</th>"
            "<th>Within 5px</th><th>Within 10px</th><th>Median du</th><th>Median dv</th>"
            "<th>Median Sync</th><th>P95 Sync</th></tr>"
        ]
        for r in results:
            rows.append(
                "<tr>"
                f"<td>{html.escape(td.PAIRING_LABELS.get(r.mode, str(r.mode)))}</td>"
                f"<td align='right'>{self._px_or_dash(r.matched_mean_px)}</td>"
                f"<td align='right'>{self._px_or_dash(r.matched_median_px)}</td>"
                f"<td align='right'>{self._pct_text(r.match_rate_mean)}</td>"
                f"<td align='right'>{self._pct_text(r.within_5px)}</td>"
                f"<td align='right'>{self._pct_text(r.within_10px)}</td>"
                f"<td align='right'>{self._px_or_dash(r.median_delta_u_px, signed=True)}</td>"
                f"<td align='right'>{self._px_or_dash(r.median_delta_v_px, signed=True)}</td>"
                f"<td align='right'>{self._signed_ms_text(r.sync_stats.median_ms)}</td>"
                f"<td align='right'>{self._ms_text(r.sync_stats.p95_abs_ms)}</td>"
                "</tr>"
            )
        rows.append("</table>")
        rows.append(
            "<span style='color:palette(mid)'>sync_offset_ms = camera_acquisition_time - lidar_reference_time. "
            "Positive means the camera timestamp is later.</span>"
        )
        return "".join(rows)

    def _timestamp_diagnostic_detail_html(self, results: list) -> str:
        parts = []
        for r in results:
            rows = [
                ("Frames valid / total", f"{r.n_valid} / {r.n_total}"),
                ("Penalty Mean / Median / P95", f"{r.mean_px:.2f} / {r.median_px:.2f} / {r.p95_px:.2f} px"),
                ("Matched P95 / P99 / Max", f"{self._px_or_dash(r.matched_p95_px)} / {self._px_or_dash(r.matched_p99_px)} / {self._px_or_dash(r.matched_max_px)}"),
                ("Matched STD", self._px_or_dash(r.matched_std_px)),
                ("Within 2 / 3 / 5 / 10 / 15 px",
                 f"{self._pct_text(r.within_2px)} / {self._pct_text(r.within_3px)} / "
                 f"{self._pct_text(r.within_5px)} / {self._pct_text(r.within_10px)} / {self._pct_text(r.within_15px)}"),
                ("Sync mean / median / std / P95(abs)",
                 f"{self._signed_ms_text(r.sync_stats.mean_ms)} / {self._signed_ms_text(r.sync_stats.median_ms)} / "
                 f"{self._ms_text(r.sync_stats.std_ms)} / {self._ms_text(r.sync_stats.p95_abs_ms)}"),
            ]
            if r.reason:
                rows.append(("Warning", html.escape(r.reason)))
            parts.append(f"<b>{html.escape(td.PAIRING_LABELS.get(r.mode, str(r.mode)))}</b>")
            parts.append(self._rows_table(rows))
            if r.horizontal:
                parts.append("Spatial<br>" + self._rows_table([
                    (k, self._timestamp_bin_text(v)) for k, v in r.horizontal.items()
                ]))
            if r.depth:
                wanted = [k for k in ("0-10m", "10-20m", "20-30m") if k in r.depth]
                parts.append("Depth<br>" + self._rows_table([
                    (k, self._timestamp_bin_text(r.depth[k])) for k in wanted
                ]))
        return "<br>".join(parts) if parts else "—"

    def _timestamp_bin_text(self, b: dict) -> str:
        return (
            f"{self._px_or_dash(b.get('matched_mean_px', float('nan')))} matched, "
            f"{self._pct_text(b.get('match_rate', float('nan')))} match, "
            f"du {self._px_or_dash(b.get('median_delta_u_px', float('nan')), signed=True)}, "
            f"dv {self._px_or_dash(b.get('median_delta_v_px', float('nan')), signed=True)}"
        )

    @staticmethod
    def _pct_text(value: float) -> str:
        return f"{value * 100:.1f} %" if np.isfinite(value) else "??"

    @staticmethod
    def _bin_text(stats) -> str:
        """Matched-only mean is shown first and bold -- per requirement,
        spatial/depth comparison reads matched-only by default -- with the
        existing penalty-inclusive mean kept right below it (muted, not
        removed) so neither number is lost. Signed Δu/Δv, when this bin has
        any, are shown on a third line: a region/depth-bin-local systematic
        offset is exactly the kind of thing a single overall number hides.
        """
        if stats.n_points == 0:
            return "—"
        matched = VerifyStep._px_or_dash(stats.matched_mean_px)
        rate = f"{stats.match_rate * 100:.0f}%" if np.isfinite(stats.match_rate) else "—"
        lines = [
            f"<b>{matched}</b> <span style='color:palette(mid)'>matched · {rate} match</span>",
            f"<span style='color:palette(mid)'>penalty-incl {stats.mean_px:.2f} px (n={stats.n_points:,})</span>",
        ]
        if np.isfinite(stats.mean_delta_u_px) or np.isfinite(stats.mean_delta_v_px):
            du = VerifyStep._px_or_dash(stats.mean_delta_u_px, signed=True)
            dv = VerifyStep._px_or_dash(stats.mean_delta_v_px, signed=True)
            lines.append(f"<span style='color:palette(mid)'>Δu {du} / Δv {dv}</span>")
        return "<br>".join(lines)

    def _eval_current_frame(self):
        """Score the frame already on screen.

        Projects fresh with the Evaluation Range above -- not `self._last_projection`,
        which was built from the *display* range sliders. Reusing that would
        mean nudging the display's near/far controls silently moves this
        metric too, which is exactly the hidden coupling that makes two
        calibrations "checked" at different display ranges incomparable.

        Measured at up to ~1s on a 4K frame with a realistic point count, so
        this runs on the worker thread rather than blocking the GUI here.
        """
        sol = self._effective()
        if self._image is None or self._cloud is None or sol is None or not sol.ok:
            self.window().statusBar().showMessage("먼저 화면에 프레임을 불러오세요.", 4000)
            return
        self._eval_gen += 1
        # Pinned to the frame being sent, not read back from self._image later:
        # if the user scrubs to a different (possibly different-resolution)
        # frame before this result comes back, self._image will have already
        # moved on by the time _display_edge_alignment runs.
        self._eval_frame_shape = self._image.shape[:2]
        self._eval_result, self._eval_spatial = None, None
        self._eval_T = (sol.R.copy(), sol.t.copy())
        self.eval_run_btn.setEnabled(False)
        self.eval_edge_label.setText("<span style='color:palette(mid)'>평가 중…</span>")
        self.request_eval_current.emit(
            self._eval_gen, self._image, self._cloud, sol, self.project.camera,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
        )

    def _on_eval_ready(self, gen: int, result: "ea.EdgeAlignmentResult"):
        if gen != self._eval_gen:
            return
        self.eval_run_btn.setEnabled(True)
        self._display_edge_alignment(result)

    def _on_eval_failed(self, gen: int, msg: str):
        if gen != self._eval_gen:
            return
        self.eval_run_btn.setEnabled(True)
        self.eval_edge_label.setText(f"<span style='color:#d9534f'>{msg}</span>")
        self.eval_spatial_label.setText("—")
        self.eval_depth_label.setText("—")

    @staticmethod
    def _edge_alignment_summary_html(result: "ea.EdgeAlignmentResult") -> str:
        """Penalty-inclusive and Matched-only are shown as two clearly
        labelled, separate blocks -- never merged into one number -- per the
        requirement that a badly-matched scene's inflated Penalty-inclusive
        Mean must never be read as if it were the same thing as
        Matched-only Mean. See edge_alignment.EdgeAlignmentResult's own
        docstring for what each family means.
        """
        px = VerifyStep._px_or_dash
        out = (
            "<b>Penalty-inclusive</b> "
            "<span style='color:palette(mid)'>(unmatched points count at the largest search radius)</span><br>"
            + VerifyStep._rows_table([
                ("Mean", f"{result.mean_px:.2f} px"),
                ("Median", f"{result.median_px:.2f} px"),
                ("P95", f"{result.p95_px:.2f} px"),
                ("Max", f"{result.max_px:.2f} px"),
            ])
            + "<b>Matched-only</b> <span style='color:palette(mid)'>(unmatched points excluded entirely)</span><br>"
            + VerifyStep._rows_table([
                ("Mean", px(result.matched_mean_px)),
                ("Median", px(result.matched_median_px)),
                ("P95", px(result.matched_p95_px)),
                ("P99", px(result.matched_p99_px)),
                ("Max", px(result.matched_max_px)),
                ("Matched Count", f"{result.n_matched:,}"),
            ])
            + VerifyStep._rows_table([
                ("Match Rate", f"{result.match_rate * 100:.1f} %"),
                ("Edge Points", f"{result.n_edge_points:,}"),
                ("Matched / Unmatched", f"{result.n_matched:,} / {result.n_unmatched:,}"),
            ])
        )
        if result.match_rate_by_threshold_px:
            out += "<b>Match Rate by Threshold</b><br>" + VerifyStep._rows_table([
                (f"Within {th:.0f} px", f"{rate * 100:.1f} %")
                for th, rate in sorted(result.match_rate_by_threshold_px.items())
            ])
        out += (
            "<b>Signed Offset</b> <span style='color:palette(mid)'>(matched only -- sign shows bias direction, "
            "not just magnitude)</span><br>"
            + VerifyStep._rows_table([
                ("Δu Mean", px(result.mean_delta_u_px, signed=True)),
                ("Δu Median", px(result.median_delta_u_px, signed=True)),
                ("Δv Mean", px(result.mean_delta_v_px, signed=True)),
                ("Δv Median", px(result.median_delta_v_px, signed=True)),
            ])
        )
        return out

    def _display_edge_alignment(self, result: "ea.EdgeAlignmentResult"):
        self._eval_result = result
        if not result.ok:
            self.eval_edge_label.setText(f"<span style='color:palette(mid)'>{result.reason}</span>")
            self.eval_spatial_label.setText("—")
            self.eval_depth_label.setText("—")
            return

        self.eval_edge_label.setText(self._edge_alignment_summary_html(result))

        h, w = self._eval_frame_shape
        spatial = sa.analyze_spatial(result, w, h)
        self._eval_spatial = spatial
        if spatial is None:
            self.eval_spatial_label.setText("—")
            self.eval_depth_label.setText("—")
            return

        self.eval_spatial_label.setText(
            "수평<br>"
            + self._rows_table([(label, self._bin_text(spatial.horizontal[label])) for label in sa.HORIZONTAL_REGIONS])
            + "수직<br>"
            + self._rows_table([(label, self._bin_text(spatial.vertical[label])) for label in sa.VERTICAL_REGIONS])
        )
        self.eval_depth_label.setText(
            self._rows_table([(label, self._bin_text(spatial.depth_bins[label])) for label in sa.DEPTH_BIN_LABELS])
        )

    def _run_multiframe(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        if not self._times:
            self.window().statusBar().showMessage("bag 이 아직 로드되지 않았습니다.", 4000)
            return

        eval_times = list(self._times)
        if self.board_motion_stable_only_check.isChecked():
            if not self._board_motion_stable_lidar_times:
                self.window().statusBar().showMessage("Run Board Motion Analysis first; no stable frames are available.", 5000)
                return
            eval_times = list(self._board_motion_stable_lidar_times)

        n_samples = min(self.mf_count_spin.value(), len(eval_times))
        self._mf_gen += 1
        self.mf_run_btn.setEnabled(False)
        self.mf_cancel_btn.setVisible(True)
        self.mf_progress_label.setText(f"평가 중 0 / {n_samples}")
        self.mf_summary_label.setText("—")
        self.mf_worst_table.setRowCount(0)
        self._mf_result = None
        self._mf_T = (sol.R.copy(), sol.t.copy())
        self._mf_sync_limit_ms = float(self.max_sync_spin.value())

        p = self.project
        self.request_multiframe.emit(
            self._mf_gen, self._current_bag(), p.lidar_topic, p.camera_topic,
            p.camera, sol, eval_times, n_samples,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
            float(self.max_sync_spin.value()),
        )

    def _cancel_multiframe(self):
        self._worker.cancel_multiframe()
        self.mf_progress_label.setText(self.mf_progress_label.text() + "  (취소 중…)")

    def _on_mf_progress(self, gen: int, done: int, total: int):
        if gen != self._mf_gen:
            return
        self.mf_progress_label.setText(f"평가 중 {done} / {total}")

    def _on_mf_ready(self, gen: int, result: "mc.MultiFrameConsistencyResult"):
        if gen != self._mf_gen:
            return
        self.mf_run_btn.setEnabled(True)
        self.mf_cancel_btn.setVisible(False)
        self._mf_result = result
        self._display_multiframe(result)

    def _on_mf_failed(self, gen: int, msg: str):
        if gen != self._mf_gen:
            return
        self.mf_run_btn.setEnabled(True)
        self.mf_cancel_btn.setVisible(False)
        self.mf_progress_label.setText("")
        self.mf_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _display_multiframe(self, result: "mc.MultiFrameConsistencyResult"):
        self.mf_progress_label.setText(f"완료 — {result.n_total} frame")
        if result.reason:
            # Still worth showing what was collected -- e.g. a high
            # Sync Rejected count is often *why* too few frames were valid.
            self.mf_summary_label.setText(
                f"<span style='color:palette(mid)'>{result.reason}</span>"
                + self._rows_table([
                    ("Frames", f"{result.n_total}"),
                    ("Failed", f"{result.n_failed}"),
                    ("Sync Rejected", f"{result.n_sync_rejected}"),
                    ("Median Sync Offset (abs)", self._ms_text(result.sync_offset_median_ms)),
                    ("Max Sync Offset (abs)", self._ms_text(result.sync_offset_max_ms)),
                    ("Signed Sync Median", self._signed_ms_text(result.sync_offset_signed_median_ms)),
                ])
            )
            self.mf_worst_table.setRowCount(0)
            return

        px = self._px_or_dash
        self.mf_summary_label.setText(
            self._rows_table([
                ("Frames", f"{result.n_total}"),
                ("Valid", f"{result.n_valid}"),
                ("Failed", f"{result.n_failed}"),
                ("Outlier", f"{result.n_outlier}"),
                ("Sync Rejected", f"{result.n_sync_rejected}"),
            ])
            + "<b>Penalty-inclusive</b> "
            "<span style='color:palette(mid)'>(per-frame mean, unmatched points at the search-radius penalty)</span><br>"
            + self._rows_table([
                ("Mean", f"{result.mean_px:.2f} px"),
                ("Median", f"{result.median_px:.2f} px"),
                ("STD", f"{result.std_px:.2f} px"),
                ("P95", f"{result.p95_px:.2f} px"),
                ("Max", f"{result.max_px:.2f} px"),
                ("MAD", f"{result.mad_px:.2f} px"),
                ("IQR", f"{result.iqr_px:.2f} px"),
            ])
            + "<b>Matched-only</b> <span style='color:palette(mid)'>(per-frame matched mean, across valid frames)</span><br>"
            + self._rows_table([
                ("Mean", px(result.matched_mean_px)),
                ("Median", px(result.matched_median_px)),
                ("STD", px(result.matched_std_px)),
                ("P95", px(result.matched_p95_px)),
                ("Max", px(result.matched_max_px)),
            ])
            + "<b>Match Rate</b> <span style='color:palette(mid)'>(per-frame, across valid frames)</span><br>"
            + self._rows_table([
                ("Mean Match Rate", f"{result.match_rate_mean * 100:.1f} %" if np.isfinite(result.match_rate_mean) else "—"),
                ("Median Match Rate", f"{result.match_rate_median * 100:.1f} %" if np.isfinite(result.match_rate_median) else "—"),
                ("Min Match Rate", f"{result.match_rate_min * 100:.1f} %" if np.isfinite(result.match_rate_min) else "—"),
                ("P5 Match Rate", f"{result.match_rate_p5 * 100:.1f} %" if np.isfinite(result.match_rate_p5) else "—"),
            ])
            + self._rows_table([
                ("Valid Ratio", f"{result.valid_ratio * 100:.1f} %"),
                ("Failure Ratio", f"{result.failure_ratio * 100:.1f} %"),
                ("Outlier Ratio", f"{result.outlier_ratio * 100:.1f} %"),
                ("Sync Rejected Ratio", f"{result.sync_rejected_ratio * 100:.1f} %"),
                ("Median Sync Offset (abs)", self._ms_text(result.sync_offset_median_ms)),
                ("P95 Sync Offset (abs)", self._ms_text(result.sync_offset_p95_ms)),
                ("Max Sync Offset (abs)", self._ms_text(result.sync_offset_max_ms)),
                ("Signed Sync Median", self._signed_ms_text(result.sync_offset_signed_median_ms)),
            ])
        )

        self.mf_worst_table.setRowCount(len(result.worst_frames))
        for r, f in enumerate(result.worst_frames):
            self.mf_worst_table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{f.offset_s:.2f}s"))
            self.mf_worst_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{f.mean_px:.2f} px"))
            self.mf_worst_table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{f.match_rate * 100:.0f}%"))
            self.mf_worst_table.setItem(r, 3, QtWidgets.QTableWidgetItem(self._ms_text(f.abs_sync_offset_ms)))
            btn = QtWidgets.QPushButton("이동")
            # Pinning the exact camera timestamp this frame was evaluated with
            # is what guarantees the jump shows the same Camera/LiDAR pair the
            # Multi-frame run scored -- see pick_camera_for_frame's docstring.
            btn.clicked.connect(
                lambda _checked=False, idx=f.timeline_index, cam_t=f.camera_timestamp_ns, lid_t=f.lidar_timestamp_ns:
                    self._goto_timeline_index(idx, cam_t, lid_t)
            )
            self.mf_worst_table.setCellWidget(r, 4, btn)
        self.mf_worst_table.resizeColumnsToContents()

    @staticmethod
    def _ms_text(value_ms: float) -> str:
        return f"{value_ms:.1f} ms" if np.isfinite(value_ms) else "—"

    @staticmethod
    def _signed_ms_text(value_ms: float) -> str:
        return f"{value_ms:+.1f} ms" if np.isfinite(value_ms) else "—"

    def _goto_timeline_index(self, idx: int, camera_t_ns: "int | None" = None,
                             lidar_t_ns: "int | None" = None):
        """Where a Worst-Frame row's "이동" button lands: reuse the same
        slider -> debounce -> _request path a manual drag already takes.

        `camera_t_ns`, when given, pins `_request()`'s next load to the exact
        camera message Multi-frame evaluation scored this frame with (see
        `pick_camera_for_frame`) -- without it, `_request()` would pair this
        LiDAR moment with `first_after` instead, which is not necessarily the
        same image Multi-frame used (it pairs by nearest timestamp).
        """
        if not self._times:
            return
        if lidar_t_ns is not None:
            idx = min(range(len(self._times)), key=lambda i: abs(self._times[i] - lidar_t_ns))
        if not (0 <= idx < len(self._times)):
            return
        self.side_tabs.setCurrentIndex(0)
        self._pinned_camera_t_ns = camera_t_ns
        if idx == self.slider.value():
            self._request()
        else:
            self.slider.setValue(idx)

    # ----------------------------------------------------- perturbation sensitivity
    #
    # Diagnostic only: re-evaluates the existing Edge Alignment metric at
    # small, temporary copies of the current T. Never writes back to
    # self._sol, the project, or any file -- see
    # gui.core.evaluation.perturbation's module docstring for the guarantee
    # this whole section rests on.

    def _run_perturbation(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return

        mode = "current_frame" if self.pert_mode_current_radio.isChecked() else "multi_frame"
        if mode == "current_frame":
            if self._image is None or self._cloud is None:
                self.window().statusBar().showMessage("먼저 화면에 프레임을 불러오세요.", 4000)
                return
        elif not self._times:
            self.window().statusBar().showMessage("bag 이 아직 로드되지 않았습니다.", 4000)
            return

        self._pert_gen += 1
        self.pert_run_btn.setEnabled(False)
        self.pert_cancel_btn.setVisible(True)
        self.pert_progress_label.setText("분석 준비 중…")
        self.pert_table_label.setText("—")
        self.pert_frame_table_label.setText("—")
        self.pert_summary_label.setText("—")
        self.pert_spatial_label.setText("—")
        self._pert_result = None
        self._pert_T = (sol.R.copy(), sol.t.copy())
        self._pert_sync_limit_ms = float(self.max_sync_spin.value())
        self._pert_search_source = "quick" if self.pert_search_quick_radio.isChecked() else "full"
        self._pert_discovery_bag = self._current_bag()
        self._pert_discovery_timestamps = self._discovery_frame_timestamps(mode)

        rotation_deltas, translation_deltas = self._pert_search_deltas()
        p = self.project
        self.request_perturbation.emit(
            self._pert_gen, mode, self._current_bag(), p.lidar_topic, p.camera_topic,
            p.camera, sol.R, sol.t, list(self._times), self.pert_frames_spin.value(),
            rotation_deltas, translation_deltas,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
            float(self.max_sync_spin.value()), self._image, self._cloud,
        )

    def _note_discovery_success(self):
        """A Perturbation Grid or Fine Scan run has just *successfully*
        produced a result -- not merely been launched (see
        `_on_pert_ready`/`_on_fine_ready`, the only two call sites, each
        already checked their result is neither cancelled nor empty before
        calling this). Bumps the project-persistent `discovery_revision`
        counter (see gui.core.evaluation.holdout.CandidateProvenance's
        docstring) that any existing hold-out result's locked-in revision is
        compared against (see `_holdout_status`), writes the new value
        through to `self.project.validation_history` immediately
        (`_persist_validation_history`), and refreshes that result's own
        display immediately so a validation status shown on screen never
        silently goes stale after the moment it changes."""
        self._discovery_revision += 1
        self._persist_validation_history()
        if self._holdout_result is not None:
            self._display_holdout(self._holdout_result)

    def _persist_validation_history(self):
        """Write-through + autosave signal, shared by both kinds of
        validation provenance this page tracks: a Hold-out reveal
        (`_on_holdout_ready`) and a successful Discovery run
        (`_note_discovery_success`). `self.project.validation_history` (in
        memory) is updated immediately either way, and
        `validation_history_changed` asks MainWindow to patch just that key
        to disk (see `Project.patch_validation_history`) -- never a full
        project save, so an unrelated unsaved edit elsewhere in the project
        is never silently promoted to "saved" by this."""
        self.project.validation_history = ho.serialize_exposure_registry(
            self._holdout_exposure_registry, self._discovery_revision,
        )
        self.validation_history_changed.emit()

    def _discovery_frame_timestamps(self, mode: str) -> frozenset:
        """Which lidar timestamps a Perturbation/Fine Scan run is about to
        score, captured before the request is even sent -- `sample_frame_indices`
        is a deterministic function of `len(self._times)` alone, so this needs
        no bag I/O. Used only later, to let a Hold-out run check contamination
        (see gui.core.evaluation.holdout.check_overlap); never influences what
        the Discovery run itself scores.
        """
        if not self._times:
            return frozenset()
        if mode == "current_frame":
            return frozenset({self._times[self.slider.value()]})
        indices = mc.sample_frame_indices(len(self._times), self.pert_frames_spin.value())
        return frozenset(self._times[i] for i in indices)

    def _pert_search_deltas(self) -> "tuple[tuple, tuple]":
        if self.pert_search_quick_radio.isChecked():
            return pert.QUICK_ROTATION_DELTAS_DEG, pert.QUICK_TRANSLATION_DELTAS_MM
        return pert.DEFAULT_ROTATION_DELTAS_DEG, pert.DEFAULT_TRANSLATION_DELTAS_MM

    def _update_pert_trial_count(self):
        """Trial/evaluation count shown before running, per the earlier
        request to see the cost up front (no time estimate -- just the count,
        since past-run timing isn't tracked)."""
        rotation_deltas, translation_deltas = self._pert_search_deltas()
        n_trials = 3 * (len(rotation_deltas) - 1) + 3 * (len(translation_deltas) - 1) + 1
        n_frames = 1 if self.pert_mode_current_radio.isChecked() else self.pert_frames_spin.value()
        self.pert_trial_count_label.setText(
            f"Frames: {n_frames}   Trials: {n_trials}   Total Evaluations: {n_frames * n_trials}"
        )

    def _cancel_perturbation(self):
        self._worker.cancel_perturbation()
        self.pert_progress_label.setText(self.pert_progress_label.text() + "  (취소 중…)")

    def _on_pert_progress(self, gen: int, done: int, total: int):
        if gen != self._pert_gen:
            return
        self.pert_progress_label.setText(f"평가 중 {done} / {total}")

    def _on_pert_ready(self, gen: int, result: "pert.PerturbationResult"):
        if gen != self._pert_gen:
            return
        self.pert_run_btn.setEnabled(True)
        self.pert_cancel_btn.setVisible(False)
        self._pert_result = result
        self._display_perturbation(result)
        self._refresh_holdout_candidate_deltas()  # new tested deltas may now be available to lock
        if not result.cancelled and result.axes:
            self._note_discovery_success()

    def _on_pert_failed(self, gen: int, msg: str):
        if gen != self._pert_gen:
            return
        self.pert_run_btn.setEnabled(True)
        self.pert_cancel_btn.setVisible(False)
        self.pert_progress_label.setText("")
        self.pert_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _display_perturbation(self, result: "pert.PerturbationResult"):
        self.pert_progress_label.setText(
            f"완료 — {result.n_frames_used} frame 사용"
            + (f", sync 제외 {result.n_sync_rejected}" if result.n_sync_rejected else "")
            + (" (취소됨 — 일부 축만 계산됨)" if result.cancelled else "")
        )
        if result.reason or not result.axes:
            self.pert_table_label.setText(
                f"<span style='color:palette(mid)'>{result.reason or '결과가 없습니다.'}</span>"
            )
            self.pert_summary_label.setText("—")
            self.pert_spatial_label.setText("—")
            return
        self._display_perturbation_axis()

    def _display_perturbation_axis(self):
        """Re-render the table/summary/spatial panels for whichever axis is
        currently selected -- called after a run finishes and every time the
        axis combo changes, with no new computation either time (everything
        for every axis was already computed by `evaluate_perturbation_grid`)."""
        result = self._pert_result
        if result is None or not result.axes:
            return
        axis = result.axes.get(self.pert_axis_combo.currentData())
        if axis is None:
            self.pert_table_label.setText(
                "<span style='color:palette(mid)'>이 축은 취소로 인해 계산되지 않았습니다.</span>"
            )
            self.pert_frame_table_label.setText("—")
            self.pert_summary_label.setText("—")
            self.pert_spatial_label.setText("—")
            return
        self.pert_table_label.setText(self._perturbation_table_html(axis))
        self.pert_frame_table_label.setText(self._perturbation_frame_table_html(axis))
        self.pert_summary_label.setText(self._perturbation_summary_html(axis))
        self.pert_spatial_label.setText(self._perturbation_spatial_html(axis))

    @staticmethod
    def _perturbation_table_html(axis: "pert.AxisSensitivity") -> str:
        """`Mean` is Penalty-inclusive (unchanged, pooled over every usable
        frame's edge points -- see edge_alignment.EdgeAlignmentResult's
        docstring), `Matched Mean` is the matched-only equivalent, and
        `Match` is the fraction they should always be read alongside: a
        delta whose Matched Mean looks great on a collapsing Match Rate is
        describing a shrinking, possibly biased subset of points, not "the
        rest were fine". Median/P95/ΔP95/Match were already here and are
        unchanged.
        """
        unit = "°" if axis.unit == "deg" else "mm"
        header = "".join(
            f"<th style='text-align:right;padding:0 6px'>{h}</th>"
            for h in ("Δ", "Mean", "Matched Mean", "Median", "P95", "ΔP95", "Match")
        )
        rows = [f"<tr>{header}</tr>"]
        for p in axis.points:
            weight = "font-weight:bold;" if p.delta == 0.0 else ""
            delta_text = f"{p.delta:+.2f}{unit}" if p.delta != 0.0 else f"0{unit} (baseline)"
            if not p.ok:
                rows.append(
                    f"<tr style='{weight}'><td>{delta_text}</td>"
                    f"<td colspan='6' style='color:#d9534f'>{p.reason or '평가 실패'}</td></tr>"
                )
                continue
            dp95 = f"{p.delta_p95_px:+.2f}" if np.isfinite(p.delta_p95_px) else "—"
            matched_mean = f"{p.matched_mean_px:.2f}" if np.isfinite(p.matched_mean_px) else "—"
            rows.append(
                f"<tr style='{weight}'>"
                f"<td>{delta_text}</td>"
                f"<td align='right'>{p.mean_px:.2f}</td>"
                f"<td align='right'>{matched_mean}</td>"
                f"<td align='right'>{p.median_px:.2f}</td>"
                f"<td align='right'>{p.p95_px:.2f}</td>"
                f"<td align='right'>{dp95}</td>"
                f"<td align='right'>{p.match_rate * 100:.0f}%</td>"
                f"</tr>"
            )
        html = "<table cellspacing='4'>" + "".join(rows) + "</table>"
        if any(p.coverage_warning for p in axis.points):
            html += "<div style='color:#b26a00'>Comparable frame coverage differs from baseline for some perturbations.</div>"
        return html

    @staticmethod
    def _perturbation_frame_table_html(axis: "pert.AxisSensitivity") -> str:
        """Frame-balanced view: every frame counts once regardless of how many
        edge points it contributed, and Δ is always paired same-frame-to-same-
        frame against baseline (see `_pair_frame_metrics`) -- this is what
        Multi-frame mode's Lowest-Tested/Local-Minimum judgment is based on,
        not the pooled table above."""
        unit = "°" if axis.unit == "deg" else "mm"
        header = "".join(
            f"<th style='text-align:right;padding:0 6px'>{h}</th>"
            for h in ("Δ", "Frame P95(med)", "ΔFrame-P95(med)", "Improved", "Worsened", "Unchanged", "Ratio", "Both valid")
        )
        rows = [f"<tr>{header}</tr>"]
        baseline_valid = axis.baseline.n_valid_frames
        for p in axis.points:
            weight = "font-weight:bold;" if p.delta == 0.0 else ""
            delta_text = f"{p.delta:+.2f}{unit}" if p.delta != 0.0 else f"0{unit} (baseline)"
            if p.delta == 0.0:
                rows.append(
                    f"<tr style='{weight}'><td>{delta_text}</td>"
                    f"<td align='right'>{p.median_frame_p95_px:.2f}</td>"
                    f"<td colspan='5' align='center' style='color:palette(mid)'>—</td>"
                    f"<td align='right'>{baseline_valid}/{baseline_valid}</td></tr>"
                )
                continue
            if not p.ok:
                rows.append(
                    f"<tr style='{weight}'><td>{delta_text}</td>"
                    f"<td colspan='7' style='color:#d9534f'>{p.reason or '평가 실패'}</td></tr>"
                )
                continue
            d_frame_p95 = f"{p.median_delta_frame_p95_px:+.2f}" if np.isfinite(p.median_delta_frame_p95_px) else "—"
            ratio = f"{p.improved_frame_ratio * 100:.0f}%" if np.isfinite(p.improved_frame_ratio) else "—"
            # Named by which side actually failed -- a frame baseline scored
            # but this candidate could not is "Candidate failed", never lumped
            # in with "Candidate-only valid" (see frame_coverage_breakdown).
            coverage = f"{p.n_comparable_frames}/{baseline_valid}"
            extra = [f"{label} {n}" for label, n in pert.frame_coverage_breakdown(p)[1:] if n]
            if extra:
                coverage += "<br>" + "<br>".join(extra)
            rows.append(
                f"<tr style='{weight}'>"
                f"<td>{delta_text}</td>"
                f"<td align='right'>{p.median_frame_p95_px:.2f}</td>"
                f"<td align='right'>{d_frame_p95}</td>"
                f"<td align='right'>{p.n_improved_frames}</td>"
                f"<td align='right'>{p.n_worsened_frames}</td>"
                f"<td align='right'>{p.n_unchanged_frames}</td>"
                f"<td align='right'>{ratio}</td>"
                f"<td align='right'>{coverage}</td>"
                f"</tr>"
            )
        legend = (
            "<div style='color:palette(mid)'>Both valid = baseline·candidate 모두 평가된 frame (paired 비교 대상) / "
            "baseline 유효 frame. Candidate failed = baseline만 유효, Candidate-only valid = candidate만 유효, "
            "Both failed = 둘 다 실패.</div>"
        )
        return "<table cellspacing='4'>" + "".join(rows) + "</table>" + legend

    @staticmethod
    def _perturbation_summary_html(axis: "pert.AxisSensitivity") -> str:
        unit = "°" if axis.unit == "deg" else "mm"
        name = axis.ranking_metric_name
        prefix = "Frame-balanced — " if axis.ranking_metric == "median_frame_p95" else ""
        lines = [f"<span style='color:palette(mid)'>Ranking basis: {prefix}{name}</span>"]
        lines.append(f"Baseline ({name}): {axis.ranking_baseline_px:.2f} px")
        if axis.lowest_point is axis.baseline:
            lines.append("Lowest tested: baseline (0)")
        else:
            lines.append(f"Lowest tested: {axis.lowest_point.delta:+.2f}{unit} → {axis.ranking_best_px:.2f} px")
            lines.append(f"Improvement ({name}): {axis.ranking_improvement_px:.2f} px")
        lines.append(f"Baseline local minimum: <b>{'Yes' if axis.is_local_minimum else 'No'}</b>")
        return "<br>".join(lines)

    @staticmethod
    def _perturbation_spatial_html(axis: "pert.AxisSensitivity") -> str:
        b_spatial, l_spatial = axis.baseline.spatial, axis.lowest_point.spatial
        if b_spatial is None or l_spatial is None:
            return "<span style='color:palette(mid)'>—</span>"
        # HORIZONTAL_REGIONS and VERTICAL_REGIONS both contain "CENTER" (a
        # different statistic in each), so the two dicts must stay separate
        # here rather than merged into one lookup that would collide on it.
        rows = ["<tr><th></th><th>Baseline</th><th>Lowest tested</th></tr>"]
        for group, b_dict, l_dict in (
            ("수평", b_spatial.horizontal, l_spatial.horizontal),
            ("수직", b_spatial.vertical, l_spatial.vertical),
        ):
            rows.append(f"<tr><td colspan='3' style='color:palette(mid)'>{group}</td></tr>")
            for label, b in b_dict.items():
                l = l_dict[label]
                rows.append(
                    f"<tr><td>{label}</td>"
                    f"<td align='right'>{VerifyStep._bin_text(b)}</td>"
                    f"<td align='right'>{VerifyStep._bin_text(l)}</td></tr>"
                )
        return "<table cellspacing='4'>" + "".join(rows) + "</table>"

    # --------------------------------------------------------------- fine scan
    #
    # Zooms into whichever axis is selected in the grid's own axis combo, over
    # a caller-chosen range/step -- shares evaluate_single_axis with the grid's
    # evaluate_perturbation_grid (see gui.core.evaluation.perturbation), so a
    # delta scored here and one scored by the grid are never computed two
    # different ways.

    def _on_pert_axis_changed_for_fine_scan(self):
        axis_key = self.pert_axis_combo.currentData()
        if axis_key is None:
            return
        self.fine_axis_label.setText(f"Selected Axis: {pert.AXIS_LABELS[axis_key]}")
        is_rotation = axis_key in pert.ROTATION_AXES
        suffix = " deg" if is_rotation else " mm"
        for spin in (self.fine_min_spin, self.fine_max_spin, self.fine_step_spin):
            spin.setSuffix(suffix)
        if is_rotation:
            self.fine_min_spin.setValue(-0.2)
            self.fine_max_spin.setValue(0.2)
            self.fine_step_spin.setValue(0.05)
        else:
            self.fine_min_spin.setValue(-10.0)
            self.fine_max_spin.setValue(10.0)
            self.fine_step_spin.setValue(2.0)

    def _run_fine_scan(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        axis_key = self.pert_axis_combo.currentData()
        if axis_key is None:
            return

        mode = "current_frame" if self.pert_mode_current_radio.isChecked() else "multi_frame"
        if mode == "current_frame":
            if self._image is None or self._cloud is None:
                self.window().statusBar().showMessage("먼저 화면에 프레임을 불러오세요.", 4000)
                return
        elif not self._times:
            self.window().statusBar().showMessage("bag 이 아직 로드되지 않았습니다.", 4000)
            return

        if self.fine_max_spin.value() < self.fine_min_spin.value():
            self.window().statusBar().showMessage("Fine Scan Max가 Min보다 작습니다.", 4000)
            return
        deltas = pert.make_delta_range(
            self.fine_min_spin.value(), self.fine_max_spin.value(), self.fine_step_spin.value(),
        )

        self._fine_gen += 1
        self.fine_run_btn.setEnabled(False)
        self.fine_cancel_btn.setVisible(True)
        self.fine_progress_label.setText(f"분석 준비 중… (0 / {len(deltas)})")
        self.fine_table_label.setText("—")
        self.fine_frame_table_label.setText("—")
        self.fine_summary_label.setText("—")
        self.fine_spatial_label.setText("—")
        self._fine_result = None
        self._fine_T = (sol.R.copy(), sol.t.copy())
        self._fine_mode = mode
        self._fine_discovery_bag = self._current_bag()
        self._fine_discovery_timestamps = self._discovery_frame_timestamps(mode)

        p = self.project
        self.request_fine_scan.emit(
            self._fine_gen, mode, self._current_bag(), p.lidar_topic, p.camera_topic,
            p.camera, sol.R, sol.t, list(self._times), self.pert_frames_spin.value(),
            axis_key, deltas,
            self.eval_min_range_spin.value(), self.eval_max_range_spin.value(),
            float(self.max_sync_spin.value()), self._image, self._cloud,
        )

    def _cancel_fine_scan(self):
        self._worker.cancel_fine_scan()
        self.fine_progress_label.setText(self.fine_progress_label.text() + "  (취소 중…)")

    def _on_fine_progress(self, gen: int, done: int, total: int):
        if gen != self._fine_gen:
            return
        self.fine_progress_label.setText(f"평가 중 {done} / {total}")

    def _on_fine_ready(self, gen: int, axis: "pert.AxisSensitivity"):
        if gen != self._fine_gen:
            return
        self.fine_run_btn.setEnabled(True)
        self.fine_cancel_btn.setVisible(False)
        self._fine_result = axis
        self._display_fine_scan(axis)
        self._refresh_holdout_candidate_deltas()  # new tested deltas may now be available to lock
        if axis.baseline is not None:
            self._note_discovery_success()

    def _on_fine_failed(self, gen: int, msg: str):
        if gen != self._fine_gen:
            return
        self.fine_run_btn.setEnabled(True)
        self.fine_cancel_btn.setVisible(False)
        self.fine_progress_label.setText("")
        self.fine_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _display_fine_scan(self, axis: "pert.AxisSensitivity"):
        self.fine_progress_label.setText(f"완료 — {len(axis.points)} delta 평가")
        if axis.baseline is None:
            self.fine_table_label.setText(
                "<span style='color:palette(mid)'>baseline 평가 전에 취소되었습니다.</span>"
            )
            self.fine_frame_table_label.setText("—")
            self.fine_summary_label.setText("—")
            self.fine_spatial_label.setText("—")
            return
        self.fine_table_label.setText(self._perturbation_table_html(axis))
        self.fine_frame_table_label.setText(self._perturbation_frame_table_html(axis))
        self.fine_summary_label.setText(self._perturbation_summary_html(axis))
        self.fine_spatial_label.setText(self._perturbation_spatial_html(axis))

    # ------------------------------------------------------ independent hold-out validation
    #
    # See gui.core.evaluation.holdout's module docstring for the one rule
    # this whole section exists to enforce: hold-out data is never used to
    # fit T or to search for a candidate. A candidate is only ever read off
    # an already-finished Perturbation/Fine Scan result above, frozen into
    # the request the moment "Hold-out 평가 실행" is pressed, and never
    # written back to self._sol, the project, or a file.

    def _on_holdout_source_changed(self):
        is_bag = self.holdout_source_bag_radio.isChecked()
        self.holdout_bag_combo.setEnabled(is_bag)
        self.holdout_bag_status_label.setVisible(is_bag)
        self.holdout_range_start_spin.setEnabled(not is_bag)
        self.holdout_range_end_spin.setEnabled(not is_bag)
        self.holdout_range_note.setVisible(not is_bag)
        self._update_holdout_run_enabled()
        self._update_holdout_preview()

    def _update_holdout_run_enabled(self):
        """Whether "Hold-out 평가 실행" has anything to run against -- kept as
        its own step (rather than set inline wherever a bag finishes loading)
        because it depends on which source radio is active *and* on
        different state for each: a separate bag needs its own timeline
        fetched, while "Current Bag Time Range" only needs the main view's
        timeline, which may already be loaded before this section is even
        touched."""
        if self.holdout_source_bag_radio.isChecked():
            self.holdout_run_btn.setEnabled(bool(self._holdout_bag_path) and bool(self._holdout_times))
        else:
            self.holdout_run_btn.setEnabled(bool(self._times))

    def _on_holdout_bag_combo_changed(self):
        path = self.holdout_bag_combo.currentData()
        if path is None:  # "찾아보기…"
            start = str(Path(self.project.bag_path).parent) if self.project.bag_path else ""
            picked, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Hold-out Bag 선택", start, "ROS bag (*.bag)")
            if not picked:
                if self.holdout_bag_combo.count() > 1:
                    self.holdout_bag_combo.blockSignals(True)
                    self.holdout_bag_combo.setCurrentIndex(0)
                    self.holdout_bag_combo.blockSignals(False)
                return
            self.holdout_bag_combo.blockSignals(True)
            self.holdout_bag_combo.insertItem(self.holdout_bag_combo.count() - 1, Path(picked).name, picked)
            self.holdout_bag_combo.setCurrentIndex(self.holdout_bag_combo.count() - 2)
            self.holdout_bag_combo.blockSignals(False)
            path = picked
        self._open_holdout_bag(path)

    def _open_holdout_bag(self, path: str):
        self._holdout_bag_path = path
        self._holdout_times = []
        self._update_holdout_run_enabled()
        self.holdout_bag_status_label.setText("여는 중…")
        self._holdout_bag_gen += 1
        self.request_holdout_open.emit(self._holdout_bag_gen, path, self.project.lidar_topic)

    def _on_holdout_bag_progress(self, gen: int, done: int, total: int):
        if gen != self._holdout_bag_gen:
            return
        self.holdout_bag_status_label.setText(f"여는 중… {done} / {total}")

    def _on_holdout_bag_ready(self, gen: int, times: list):
        if gen != self._holdout_bag_gen:
            return
        self._holdout_times = list(times)
        name = Path(self._holdout_bag_path).name if self._holdout_bag_path else "—"
        self.holdout_bag_status_label.setText(f"{name}  ({len(self._holdout_times)} frames)")
        self._update_holdout_run_enabled()
        self._update_holdout_preview()

    def _on_holdout_bag_failed(self, gen: int, msg: str):
        if gen != self._holdout_bag_gen:
            return
        self._holdout_times = []
        self._update_holdout_run_enabled()
        self.holdout_bag_status_label.setText(f"<span style='color:#d9534f'>{msg}</span>")

    def _update_holdout_range_bounds(self):
        """Keep the "Current Bag Time Range" spinboxes inside the bag
        actually loaded in the main view -- called whenever that timeline
        changes, not only while the hold-out section happens to be visible,
        so a stale range is never silently carried into a different bag."""
        if not self._times:
            self.holdout_range_note.setText("")
            self._update_holdout_run_enabled()
            return
        duration_s = (self._times[-1] - self._times[0]) / 1e9
        self.holdout_range_start_spin.setRange(0.0, duration_s)
        self.holdout_range_end_spin.setRange(0.0, duration_s)
        if self.holdout_range_end_spin.value() <= self.holdout_range_start_spin.value():
            self.holdout_range_end_spin.setValue(duration_s)
        self.holdout_range_note.setText(f"현재 bag 길이: {duration_s:.1f}s")
        self._update_holdout_run_enabled()
        self._update_holdout_preview()

    def _refresh_holdout_candidate_deltas(self):
        """Every non-zero, successfully-evaluated delta the selected
        Discovery source (Perturbation Grid or Fine Scan) actually tested for
        the axis currently selected in `pert_axis_combo` -- a hold-out
        candidate can only ever be one of these, never a typed-in value, so
        it is always traceable back to a real Discovery result."""
        axis_key = self.pert_axis_combo.currentData()
        self.holdout_delta_combo.blockSignals(True)
        self.holdout_delta_combo.clear()
        src = self.holdout_candidate_source_combo.currentData()
        axis_result = None
        if src == "fine" and self._fine_result is not None and self._fine_result.axis == axis_key:
            axis_result = self._fine_result
        elif src == "grid" and self._pert_result is not None and axis_key is not None:
            axis_result = self._pert_result.axes.get(axis_key)
        if axis_result is not None:
            unit = "°" if axis_result.unit == "deg" else " mm"
            for point in axis_result.points:
                if point.delta == 0.0 or not point.ok:
                    continue
                self.holdout_delta_combo.addItem(f"{point.delta:+.3f}{unit}", point.delta)
        self.holdout_delta_combo.blockSignals(False)
        self._update_holdout_preview()

    def _update_holdout_preview(self):
        axis_key = self.pert_axis_combo.currentData()
        delta = self.holdout_delta_combo.currentData()
        src = self.holdout_candidate_source_combo.currentData()
        if axis_key is None or delta is None:
            self.holdout_candidate_label.setText(
                "<span style='color:palette(mid)'>Perturbation Sensitivity 또는 Fine Scan을 먼저 실행하세요.</span>"
            )
            self.holdout_overlap_label.setText("")
            return
        unit = "°" if axis_key in pert.ROTATION_AXES else " mm"
        self.holdout_candidate_label.setText(
            f"Axis: <b>{pert.AXIS_LABELS[axis_key]}</b><br>"
            f"Delta: <b>{delta:+.3f}{unit}</b><br>"
            f"Source: <b>{'Fine Scan' if src == 'fine' else 'Perturbation Grid'}</b>"
        )
        self._refresh_holdout_overlap_warning()

    def _holdout_plan(self):
        """(bag_path, times, error) for the currently configured hold-out
        source; `error` is None exactly when the other two are usable.

        Pure UI-thread bookkeeping -- no bag is read here. "Separate Bag"'s
        `times` were already fetched by `_on_holdout_bag_ready`; "Current Bag
        Time Range"'s is a sub-list of `self._times`, already held in memory.
        """
        if self.holdout_source_bag_radio.isChecked():
            if not self._holdout_bag_path:
                return None, None, "Hold-out bag을 선택하세요."
            if not self._holdout_times:
                return None, None, "Hold-out bag을 여는 중입니다…"
            return self._holdout_bag_path, list(self._holdout_times), None

        if not self._times:
            return None, None, "현재 bag이 아직 로드되지 않았습니다."
        base = self._times[0]
        start_ns = base + int(round(self.holdout_range_start_spin.value() * 1e9))
        end_ns = base + int(round(self.holdout_range_end_spin.value() * 1e9))
        if end_ns <= start_ns:
            return None, None, "Hold-out End가 Start보다 뒤여야 합니다."
        sub = [t for t in self._times if start_ns <= t <= end_ns]
        if not sub:
            return None, None, "선택한 시간 구간에 frame이 없습니다."
        return self._current_bag(), sub, None

    def _holdout_discovery(self):
        """(bag, timestamps) of whichever Discovery run -- Perturbation Grid
        or Fine Scan -- the currently selected candidate source refers to;
        what `holdout.check_overlap`/`partition_independent_frames` compares
        the hold-out frame set against."""
        if self.holdout_candidate_source_combo.currentData() == "fine":
            return self._fine_discovery_bag, self._fine_discovery_timestamps
        return self._pert_discovery_bag, self._pert_discovery_timestamps

    def _holdout_sample(self, times: list):
        """The exact hold-out frame timestamps a run would sample from
        `times`, *before* Discovery-exclusion filtering -- the same
        deterministic `sample_frame_indices` every other frame-sampling path
        in this module uses, computed here purely to preview/check
        independence before anything is actually read from a bag."""
        n_frames = min(self.holdout_frames_spin.value(), len(times))
        indices = mc.sample_frame_indices(len(times), n_frames)
        return n_frames, [times[i] for i in indices]

    def _holdout_independence_partition(self):
        """(bag, partition, error) -- `partition` is a
        `holdout.IndependencePartition` splitting the sampled hold-out frames
        into usable / excluded-near-Discovery, per the configured Discovery
        Exclusion Margin. `error` is None exactly when `bag`/`partition` are
        usable; still returned even when `partition.n_usable == 0`, so a
        caller can show *why* before refusing to run.
        """
        bag, times, error = self._holdout_plan()
        if error:
            return None, None, error
        _, sampled = self._holdout_sample(times)
        disc_bag, disc_ts = self._holdout_discovery()
        margin_s = self.holdout_margin_spin.value()
        partition = ho.partition_independent_frames(sampled, disc_bag or "", disc_ts, bag, margin_s)
        return bag, partition, None

    def _refresh_holdout_overlap_warning(self):
        bag, partition, error = self._holdout_independence_partition()
        if error:
            self.holdout_overlap_label.setText(f"<span style='color:palette(mid)'>{error}</span>")
            return
        disc_bag, _ = self._holdout_discovery()
        lines = [f"Requested: {partition.requested}   Excluded near Discovery: {partition.n_excluded}   "
                 f"Usable: {partition.n_usable}"]
        if partition.n_usable == 0:
            lines.append(
                "<b>Hold-out overlaps with discovery data. Independent validation is not valid.</b>"
            )
            self.holdout_overlap_label.setText(f"<span style='color:#d9534f'>{'<br>'.join(lines)}</span>")
            return
        if not disc_bag:
            lines.append("Discovery 결과가 없습니다 — Perturbation/Fine Scan을 먼저 실행하세요.")
        elif partition.independence_limited:
            lines.append(
                f"<b>Independence limited</b> -- fewer than {ho.DEFAULT_MIN_INDEPENDENT_FRAMES} independent "
                "frames available; read a result from this configuration cautiously."
            )
        self.holdout_overlap_label.setText(f"<span style='color:palette(mid)'>{'<br>'.join(lines)}</span>")

    def _run_holdout(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return

        axis_key = self.pert_axis_combo.currentData()
        delta = self.holdout_delta_combo.currentData()
        src = self.holdout_candidate_source_combo.currentData()
        if axis_key is None or delta is None:
            self.window().statusBar().showMessage(
                "Candidate가 없습니다. Perturbation Sensitivity 또는 Fine Scan을 먼저 실행하세요.", 5000
            )
            return

        # The candidate must have been discovered against the extrinsic shown
        # now -- otherwise Baseline (current T) and Candidate (baseline+delta
        # read off a stale discovery run) would not describe the same T_original.
        discovery_T = self._fine_T if src == "fine" else self._pert_T
        if discovery_T is None or not dg.extrinsic_matches(discovery_T[0], discovery_T[1], sol.R, sol.t):
            self.window().statusBar().showMessage(
                "선택한 Discovery 결과가 현재 extrinsic과 다릅니다. Perturbation/Fine Scan을 다시 실행하세요.", 6000
            )
            return

        bag, partition, error = self._holdout_independence_partition()
        if error:
            self.window().statusBar().showMessage(error, 5000)
            return
        if partition.n_usable == 0:
            self.window().statusBar().showMessage(
                "Hold-out overlaps with discovery data. Independent validation is not valid.", 8000
            )
            return

        source_label = self._pert_search_source if src == "grid" else "fine"
        disc_bag, disc_ts = self._holdout_discovery()

        self._holdout_gen += 1
        self.holdout_run_btn.setEnabled(False)
        self.holdout_cancel_btn.setVisible(True)
        self.holdout_progress_label.setText("Hold-out 프레임 준비 중…")
        self.holdout_status_label.setText("")
        self.holdout_summary_label.setText("—")
        self.holdout_spatial_label.setText("—")
        self.holdout_depth_label.setText("—")
        self.ab_export_btn.setEnabled(False)
        self._holdout_result = None
        # Locked here, at the moment Run is pressed: axis/delta/source and
        # the baseline T they are compared against are frozen into this call
        # and never re-read from the UI afterward -- see "Candidate lock" in
        # the module docstring. Changing the combos above after this point
        # can only affect the *next* run, never the one now in flight.
        self._holdout_T = (sol.R.copy(), sol.t.copy())
        self._holdout_locked_discovery_revision = self._discovery_revision

        p = self.project
        req = _HoldoutRequest(
            gen=self._holdout_gen, bag=bag, lidar_topic=p.lidar_topic, camera_topic=p.camera_topic, camera=p.camera,
            baseline_R=sol.R.copy(), baseline_t=sol.t.copy(), axis=axis_key, delta=float(delta),
            source_label=source_label, times=list(partition.usable_timestamps),
            min_range=self.eval_min_range_spin.value(), max_range=self.eval_max_range_spin.value(),
            max_sync_offset_ms=float(self.max_sync_spin.value()),
            discovery_bag=disc_bag or "", discovery_timestamps=tuple(disc_ts),
            exclusion_margin_s=self.holdout_margin_spin.value(),
            # usable + excluded reconstructs exactly what was sampled before
            # exclusion filtering (see IndependencePartition: every sampled
            # timestamp ends up in exactly one of the two).
            requested_timestamps=partition.usable_timestamps + partition.excluded_timestamps,
            excluded_near_discovery=partition.excluded_timestamps,
            bootstrap_iterations=self.holdout_bootstrap_iter_spin.value(),
            bootstrap_block_s=self.holdout_bootstrap_block_spin.value(),
            discovery_revision=self._discovery_revision,
        )
        self.request_holdout.emit(req)

    def _cancel_holdout(self):
        self._worker.cancel_holdout()
        self.holdout_progress_label.setText(self.holdout_progress_label.text() + "  (취소 중…)")

    def _on_holdout_progress(self, gen: int, phase: str, done: int, total: int):
        if gen != self._holdout_gen:
            return
        label = "프레임 준비" if phase == "prepare" else "평가"
        self.holdout_progress_label.setText(f"{label} 중 {done} / {total}")

    def _on_holdout_ready(self, gen: int, result: "ho.HoldoutResult"):
        if gen != self._holdout_gen:
            return
        self.holdout_run_btn.setEnabled(True)
        self.holdout_cancel_btn.setVisible(False)
        self._holdout_result = result
        # Exposure is recorded here -- the moment a *successful, uncancelled*
        # result is actually about to be shown to a person -- never merely
        # because the worker finished computing. A cancelled or failed run
        # revealed nothing (see holdout.record_exposure's docstring).
        if result.ok and not result.cancelled and result.provenance is not None and result.provenance.bag \
                and result.candidate_identity is not None and result.candidate_provenance is not None:
            # The guard band in effect *for this run* (see result.discovery),
            # not necessarily whatever the spinbox shows by the time the
            # result arrives -- reused as the Hold-out reveal margin too
            # (see gui.core.evaluation.holdout.determine_and_record_exposure's
            # docstring for why this is a distinct, if numerically copied,
            # setting from Discovery Exclusion Margin).
            reveal_margin_s = result.discovery.exclusion_margin_s if result.discovery is not None \
                else self.holdout_margin_spin.value()
            self._holdout_reveal_status = ho.determine_and_record_exposure(
                self._holdout_exposure_registry, result.provenance.bag, result.signature,
                result.provenance.usable_timestamps, result.candidate_identity, result.candidate_provenance,
                reveal_margin_s,
            )
            # Reflected into the Project object immediately, in memory, and
            # patched to disk right away -- validation exposure history is
            # provenance, not an ordinary UI setting a person might casually
            # discard -- unlike scenes/filters/etc. (only written to disk on
            # an explicit "저장…"). See `_persist_validation_history`: it
            # patches ONLY the validation_history key on disk (see
            # `Project.patch_validation_history`), never a full project
            # save, and is a no-op until this project has been saved at
            # least once (see `_refresh_holdout_persistence_note`).
            self._persist_validation_history()
        else:
            self._holdout_reveal_status = ho.STATUS_INDEPENDENT
        self._display_holdout(result)

    def _on_holdout_failed(self, gen: int, msg: str):
        if gen != self._holdout_gen:
            return
        self.holdout_run_btn.setEnabled(True)
        self.holdout_cancel_btn.setVisible(False)
        self.holdout_progress_label.setText("")
        self.holdout_summary_label.setText(f"<span style='color:#d9534f'>{msg}</span>")
        self.ab_export_btn.setEnabled(False)

    def _holdout_status(self) -> str:
        """Combines two independent exposure checks -- either alone is
        enough to call this Exploratory:

          (A) Live staleness: has Discovery moved on since *this* result's
              candidate was locked? Checked fresh every call against the
              project-persistent `discovery_revision` counter, so an
              already-displayed Independent result correctly turns
              Exploratory the moment Perturbation/Fine Scan next
              *successfully* completes (see `_note_discovery_success`),
              without needing to be re-shown -- and this comparison stays
              correct across an app restart, since the counter is restored
              from disk rather than reset to 0.
          (B) Historical reuse: was this candidate the first one ever
              revealed for this exact Hold-out dataset in this session?
              Classified once, at the moment this result was first shown
              (`_on_holdout_ready` -> `ho.determine_and_record_exposure`),
              and frozen in `self._holdout_reveal_status` from then on --
              catches candidate-switching on a dataset that was peeked at
              before, which a Discovery-generation check alone cannot see.

        `STATUS_INDEPENDENT` (not "no result yet") when there is no result at
        all -- there is simply nothing to have been exposed.
        """
        if self._holdout_result is None:
            return ho.STATUS_INDEPENDENT
        if self._discovery_revision > self._holdout_locked_discovery_revision:
            return ho.STATUS_EXPOSED_DISCOVERY_UPDATED
        return self._holdout_reveal_status

    def _display_holdout(self, result: "ho.HoldoutResult"):
        self.holdout_progress_label.setText(f"완료 — {result.n_frames_used} / {result.n_frames_requested} frame")
        status = self._holdout_status()
        label = ho.STATUS_LABELS.get(status, status)
        if status in ho.EXPOSED_STATUSES:
            self.holdout_status_label.setText(f"<span style='color:#b26a00'><b>{label}</b></span>")
        else:
            self.holdout_status_label.setText(f"<span style='color:palette(mid)'>{label}</span>")

        if result.cancelled or not result.ok:
            color = "palette(mid)" if result.cancelled else "#d9534f"
            self.holdout_summary_label.setText(f"<span style='color:{color}'>{result.reason}</span>")
            self.holdout_spatial_label.setText("—")
            self.holdout_depth_label.setText("—")
            self.ab_export_btn.setEnabled(False)
            return

        self.holdout_summary_label.setText(self._holdout_summary_html(result))
        self.holdout_spatial_label.setText(
            "수평<br>" + self._holdout_compare_table_html([r for r in result.horizontal])
            + "수직<br>" + self._holdout_compare_table_html([r for r in result.vertical])
        )
        self.holdout_depth_label.setText(self._holdout_compare_table_html(result.depth))
        # Independent vs Exploratory is not a gate here -- see
        # _export_ab_package/ab_export's own module docstring: Exploratory
        # export is allowed, just clearly labelled, never blocked outright.
        self.ab_export_btn.setEnabled(True)

    @staticmethod
    def _holdout_summary_html(result: "ho.HoldoutResult") -> str:
        cand = result.candidate
        baseline, candidate_point = result.baseline_point, result.candidate_point
        bs = result.bootstrap
        unit = "°" if cand.unit == "deg" else " mm"
        rows = [
            ("Candidate", f"{cand.label} {cand.delta:+.3f}{unit}  ({cand.source.capitalize()})"),
        ]
        if result.candidate_provenance is not None:
            rows.append(("Discovery revision (locked)", str(result.candidate_provenance.discovery_revision)))
        if result.provenance is not None:
            rows.append(("Requested / Excluded near Discovery / Usable",
                         f"{len(result.provenance.requested_timestamps)} / "
                         f"{len(result.provenance.excluded_near_discovery)} / {result.n_frames_requested}"))
        if result.discovery is not None:
            rows.append(("Discovery Exclusion Margin", f"{result.discovery.exclusion_margin_s:.1f} sec"))
        rows += [
            ("Hold-out frames (usable / requested)", f"{result.n_frames_used} / {result.n_frames_requested}"),
            ("Sync Rejected", str(result.n_sync_rejected)),
            ("Baseline Median Frame-P95", f"{baseline.median_frame_p95_px:.2f} px"),
            ("Candidate Median Frame-P95", f"{candidate_point.median_frame_p95_px:.2f} px"),
            ("Median ΔFrame-P95", f"{candidate_point.median_delta_frame_p95_px:+.2f} px"
                if np.isfinite(candidate_point.median_delta_frame_p95_px) else "—"),
            ("P95 ΔFrame-P95", f"{candidate_point.p95_delta_frame_p95_px:+.2f} px"
                if np.isfinite(candidate_point.p95_delta_frame_p95_px) else "—"),
        ]
        if bs is not None:
            rows += [
                ("Bootstrap method", "Non-overlapping Block Bootstrap"),
                ("Comparable frames", str(bs.n_comparable_frames)),
                ("Median Frame Interval", f"{bs.median_frame_interval_s:.2f} s"
                    if np.isfinite(bs.median_frame_interval_s) else "—"),
                ("Bootstrap Block Duration", f"{bs.block_s:.1f} s"),
                ("Effective Temporal Blocks", str(bs.n_blocks)),
            ]
            if bs.n_blocks:
                rows.append((
                    "Frames / Block",
                    f"median {bs.frames_per_block_median:.1f}, range {bs.frames_per_block_min}-{bs.frames_per_block_max}",
                ))
            if bs.sparse_sampling:
                rows.append(("Sparse sampling", "Most bootstrap blocks contain only one evaluated frame. "
                                                 "Temporal dependence may be weakly represented."))
        if bs is not None and bs.ok:
            rows.append((
                "95% Block-Bootstrap CI (median ΔP95)",
                f"[{bs.median_delta_ci_low_px:+.2f}, {bs.median_delta_ci_high_px:+.2f}] px "
                f"({bs.iterations} iters)",
            ))
            rows.append((
                "95% CI (Improved Ratio)",
                f"[{bs.improved_ratio_ci_low * 100:.1f}, {bs.improved_ratio_ci_high * 100:.1f}] %",
            ))
            if bs.median_ci_excludes_zero is False:
                rows.append(("CI interpretation", "The estimated improvement is uncertain (interval includes zero)."))
        elif bs is not None:
            rows.append(("95% Block-Bootstrap CI", bs.reason or "unavailable"))
        rows += [
            ("Improved", f"{candidate_point.n_improved_frames} / {candidate_point.n_comparable_frames}"),
            ("Worsened", f"{candidate_point.n_worsened_frames} / {candidate_point.n_comparable_frames}"),
            ("Unchanged", f"{candidate_point.n_unchanged_frames} / {candidate_point.n_comparable_frames}"),
            ("Improved Ratio", f"{candidate_point.improved_frame_ratio * 100:.1f} %"
                if np.isfinite(candidate_point.improved_frame_ratio) else "—"),
            ("Comparable", f"{candidate_point.n_comparable_frames} / {baseline.n_valid_frames}"),
        ]
        if candidate_point.n_baseline_only_valid:
            rows.append(("Baseline-only valid", str(candidate_point.n_baseline_only_valid)))
        if candidate_point.n_candidate_only_valid:
            rows.append(("Candidate-only valid", str(candidate_point.n_candidate_only_valid)))
        if candidate_point.n_both_failed:
            rows.append(("Both failed", str(candidate_point.n_both_failed)))
        html_out = VerifyStep._rows_table(rows)
        if result.signature_warning:
            html_out += f"<div style='color:#d9534f'>{result.signature_warning}</div>"
        if result.cross_bag_time_overlap_note:
            html_out += f"<div style='color:palette(mid)'>{result.cross_bag_time_overlap_note}</div>"
        return html_out

    @staticmethod
    def _holdout_compare_table_html(rows: list) -> str:
        if not rows:
            return "<span style='color:palette(mid)'>—</span>"
        out = ["<table cellspacing='4'><tr><th></th><th>Baseline</th><th>Candidate</th><th>Δ</th></tr>"]
        for r in rows:
            if r.baseline_n_points == 0 and r.candidate_n_points == 0:
                out.append(f"<tr><td>{r.label}</td><td colspan='3' style='color:palette(mid)'>—</td></tr>")
                continue
            delta_text = f"{r.delta_p95_px:+.2f}" if np.isfinite(r.delta_p95_px) else "—"
            out.append(
                f"<tr><td>{r.label}</td>"
                f"<td align='right'>{r.baseline_p95_px:.2f} "
                f"<span style='color:palette(mid)'>(n={r.baseline_n_points:,})</span></td>"
                f"<td align='right'>{r.candidate_p95_px:.2f} "
                f"<span style='color:palette(mid)'>(n={r.candidate_n_points:,})</span></td>"
                f"<td align='right'>{delta_text}</td></tr>"
            )
        out.append("</table>")
        return "".join(out)

    # --------------------------------------------- candidate A/B test package export
    #
    # See gui.core.evaluation.ab_export's module docstring: this only ever
    # repackages the Hold-out result already displayed above. The actual
    # "is this still valid to export" checks (current extrinsic == locked
    # baseline, recomputed candidate == locked candidate fingerprint) run
    # inside ab_export.build_snapshot -- never duplicated or re-decided
    # here, so there is exactly one place that decides it.

    def _export_ab_package(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        block_reason = self._symmetry_export_block_reason()
        if block_reason:
            self.ab_export_status_label.setText(f"<span style='color:#d9534f'>{html.escape(block_reason)}</span>")
            self.window().statusBar().showMessage("Calibration export blocked by correspondence symmetry gate.", 7000)
            return
        if self._holdout_result is None or not self._holdout_result.ok or self._holdout_result.cancelled:
            self.window().statusBar().showMessage("먼저 Hold-out 평가를 실행하세요.", 5000)
            return

        p = self.project
        snap = abx.build_snapshot(
            project_name=p.name, baseline_solution=sol, holdout=self._holdout_result,
            validation_status=self._holdout_status(), camera=p.camera, target=p.target,
            lidar_topic=p.lidar_topic, camera_topic=p.camera_topic, bag_paths=list(p.bag_paths),
            method=self._method(),
        )
        if not snap.ok:
            self.ab_export_status_label.setText(f"<span style='color:#d9534f'>{html.escape(snap.reason)}</span>")
            return

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            outcome = abx.export_package(snap, RESULT_DIR)
        except Exception as exc:  # noqa: BLE001 - shown to the user, never crashes the app
            self.ab_export_status_label.setText(
                f"<span style='color:#d9534f'>{html.escape(f'{type(exc).__name__}: {exc}')}</span>"
            )
            return
        if not outcome.ok:
            self.ab_export_status_label.setText(f"<span style='color:#d9534f'>{html.escape(outcome.reason)}</span>")
            return

        self._last_ab_export_dir = outcome.out_dir
        self.ab_export_open_btn.setEnabled(True)
        self.ab_export_status_label.setText(
            "<span style='color:palette(mid)'>A/B package exported<br>"
            f"{html.escape(str(outcome.out_dir))}</span>"
        )
        self.window().statusBar().showMessage(f"A/B package exported: {outcome.out_dir.name}", 8000)

    def _open_ab_export_folder(self):
        path = self._last_ab_export_dir
        if path is None:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------ diagnostic evidence
    #
    # Only gathers what the sections above already computed and hands it to
    # gui.core.evaluation.diagnostics -- every rule lives there, not here.
    # Nothing in this section writes to self._sol, the project, or a file.

    def _diagnostic_inputs(self) -> "dg.DiagnosticInputs":
        sol = self._effective()
        R, t = (sol.R, sol.t) if sol is not None and sol.ok else (None, None)
        notes: list = []

        def fresh(result, T, name):
            if result is None:
                return None
            if T is None or not dg.extrinsic_matches(T[0], T[1], R, t):
                notes.append(f"{name} result excluded: it was computed for a different extrinsic -- rerun it.")
                return None
            return result

        pert_result = fresh(self._pert_result, self._pert_T, "Perturbation Sensitivity")
        fine_result = fresh(self._fine_result, self._fine_T, "Fine Scan")
        mf_result = fresh(self._mf_result, self._mf_T, "Multi-frame evaluation")
        eval_result = fresh(self._eval_result, self._eval_T, "Current-frame evaluation")
        eval_spatial = self._eval_spatial if eval_result is not None else None
        # Independent Hold-out Validation compares its own frozen baseline
        # (self._holdout_T, captured when Run was pressed) against the T
        # shown now -- a stale hold-out result describes a candidate that was
        # locked against a T_original this page no longer shows.
        holdout_result = fresh(self._holdout_result, self._holdout_T, "Hold-out validation")

        calibration = dg.CalibrationContext()
        step = self._calibrate_step()
        if self._loaded_from is not None:
            calibration.unavailable_reason = (
                "The extrinsic was loaded from a file; Step 6 leave-one-out describes a different solution."
            )
        elif step is None or getattr(step, "solution", None) is not self._sol or self._sol is None:
            calibration.unavailable_reason = "No Step 6 solution matches the extrinsic being verified."
        else:
            calibration.loo = getattr(step, "loo", None)
            calibration.rmse_m = self._sol.rmse
            calibration.n_scenes = len(self._sol.scene_ids)
            if self._flipped:
                notes.append(
                    "The shown extrinsic is the half-turn flipped version; Step 6 leave-one-out describes the "
                    "unflipped solve."
                )

        return dg.DiagnosticInputs(
            perturbation=pert_result,
            fine_scan=fine_result,
            fine_scan_mode=self._fine_mode,
            multiframe=mf_result,
            multiframe_sync_limit_ms=self._mf_sync_limit_ms or None,
            perturbation_sync_limit_ms=self._pert_sync_limit_ms or None,
            current_frame=eval_result,
            current_frame_spatial=eval_spatial,
            calibration=calibration,
            holdout=holdout_result,
            holdout_status=self._holdout_status(),
            notes=notes,
        )

    def _generate_diagnostics(self):
        sol = self._effective()
        if sol is None or not sol.ok:
            self.window().statusBar().showMessage(
                "6단계 계산 또는 extrinsic 불러오기가 먼저 필요합니다.", 5000
            )
            return
        self._diag_report = dg.build_diagnostic_report(self._diagnostic_inputs())
        self.diag_label.setText(self._diagnostic_html(self._diag_report))
        self.diag_copy_btn.setEnabled(True)

    def _copy_diagnostics(self):
        if self._diag_report is None:
            return
        QtWidgets.QApplication.clipboard().setText(dg.format_report_text(self._diag_report))
        self.window().statusBar().showMessage("진단 결과를 복사했습니다.", 3000)

    @staticmethod
    def _diagnostic_html(report: "dg.DiagnosticReport") -> str:
        esc = lambda s: html.escape(s).replace("\n", "<br>")  # noqa: E731
        muted = {dg.NOT_OBSERVED, dg.INSUFFICIENT, dg.UNAVAILABLE}

        def strength(item) -> str:
            style = "color:palette(mid)" if item.strength in muted else "font-weight:bold"
            text = f"<span style='{style}'>{esc(item.strength)}</span>"
            if item.mixed:
                text += " <span style='color:#b26a00;font-weight:bold'>(Mixed)</span>"
            return text

        mode_text = {dg.MODE_MULTI: "Multi-frame", dg.MODE_SINGLE: "Single-frame diagnostic",
                     dg.MODE_NONE: "Insufficient data"}
        parts = [f"<div style='color:palette(mid)'>Mode: {mode_text.get(report.mode, report.mode)}</div>"]
        parts += [f"<div style='color:palette(mid)'>• {esc(n)}</div>" for n in report.notes]
        parts.append(f"<p><b>{esc(report.headline)}</b></p>")

        parts.append("<p><b>Root Cause Candidates</b><br><span style='color:palette(mid)'>"
                     "Ordered by evidence strength and frame consistency; order is not a probability.</span></p>")
        if report.candidates:
            items = "".join(
                f"<li>{esc(c.title)} — Evidence: {strength(c)}"
                + "".join(f"<br>{esc(t)}" for t in c.interpretations)
                + "</li>"
                for c in report.candidates
            )
            parts.append(f"<ol>{items}</ol>")
        else:
            parts.append("<div style='color:palette(mid)'>(none)</div>")

        for category in dg.CATEGORY_ORDER:
            items = [i for i in report.evidence if i.category == category]
            if not items:
                continue
            parts.append(f"<hr><b>{esc(category)}</b>")
            for item in items:
                block = [f"<p><b>{esc(item.title)}</b><br>Evidence: {strength(item)}</p>"]
                if item.metrics:
                    block.append(VerifyStep._rows_table([(esc(k), esc(v)) for k, v in item.metrics]))
                if item.observations:
                    block.append("Observed:<ul>" + "".join(f"<li>{esc(o)}</li>" for o in item.observations) + "</ul>")
                if item.interpretations:
                    block.append("Possible interpretation:<ul>"
                                 + "".join(f"<li>{esc(t)}</li>" for t in item.interpretations) + "</ul>")
                if item.caveats:
                    block.append("<ul style='color:palette(mid)'>"
                                 + "".join(f"<li>{esc(c)}</li>" for c in item.caveats) + "</ul>")
                parts.append("".join(block))
        return "".join(parts)

    # ---------------------------------------------------------------- export

    def _export_text(self) -> str:
        sol = self._effective()
        if sol is None:
            return ""
        kind = self.export_combo.currentData()
        cam = self.project.camera
        if kind == "livo2":
            h, w = (self._image.shape[:2] if self._image is not None else (0, 0))
            return to_fast_livo2(sol, cam, w, h)
        if kind == "yaml":
            note = "half-turn flipped by hand" if self._flipped else ""
            return verify.to_yaml(
                sol, cam, self.project.target, note,
                project=self.project, method=self._method(),
            )
        if kind == "rt":
            return verify.to_rt_matrix(sol, cam, project=self.project, method=self._method())
        return verify.to_static_transform(sol) + "\n"

    def _update_export(self):
        self.export_text.setPlainText(self._export_text())

    def _capture_note(self) -> str:
        """What the picture was made from.

        A projection image is evidence, and evidence is only readable with its
        conditions attached: which bag, which moment, which extrinsic, and what
        was filtered out before drawing. Written beside the image so the two
        cannot drift apart.
        """
        p = self.project
        t_ns = self._times[self.slider.value()] if self._times else 0
        rel = (t_ns - self._times[0]) / 1e9 if self._times else 0.0
        pr = self._last_projection
        cam = p.camera
        lines = [
            f"프로젝트   : {p.name}",
            f"bag        : {Path(self._current_bag()).name}",
            f"시각       : {rel:.2f} s (t_ns {t_ns})",
            f"LiDAR 토픽 : {p.lidar_topic}",
            f"카메라 토픽: {p.camera_topic}",
            f"카메라     : fx {cam.fx:.4f} fy {cam.fy:.4f} cx {cam.cx:.4f} cy {cam.cy:.4f}",
            f"왜곡       : {'rational_polynomial(8)' if cam.rational else 'plumb_bob(5)'} "
            + " ".join(f"{v:.6f}" for v in cam.dist()),
            f"extrinsic  : {self._loaded_from or '6단계 계산 결과'}"
            + (f" (RMSE {self._sol.rmse * 1000:.2f} mm)" if self._sol else ""),
            "",
            "그리기 조건",
            f"  거리 필터 : {self.near_spin.value():.2f} ~ {self.far_spin.value():.2f} m"
            + ("  (0 = 제한 없음)" if not self.far_spin.value() else ""),
            f"  색상      : {self.colour_combo.currentText()}",
            f"  점 크기   : {self.size_spin.value():.1f} px   배경 {self.dim_slider.value()}%",
        ]
        if pr is not None:
            lines += [
                "",
                "점 개수",
                f"  입력        {pr.n_input:,}",
                f"  카메라 뒤   {pr.n_behind:,}",
                f"  시야 밖(왜곡 모델 유효 범위 초과) {pr.n_folded:,}",
                f"  화면 밖     {pr.n_outside:,}",
                f"  그린 점     {pr.n_visible:,}",
            ]
        return "\n".join(lines) + "\n"

    def _save_capture(self):
        # `_pix` only exists once something has been drawn; _apply_zoom guards the
        # same way.
        pix = getattr(self, "_pix", None)
        if pix is None or pix.isNull():
            self.window().statusBar().showMessage("아직 그릴 그림이 없습니다.", 4000)
            return
        stem = self.project.name if self.project.name != "untitled" else "capture"
        t_ns = self._times[self.slider.value()] if self._times else 0
        rel = (t_ns - self._times[0]) / 1e9 if self._times else 0.0
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        default = RESULT_DIR / f"{stem}_projection_{rel:07.2f}s.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "캡처 저장", str(default), "PNG 이미지 (*.png)"
        )
        if not path:
            return
        if not pix.save(path, "PNG"):
            self.window().statusBar().showMessage("저장하지 못했습니다.", 5000)
            return
        note = Path(path).with_suffix(".txt")
        note.write_text(self._capture_note())
        self.window().statusBar().showMessage(
            f"저장했습니다: {path}  (조건은 {note.name})", 6000
        )

    def _copy_export(self):
        text = self.export_text.toPlainText()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.window().statusBar().showMessage("복사했습니다.", 3000)

    def _save_export(self):
        text = self.export_text.toPlainText()
        if not text:
            return
        block_reason = self._symmetry_export_block_reason()
        if block_reason:
            QtWidgets.QMessageBox.warning(self, "Calibration ambiguous", block_reason)
            self.window().statusBar().showMessage("Calibration export blocked by correspondence symmetry gate.", 7000)
            return
        kind = self.export_combo.currentData()
        stem = self.project.name if self.project.name != "untitled" else "extrinsic"
        name = {"livo2": f"{stem}_fast_livo2.txt", "yaml": f"{stem}.yaml",
                "tf": f"{stem}_static_transform.sh", "rt": f"{stem}_rt.yaml"}[kind]
        # calib_result/ holds one extrinsic per vehicle and is committed: this is
        # the file the rest of the team uses, so its history is worth keeping.
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "extrinsic 저장", str(RESULT_DIR / name))
        if path:
            with open(path, "w") as fh:
                fh.write(text)
            self.window().statusBar().showMessage(f"저장했습니다: {path}", 5000)

    # ------------------------------------------------------------------ state

    def is_complete(self) -> bool:
        return self._sol is not None and self._sol.ok

    def status_text(self) -> str:
        if self._sol is None or not self._sol.ok:
            return "6단계 필요"
        if self._loaded_from is not None:
            return f"불러온 extrinsic 검증 ({self._loaded_from})"
        return "뒤집어 확인함" if self._flipped else "투영 확인"

    def shutdown(self):
        self._worker.cancel_multiframe()  # let a long multi-frame run exit early rather than block close
        self._worker.cancel_board_motion()
        self._worker.cancel_timestamp_diagnostic()
        self._worker.cancel_board_timestamp_diagnostic()
        self._worker.cancel_holdout()
        self._worker.shutdown()
        self._thread.quit()
        self._thread.wait(1500)

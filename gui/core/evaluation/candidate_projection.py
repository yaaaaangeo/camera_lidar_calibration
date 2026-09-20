"""Projection evidence for symmetry candidate selection.

This module evaluates already-solved extrinsic candidates on validation frames.
It does not change the production extrinsic, write YAML, optimize time offset,
or deskew points.  Its output is evidence that can be attached to the
symmetry-aware Step 6 gate.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from gui.core import verify
from gui.core.bag_reader import BagSource, accumulate_cloud
from gui.core.decode import cloud_to_struct, image_to_bgr
from gui.core.evaluation import edge_alignment as ea
from gui.core.evaluation.edge_correspondence import compute_edge_orientation_map, match_lidar_edges_to_image
from gui.core.solve import Solution


@dataclass
class ValidationFrameSpec:
    frame_index: int
    lidar_record_ns: int
    camera_record_ns: int
    camera_header_ns: int
    timestamp_source: str
    selection_reason: str
    board_median_ns: int | None = None
    nearest_offset_ms: float = float("nan")


@dataclass
class CachedValidationFrame:
    spec: ValidationFrameSpec
    image_bgr: np.ndarray
    cloud_xyz: np.ndarray
    edge_cache: Any
    organized_shape: tuple[int, int] | None = None
    lidar_edge_indices: np.ndarray | None = None
    lidar_edge_xyz: np.ndarray | None = None
    lidar_edge_count_before_subsample: int = 0


@dataclass
class CandidateFrameMetric:
    candidate: str
    frame_index: int
    ok: bool
    reason: str = ""
    matched_mean_px: float = float("nan")
    matched_median_px: float = float("nan")
    matched_p95_px: float = float("nan")
    match_rate: float = float("nan")
    within_2px: float = float("nan")
    within_3px: float = float("nan")
    within_5px: float = float("nan")
    within_10px: float = float("nan")
    median_delta_u_px: float = float("nan")
    median_delta_v_px: float = float("nan")
    n_projected: int = 0
    n_edge_points: int = 0
    invalid_reason_code: str = ""


@dataclass
class ProjectionStageAudit:
    candidate: str
    frame_index: int
    valid: bool
    invalid_reason_code: str
    invalid_reason: str
    raw_points: int
    finite_input_points: int
    finite_transformed_points: int
    z_positive_points: int
    folded_points: int
    range_kept_points: int
    project_attempt_points: int
    finite_projected_points: int
    in_image_points: int
    metric_input_points: int
    matched_points: int
    image_width: int
    image_height: int
    x_camera_min: float = float("nan")
    x_camera_median: float = float("nan")
    x_camera_max: float = float("nan")
    y_camera_min: float = float("nan")
    y_camera_median: float = float("nan")
    y_camera_max: float = float("nan")
    z_camera_min: float = float("nan")
    z_camera_median: float = float("nan")
    z_camera_max: float = float("nan")
    z_positive_ratio: float = float("nan")
    u_min: float = float("nan")
    u_median: float = float("nan")
    u_max: float = float("nan")
    v_min: float = float("nan")
    v_median: float = float("nan")
    v_max: float = float("nan")
    in_image_ratio: float = float("nan")
    evaluation_mode: str = "FULL"
    lidar_edge_points_before_subsample: int = 0
    lidar_edge_points_after_subsample: int = 0


@dataclass
class CandidateProjectionEvidence:
    validation_frames: list[ValidationFrameSpec]
    per_frame: list[CandidateFrameMetric]
    audits: list[ProjectionStageAudit]
    aggregate: dict[str, dict[str, float]]
    pairwise_wins: dict[str, dict[str, int]]
    candidate_summaries: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionValidationParams:
    n_frames: int = 17
    max_cloud_points: int = 120_000
    evaluation_mode: str = "FULL"
    max_edge_points: int = 80_000
    edge_subsample_seed: int = 407
    edge_azimuth_bins: int = 72
    edge_elevation_bins: int = 16
    min_range_m: float = 0.0
    max_range_m: float = 0.0
    max_abs_board_camera_offset_ms: float = 60.0
    edge_params: ea.EdgeAlignmentParams = field(default_factory=ea.EdgeAlignmentParams)


def select_validation_frames_from_board_csv(
    path: str | Path,
    n_frames: int = 17,
    max_abs_offset_ms: float = 60.0,
) -> list[ValidationFrameSpec]:
    """Pick board-time/header-nearest validation frames from an existing CSV."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("board_timestamp_valid", "1") in {"0", "False", "false"}:
                continue
            try:
                nearest_offset = float(row.get("nearest_offset_ms") or row.get("camera_to_board_median_ms"))
                if abs(nearest_offset) > max_abs_offset_ms:
                    continue
                rows.append((
                    abs(nearest_offset),
                    int(row["frame_index"]),
                    ValidationFrameSpec(
                        frame_index=int(row["frame_index"]),
                        lidar_record_ns=int(row["lidar_record_ns"]),
                        camera_record_ns=0,
                        camera_header_ns=int(row.get("nearest_camera_header_ns") or row["camera_header_ns"]),
                        board_median_ns=int(row["board_median_ns"]),
                        nearest_offset_ms=nearest_offset,
                        timestamp_source="board_median_to_nearest_camera_header",
                        selection_reason=(
                            "board timestamp valid; camera selected by nearest header.stamp to LiDAR board median"
                        ),
                    ),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda item: (item[0], item[1]))
    if n_frames > 0:
        rows = rows[:n_frames]
    return [spec for _, _, spec in rows]


def load_cached_validation_frames(
    bag_path: str | Path,
    lidar_topic: str,
    camera_topic: str,
    specs: list[ValidationFrameSpec],
    params: ProjectionValidationParams = ProjectionValidationParams(),
) -> list[CachedValidationFrame]:
    """Load frames once and compute image-side edge cache once per frame."""
    out: list[CachedValidationFrame] = []
    with BagSource(bag_path) as src:
        header_index = _camera_header_index(src, camera_topic)
        for spec in specs:
            cam = header_index.get(spec.camera_header_ns)
            if cam is None:
                continue
            camera_record_ns, camera_msg = cam
            spec.camera_record_ns = int(camera_record_ns)
            image = image_to_bgr(camera_msg)
            cloud, organized_shape, edge_idx, edge_xyz, edge_count = _load_lidar_cloud_and_edges(
                src, lidar_topic, spec.lidar_record_ns, params
            )
            cache = compute_edge_orientation_map(
                image, params.edge_params.canny_low, params.edge_params.canny_high
            )
            out.append(CachedValidationFrame(
                spec=spec,
                image_bgr=image,
                cloud_xyz=cloud,
                edge_cache=cache,
                organized_shape=organized_shape,
                lidar_edge_indices=edge_idx,
                lidar_edge_xyz=edge_xyz,
                lidar_edge_count_before_subsample=edge_count,
            ))
    return out


def evaluate_candidate_projection(
    frames: list[CachedValidationFrame],
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    camera,
    params: ProjectionValidationParams = ProjectionValidationParams(),
) -> CandidateProjectionEvidence:
    per_frame: list[CandidateFrameMetric] = []
    audits: list[ProjectionStageAudit] = []
    for name, (R, t) in candidates.items():
        sol = Solution(ok=True, R=np.asarray(R, np.float64), t=np.asarray(t, np.float64), scene_ids=[], n_pairs=0)
        for frame in frames:
            h, w = frame.image_bgr.shape[:2]
            audit = audit_projection_stage(
                name, frame, sol, camera, params.min_range_m, params.max_range_m,
                params.edge_params, params.evaluation_mode,
            )
            audits.append(audit)
            eval_cloud = _metric_cloud(frame, params)
            pr = verify.project_cloud(
                eval_cloud, sol, camera, w, h,
                min_range=params.min_range_m,
                max_range=params.max_range_m,
            )
            if pr.n_visible == 0:
                per_frame.append(CandidateFrameMetric(
                    candidate=name, frame_index=frame.spec.frame_index, ok=False,
                    reason=audit.invalid_reason,
                    invalid_reason_code=audit.invalid_reason_code,
                    n_projected=pr.n_visible,
                ))
                continue
            er = (
                _evaluate_precomputed_lidar_edges(frame.image_bgr, pr.uv, pr.depth, params.edge_params, frame.edge_cache)
                if _use_edge_subsampled(params) else
                ea.evaluate_edge_alignment(
                    frame.image_bgr, pr.uv, pr.depth, params.edge_params,
                    cached_orientation_map=frame.edge_cache,
                )
            )
            metric = _metric_from_edge_result(name, frame.spec.frame_index, er)
            if not metric.ok:
                metric.invalid_reason_code = audit.invalid_reason_code
            per_frame.append(metric)
    aggregate = _aggregate_by_candidate(per_frame)
    wins = _all_pairwise_wins(per_frame, list(candidates))
    summaries = {
        name: _summary_for_gate(aggregate.get(name, {}))
        for name in candidates
    }
    return CandidateProjectionEvidence(
        validation_frames=[f.spec for f in frames],
        per_frame=per_frame,
        audits=audits,
        aggregate=aggregate,
        pairwise_wins=wins,
        candidate_summaries=summaries,
        metadata={
            "evaluation_mode": params.evaluation_mode.upper(),
            "preprocessing": (
                "FULL_RAW_CLOUD" if not _use_edge_subsampled(params)
                else "FULL_RAW_CLOUD_TO_LIDAR_EDGES_THEN_STRATIFIED_EDGE_SUBSAMPLE"
            ),
            "usable_for_candidate_selection": True,
            "raw_cloud_linear_subsampling": False,
            "max_edge_points": int(params.max_edge_points),
            "edge_subsample_seed": int(params.edge_subsample_seed),
            "edge_azimuth_bins": int(params.edge_azimuth_bins),
            "edge_elevation_bins": int(params.edge_elevation_bins),
            "candidate_names": list(candidates),
        },
    )


def attach_projection_evidence(candidates: list, evidence: CandidateProjectionEvidence) -> None:
    for c in candidates:
        summary = evidence.candidate_summaries.get(c.name)
        if summary:
            c.projection_summary = summary


def export_projection_evidence(evidence: CandidateProjectionEvidence, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "validation_frames": [f.__dict__ for f in evidence.validation_frames],
        "aggregate": evidence.aggregate,
        "pairwise_wins": evidence.pairwise_wins,
        "candidate_summaries": evidence.candidate_summaries,
        "metadata": evidence.metadata,
        "per_frame": [m.__dict__ for m in evidence.per_frame],
        "stage_audit": [a.__dict__ for a in evidence.audits],
    }
    (out / "projection_evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_frame_csv(out / "projection_frame_metrics.csv", evidence.per_frame)
    _write_audit_csv(out / "projection_stage_audit.csv", evidence.audits)
    _write_frame_comparison_csv(out / "normal_vs_half_turn.csv", evidence.per_frame)
    return out


def audit_projection_stage(
    candidate: str,
    frame: CachedValidationFrame,
    sol: Solution,
    camera,
    min_range_m: float = 0.0,
    max_range_m: float = 0.0,
    edge_params: ea.EdgeAlignmentParams = ea.EdgeAlignmentParams(),
    evaluation_mode: str = "FULL",
) -> ProjectionStageAudit:
    """Count every projection stage before the frame is collapsed to valid/invalid."""
    raw_cloud = np.asarray(frame.cloud_xyz, np.float64).reshape(-1, 3)
    cloud = np.asarray(
        frame.lidar_edge_xyz if evaluation_mode.upper() == "EDGE_SUBSAMPLED" and frame.lidar_edge_xyz is not None
        else frame.cloud_xyz,
        np.float64,
    ).reshape(-1, 3)
    h, w = frame.image_bgr.shape[:2]
    raw = int(len(raw_cloud))
    finite_input = int(np.isfinite(raw_cloud).all(axis=1).sum()) if raw else 0
    metric_input_raw = int(len(cloud))
    if raw == 0:
        return ProjectionStageAudit(candidate, frame.spec.frame_index, False, "NO_LIDAR_POINTS", "no LiDAR points",
                                    raw, finite_input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, h)
    try:
        cam_pts = sol.transform(cloud)
    except Exception as exc:  # noqa: BLE001 - diagnostic payload, not control flow
        return ProjectionStageAudit(candidate, frame.spec.frame_index, False, "PROJECTION_EXCEPTION",
                                    f"{type(exc).__name__}: {exc}", raw, finite_input, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, h)
    finite_mask = np.isfinite(cam_pts).all(axis=1)
    finite_transformed = int(finite_mask.sum())
    z_mask = finite_mask & (cam_pts[:, 2] > 1e-3)
    z_positive = int(z_mask.sum())
    stats = _camera_stats(cam_pts[finite_mask])
    if finite_transformed == 0:
        return _audit(candidate, frame, False, "TRANSFORM_FAILURE", "no finite transformed points",
                      raw, finite_input, finite_transformed, z_positive, 0, 0, 0, 0, 0, 0, 0, stats)
    if z_positive == 0:
        return _audit(candidate, frame, False, "ALL_POINTS_BEHIND_CAMERA", "all transformed points have camera Z <= 0",
                      raw, finite_input, finite_transformed, z_positive, 0, 0, 0, 0, 0, 0, 0, stats)

    off_axis = np.hypot(cam_pts[:, 0], cam_pts[:, 1]) / np.maximum(cam_pts[:, 2], 1e-9)
    folded = z_mask & (off_axis > verify.radial_limit(camera))
    keep = z_mask & ~folded
    if max_range_m > 0 or min_range_m > 0:
        dist = np.linalg.norm(cam_pts, axis=1)
        if max_range_m > 0:
            keep &= dist <= max_range_m
        if min_range_m > 0:
            keep &= dist >= min_range_m
    range_kept = int(keep.sum())
    if range_kept == 0:
        return _audit(candidate, frame, False, "NO_RANGE_POINTS", "no points remain after range/folded filtering",
                      raw, finite_input, finite_transformed, z_positive, int(folded.sum()), range_kept,
                      0, 0, 0, 0, 0, stats)

    project_pts = cam_pts[keep]
    uv, _ = cv2.projectPoints(project_pts, np.zeros(3), np.zeros(3), camera.matrix(), camera.dist())
    uv = uv.reshape(-1, 2)
    finite_uv = np.isfinite(uv).all(axis=1)
    finite_projected = int(finite_uv.sum())
    in_img = finite_uv & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    in_image = int(in_img.sum())
    uv_stats = _uv_stats(uv[finite_uv], in_image / max(len(uv), 1))
    if finite_projected == 0:
        return _audit(candidate, frame, False, "NO_FINITE_PROJECTION", "cv2.projectPoints produced no finite pixels",
                      raw, finite_input, finite_transformed, z_positive, int(folded.sum()), range_kept,
                      len(project_pts), finite_projected, in_image, 0, 0, stats, uv_stats)
    if in_image == 0:
        return _audit(candidate, frame, False, "NO_IN_IMAGE_POINTS", "no projected points fall inside the image",
                      raw, finite_input, finite_transformed, z_positive, int(folded.sum()), range_kept,
                      len(project_pts), finite_projected, in_image, 0, 0, stats, uv_stats)

    pr = verify.project_cloud(cloud, sol, camera, w, h, min_range=min_range_m, max_range=max_range_m)
    er = (
        _evaluate_precomputed_lidar_edges(frame.image_bgr, pr.uv, pr.depth, edge_params, frame.edge_cache)
        if evaluation_mode.upper() == "EDGE_SUBSAMPLED" else
        ea.evaluate_edge_alignment(frame.image_bgr, pr.uv, pr.depth, edge_params, cached_orientation_map=frame.edge_cache)
    )
    code = ""
    reason = ""
    if not er.ok:
        code = "NO_EDGE_POINTS" if er.n_edge_points < edge_params.min_edge_points else "TOO_FEW_MATCHES"
        reason = er.reason
    return _audit(candidate, frame, er.ok, code, reason, raw, finite_input, finite_transformed, z_positive,
                  int(folded.sum()), range_kept, len(project_pts), finite_projected, in_image,
                  er.n_edge_points, er.n_matched, stats, uv_stats, evaluation_mode=evaluation_mode,
                  lidar_edge_before=frame.lidar_edge_count_before_subsample,
                  lidar_edge_after=0 if frame.lidar_edge_xyz is None else len(frame.lidar_edge_xyz))


def _camera_header_index(src: BagSource, camera_topic: str) -> dict[int, tuple[int, object]]:
    from gui.core.evaluation.timestamp_diagnostic import header_stamp_ns

    times = src.timestamps(camera_topic)
    if not times:
        return {}
    out = {}
    for record_ns, msg in src.range(camera_topic, times[0], times[-1] + 1):
        header_ns = header_stamp_ns(msg)
        if header_ns is not None:
            out[int(header_ns)] = (int(record_ns), msg)
    return out


def _use_edge_subsampled(params: ProjectionValidationParams) -> bool:
    return params.evaluation_mode.upper() == "EDGE_SUBSAMPLED"


def _metric_cloud(frame: CachedValidationFrame, params: ProjectionValidationParams) -> np.ndarray:
    if _use_edge_subsampled(params) and frame.lidar_edge_xyz is not None:
        return frame.lidar_edge_xyz
    return frame.cloud_xyz


def _load_lidar_cloud_and_edges(
    src: BagSource,
    lidar_topic: str,
    lidar_record_ns: int,
    params: ProjectionValidationParams,
) -> tuple[np.ndarray, tuple[int, int] | None, np.ndarray | None, np.ndarray | None, int]:
    _, msg = src.nearest(lidar_topic, lidar_record_ns)
    if msg is None:
        cloud, _, _, _, _ = accumulate_cloud(src, lidar_topic, lidar_record_ns, 1)
        return np.asarray(cloud, np.float32), None, None, None, 0

    rec = cloud_to_struct(msg)
    xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    cloud = xyz[finite]
    shape = (int(getattr(msg, "height", 0)), int(getattr(msg, "width", 0)))
    edge_idx = _organized_lidar_edge_indices(
        xyz, shape, finite, params.edge_params.depth_jump_threshold_m
    )
    if len(edge_idx) == 0 and rec.dtype.names and "ring" in rec.dtype.names:
        edge_idx, ring_shape = _ring_lidar_edge_indices(
            xyz, np.asarray(rec["ring"]), finite, params.edge_params.depth_jump_threshold_m
        )
        if ring_shape is not None:
            shape = ring_shape
    edge_count = int(len(edge_idx))
    if edge_count:
        edge_idx = _stratified_lidar_edge_subsample(
            xyz, edge_idx, params.max_edge_points, params.edge_azimuth_bins,
            params.edge_elevation_bins, params.edge_subsample_seed,
        )
        edge_xyz = xyz[edge_idx]
    else:
        edge_xyz = np.empty((0, 3), np.float32)
    return cloud, shape if shape[0] > 0 and shape[1] > 0 else None, edge_idx, edge_xyz, edge_count


def _organized_lidar_edge_indices(
    xyz: np.ndarray,
    shape: tuple[int, int],
    finite: np.ndarray,
    depth_jump_threshold_m: float,
) -> np.ndarray:
    h, w = shape
    if h <= 1 or w <= 1 or h * w != len(xyz):
        return np.empty(0, dtype=np.int64)
    ranges = np.linalg.norm(xyz.astype(np.float64), axis=1).reshape(h, w)
    finite_img = finite.reshape(h, w) & np.isfinite(ranges) & (ranges > 0)
    edge = np.zeros((h, w), dtype=bool)

    def mark_neighbor(a_slice, b_slice) -> None:
        ra = ranges[a_slice]
        rb = ranges[b_slice]
        ok = finite_img[a_slice] & finite_img[b_slice] & (np.abs(ra - rb) > depth_jump_threshold_m)
        near_a = ok & (ra <= rb)
        near_b = ok & (rb < ra)
        edge[a_slice] |= near_a
        edge[b_slice] |= near_b

    mark_neighbor((slice(None), slice(0, -1)), (slice(None), slice(1, None)))
    mark_neighbor((slice(0, -1), slice(None)), (slice(1, None), slice(None)))
    return np.flatnonzero(edge.reshape(-1) & finite)


def _ring_lidar_edge_indices(
    xyz: np.ndarray,
    ring: np.ndarray,
    finite: np.ndarray,
    depth_jump_threshold_m: float,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    ring = np.asarray(ring).reshape(-1)
    if len(ring) != len(xyz):
        return np.empty(0, dtype=np.int64), None
    unique, counts = np.unique(ring, return_counts=True)
    if len(unique) <= 1 or counts.min() != counts.max():
        return np.empty(0, dtype=np.int64), None
    width = int(counts[0])
    if width <= 1:
        return np.empty(0, dtype=np.int64), None

    ordered_idx = np.concatenate([np.flatnonzero(ring == r) for r in unique]).astype(np.int64)
    organized_xyz = xyz[ordered_idx]
    organized_finite = finite[ordered_idx]
    shape = (int(len(unique)), width)
    edge_org = _organized_lidar_edge_indices(
        organized_xyz, shape, organized_finite, depth_jump_threshold_m
    )
    if len(edge_org) == 0:
        return np.empty(0, dtype=np.int64), shape
    return ordered_idx[edge_org], shape


def _stratified_lidar_edge_subsample(
    xyz: np.ndarray,
    edge_idx: np.ndarray,
    max_points: int,
    azimuth_bins: int,
    elevation_bins: int,
    seed: int,
) -> np.ndarray:
    edge_idx = np.asarray(edge_idx, np.int64)
    if max_points <= 0 or len(edge_idx) <= max_points:
        return edge_idx
    pts = xyz[edge_idx].astype(np.float64)
    az = np.mod(np.arctan2(pts[:, 1], pts[:, 0]) + np.pi, 2.0 * np.pi)
    el = np.arctan2(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1]))
    az_bin = np.minimum((az / (2.0 * np.pi) * max(azimuth_bins, 1)).astype(np.int64), max(azimuth_bins - 1, 0))
    el_norm = np.clip((el + np.pi / 2.0) / np.pi, 0.0, 1.0 - 1e-12)
    el_bin = np.minimum((el_norm * max(elevation_bins, 1)).astype(np.int64), max(elevation_bins - 1, 0))
    bin_id = az_bin * max(elevation_bins, 1) + el_bin
    rng = np.random.default_rng(seed)
    groups = []
    for bid in np.unique(bin_id):
        members = edge_idx[bin_id == bid]
        groups.append(rng.permutation(members))
    groups.sort(key=lambda arr: int(arr[0]) if len(arr) else -1)

    selected: list[int] = []
    cursor = np.zeros(len(groups), dtype=np.int64)
    while len(selected) < max_points:
        progressed = False
        for gi, group in enumerate(groups):
            if len(selected) >= max_points:
                break
            if cursor[gi] < len(group):
                selected.append(int(group[cursor[gi]]))
                cursor[gi] += 1
                progressed = True
        if not progressed:
            break
    return np.asarray(selected, dtype=np.int64)


def _evaluate_precomputed_lidar_edges(
    image_bgr: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    params: ea.EdgeAlignmentParams,
    edge_cache: Any,
) -> ea.EdgeAlignmentResult:
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    depths = np.asarray(depths, np.float64).reshape(-1)
    n_projected = int(len(pixels))
    if n_projected < params.min_edge_points:
        return ea.EdgeAlignmentResult(
            ok=False,
            reason=f"LiDAR precomputed edge points {n_projected} below minimum {params.min_edge_points}.",
            n_projected=n_projected,
            n_edge_points=n_projected,
        )
    if edge_cache is None or not edge_cache.edge_mask.any():
        return ea.EdgeAlignmentResult(
            ok=False,
            reason="image Canny edge map is empty.",
            n_projected=n_projected,
            n_edge_points=n_projected,
        )
    correspondence = match_lidar_edges_to_image(
        pixels, image_bgr,
        canny_low=params.canny_low,
        canny_high=params.canny_high,
        radii_px=params.correspondence_radii_px,
        max_orientation_diff_deg=params.max_orientation_diff_deg,
        k_orientation=params.k_orientation,
        k_consistency=params.k_consistency,
        max_consistency_angle_deg=params.max_consistency_angle_deg,
        max_consistency_magnitude_ratio=params.max_consistency_magnitude_ratio,
        min_consistency_magnitude_diff_px=params.min_consistency_magnitude_diff_px,
        min_consistency_displacement_for_angle_check_px=params.min_consistency_displacement_for_angle_check_px,
        edge_mask=edge_cache.edge_mask,
        precomputed_orientation_map=edge_cache,
    )
    errors_px = correspondence.distance_px
    matched = correspondence.matched
    delta_u_px = correspondence.matched_pixels[:, 0] - pixels[:, 0]
    delta_v_px = correspondence.matched_pixels[:, 1] - pixels[:, 1]
    agg = ea.aggregate_point_stats(errors_px, matched, delta_u_px, delta_v_px)
    return ea.EdgeAlignmentResult(
        ok=True,
        mean_px=float(np.mean(errors_px)),
        median_px=float(np.median(errors_px)),
        p95_px=float(np.percentile(errors_px, 95)),
        max_px=float(np.max(errors_px)),
        n_projected=n_projected,
        n_edge_points=n_projected,
        n_matched=int(matched.sum()),
        n_unmatched=int(n_projected - matched.sum()),
        match_rate=float(matched.sum()) / max(n_projected, 1),
        matched_mean_px=agg["matched_mean_px"],
        matched_median_px=agg["matched_median_px"],
        matched_p95_px=agg["matched_p95_px"],
        matched_p99_px=agg["matched_p99_px"],
        matched_max_px=agg["matched_max_px"],
        match_rate_by_threshold_px=agg["match_rate_by_threshold_px"],
        mean_delta_u_px=agg["mean_delta_u_px"],
        median_delta_u_px=agg["median_delta_u_px"],
        mean_delta_v_px=agg["mean_delta_v_px"],
        median_delta_v_px=agg["median_delta_v_px"],
        edge_pixels=pixels,
        edge_depths=depths,
        edge_errors_px=errors_px,
        edge_matched=matched,
        edge_delta_u_px=delta_u_px,
        edge_delta_v_px=delta_v_px,
    )


def _audit(
    candidate: str,
    frame: CachedValidationFrame,
    valid: bool,
    code: str,
    reason: str,
    raw: int,
    finite_input: int,
    finite_transformed: int,
    z_positive: int,
    folded: int,
    range_kept: int,
    project_attempt: int,
    finite_projected: int,
    in_image: int,
    metric_input: int,
    matched: int,
    cam_stats: dict[str, float],
    uv_stats: dict[str, float] | None = None,
    evaluation_mode: str = "FULL",
    lidar_edge_before: int = 0,
    lidar_edge_after: int = 0,
) -> ProjectionStageAudit:
    h, w = frame.image_bgr.shape[:2]
    uv_stats = uv_stats or {}
    return ProjectionStageAudit(
        candidate=candidate,
        frame_index=frame.spec.frame_index,
        valid=valid,
        invalid_reason_code="" if valid else code,
        invalid_reason="" if valid else reason,
        raw_points=int(raw),
        finite_input_points=int(finite_input),
        finite_transformed_points=int(finite_transformed),
        z_positive_points=int(z_positive),
        folded_points=int(folded),
        range_kept_points=int(range_kept),
        project_attempt_points=int(project_attempt),
        finite_projected_points=int(finite_projected),
        in_image_points=int(in_image),
        metric_input_points=int(metric_input),
        matched_points=int(matched),
        image_width=w,
        image_height=h,
        x_camera_min=cam_stats.get("x_min", float("nan")),
        x_camera_median=cam_stats.get("x_median", float("nan")),
        x_camera_max=cam_stats.get("x_max", float("nan")),
        y_camera_min=cam_stats.get("y_min", float("nan")),
        y_camera_median=cam_stats.get("y_median", float("nan")),
        y_camera_max=cam_stats.get("y_max", float("nan")),
        z_camera_min=cam_stats.get("z_min", float("nan")),
        z_camera_median=cam_stats.get("z_median", float("nan")),
        z_camera_max=cam_stats.get("z_max", float("nan")),
        z_positive_ratio=float(z_positive) / max(int(finite_transformed), 1),
        u_min=uv_stats.get("u_min", float("nan")),
        u_median=uv_stats.get("u_median", float("nan")),
        u_max=uv_stats.get("u_max", float("nan")),
        v_min=uv_stats.get("v_min", float("nan")),
        v_median=uv_stats.get("v_median", float("nan")),
        v_max=uv_stats.get("v_max", float("nan")),
        in_image_ratio=uv_stats.get("in_image_ratio", float(in_image) / max(int(project_attempt), 1)),
        evaluation_mode=evaluation_mode,
        lidar_edge_points_before_subsample=int(lidar_edge_before),
        lidar_edge_points_after_subsample=int(lidar_edge_after),
    )


def _camera_stats(points: np.ndarray) -> dict[str, float]:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return {}
    out = {}
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        vals = pts[:, idx]
        out[f"{axis}_min"] = float(np.min(vals))
        out[f"{axis}_median"] = float(np.median(vals))
        out[f"{axis}_max"] = float(np.max(vals))
    return out


def _uv_stats(uv: np.ndarray, in_image_ratio: float) -> dict[str, float]:
    pts = np.asarray(uv, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return {"in_image_ratio": float(in_image_ratio)}
    return {
        "u_min": float(np.min(pts[:, 0])),
        "u_median": float(np.median(pts[:, 0])),
        "u_max": float(np.max(pts[:, 0])),
        "v_min": float(np.min(pts[:, 1])),
        "v_median": float(np.median(pts[:, 1])),
        "v_max": float(np.max(pts[:, 1])),
        "in_image_ratio": float(in_image_ratio),
    }


def _subsample_cloud(cloud: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(cloud) <= max_points:
        return cloud
    idx = np.linspace(0, len(cloud) - 1, max_points).astype(np.int64)
    return cloud[idx]


def _metric_from_edge_result(candidate: str, frame_index: int, er: ea.EdgeAlignmentResult) -> CandidateFrameMetric:
    if not er.ok:
        return CandidateFrameMetric(candidate=candidate, frame_index=frame_index, ok=False, reason=er.reason)
    rates = er.match_rate_by_threshold_px or {}
    return CandidateFrameMetric(
        candidate=candidate,
        frame_index=frame_index,
        ok=True,
        matched_mean_px=er.matched_mean_px,
        matched_median_px=er.matched_median_px,
        matched_p95_px=er.matched_p95_px,
        match_rate=er.match_rate,
        within_2px=rates.get(2.0, float("nan")),
        within_3px=rates.get(3.0, float("nan")),
        within_5px=rates.get(5.0, float("nan")),
        within_10px=rates.get(10.0, float("nan")),
        median_delta_u_px=er.median_delta_u_px,
        median_delta_v_px=er.median_delta_v_px,
        n_projected=er.n_projected,
        n_edge_points=er.n_edge_points,
    )


def _aggregate_by_candidate(rows: list[CandidateFrameMetric]) -> dict[str, dict[str, float]]:
    out = {}
    for name in sorted({r.candidate for r in rows}):
        valid = [r for r in rows if r.candidate == name and r.ok]
        out[name] = {
            "valid_frames": float(len(valid)),
            "matched_mean_median_px": _nanpercentile([r.matched_mean_px for r in valid], 50),
            "matched_mean_p90_px": _nanpercentile([r.matched_mean_px for r in valid], 90),
            "matched_mean_p95_px": _nanpercentile([r.matched_mean_px for r in valid], 95),
            "matched_p95_median_px": _nanpercentile([r.matched_p95_px for r in valid], 50),
            "matched_p95_p90_px": _nanpercentile([r.matched_p95_px for r in valid], 90),
            "match_rate_median": _nanpercentile([r.match_rate for r in valid], 50),
            "match_rate_p10": _nanpercentile([r.match_rate for r in valid], 10),
            "within_5px_median": _nanpercentile([r.within_5px for r in valid], 50),
            "within_10px_median": _nanpercentile([r.within_10px for r in valid], 50),
            "median_delta_u_px": _nanpercentile([r.median_delta_u_px for r in valid], 50),
            "median_delta_v_px": _nanpercentile([r.median_delta_v_px for r in valid], 50),
        }
    return out


def _summary_for_gate(agg: dict[str, float]) -> dict[str, Any]:
    if not agg or agg.get("valid_frames", 0.0) <= 0:
        return {"ok": False, "reason": "no valid projection frames"}
    return {
        "ok": True,
        "matched_mean_px": agg["matched_mean_median_px"],
        "matched_median_px": agg["matched_mean_median_px"],
        "matched_p95_px": agg["matched_p95_median_px"],
        "match_rate": agg["match_rate_median"],
        "within_5px": agg["within_5px_median"],
        "within_10px": agg["within_10px_median"],
        "median_delta_u_px": agg["median_delta_u_px"],
        "median_delta_v_px": agg["median_delta_v_px"],
        "n_validation_frames": int(agg["valid_frames"]),
    }


def _pairwise_wins(rows: list[CandidateFrameMetric], a: str, b: str) -> dict[str, dict[str, int]]:
    by = {(r.candidate, r.frame_index): r for r in rows if r.ok}
    frame_ids = sorted({idx for cand, idx in by if cand in {a, b}})
    out = {
        "matched_mean": {a: 0, b: 0, "ties": 0},
        "matched_p95": {a: 0, b: 0, "ties": 0},
        "match_rate": {a: 0, b: 0, "ties": 0},
    }
    for idx in frame_ids:
        ra = by.get((a, idx))
        rb = by.get((b, idx))
        if ra is None or rb is None:
            continue
        _count_lower(out["matched_mean"], a, b, ra.matched_mean_px, rb.matched_mean_px)
        _count_lower(out["matched_p95"], a, b, ra.matched_p95_px, rb.matched_p95_px)
        _count_higher(out["match_rate"], a, b, ra.match_rate, rb.match_rate)
    return out


def _all_pairwise_wins(rows: list[CandidateFrameMetric], candidates: list[str]) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            out[f"{a}_vs_{b}"] = _pairwise_wins(rows, a, b)
    return out


def _count_lower(bucket: dict[str, int], a: str, b: str, va: float, vb: float) -> None:
    if not np.isfinite(va) or not np.isfinite(vb) or abs(va - vb) < 1e-9:
        bucket["ties"] += 1
    elif va < vb:
        bucket[a] += 1
    else:
        bucket[b] += 1


def _count_higher(bucket: dict[str, int], a: str, b: str, va: float, vb: float) -> None:
    if not np.isfinite(va) or not np.isfinite(vb) or abs(va - vb) < 1e-9:
        bucket["ties"] += 1
    elif va > vb:
        bucket[a] += 1
    else:
        bucket[b] += 1


def _nanpercentile(values: list[float], q: float) -> float:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _write_frame_csv(path: Path, rows: list[CandidateFrameMetric]) -> None:
    fields = list(CandidateFrameMetric.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_audit_csv(path: Path, rows: list[ProjectionStageAudit]) -> None:
    fields = list(ProjectionStageAudit.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_frame_comparison_csv(path: Path, rows: list[CandidateFrameMetric]) -> None:
    by = {(r.candidate, r.frame_index): r for r in rows}
    frame_ids = sorted({r.frame_index for r in rows})
    fields = [
        "frame_index",
        "normal_mean", "half_turn_mean",
        "normal_p95", "half_turn_p95",
        "normal_match_rate", "half_turn_match_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx in frame_ids:
            n = by.get(("NORMAL", idx))
            h = by.get(("HALF_TURN", idx))
            writer.writerow({
                "frame_index": idx,
                "normal_mean": getattr(n, "matched_mean_px", float("nan")),
                "half_turn_mean": getattr(h, "matched_mean_px", float("nan")),
                "normal_p95": getattr(n, "matched_p95_px", float("nan")),
                "half_turn_p95": getattr(h, "matched_p95_px", float("nan")),
                "normal_match_rate": getattr(n, "match_rate", float("nan")),
                "half_turn_match_rate": getattr(h, "match_rate", float("nan")),
            })

"""Tests for gui.core.evaluation.diagnostics (+ the Phase-0 naming cleanup in
perturbation.py it builds on).

Every fixture here is a hand-built result object -- the whole point of the
diagnostics module is that it only *reads* results other modules already
produced, so nothing below needs an image, a cloud, or a projection.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import gui.core.evaluation.diagnostics as diag
from gui.core.evaluation.diagnostics import (
    CalibrationContext,
    DiagnosticInputs,
    INSUFFICIENT,
    MODERATE,
    NOT_OBSERVED,
    STRONG,
    UNAVAILABLE,
    WEAK,
    assess_axis,
    build_diagnostic_report,
    extrinsic_matches,
    format_report_text,
)
from gui.core.evaluation.multiframe_consistency import MultiFrameConsistencyResult
from gui.core.evaluation.perturbation import (
    AXIS_LABELS,
    FrameMetric,
    PerturbationPoint,
    PerturbationResult,
    frame_coverage_breakdown,
    summarize_axis,
)
from gui.core.evaluation.spatial_analysis import (
    DEPTH_BIN_LABELS,
    HORIZONTAL_REGIONS,
    VERTICAL_REGIONS,
    BinStats,
    SpatialAnalysisResult,
)

NAN = float("nan")

# ------------------------------------------------------------------ fixtures


def _bins(labels, values, n=500):
    return {l: BinStats(l, v, v, v, 0.1, n, n, 0) for l, v in zip(labels, values)}


def _spatial(h=(2.0, 2.0, 2.0), v=(2.0, 2.0, 2.0), d=(2.0, 2.0, 2.0, 2.0, 2.0)):
    return SpatialAnalysisResult(
        depth_bins=_bins(DEPTH_BIN_LABELS, d),
        horizontal=_bins(HORIZONTAL_REGIONS, h),
        vertical=_bins(VERTICAL_REGIONS, v),
    )


def _per_frame(values):
    """None = that frame failed for this trial."""
    return {
        i: FrameMetric(ok=v is not None, p95_px=v if v is not None else NAN, median_px=v * 0.8 if v is not None else NAN)
        for i, v in enumerate(values)
    }


def _point(delta, frame_values, pooled_p95, spatial=None):
    per = _per_frame(frame_values)
    n_ok = sum(1 for m in per.values() if m.ok)
    return PerturbationPoint(
        delta=delta, ok=n_ok > 0, p95_px=pooled_p95, median_px=pooled_p95 * 0.8,
        n_valid_frames=n_ok, n_failed_frames=len(per) - n_ok, per_frame=per, spatial=spatial,
    )


def _axis(axis_key, baseline_values, candidate_values, delta=-0.1,
          pooled_base=5.0, pooled_cand=None, spatial_b=None, spatial_c=None, mode="multi_frame"):
    """baseline + one candidate + one clearly-worse opposite-sign point, run
    through the real summarize_axis so every derived field is the genuine
    article rather than hand-set."""
    unit = "deg" if axis_key in ("roll", "pitch", "yaw") else "mm"
    if pooled_cand is None:
        pooled_cand = float(np.median([v for v in candidate_values if v is not None]))
    baseline = _point(0.0, baseline_values, pooled_base, spatial_b)
    cand = _point(delta, candidate_values, pooled_cand, spatial_c)
    other = _point(-delta, [v + 1.0 for v in baseline_values], pooled_base + 1.0, spatial_b)
    extra = _point(-2 * delta, [v + 2.0 for v in baseline_values], pooled_base + 2.0, spatial_b)
    ranking = "median_frame_p95" if mode == "multi_frame" else "pooled_p95"
    return summarize_axis(axis_key, unit, [baseline, cand, other, extra], baseline, 0.05, ranking)


def _flat_axis(axis_key, n=30, spatial=None):
    """Baseline is already the lowest tested point for this axis."""
    base = [5.0] * n
    return _axis(axis_key, base, [v + 0.5 for v in base], delta=0.1, pooled_cand=5.5,
                 spatial_b=spatial, spatial_c=spatial)


def _strong_yaw_values(n=30):
    base = [5.0] * n
    cand = [4.4] * 25 + [5.2] * 4 + [5.0]   # 25 improved, 4 worsened, 1 unchanged
    return base, cand


def _pert_result(axes, mode="multi_frame", n_frames=30):
    return PerturbationResult(mode=mode, axes=axes, n_frames_used=n_frames)


def _all_axes(**overrides):
    axes = {k: _flat_axis(k) for k in ("roll", "pitch", "yaw", "tx", "ty", "tz")}
    axes.update(overrides)
    return axes


def _mf(**kwargs):
    defaults = dict(
        n_total=30, n_valid=30, n_failed=0, n_outlier=0, n_sync_rejected=0,
        valid_ratio=1.0, failure_ratio=0.0, outlier_ratio=0.0, sync_rejected_ratio=0.0,
        mean_px=2.0, median_px=2.0, std_px=0.2, p95_px=2.3, max_px=2.5, mad_px=0.2, iqr_px=0.3,
        sync_offset_median_ms=2.0, sync_offset_p95_ms=4.0, sync_offset_max_ms=5.0,
        sync_offset_signed_median_ms=0.5,
    )
    defaults.update(kwargs)
    return MultiFrameConsistencyResult(**defaults)


# ------------------------------------------------ Phase 0: naming cleanup


def test_summarize_axis_fills_generalised_ranking_fields():
    base, cand = _strong_yaw_values()
    axis = _axis("yaw", base, cand)
    assert axis.ranking_metric == "median_frame_p95"
    assert axis.ranking_metric_name == "Median of per-frame P95"
    assert np.isclose(axis.ranking_baseline_px, 5.0)
    assert np.isclose(axis.ranking_best_px, 4.4)
    assert np.isclose(axis.ranking_improvement_px, 0.6)
    # Legacy field kept, same value, for existing callers.
    assert axis.improvement_p95_px == axis.ranking_improvement_px


def test_ranking_metric_name_for_pooled_mode():
    axis = _axis("yaw", [5.0], [4.0], mode="current_frame")
    assert axis.ranking_metric_name == "Pooled P95"


def test_coverage_breakdown_labels_candidate_failure_correctly():
    """A frame baseline scored but the candidate failed must be labelled
    'Candidate failed', never 'Candidate-only valid'."""
    axis = _axis("yaw", [5.0, 5.0, 5.0], [4.0, None, 4.0])
    cand = next(p for p in axis.points if p.delta == -0.1)
    breakdown = dict(frame_coverage_breakdown(cand))
    assert breakdown["Both valid"] == 2
    assert breakdown["Candidate failed"] == 1
    assert breakdown["Candidate-only valid"] == 0
    assert breakdown["Both failed"] == 0


# ------------------------------------------------------- 1. strong axis


def test_axis_improving_on_most_frames_is_strong():
    base, cand = _strong_yaw_values()
    item = assess_axis(_axis("yaw", base, cand), "multi_frame")
    assert item.strength == STRONG
    assert not item.mixed
    metrics = dict(item.metrics)
    assert metrics["Improved frames"] == "25 / 30"
    assert metrics["Worsened frames"] == "4 / 30"
    assert metrics["Comparable frames"] == "30 / 30"


def test_strongest_candidate_is_worded_as_evidence_not_a_verdict():
    base, cand = _strong_yaw_values()
    report = build_diagnostic_report(DiagnosticInputs(perturbation=_pert_result(_all_axes(yaw=_axis("yaw", base, cand)))))
    assert report.candidates[0].key == "axis:yaw:grid"
    assert report.headline.startswith("Strongest observed evidence:")
    text = format_report_text(report).lower()
    for forbidden in ("most likely", "root cause is", "is wrong", "apply", "corrected extrinsic"):
        assert forbidden not in text
    assert "evidence strength, not a calibration correctness score" in text


# ------------------------------------------ 2. one or two frames only


def test_improvement_in_two_frames_does_not_become_strong_global_evidence():
    # Baseline median 5; two frames 6 -> 3.9 pull the candidate median to 4.
    base = [4.0] * 5 + [6.0] * 5
    cand = [4.0] * 5 + [3.9, 3.9] + [6.0] * 3
    item = assess_axis(_axis("yaw", base, cand), "multi_frame")
    assert item.strength == WEAK
    assert any("concentrated in a small subset" in o for o in item.observations)


def test_two_huge_frame_improvements_without_median_shift_is_not_observed():
    base = [5.0] * 30
    cand = [1.0, 1.0] + [5.0] * 28   # pooled could drop a lot; the frame median does not move
    item = assess_axis(_axis("yaw", base, cand, pooled_cand=3.0), "multi_frame")
    assert item.strength == NOT_OBSERVED
    assert not item.is_candidate


# ---------------------------------------------------------- 3. coverage


def test_low_candidate_coverage_lowers_strength():
    base = [5.0] * 30
    full = assess_axis(_axis("yaw", base, [4.4] * 30), "multi_frame")
    partial = assess_axis(_axis("yaw", base, [4.4] * 20 + [None] * 10), "multi_frame")   # 20/30 comparable
    thin = assess_axis(_axis("yaw", base, [4.4] * 14 + [None] * 16), "multi_frame")      # 14/30 comparable
    assert full.strength == STRONG
    assert partial.strength == WEAK
    assert thin.strength == INSUFFICIENT
    assert dict(partial.metrics)["Candidate failed (baseline-only valid)"] == "10"


# --------------------------------------------------------- 4. tolerance


def test_improvement_within_tolerance_is_not_a_candidate():
    item = assess_axis(_axis("tx", [5.0] * 30, [4.96] * 30, delta=10.0), "multi_frame")
    assert item.strength == NOT_OBSERVED
    assert not item.is_candidate


def test_small_but_consistent_improvement_is_capped_below_strong():
    item = assess_axis(_axis("tx", [5.0] * 30, [4.9] * 30, delta=10.0), "multi_frame")
    assert item.strength == MODERATE


# ------------------------------------------------------ 5. spatial mixed


def test_spatial_improvement_and_degradation_is_mixed():
    base, cand = _strong_yaw_values()
    axis = _axis(
        "yaw", base, cand,
        spatial_b=_spatial(h=(2.1, 2.7, 5.8)),
        spatial_c=_spatial(h=(3.5, 2.5, 3.9)),   # RIGHT much better, LEFT much worse
    )
    item = assess_axis(axis, "multi_frame")
    assert item.mixed
    assert item.strength == WEAK
    assert item.strength_label == "Weak (Mixed)"
    assert any("RIGHT-region P95 decreased" in o for o in item.observations)
    assert any("LEFT-region P95 increased" in o for o in item.observations)
    assert any("mixed" in o.lower() for o in item.observations)


def test_consistent_spatial_improvement_is_supporting_not_mixed():
    base, cand = _strong_yaw_values()
    axis = _axis("yaw", base, cand, spatial_b=_spatial(h=(2.1, 2.7, 5.8)), spatial_c=_spatial(h=(2.2, 2.5, 3.9)))
    item = assess_axis(axis, "multi_frame")
    assert not item.mixed
    assert item.strength == STRONG
    assert any("RIGHT-region P95 decreased by 1.90 px" in o for o in item.observations)


# ------------------------------------------ 6. pooled vs frame-balanced


def test_pooled_only_improvement_is_not_strong():
    # Per-frame values identical -> frame-balanced unchanged; pooled drops 1 px.
    item = assess_axis(_axis("yaw", [5.0] * 30, [5.0] * 30, pooled_base=5.0, pooled_cand=4.0), "multi_frame")
    assert item.strength == NOT_OBSERVED
    assert item.mixed
    assert any("Pooled P95 decreased" in o for o in item.observations)


def test_frame_balanced_improves_but_pooled_worsens_is_mixed():
    base, cand = _strong_yaw_values()
    item = assess_axis(_axis("yaw", base, cand, pooled_base=5.0, pooled_cand=6.0), "multi_frame")
    assert item.mixed
    assert item.strength != STRONG


# ----------------------------------------------------------- 7. sync only


def test_sync_offsets_alone_never_diagnose_an_extrinsic_axis():
    mf = _mf(
        n_valid=23, n_sync_rejected=7, sync_rejected_ratio=7 / 30,
        sync_offset_median_ms=35.0, sync_offset_p95_ms=80.0, sync_offset_max_ms=120.0,
        sync_offset_signed_median_ms=34.0,
    )
    report = build_diagnostic_report(DiagnosticInputs(multiframe=mf, multiframe_sync_limit_ms=50.0))
    temporal = next(i for i in report.evidence if i.key == "temporal")
    assert temporal.strength == WEAK   # "review", never above Weak
    assert "7 / 30 frames rejected by configured 50 ms sync limit." in temporal.observations
    for item in report.evidence:
        if item.category in (diag.CAT_ROTATION, diag.CAT_TRANSLATION):
            assert item.strength == UNAVAILABLE
    for label in AXIS_LABELS.values():
        assert label not in report.headline
        assert all(label not in text for text in temporal.observations + temporal.interpretations)
    assert "Sync is the root cause" not in format_report_text(report)


# --------------------------------------------------------- 8. depth only


def test_depth_trend_alone_does_not_establish_translation_error():
    spatial = _spatial(d=(2.1, 2.4, 3.1, 4.8, 5.5))
    axes = _all_axes()
    for ax in axes.values():
        ax.baseline.spatial = spatial
    report = build_diagnostic_report(DiagnosticInputs(perturbation=_pert_result(axes)))
    depth = next(i for i in report.evidence if i.key == "depth")
    assert depth.strength in (WEAK, MODERATE)            # observed, but capped
    assert not depth.is_candidate                         # supporting only
    assert any("increases with distance" in o for o in depth.observations)
    assert "Possible contributors" in depth.interpretations[0]
    for item in report.evidence:
        if item.category == diag.CAT_TRANSLATION:
            assert item.strength == NOT_OBSERVED
    assert all(c.category != diag.CAT_DEPTH for c in report.candidates)


# --------------------------------------------------- 9. current frame


def test_current_frame_result_is_never_promoted_to_global_diagnosis():
    axes = {k: _axis(k, [5.0], [3.0 if k == "yaw" else 5.5], mode="current_frame") for k in
            ("roll", "pitch", "yaw", "tx", "ty", "tz")}
    report = build_diagnostic_report(DiagnosticInputs(perturbation=_pert_result(axes, mode="current_frame", n_frames=1)))
    yaw = next(i for i in report.evidence if i.key == "axis:yaw:grid")
    assert yaw.scope == diag.SCOPE_SINGLE
    assert yaw.strength == WEAK
    assert not yaw.is_candidate
    assert report.mode == diag.MODE_SINGLE
    assert report.candidates == []
    assert diag.SINGLE_FRAME_NOTE in report.notes


# ---------------------------------------------------- 10. Step 6 LOO


def test_missing_loo_is_reported_unavailable():
    for ctx in (None, CalibrationContext(loo=None), CalibrationContext(loo={}),
                CalibrationContext(unavailable_reason="extrinsic loaded from file")):
        report = build_diagnostic_report(DiagnosticInputs(calibration=ctx))
        calib = next(i for i in report.evidence if i.key == "calibration")
        assert calib.strength == UNAVAILABLE
        assert "Calibration-scene stability: unavailable" in calib.observations[0]


def test_loo_scene_sensitivity_is_reported_from_existing_numbers():
    loo = {
        "01": {"shift_mm": 5.0, "rotation_deg": 0.05, "rmse_without": 0.01, "rmse_delta": 0.0},
        "02": {"shift_mm": 8.0, "rotation_deg": 0.10, "rmse_without": 0.01, "rmse_delta": 0.0},
        "03": {"shift_mm": 6.0, "rotation_deg": 0.08, "rmse_without": 0.01, "rmse_delta": 0.0},
        "04": {"shift_mm": 42.0, "rotation_deg": 0.72, "rmse_without": 0.01, "rmse_delta": 0.0},
    }
    item = diag.assess_calibration(CalibrationContext(loo=loo, rmse_m=0.012, n_scenes=4))
    assert item.strength == MODERATE
    assert "Removing scene 04: translation shift 42 mm, rotation shift 0.72°." in item.observations
    assert "Calibration solution is sensitive to scene selection." in item.observations


# ------------------------------------------- 11. no re-evaluation at all


def test_diagnostics_never_calls_projection_edge_or_solver(monkeypatch):
    import cv2

    import gui.core.evaluation.edge_alignment as ea_mod
    import gui.core.evaluation.perturbation as pert_mod
    import gui.core.evaluation.spatial_analysis as sa_mod
    import gui.core.solve as solve_mod
    import gui.core.verify as verify_mod

    def boom(*_a, **_k):
        raise AssertionError("diagnostics must not recompute anything")

    for module, name in (
        (verify_mod, "project_cloud"),
        (ea_mod, "evaluate_edge_alignment"),
        (pert_mod, "evaluate_edge_alignment"),
        (pert_mod, "_evaluate_trial"),
        (pert_mod, "_run_trial_point"),
        (pert_mod, "evaluate_perturbation_grid"),
        (pert_mod, "evaluate_single_axis"),
        (pert_mod, "compute_edge_orientation_map"),
        (sa_mod, "analyze_spatial"),
        (solve_mod, "solve"),
        (solve_mod, "leave_one_out"),
        (cv2, "Canny"),
    ):
        monkeypatch.setattr(module, name, boom)

    base, cand = _strong_yaw_values()
    yaw = _axis("yaw", base, cand, spatial_b=_spatial(), spatial_c=_spatial())
    report = build_diagnostic_report(DiagnosticInputs(
        perturbation=_pert_result(_all_axes(yaw=yaw)),
        fine_scan=_axis("yaw", base, cand),
        multiframe=_mf(),
        current_frame_spatial=_spatial(),
        calibration=CalibrationContext(loo={"a": {"shift_mm": 1.0, "rotation_deg": 0.01}}, n_scenes=3),
    ))
    format_report_text(report)
    assert report.candidates


# ----------------------------------------------- 12. T is never modified


def _deep_equal(a, b) -> bool:
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _deep_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_diagnostics_never_modifies_extrinsic_or_inputs():
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t = np.array([0.1, -0.2, 0.05])
    R0, t0 = R.copy(), t.copy()

    base, cand = _strong_yaw_values()
    inputs = DiagnosticInputs(
        perturbation=_pert_result(_all_axes(yaw=_axis("yaw", base, cand, spatial_b=_spatial(), spatial_c=_spatial()))),
        multiframe=_mf(),
    )
    snapshot = copy.deepcopy(inputs)

    assert extrinsic_matches(R, t, R.copy(), t.copy())
    report = build_diagnostic_report(inputs)
    format_report_text(report)

    assert np.array_equal(R, R0) and np.array_equal(t, t0)
    assert _deep_equal(inputs, snapshot)
    # The report carries text/metrics only -- no transform to apply.
    for item in report.evidence:
        for value in (*[v for _, v in item.metrics], *item.observations, *item.interpretations):
            assert isinstance(value, str)
    assert not any(name.startswith(("apply", "update", "save", "correct")) for name in dir(diag))


def test_extrinsic_matches_detects_a_different_transform():
    R, t = np.eye(3), np.zeros(3)
    assert not extrinsic_matches(R, t, R, t + 1e-3)
    assert not extrinsic_matches(R, t, None, t)


# ---------------------------------------------------- supporting checks


def test_fine_scan_disagreeing_with_grid_marks_both_mixed():
    base, cand = _strong_yaw_values()
    grid = _axis("yaw", base, cand, delta=-0.1)
    fine = _axis("yaw", base, cand, delta=+0.1)
    report = build_diagnostic_report(DiagnosticInputs(perturbation=_pert_result(_all_axes(yaw=grid)), fine_scan=fine))
    items = [i for i in report.evidence if i.key.startswith("axis:yaw")]
    assert len(items) == 2
    assert all(i.mixed and i.strength == WEAK for i in items)


def test_isolated_outlier_frames_reported_as_scene_dependent():
    item = diag.assess_stability(_mf(n_outlier=6, outlier_ratio=6 / 30))
    assert item.strength == MODERATE
    assert any("isolated frames have high error" in o for o in item.observations)


def test_rotation_and_translation_are_never_combined_into_one_number():
    base, cand = _strong_yaw_values()
    report = build_diagnostic_report(DiagnosticInputs(perturbation=_pert_result(_all_axes(
        yaw=_axis("yaw", base, cand), tx=_axis("tx", base, cand, delta=10.0),
    ))))
    for item in report.evidence:
        labels = [k.lower() for k, _ in item.metrics]
        assert not any("per deg" in l or "per mm" in l or "/°" in l for l in labels)


@pytest.mark.parametrize("mode", ["multi_frame", "current_frame"])
def test_report_text_renders_for_every_mode(mode):
    axes = {k: _axis(k, [5.0] * 10, [4.0] * 10, mode=mode) for k in ("roll", "pitch", "yaw", "tx", "ty", "tz")}
    text = format_report_text(build_diagnostic_report(DiagnosticInputs(perturbation=_pert_result(axes, mode=mode))))
    assert "Diagnostic Evidence" in text and "Root Cause Candidates" in text

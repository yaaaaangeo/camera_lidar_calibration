"""Diagnostic Evidence / Root Cause Candidates -- reading results that already exist.

Every other module in this package *measures* something. This one only
*reads*: it takes result objects the other modules have already produced
(Perturbation Sensitivity, Fine Scan, Multi-frame consistency, current-frame
Edge Alignment + Spatial, Step 6 leave-one-out) and sorts what they show into
evidence categories a person can scan. It never projects a cloud, runs Canny,
scores edges, re-runs a perturbation trial, or re-solves the extrinsic -- if a
result is missing, the matching category says "unavailable" instead of
computing it.

What it deliberately does NOT do:
  - confirm a cause. "Strong" means "the data here consistently points this
    way", not "this axis is wrong". Every level is an *evidence strength*,
    never a calibration correctness score.
  - suggest or apply a correction. "Lowest tested" is a probe position on a
    fixed grid, not a new calibration result, and nothing here returns,
    stores, or writes a modified T_cam_lidar.
  - merge rotation and translation into one sensitivity number. Degrees and
    millimetres are different physical quantities, and the probe step sizes
    are arbitrary per unit; candidates are ordered by evidence strength and
    frame consistency (unitless ratios), never by px-per-degree vs px-per-mm.

Observed facts ("RIGHT-region P95 decreased by 1.8 px") and possible
interpretations ("a Cam-Y rotation component may contribute") are kept in
separate lists on every `EvidenceItem`, so the UI can never present the second
as if it were the first.

Every rule below is a plain threshold on an existing number, gathered in
`DiagnosticThresholds` so the numbers are visible in one place and adjustable
in tests; nothing is a learned or weighted score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gui.core.evaluation.multiframe_consistency import DEFAULT_HAMPEL_K, compute_robust_stats, flag_outliers_hampel
from gui.core.evaluation.perturbation import AXIS_LABELS, DEFAULT_LOCAL_MIN_TOLERANCE_PX, ROTATION_AXES, TRANSLATION_AXES
from gui.core.evaluation.spatial_analysis import DEPTH_BIN_LABELS, HORIZONTAL_REGIONS, VERTICAL_REGIONS

# ------------------------------------------------------------------ vocabulary

STRONG = "Strong"
MODERATE = "Moderate"
WEAK = "Weak"
NOT_OBSERVED = "Not observed"   # enough data to look, and nothing beyond tolerance was seen
INSUFFICIENT = "Insufficient"   # the result exists but is too thin to judge (e.g. too few comparable frames)
UNAVAILABLE = "Unavailable"     # the result this category reads was never computed

_STRENGTH_RANK = {STRONG: 4, MODERATE: 3, WEAK: 2, NOT_OBSERVED: 1, INSUFFICIENT: 0, UNAVAILABLE: -1}
CANDIDATE_STRENGTHS = (STRONG, MODERATE, WEAK)

CAT_ROTATION = "Rotation sensitivity"
CAT_TRANSLATION = "Translation sensitivity"
CAT_SPATIAL = "Spatial asymmetry"
CAT_DEPTH = "Depth dependence"
CAT_TEMPORAL = "Temporal pairing quality"
CAT_STABILITY = "Multi-frame stability"
CAT_CALIBRATION = "Calibration-scene stability"
CATEGORY_ORDER = (CAT_ROTATION, CAT_TRANSLATION, CAT_SPATIAL, CAT_DEPTH, CAT_TEMPORAL, CAT_STABILITY, CAT_CALIBRATION)

SCOPE_MULTI = "multi_frame"
SCOPE_SINGLE = "current_frame"
SCOPE_CALIBRATION = "calibration"   # Step 6 target scenes, not the evaluation timeline

MODE_MULTI = "multi_frame"
MODE_SINGLE = "single_frame"
MODE_NONE = "insufficient"

DISCLAIMER = (
    "Evidence strength, not a calibration correctness score. "
    "Nothing here confirms a cause or modifies T_cam_lidar."
)
SINGLE_FRAME_NOTE = "Current-frame evidence only. Not sufficient to infer global extrinsic behavior."
PROBE_NOTE = "Lowest tested is a diagnostic probe, not a new calibration result."
UNITS_NOTE = (
    "Rotation (deg) and translation (mm) probes are never merged into one sensitivity number; "
    "ordering uses evidence strength and frame consistency, not px per unit."
)


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Every cut-off the rules use. Defaults are deliberately conservative:
    Strong needs a clear majority of frames improving, near-full coverage, few
    frames getting worse, *and* a paired per-frame improvement -- one pooled
    number moving is never enough on its own."""

    # Same tolerance Perturbation Sensitivity uses for "local minimum" and for
    # counting a frame as improved/worsened -- a smaller change is noise.
    improvement_tolerance_px: float = DEFAULT_LOCAL_MIN_TOLERANCE_PX

    # --- extrinsic-axis perturbation, Multi-frame ---
    min_comparable_frames: int = 5
    insufficient_coverage_ratio: float = 0.50     # comparable / baseline-valid below this -> Insufficient
    strong_improved_ratio: float = 0.70
    strong_min_coverage_ratio: float = 0.80
    strong_max_worsened_ratio: float = 0.20
    moderate_improved_ratio: float = 0.55
    moderate_min_coverage_ratio: float = 0.70
    moderate_max_worsened_ratio: float = 0.30
    min_improvement_for_strong_px: float = 0.20   # below this, a consistent effect is still capped at Moderate
    concentrated_improved_ratio: float = 0.25     # fewer improved frames than this -> "concentrated in a subset"
    mixed_worsened_ratio: float = 0.30            # this many frames worsening alongside an improvement -> Mixed

    # --- per-region / per-depth-bin change, baseline vs lowest tested ---
    region_min_points: int = 50
    region_change_min_px: float = 0.30
    region_change_min_ratio: float = 0.10         # of the baseline region value; the larger of the two applies

    # --- spatial asymmetry at baseline (supporting only, capped at Moderate) ---
    asymmetry_weak_ratio: float = 1.25
    asymmetry_moderate_ratio: float = 1.50
    asymmetry_min_diff_px: float = 0.50

    # --- depth dependence at baseline (supporting only, capped at Moderate) ---
    depth_min_bins: int = 3
    depth_weak_ratio: float = 1.25
    depth_moderate_ratio: float = 1.50
    depth_min_diff_px: float = 0.50

    # --- temporal pairing (never above Weak: no temporal sensitivity exists) ---
    sync_review_abs_p95_ms: float = 10.0
    sync_review_rejected_ratio: float = 0.10

    # --- multi-frame stability (capped at Moderate) ---
    stability_min_frames: int = 5
    stability_outlier_weak_ratio: float = 0.05
    stability_outlier_moderate_ratio: float = 0.15
    stability_spread_weak: float = 0.25           # MAD / median
    stability_spread_moderate: float = 0.40
    stability_failure_note_ratio: float = 0.20

    # --- Step 6 leave-one-out; mirrors step_calibrate's red highlight (>30 mm or >0.5 deg) ---
    loo_shift_warn_mm: float = 30.0
    loo_rotation_warn_deg: float = 0.5
    loo_min_scenes_for_strong: int = 4


DEFAULT_THRESHOLDS = DiagnosticThresholds()


# ---------------------------------------------------------------- data model


@dataclass
class EvidenceItem:
    category: str
    title: str
    strength: str
    scope: str
    key: str = ""
    mixed: bool = False
    # False for categories that may only *support* another finding (Spatial,
    # Depth) -- they are shown as evidence but never listed as a candidate.
    candidate_eligible: bool = True
    metrics: list = field(default_factory=list)          # (label, value text)
    observations: list = field(default_factory=list)     # facts read off the numbers
    interpretations: list = field(default_factory=list)  # hedged, "possible ..." only
    caveats: list = field(default_factory=list)
    consistency: float = float("nan")                    # improved-frame ratio, for tie-breaking only

    @property
    def strength_label(self) -> str:
        return f"{self.strength} (Mixed)" if self.mixed else self.strength

    @property
    def is_candidate(self) -> bool:
        return (
            self.candidate_eligible
            and self.scope != SCOPE_SINGLE
            and self.strength in CANDIDATE_STRENGTHS
        )


@dataclass
class CalibrationContext:
    """What Step 6 already computed -- passed through, never recomputed."""

    loo: Optional[dict] = None          # solve.leave_one_out() output; None/{} = not available
    rmse_m: float = float("nan")
    n_scenes: int = 0
    unavailable_reason: str = ""


@dataclass
class DiagnosticInputs:
    """Already-computed results. Any may be None; the matching category then
    reports Unavailable instead of computing anything."""

    perturbation: Optional[object] = None          # perturbation.PerturbationResult
    fine_scan: Optional[object] = None             # perturbation.AxisSensitivity
    fine_scan_mode: str = "multi_frame"
    multiframe: Optional[object] = None            # multiframe_consistency.MultiFrameConsistencyResult
    multiframe_sync_limit_ms: Optional[float] = None   # None / 0 = no limit configured
    perturbation_sync_limit_ms: Optional[float] = None
    current_frame: Optional[object] = None         # edge_alignment.EdgeAlignmentResult
    current_frame_spatial: Optional[object] = None  # spatial_analysis.SpatialAnalysisResult
    calibration: Optional[CalibrationContext] = None
    notes: list = field(default_factory=list)      # caller context, e.g. results excluded as stale


@dataclass
class DiagnosticReport:
    mode: str
    headline: str
    evidence: list = field(default_factory=list)    # every EvidenceItem, CATEGORY_ORDER
    candidates: list = field(default_factory=list)  # ranked subset; order is not a probability
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------- helpers


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


def _px(v: float) -> str:
    return f"{v:.2f} px" if _finite(v) else "—"


def _ms(v: float, signed: bool = False) -> str:
    if not _finite(v):
        return "—"
    return f"{v:+.1f} ms" if signed else f"{v:.1f} ms"


def _ratio_text(n: int, d: int) -> str:
    return f"{n} / {d}" if d else "—"


def cap_strength(strength: str, cap: str) -> str:
    """Lower `strength` to `cap` if it is above it; never raises a level."""
    return cap if _STRENGTH_RANK[strength] > _STRENGTH_RANK[cap] else strength


def extrinsic_matches(R_a, t_a, R_b, t_b, atol: float = 1e-9) -> bool:
    """Whether a stored result was computed against the same extrinsic as the
    one now shown -- so a stale result (computed before a flip, a re-solve,
    or loading another file) can be left out instead of silently diagnosed.
    Read-only; the arrays are only compared."""
    if R_a is None or R_b is None or t_a is None or t_b is None:
        return False
    return bool(np.allclose(R_a, R_b, rtol=0.0, atol=atol) and np.allclose(t_a, t_b, rtol=0.0, atol=atol))


def _significant_change(baseline: float, candidate: float, th: DiagnosticThresholds) -> int:
    """-1 improved, +1 worsened, 0 within noise."""
    limit = max(th.region_change_min_px, th.region_change_min_ratio * abs(baseline))
    d = candidate - baseline
    if d <= -limit:
        return -1
    if d >= limit:
        return 1
    return 0


def _compare_bins(b_dict: dict, l_dict: dict, labels: tuple, th: DiagnosticThresholds) -> list:
    """(label, baseline P95, candidate P95, direction) per bin with enough
    points on both sides; bins without are left out rather than guessed."""
    out = []
    for label in labels:
        b, l = b_dict.get(label), l_dict.get(label)
        if b is None or l is None:
            continue
        if b.n_points < th.region_min_points or l.n_points < th.region_min_points:
            continue
        if not (_finite(b.p95_px) and _finite(l.p95_px)):
            continue
        out.append((label, b.p95_px, l.p95_px, _significant_change(b.p95_px, l.p95_px, th)))
    return out


def _describe_changes(changes: list, suffix: str) -> list:
    lines = []
    stable = []
    for label, b, l, direction in changes:
        if direction < 0:
            lines.append(f"{label}{suffix} P95 decreased by {b - l:.2f} px ({b:.2f} → {l:.2f}).")
        elif direction > 0:
            lines.append(f"{label}{suffix} P95 increased by {l - b:.2f} px ({b:.2f} → {l:.2f}).")
        else:
            stable.append(label)
    if stable and len(stable) < len(changes):
        lines.append(f"{', '.join(stable)} remained approximately stable.")
    return lines


# ----------------------------------------------------------- axis evidence


def assess_axis(axis, mode: str, th: DiagnosticThresholds = DEFAULT_THRESHOLDS, source: str = "grid") -> EvidenceItem:
    """Evidence that one extrinsic parameter is sensitive, from an
    `AxisSensitivity` already computed by Perturbation Sensitivity (source
    "grid") or Fine Scan (source "fine"). Reads only; `axis` is not modified.

    Multi-frame rules, in order (see DiagnosticThresholds for the numbers):
      1. The frame-balanced ranking metric must drop by more than the
         tolerance, or nothing is claimed (Not observed).
      2. Too few comparable frames, or too many candidate failures, and the
         result is Insufficient -- a candidate cannot look good by failing on
         the hard frames.
      3. Strong / Moderate need a majority of comparable frames improving, few
         worsening, enough coverage, and a negative paired median delta.
      4. Caps: improvement concentrated in a small subset -> Weak; any mixed
         signal -> Weak; small absolute improvement -> at most Moderate.
    Current-frame results are never above Weak and never a candidate.
    """
    is_rotation = axis.axis in ROTATION_AXES
    kind = "rotation" if is_rotation else "translation"
    label = AXIS_LABELS.get(axis.axis, axis.axis)
    title = f"{label} {kind} sensitivity" + (" — Fine Scan" if source == "fine" else "")
    scope = SCOPE_MULTI if mode == "multi_frame" else SCOPE_SINGLE
    item = EvidenceItem(
        category=CAT_ROTATION if is_rotation else CAT_TRANSLATION,
        title=title, strength=INSUFFICIENT, scope=scope, key=f"axis:{axis.axis}:{source}",
    )
    if scope == SCOPE_SINGLE:
        item.caveats.append(SINGLE_FRAME_NOTE)

    baseline = axis.baseline
    if baseline is None or not baseline.ok:
        item.observations.append("Baseline could not be evaluated for this axis.")
        return item

    unit = "°" if axis.unit == "deg" else " mm"
    tol = th.improvement_tolerance_px
    lowest = axis.lowest_point or baseline
    improvement = axis.ranking_improvement_px if _finite(axis.ranking_improvement_px) else axis.improvement_p95_px
    metric = axis.ranking_metric_name
    tested = [p.delta for p in axis.points if p.delta != 0.0]

    item.metrics.append(("Ranking metric", metric))
    item.metrics.append(("Baseline", _px(axis.ranking_baseline_px)))

    if lowest is baseline or not _finite(improvement) or improvement <= tol:
        item.strength = NOT_OBSERVED
        if lowest is baseline:
            item.metrics.append(("Lowest tested", "baseline (0)"))
        else:
            item.metrics.append(("Lowest tested", f"{lowest.delta:+.2f}{unit}"))
            item.metrics.append(("Improvement", _px(improvement)))
        item.observations.append(
            f"No tested {label} offset lowered the {metric} by more than the {tol:.2f} px tolerance."
        )
        if scope == SCOPE_MULTI:
            pooled = [p for p in axis.points if p is not baseline and p.ok and _finite(p.delta_p95_px)]
            best_pooled = min(pooled, key=lambda p: p.delta_p95_px) if pooled else None
            if best_pooled is not None and best_pooled.delta_p95_px < -tol:
                item.mixed = True
                item.observations.append(
                    f"Pooled P95 decreased by {-best_pooled.delta_p95_px:.2f} px at {best_pooled.delta:+.2f}{unit}, "
                    "but the frame-balanced metric did not -- the pooled change is likely driven by a few "
                    "edge-rich frames, not by the frame set as a whole."
                )
        item.interpretations.append("No sensitivity evidence for this parameter within the tested range.")
        return item

    item.metrics.append(("Lowest tested", f"{lowest.delta:+.2f}{unit}"))
    item.metrics.append(("Candidate", _px(axis.ranking_best_px)))
    item.metrics.append(("Improvement", _px(improvement)))

    limits: list = []   # why the level is not higher -- shown, so the rule is never hidden
    if scope == SCOPE_MULTI:
        n_base = baseline.n_valid_frames
        n_cmp = lowest.n_comparable_frames
        coverage = n_cmp / n_base if n_base else float("nan")
        imp_ratio = lowest.n_improved_frames / n_cmp if n_cmp else float("nan")
        worse_ratio = lowest.n_worsened_frames / n_cmp if n_cmp else float("nan")
        paired_median = lowest.median_delta_frame_p95_px
        item.consistency = imp_ratio

        item.metrics += [
            ("Improved frames", _ratio_text(lowest.n_improved_frames, n_cmp)),
            ("Worsened frames", _ratio_text(lowest.n_worsened_frames, n_cmp)),
            ("Unchanged frames", _ratio_text(lowest.n_unchanged_frames, n_cmp)),
            ("Comparable frames", _ratio_text(n_cmp, n_base)),
            ("Median paired ΔFrame-P95", f"{paired_median:+.2f} px" if _finite(paired_median) else "—"),
        ]
        if lowest.n_baseline_only_valid:
            item.metrics.append(("Candidate failed (baseline-only valid)", str(lowest.n_baseline_only_valid)))
        if lowest.n_candidate_only_valid:
            item.metrics.append(("Candidate-only valid", str(lowest.n_candidate_only_valid)))
        if lowest.n_both_failed:
            item.metrics.append(("Both failed", str(lowest.n_both_failed)))

        if n_cmp < th.min_comparable_frames or not _finite(coverage) or coverage < th.insufficient_coverage_ratio:
            item.strength = INSUFFICIENT
            item.observations.append(
                f"Only {n_cmp} of {n_base} baseline-valid frames are comparable at {lowest.delta:+.2f}{unit} "
                f"(need at least {th.min_comparable_frames} and {th.insufficient_coverage_ratio:.0%})."
            )
            item.caveats.append(PROBE_NOTE)
            return item

        strong_ok = (
            imp_ratio >= th.strong_improved_ratio
            and coverage >= th.strong_min_coverage_ratio
            and worse_ratio <= th.strong_max_worsened_ratio
            and _finite(paired_median) and paired_median <= -tol
        )
        moderate_ok = (
            imp_ratio >= th.moderate_improved_ratio
            and coverage >= th.moderate_min_coverage_ratio
            and worse_ratio <= th.moderate_max_worsened_ratio
            and _finite(paired_median) and paired_median < 0
        )
        item.strength = STRONG if strong_ok else MODERATE if moderate_ok else WEAK
        if not strong_ok:
            if imp_ratio < th.strong_improved_ratio:
                limits.append(f"improved frames {imp_ratio:.0%} < {th.strong_improved_ratio:.0%}")
            if coverage < th.strong_min_coverage_ratio:
                limits.append(f"coverage {coverage:.0%} < {th.strong_min_coverage_ratio:.0%}")
            if worse_ratio > th.strong_max_worsened_ratio:
                limits.append(f"worsened frames {worse_ratio:.0%} > {th.strong_max_worsened_ratio:.0%}")
            if not (_finite(paired_median) and paired_median <= -tol):
                limits.append("median paired per-frame change is not an improvement beyond tolerance")

        item.observations.append(
            f"At {lowest.delta:+.2f}{unit} the {metric} decreased by {improvement:.2f} px; "
            f"{lowest.n_improved_frames} of {n_cmp} comparable frames improved."
        )
        if imp_ratio < th.concentrated_improved_ratio:
            item.strength = cap_strength(item.strength, WEAK)
            item.observations.append(
                f"Improvement is concentrated in a small subset of frames ({lowest.n_improved_frames} / {n_cmp}). "
                "Evidence for a global extrinsic-axis issue is weak; a scene-dependent effect is also possible."
            )
        if worse_ratio >= th.mixed_worsened_ratio:
            item.mixed = True
            item.observations.append(
                f"{lowest.n_worsened_frames} of {n_cmp} frames got worse at the same offset."
            )
        if _finite(lowest.delta_p95_px) and lowest.delta_p95_px > tol:
            item.mixed = True
            item.observations.append(
                f"Frame-balanced metric improved, but pooled P95 at the same offset increased by "
                f"{lowest.delta_p95_px:.2f} px."
            )
        if lowest.coverage_warning:
            item.caveats.append(lowest.coverage_warning)
    else:
        item.strength = WEAK
        item.observations.append(
            f"At {lowest.delta:+.2f}{unit} the {metric} of the current frame decreased by {improvement:.2f} px."
        )
        limits.append("single frame")

    # Spatial / depth changes: supporting observations only, never the reason
    # a level is raised -- but a contradiction between regions does lower it.
    b_sp, l_sp = baseline.spatial, lowest.spatial
    if b_sp is not None and l_sp is not None:
        for group, labels, b_dict, l_dict in (
            ("horizontal", HORIZONTAL_REGIONS, b_sp.horizontal, l_sp.horizontal),
            ("vertical", VERTICAL_REGIONS, b_sp.vertical, l_sp.vertical),
        ):
            changes = _compare_bins(b_dict, l_dict, labels, th)
            item.observations += _describe_changes(changes, "-region")
            directions = {c[3] for c in changes}
            if -1 in directions and 1 in directions:
                item.mixed = True
                item.observations.append(f"Spatial change is mixed: some {group} regions improved while others worsened.")
        depth_changes = _compare_bins(b_sp.depth_bins, l_sp.depth_bins, DEPTH_BIN_LABELS, th)
        item.observations += _describe_changes(depth_changes, " depth-bin")
        item.caveats.append(
            "Region/depth values pool every evaluated frame's edge points; scene geometry, intrinsic "
            "distortion and timing can produce similar regional patterns."
        )

    if tested and lowest.delta in (min(tested), max(tested)):
        item.caveats.append(
            "Lowest tested sits at the edge of the tested range; the minimum may lie beyond it (Fine Scan can widen it)."
        )
    if item.mixed:
        item.strength = cap_strength(item.strength, WEAK)
        item.observations.append("Global metric improved, but spatial/frame consistency is mixed.")
        limits.append("mixed evidence")
    if improvement < th.min_improvement_for_strong_px:
        item.observations.append(
            f"Absolute improvement is small ({improvement:.2f} px < {th.min_improvement_for_strong_px:.2f} px); "
            "little weight should be placed on it by itself."
        )
        if item.strength == STRONG:
            item.strength = MODERATE
            limits.append(f"absolute improvement {improvement:.2f} px < {th.min_improvement_for_strong_px:.2f} px")
    if limits and item.strength != STRONG:
        item.caveats.append("Not rated higher because: " + "; ".join(limits) + ".")

    if item.strength in (STRONG, MODERATE):
        item.interpretations.append(
            f"A {kind} error in {label} may contribute to the residual edge misalignment."
        )
    else:
        item.interpretations.append(
            f"A {label} {kind} component is possible, but the evidence is weak or inconsistent."
        )
    item.caveats.append(
        "Each parameter is perturbed on its own; an error in another parameter, in timing or in the "
        "intrinsics can also make this one look sensitive."
    )
    item.caveats.append(PROBE_NOTE)
    return item


def _unavailable_axis(axis_key: str, reason: str) -> EvidenceItem:
    is_rotation = axis_key in ROTATION_AXES
    kind = "rotation" if is_rotation else "translation"
    return EvidenceItem(
        category=CAT_ROTATION if is_rotation else CAT_TRANSLATION,
        title=f"{AXIS_LABELS[axis_key]} {kind} sensitivity", strength=UNAVAILABLE, scope=SCOPE_MULTI,
        key=f"axis:{axis_key}:grid", observations=[reason],
    )


def _cross_check_fine(grid_item: EvidenceItem, fine_item: EvidenceItem, grid_axis, fine_axis) -> None:
    """Grid and Fine Scan probing the same axis should point the same way; if
    both claim an improvement in opposite directions, neither is trusted."""
    if grid_item.strength not in CANDIDATE_STRENGTHS or fine_item.strength not in CANDIDATE_STRENGTHS:
        return
    g, f = grid_axis.lowest_point.delta, fine_axis.lowest_point.delta
    if g * f < 0:
        for item in (grid_item, fine_item):
            item.mixed = True
            item.strength = cap_strength(item.strength, WEAK)
            item.observations.append("Grid and Fine Scan disagree on the direction of the lowest tested offset.")
    else:
        fine_item.observations.append("Direction of the lowest tested offset agrees with the grid result.")


# ------------------------------------------------------ supporting categories


def assess_spatial(spatial, scope: str, th: DiagnosticThresholds = DEFAULT_THRESHOLDS, source: str = "") -> EvidenceItem:
    """Is baseline error uneven across the image? Supporting evidence only:
    an uneven pattern has several possible causes, so it is never matched to
    an extrinsic axis by a fixed rule, and never exceeds Moderate."""
    item = EvidenceItem(CAT_SPATIAL, "Spatial asymmetry (baseline)", UNAVAILABLE, scope,
                        key="spatial", candidate_eligible=False)
    if spatial is None:
        item.observations.append("No baseline spatial breakdown is available.")
        return item
    if source:
        item.metrics.append(("Source", source))
    levels = []
    for group, labels, stats in (
        ("Horizontal", HORIZONTAL_REGIONS, spatial.horizontal),
        ("Vertical", VERTICAL_REGIONS, spatial.vertical),
    ):
        populated = [(l, stats[l].p95_px) for l in labels
                     if stats[l].n_points >= th.region_min_points and _finite(stats[l].p95_px)]
        item.metrics += [(f"{group} {l} P95", _px(v)) for l, v in populated]
        if len(populated) < 2:
            continue
        hi_label, hi = max(populated, key=lambda x: x[1])
        lo_label, lo = min(populated, key=lambda x: x[1])
        ratio = hi / lo if lo > 0 else float("inf")
        diff = hi - lo
        if diff >= th.asymmetry_min_diff_px and ratio >= th.asymmetry_moderate_ratio:
            levels.append(MODERATE)
        elif diff >= th.asymmetry_min_diff_px and ratio >= th.asymmetry_weak_ratio:
            levels.append(WEAK)
        else:
            levels.append(NOT_OBSERVED)
            continue
        item.observations.append(f"{group}: {hi_label} P95 {hi:.2f} px vs {lo_label} {lo:.2f} px ({ratio:.1f}×).")

    if not levels:
        item.strength = INSUFFICIENT
        item.observations.append(f"Fewer than two image regions have {th.region_min_points}+ edge points.")
        return item
    item.strength = max(levels, key=lambda s: _STRENGTH_RANK[s])
    if item.strength == NOT_OBSERVED:
        item.observations.append("Error is roughly even across image regions.")
    else:
        item.interpretations.append(
            "Error is unevenly distributed across the image. Possible contributors include extrinsic rotation, "
            "lens distortion / intrinsic error toward the image edges, uneven edge content in the scene, and timing "
            "error while turning. Spatial asymmetry alone does not identify which."
        )
    item.caveats.append("Supporting evidence only -- never used on its own to name an extrinsic axis.")
    if scope == SCOPE_SINGLE:
        item.caveats.append(SINGLE_FRAME_NOTE)
    return item


def assess_depth(spatial, scope: str, th: DiagnosticThresholds = DEFAULT_THRESHOLDS, source: str = "") -> EvidenceItem:
    """Does baseline error trend with distance? Supporting evidence only,
    capped at Moderate: a depth trend has several possible causes and is
    never mapped to "translation error" by rule."""
    item = EvidenceItem(CAT_DEPTH, "Depth dependence (baseline)", UNAVAILABLE, scope,
                        key="depth", candidate_eligible=False)
    if spatial is None:
        item.observations.append("No baseline depth breakdown is available.")
        return item
    if source:
        item.metrics.append(("Source", source))
    bins = [(l, spatial.depth_bins[l]) for l in DEPTH_BIN_LABELS]
    populated = [(l, s.p95_px) for l, s in bins if s.n_points >= th.region_min_points and _finite(s.p95_px)]
    item.metrics += [(f"{l} P95", _px(v)) for l, v in populated]
    if len(populated) < th.depth_min_bins:
        item.strength = INSUFFICIENT
        item.observations.append(
            f"Only {len(populated)} depth bin(s) have {th.region_min_points}+ edge points (need {th.depth_min_bins})."
        )
        return item

    values = [v for _, v in populated]
    steps = np.diff(values)
    noise = th.region_change_min_px
    first, last = values[0], values[-1]
    increasing = bool(np.all(steps >= -noise)) and last - first >= th.depth_min_diff_px
    decreasing = bool(np.all(steps <= noise)) and first - last >= th.depth_min_diff_px
    trend = "\n".join(f"{l}: {v:.2f} px" for l, v in populated)
    if increasing or decreasing:
        low, high = (first, last) if increasing else (last, first)
        ratio = high / low if low > 0 else float("inf")
        if ratio >= th.depth_moderate_ratio:
            item.strength = MODERATE
        elif ratio >= th.depth_weak_ratio:
            item.strength = WEAK
        else:
            item.strength = NOT_OBSERVED
    else:
        item.strength = NOT_OBSERVED

    if item.strength == NOT_OBSERVED:
        item.observations.append("No clear monotonic trend of error with distance.")
        return item
    direction = "increases" if increasing else "decreases"
    item.observations.append(f"Error {direction} with distance:\n{trend}")
    item.interpretations.append(
        "Depth-dependent alignment error observed. Possible contributors include extrinsic translation or "
        "rotation, intrinsic calibration, scene geometry, LiDAR sparsity at range, or synchronization."
    )
    item.caveats.append(
        "For reference only: under a pinhole model a pure translation offset gives pixel error that shrinks with "
        "distance (∝ 1/z) and a pure rotation offset gives roughly distance-independent error, but real edge error "
        "also depends on edge density and LiDAR sparsity -- a depth trend alone does not identify the cause."
    )
    item.caveats.append("Supporting evidence only -- never used on its own to name an extrinsic axis.")
    if scope == SCOPE_SINGLE:
        item.caveats.append(SINGLE_FRAME_NOTE)
    return item


def assess_temporal(mf, sync_limit_ms, perturbation=None, perturbation_sync_limit_ms=None,
                    th: DiagnosticThresholds = DEFAULT_THRESHOLDS) -> EvidenceItem:
    """Camera/LiDAR timestamp offsets. Never above Weak and never tied to an
    extrinsic axis: no temporal-offset sensitivity is computed anywhere, so
    offsets can only say "worth reviewing", not "this is the cause"."""
    item = EvidenceItem(CAT_TEMPORAL, "Temporal pairing quality", UNAVAILABLE, SCOPE_MULTI, key="temporal")
    no_sensitivity = (
        "No temporal-offset sensitivity was computed, so timestamp offsets alone cannot establish timing as a "
        "cause, and they are never used here to point at an extrinsic axis."
    )

    if mf is not None and mf.n_total:
        n_total, n_rej = mf.n_total, mf.n_sync_rejected
        item.metrics += [
            ("Signed Sync Median", _ms(mf.sync_offset_signed_median_ms, signed=True)),
            ("Absolute Sync Median", _ms(mf.sync_offset_median_ms)),
            ("Absolute Sync P95", _ms(mf.sync_offset_p95_ms)),
            ("Absolute Sync Max", _ms(mf.sync_offset_max_ms)),
            ("Sync Rejected Frames", _ratio_text(n_rej, n_total)),
        ]
        if not _finite(mf.sync_offset_p95_ms):
            item.strength = INSUFFICIENT
            item.observations.append("No camera/LiDAR pairing offsets were collected.")
            item.caveats.append(no_sensitivity)
            return item
        if sync_limit_ms:
            item.observations.append(
                f"{n_rej} / {n_total} frames rejected by configured {sync_limit_ms:.0f} ms sync limit."
            )
        else:
            item.observations.append("No sync limit was configured; no frame was rejected on sync alone.")
        review = (
            mf.sync_offset_p95_ms >= th.sync_review_abs_p95_ms
            or (n_rej / n_total) >= th.sync_review_rejected_ratio
        )
        signed, absolute = mf.sync_offset_signed_median_ms, mf.sync_offset_median_ms
        if _finite(signed) and _finite(absolute) and absolute > 1.0 and abs(signed) >= 0.5 * absolute:
            later = "later" if signed > 0 else "earlier"
            item.observations.append(
                f"Offsets have a consistent sign (signed median {signed:+.1f} ms: camera image {later} than LiDAR)."
            )
        if review:
            item.strength = WEAK
            item.observations.append(
                "Temporal alignment should be reviewed: absolute timestamp offsets are non-negligible in this "
                "evaluation set."
            )
            item.interpretations.append(
                "On a moving platform, a timing offset shifts edges much like a small rotation or translation error "
                "would; it may contribute alongside, or instead of, an extrinsic error."
            )
        else:
            item.strength = NOT_OBSERVED
            item.observations.append(
                f"Absolute timestamp offsets are small in this evaluation set "
                f"(P95 {mf.sync_offset_p95_ms:.1f} ms < {th.sync_review_abs_p95_ms:.0f} ms review level)."
            )
        item.caveats.append("Sync-rejected frames are excluded from every geometric metric.")
        item.caveats.append(no_sensitivity)
        return item

    if perturbation is not None and perturbation.mode == "multi_frame":
        n_rej = perturbation.n_sync_rejected
        n_total = n_rej + perturbation.n_frames_used
        item.metrics.append(("Sync Rejected Frames (perturbation set)", _ratio_text(n_rej, n_total)))
        if perturbation_sync_limit_ms:
            item.observations.append(
                f"{n_rej} / {n_total} frames rejected by configured {perturbation_sync_limit_ms:.0f} ms sync limit."
            )
        if n_total and n_rej / n_total >= th.sync_review_rejected_ratio:
            item.strength = WEAK
            item.observations.append(
                "Temporal alignment should be reviewed: a notable share of sampled frames exceeded the sync limit."
            )
        else:
            item.strength = INSUFFICIENT
            item.observations.append("Offset statistics need a Multi-frame evaluation run.")
        item.caveats.append(no_sensitivity)
        return item

    item.observations.append("Run Multi-frame evaluation to collect camera/LiDAR timestamp offsets.")
    return item


def _stability_from_values(values: np.ndarray) -> "tuple[float, float, int]":
    robust = compute_robust_stats(values)
    is_outlier, _ = flag_outliers_hampel(values, robust, k=DEFAULT_HAMPEL_K)
    spread = robust["mad"] / robust["median"] if robust["median"] > 0 else float("nan")
    return robust["median"], spread, int(is_outlier.sum())


def assess_stability(mf, perturbation=None, th: DiagnosticThresholds = DEFAULT_THRESHOLDS) -> EvidenceItem:
    """How much error varies frame to frame at the fixed baseline T. A single
    fixed extrinsic offset tends to affect every frame similarly; large spread
    or isolated bad frames point (weakly) toward scene-dependent effects.
    Capped at Moderate."""
    item = EvidenceItem(CAT_STABILITY, "Scene-dependent variation (multi-frame stability)", UNAVAILABLE,
                        SCOPE_MULTI, key="stability")

    if mf is not None and not mf.reason and mf.n_valid:
        n_valid, n_outlier = mf.n_valid, mf.n_outlier
        median, spread = mf.median_px, (mf.mad_px / mf.median_px if mf.median_px > 0 else float("nan"))
        outlier_ratio = mf.outlier_ratio
        item.metrics += [
            ("Valid frames", _ratio_text(n_valid, mf.n_total)),
            ("Median (frame mean)", _px(median)),
            ("MAD", _px(mf.mad_px)),
            ("STD", _px(mf.std_px)),
            ("P95 (frame mean)", _px(mf.p95_px)),
            ("Outlier frames", _ratio_text(n_outlier, n_valid)),
            ("Failure ratio", f"{mf.failure_ratio * 100:.1f} %" if _finite(mf.failure_ratio) else "—"),
        ]
        failure_ratio = mf.failure_ratio
        source_note = ""
    elif perturbation is not None and perturbation.mode == "multi_frame" and perturbation.axes:
        baseline = next(iter(perturbation.axes.values())).baseline
        values = np.array([m.p95_px for m in baseline.per_frame.values() if m.ok and _finite(m.p95_px)]) \
            if baseline is not None else np.array([])
        n_valid = int(values.size)
        if n_valid < th.stability_min_frames:
            item.strength = INSUFFICIENT
            item.observations.append(f"Only {n_valid} valid baseline frames in the perturbation set.")
            return item
        median, spread, n_outlier = _stability_from_values(values)
        outlier_ratio = n_outlier / n_valid
        item.metrics += [
            ("Valid frames", str(n_valid)),
            ("Median (frame P95)", _px(median)),
            ("Outlier frames", _ratio_text(n_outlier, n_valid)),
        ]
        failure_ratio = float("nan")
        source_note = "Derived from the perturbation baseline's per-frame P95 (no Multi-frame evaluation run)."
    elif mf is not None:
        item.strength = INSUFFICIENT
        item.observations.append(mf.reason or "Multi-frame evaluation produced no valid frames.")
        return item
    else:
        item.observations.append("Run Multi-frame evaluation (or Multi-frame perturbation) to assess stability.")
        return item

    if source_note:
        item.caveats.append(source_note)
    if n_valid < th.stability_min_frames:
        item.strength = INSUFFICIENT
        item.observations.append(f"Only {n_valid} valid frames (need {th.stability_min_frames}).")
        return item

    if _finite(spread) and spread >= th.stability_spread_moderate:
        item.strength = MODERATE
        item.observations.append(f"Errors vary substantially across frames (MAD / median = {spread:.2f}).")
    elif outlier_ratio >= th.stability_outlier_moderate_ratio:
        item.strength = MODERATE
        item.observations.append(
            f"Median error is stable but several isolated frames have high error ({n_outlier} / {n_valid} outliers)."
        )
    elif (_finite(spread) and spread >= th.stability_spread_weak) or outlier_ratio >= th.stability_outlier_weak_ratio:
        item.strength = WEAK
        if outlier_ratio >= th.stability_outlier_weak_ratio:
            item.observations.append(
                f"Median error is stable but a few isolated frames have high error ({n_outlier} / {n_valid} outliers)."
            )
        else:
            item.observations.append(f"Moderate frame-to-frame spread (MAD / median = {spread:.2f}).")
    else:
        item.strength = NOT_OBSERVED
        item.observations.append("Frame-to-frame error is stable.")

    if item.strength in CANDIDATE_STRENGTHS:
        item.interpretations.append(
            "Part of the error may depend on scene content (edge density, dynamic objects, "
            "lighting, range distribution) or timing, rather than on one fixed extrinsic offset that would affect "
            "every frame similarly."
        )
    if _finite(failure_ratio) and failure_ratio >= th.stability_failure_note_ratio:
        item.caveats.append(
            f"{failure_ratio:.0%} of sampled frames failed evaluation; stability is judged only over frames that succeeded."
        )
    return item


def assess_calibration(ctx: Optional[CalibrationContext], th: DiagnosticThresholds = DEFAULT_THRESHOLDS) -> EvidenceItem:
    """Step 6 leave-one-out, exactly as Step 6 already computed it. The solver
    is never re-run here; without a stored result this is Unavailable."""
    item = EvidenceItem(CAT_CALIBRATION, "Calibration-scene stability (Step 6 leave-one-out)", UNAVAILABLE,
                        SCOPE_CALIBRATION, key="calibration")
    if ctx is None or ctx.unavailable_reason or not ctx.loo:
        reason = (ctx.unavailable_reason if ctx is not None and ctx.unavailable_reason
                  else "Step 6 leave-one-out results are not available (needs 3+ scenes solved in Step 6).")
        item.observations.append(f"Calibration-scene stability: unavailable. {reason}")
        return item

    if _finite(ctx.rmse_m):
        item.metrics.append(("Target RMSE", f"{ctx.rmse_m * 1000:.2f} mm"))
    n_scenes = ctx.n_scenes or len(ctx.loo)
    item.metrics.append(("Scenes", str(n_scenes)))

    def severity(m):
        return max(m["shift_mm"] / th.loo_shift_warn_mm, m["rotation_deg"] / th.loo_rotation_warn_deg)

    ranked = sorted(ctx.loo.items(), key=lambda kv: severity(kv[1]), reverse=True)
    for sid, m in ranked[:5]:
        item.metrics.append((f"Without {sid}", f"{m['shift_mm']:.0f} mm / {m['rotation_deg']:.2f}°"))
    worst = severity(ranked[0][1])
    if worst >= 2.0:
        item.strength = STRONG
    elif worst >= 1.0:
        item.strength = MODERATE
    elif worst >= 0.5:
        item.strength = WEAK
    else:
        item.strength = NOT_OBSERVED

    if item.strength == NOT_OBSERVED:
        sid, m = ranked[0]
        item.observations.append(
            f"Removing any single scene moves the solution by at most {m['shift_mm']:.0f} mm / {m['rotation_deg']:.2f}° "
            f"(largest: {sid})."
        )
        item.caveats.append("Uses Step 6 results already computed; the solver is not re-run here.")
        return item

    for sid, m in ranked:
        if severity(m) >= 0.5:
            item.observations.append(
                f"Removing scene {sid}: translation shift {m['shift_mm']:.0f} mm, rotation shift {m['rotation_deg']:.2f}°."
            )
    item.observations.append("Calibration solution is sensitive to scene selection.")
    if n_scenes < th.loo_min_scenes_for_strong:
        item.strength = cap_strength(item.strength, MODERATE)
        item.caveats.append(
            f"With only {n_scenes} scenes each leave-one-out refit rests on {n_scenes - 1}, so large shifts are expected."
        )
    item.interpretations.append(
        "Possible contributors: a scene with a detection/correspondence error, limited board-pose diversity "
        "(see Step 6 coverage), or too few scenes."
    )
    item.caveats.append("Uses Step 6 results already computed; the solver is not re-run here.")
    return item


# -------------------------------------------------------------------- report


def _rank_key(item: EvidenceItem):
    consistency = item.consistency if _finite(item.consistency) else -1.0
    return (-_STRENGTH_RANK[item.strength], item.mixed, -consistency)


def build_diagnostic_report(inputs: DiagnosticInputs, th: DiagnosticThresholds = DEFAULT_THRESHOLDS) -> DiagnosticReport:
    """Interpret `inputs` into a report. Pure: reads the given results, calls
    nothing that projects, scores, perturbs or solves, and modifies nothing."""
    pert = inputs.perturbation
    pert_multi = pert is not None and pert.mode == "multi_frame" and bool(pert.axes)
    mf = inputs.multiframe
    has_mf = mf is not None and bool(mf.n_total)
    has_single = (pert is not None and pert.mode != "multi_frame" and bool(pert.axes)) \
        or inputs.current_frame_spatial is not None \
        or (inputs.current_frame is not None and getattr(inputs.current_frame, "ok", False))
    if pert_multi or has_mf:
        mode = MODE_MULTI
    elif has_single:
        mode = MODE_SINGLE
    else:
        mode = MODE_NONE

    evidence: list = []
    notes = [DISCLAIMER]

    # --- extrinsic axes -------------------------------------------------
    axis_items: dict = {}
    if pert is not None and pert.axes:
        pert_mode = pert.mode
        for key in (*ROTATION_AXES, *TRANSLATION_AXES):
            ax = pert.axes.get(key)
            axis_items[key] = assess_axis(ax, pert_mode, th, "grid") if ax is not None else \
                _unavailable_axis(key, "Not evaluated (perturbation run was cancelled before this axis).")
        if pert.cancelled:
            notes.append("Perturbation run was cancelled; some axes were not evaluated.")
    else:
        reason = (pert.reason if pert is not None and pert.reason
                  else "Run Perturbation Sensitivity to probe this parameter.")
        for key in (*ROTATION_AXES, *TRANSLATION_AXES):
            axis_items[key] = _unavailable_axis(key, reason)

    fine = inputs.fine_scan
    fine_item = None
    if fine is not None and fine.baseline is not None:
        fine_item = assess_axis(fine, inputs.fine_scan_mode, th, "fine")
        grid_item = axis_items.get(fine.axis)
        grid_axis = pert.axes.get(fine.axis) if pert is not None and pert.axes else None
        if grid_item is not None and grid_axis is not None:
            _cross_check_fine(grid_item, fine_item, grid_axis, fine)

    for key in ROTATION_AXES:
        evidence.append(axis_items[key])
        if fine_item is not None and fine.axis == key:
            evidence.append(fine_item)
    for key in TRANSLATION_AXES:
        evidence.append(axis_items[key])
        if fine_item is not None and fine.axis == key:
            evidence.append(fine_item)

    # --- baseline spatial / depth (supporting) ----------------------------
    # Every axis of one grid run shares the same baseline point; take the
    # first one that carries a spatial breakdown.
    pert_baseline_spatial = None
    if pert is not None and pert.axes:
        pert_baseline_spatial = next(
            (ax.baseline.spatial for ax in pert.axes.values()
             if ax.baseline is not None and ax.baseline.spatial is not None),
            None,
        )
    spatial, scope, source = None, SCOPE_MULTI, ""
    if pert_multi and pert_baseline_spatial is not None:
        spatial, source = pert_baseline_spatial, "Perturbation baseline, pooled over the multi-frame set"
    elif pert is not None and pert.mode != "multi_frame" and pert_baseline_spatial is not None:
        spatial, scope, source = pert_baseline_spatial, SCOPE_SINGLE, "Perturbation baseline, current frame"
    if spatial is None and inputs.current_frame_spatial is not None:
        spatial, scope, source = inputs.current_frame_spatial, SCOPE_SINGLE, "Current-frame Edge Alignment"
    evidence.append(assess_spatial(spatial, scope, th, source))
    evidence.append(assess_depth(spatial, scope, th, source))

    # --- temporal / stability / calibration ------------------------------
    temporal = assess_temporal(mf, inputs.multiframe_sync_limit_ms, pert, inputs.perturbation_sync_limit_ms, th)
    evidence.append(temporal)
    evidence.append(assess_stability(mf, pert, th))
    evidence.append(assess_calibration(inputs.calibration, th))

    if temporal.strength == WEAK:
        for item in evidence:
            if item.category == CAT_ROTATION and item.strength in CANDIDATE_STRENGTHS:
                item.caveats.append(
                    "Timestamp offsets in this set are non-negligible; on a moving platform a timing error can "
                    "mimic a rotation offset."
                )

    # --- candidates & headline -------------------------------------------
    candidates = sorted((i for i in evidence if i.is_candidate), key=_rank_key)
    if candidates and candidates[0].strength in (STRONG, MODERATE):
        top = candidates[0]
        headline = f"Strongest observed evidence: {top.title} ({top.strength_label})"
    elif candidates:
        top = candidates[0]
        headline = f"Only weak evidence observed. Strongest: {top.title} ({top.strength_label})"
    elif mode == MODE_SINGLE:
        headline = "Single-frame diagnostic only -- Root Cause Candidates need Multi-frame results."
    else:
        headline = "No candidate with at least weak evidence in the available results."

    if mode == MODE_SINGLE:
        notes.append(SINGLE_FRAME_NOTE)
    if pert_multi and has_mf:
        notes.append("Multi-frame evaluation and Perturbation Sensitivity may use different sampled frame sets.")
    notes.append(UNITS_NOTE)
    notes += list(inputs.notes)

    return DiagnosticReport(mode=mode, headline=headline, evidence=evidence, candidates=candidates, notes=notes)


# ----------------------------------------------------------------- plain text


def format_report_text(report: DiagnosticReport) -> str:
    """Plain-text rendering for copying out of the UI."""
    mode_text = {MODE_MULTI: "Multi-frame", MODE_SINGLE: "Single-frame diagnostic", MODE_NONE: "Insufficient data"}
    lines = ["Diagnostic Evidence", "=" * 40, f"Mode: {mode_text.get(report.mode, report.mode)}"]
    lines += [f"* {n}" for n in report.notes]
    lines += ["", report.headline, "", "Root Cause Candidates", "-" * 40]
    lines.append("Ordered by evidence strength and frame consistency; order is not a probability.")
    if not report.candidates:
        lines.append("(none)")
    for i, item in enumerate(report.candidates, start=1):
        lines.append(f"{i}. {item.title} -- Evidence: {item.strength_label}")
        for text in item.interpretations:
            lines.append(f"   {text}")

    for category in CATEGORY_ORDER:
        items = [i for i in report.evidence if i.category == category]
        if not items:
            continue
        lines += ["", category, "-" * 40]
        for item in items:
            lines.append(f"{item.title}")
            lines.append(f"  Evidence: {item.strength_label}")
            width = max((len(k) for k, _ in item.metrics), default=0)
            for k, v in item.metrics:
                lines.append(f"  {k.ljust(width)}  {v}")
            if item.observations:
                lines.append("  Observed:")
                for text in item.observations:
                    for j, part in enumerate(text.split("\n")):
                        lines.append(("    - " if j == 0 else "      ") + part)
            if item.interpretations:
                lines.append("  Possible interpretation:")
                lines += [f"    - {t}" for t in item.interpretations]
            if item.caveats:
                lines.append("  Caveats:")
                lines += [f"    - {t}" for t in item.caveats]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"

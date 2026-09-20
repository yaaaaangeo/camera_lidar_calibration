"""Independent Hold-out Validation.

Perturbation Sensitivity and Fine Scan (`gui.core.evaluation.perturbation`)
answer "is there a nearby T that scores better than baseline on *this*
sampled data" -- and that data is exactly what a person then reads to pick a
candidate offset. A candidate chosen that way can look good purely because it
happened to fit quirks of the frames it was searched against, the same way a
model can look good on the data it was trained on. This module answers a
narrower, harder question: does the *same, already-chosen* candidate still
look better than baseline on data it was never searched against.

The one rule everything here exists to enforce: **hold-out data is never used
to fit T or to search for a candidate.** There is no grid here, no delta
range, no "try nearby offsets" -- `evaluate_holdout` scores exactly two
points, baseline and one caller-supplied `CandidateSpec`, both already fixed
before this module is ever called. If a result here looks bad, the fix is to
go back to Discovery and pick a different candidate from *that* data, then
run a fresh hold-out on the new candidate -- never to loosen or re-run this
module until a candidate happens to pass.

This module also has to answer questions a first version of it did not: "is
the hold-out data actually independent" (a guard-band leakage check, a
same-bag-copy signature check, a cross-bag sensor-time-range note -- see
`check_overlap` / `partition_independent_frames` / `signatures_suspiciously_similar`),
"how uncertain is the improvement it measured" (a Non-overlapping Block
Bootstrap over paired per-frame deltas -- see `evaluate_holdout`'s
`bootstrap` field), and "has this candidate, on this data, already been seen
before" (project-persistent exposure tracking, in two layers -- a
content-based Source identity that a copied bag cannot dodge
(`resolve_source_identity`), and, within a source, an Evaluation Slice
identity that treats a heavily-overlapping resample as reuse rather than a
fresh dataset (`classify_exposure` / `HoldoutSourceExposure` /
`HoldoutSliceExposure`)). Exposure history, and the monotonic
`discovery_revision` counter `classify_exposure` compares against it (see
`CandidateProvenance`'s docstring), both survive an app restart -- they live
in `Project.validation_history` (serialized/deserialized at the bottom of
this module), not merely in memory, so switching candidates across a
restart is caught exactly as it would be within one session. All of this
exists because a single point estimate on a small, possibly-leaky,
possibly-peeked-at frame set is easy to over-read; none of it turns into a
pass/fail score -- see `STATUS_INDEPENDENT` / `EXPOSED_STATUSES` and
`BootstrapResult` for why.

`CandidateSpec` deliberately carries no R/t of its own: it names an axis and
a delta (exactly the vocabulary `gui.core.evaluation.perturbation` already
uses for Quick/Full/Fine results), and `evaluate_holdout` derives the
candidate transform from whatever baseline R/t the caller passes in, the same
way `perturbation.run_trial_point` already does. This keeps "what the
candidate is" (a probe relative to baseline) separate from "what baseline
currently is" (read fresh from the caller each time), so a hold-out result
can never be mistaken for, or silently drift from, a new calibration.

Frame preparation is intentionally *not* reimplemented here: hold-out frames
are loaded once with `perturbation.prepare_frames` (image, cloud, timestamps,
Canny/orientation cache -- all per frame, once), and `evaluate_holdout` then
runs exactly two trials (baseline, candidate) against that fixed set with
`perturbation.run_trial_point`, the identical trial-scoring code Perturbation
Sensitivity and Fine Scan already use. Nothing here projects a cloud, runs
Canny, or scores an edge a second way.

Nothing in this module writes to a `Solution`, a project, or a file. Locking
in a "better" candidate -- editing the extrinsic, re-solving, saving a YAML --
is deliberately left to a human, and to a later feature, not this one. There
is also no p-value, no pass/fail gate, and no 0-100 score anywhere here --
`BootstrapResult` reports a confidence interval as numbers to read, not a
verdict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from gui.core.bag_reader import BagSignature
from gui.core.evaluation.edge_alignment import EdgeAlignmentParams
from gui.core.evaluation.perturbation import (
    AXIS_LABELS,
    DEFAULT_LOCAL_MIN_TOLERANCE_PX,
    ROTATION_AXES,
    TRANSLATION_AXES,
    AxisSensitivity,
    perturb_rotation,
    perturb_translation,
    run_trial_point,
    summarize_axis,
)
from gui.core.evaluation.spatial_analysis import DEPTH_BIN_LABELS, HORIZONTAL_REGIONS, VERTICAL_REGIONS

CANDIDATE_SOURCES = ("quick", "full", "fine")

# Every constant below is a configurable heuristic default for this project,
# not an empirically validated universal cut-off for any particular sensor,
# mounting, or dataset -- see DiagnosticThresholds' docstring for the same
# caveat applied to the rest of Diagnostic Evidence.
DEFAULT_EXCLUSION_MARGIN_S = 2.0        # temporal guard band around Discovery frames
DEFAULT_MIN_INDEPENDENT_FRAMES = 10     # below this, "independence limited" is flagged
DEFAULT_SIGNATURE_SIZE_TOLERANCE = 0.02          # relative file-size difference
DEFAULT_SIGNATURE_DURATION_TOLERANCE_S = 1.0
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_BLOCK_S = 2.0
DEFAULT_MIN_BOOTSTRAP_BLOCKS = 5

DEFAULT_BOOTSTRAP_METHOD = "non_overlapping_block"
SUPPORTED_BOOTSTRAP_METHODS = (DEFAULT_BOOTSTRAP_METHOD,)
# Only one method exists today. To add "moving_block" or "stationary_bootstrap"
# later: add its name here and a matching branch in _run_block_bootstrap --
# BootstrapResult.method already exists so callers can tell which one produced
# a given result.

# Validation status -- whether a HoldoutResult still describes a candidate
# nobody has seen validated any other way yet. Two independent things can
# make a result Exploratory instead of Independent, and they are told apart
# by which message is shown:
#   - the SAME (or heavily overlapping) Hold-out slice, on the SAME source
#     bag, was re-evaluated against a DIFFERENT candidate after its first
#     result was shown (candidate switching -- see CandidateIdentity /
#     HoldoutSourceExposure / HoldoutSliceExposure / classify_exposure), or
#   - Discovery (Perturbation/Fine Scan) was run again after a result was
#     shown, whether or not the slice is reused (see
#     step_verify.VerifyStep._holdout_status's live staleness check).
# A known source bag but a genuinely unseen slice of it is its own status
# (STATUS_INDEPENDENT_NEW_SLICE) -- independent, but worth flagging as reuse
# of an already-touched bag rather than a wholly separate recording.
# Not stored on HoldoutResult itself -- exposure is something that can only
# be known *after* a result already exists, using session history the result
# has no way to see -- so it is tracked by the caller and passed alongside
# the result wherever it is displayed or fed to Diagnostic Evidence.
STATUS_INDEPENDENT = "independent"
STATUS_INDEPENDENT_NEW_SLICE = "independent_new_slice"
STATUS_INDEPENDENT_REPEATED = "independent_repeated"
STATUS_EXPOSED_CANDIDATE_CHANGED = "exposed_candidate_changed"
STATUS_EXPOSED_DISCOVERY_UPDATED = "exposed_discovery_updated"
EXPOSED_STATUSES = (STATUS_EXPOSED_CANDIDATE_CHANGED, STATUS_EXPOSED_DISCOVERY_UPDATED)
STATUS_LABELS = {
    STATUS_INDEPENDENT: "Independent",
    STATUS_INDEPENDENT_NEW_SLICE: "Independent — new unseen slice",
    STATUS_INDEPENDENT_REPEATED: "Independent — repeated evaluation",
    STATUS_EXPOSED_CANDIDATE_CHANGED: "Exploratory — candidate changed after this hold-out was viewed",
    STATUS_EXPOSED_DISCOVERY_UPDATED: "Exploratory — discovery was updated after this hold-out was viewed",
}


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


# ------------------------------------------------------------------ candidate


@dataclass(frozen=True)
class CandidateSpec:
    """One locked candidate: a single axis, a single delta, and where it came
    from. Never a range, never a search -- see the module docstring. `source`
    is purely descriptive (which Discovery run this delta was read off of);
    it changes nothing about how the candidate is scored."""

    axis: str
    delta: float
    source: str  # "quick" | "full" | "fine"

    def __post_init__(self):
        if self.axis not in (*ROTATION_AXES, *TRANSLATION_AXES):
            raise ValueError(f"unknown axis: {self.axis!r}")
        if self.delta == 0.0:
            raise ValueError("a candidate must be a non-zero offset from baseline (0.0 is baseline, not a candidate)")
        if self.source not in CANDIDATE_SOURCES:
            raise ValueError(f"unknown candidate source: {self.source!r}")

    @property
    def unit(self) -> str:
        return "deg" if self.axis in ROTATION_AXES else "mm"

    @property
    def label(self) -> str:
        return AXIS_LABELS[self.axis]

    @property
    def kind(self) -> str:
        return "rotation" if self.axis in ROTATION_AXES else "translation"

    def describe(self) -> str:
        unit = "°" if self.unit == "deg" else " mm"
        return f"{self.label} {self.delta:+.3f}{unit} ({self.source.capitalize()})"


def fingerprint_extrinsic(R: np.ndarray, t: np.ndarray) -> str:
    """A short, deterministic label for one (R, t) pair -- not a security
    hash, just enough to eyeball "is this the same baseline T_original as
    before" across two provenance records without printing a full matrix.
    Rounded before hashing so float noise from a copy/recompute of the same
    T does not silently produce a different fingerprint."""
    data = np.concatenate([np.asarray(R, dtype=np.float64).ravel(), np.asarray(t, dtype=np.float64).ravel()])
    data = np.round(data, 9)
    return hashlib.sha1(data.tobytes()).hexdigest()[:12]


@dataclass(frozen=True, eq=False)
class CandidateIdentity:
    """The *physical* identity of one locked candidate -- does this describe
    the same transform, independent of which session, which Discovery run,
    or which search mode (Quick/Full/Fine) produced it. Equality
    (`__eq__`/`__hash__` below) is deliberately based on
    `(baseline_fingerprint, candidate_fingerprint)` alone: two candidates
    with the same baseline T and the same resulting candidate T *are* the
    same candidate, even across an app restart where nothing else about the
    session survived, or between a Quick and a Fine Scan result that happen
    to land on the exact same delta.

    `axis`/`delta` are carried for provenance/sanity (human-readable, and a
    guard against a fingerprint collision being misread as sameness) but
    never participate in equality: if they ever disagreed with the
    fingerprints (which should not happen, since the fingerprints are
    derived *from* axis+delta+baseline), the fingerprints -- the actual
    transform -- are what "the same candidate" means here, not the label.

    Deliberately does NOT include `discovery_revision` or `source` -- those
    are *provenance* about how this candidate was found, not part of what
    candidate it physically is (see `CandidateProvenance`). Provenance data
    inside an equality-bearing identity is exactly what made exposure
    tracking fail to recognise the same candidate across an app restart in
    an earlier version of this module -- and, separately, exactly what made
    the "has Discovery moved on" check itself unreliable across a restart
    until `discovery_revision` became a persisted project counter instead of
    a session-local one (see `CandidateProvenance`'s docstring).
    """

    baseline_fingerprint: str
    candidate_fingerprint: str
    axis: str
    delta: float

    def __eq__(self, other) -> bool:
        if not isinstance(other, CandidateIdentity):
            return NotImplemented
        return (self.baseline_fingerprint, self.candidate_fingerprint) == \
            (other.baseline_fingerprint, other.candidate_fingerprint)

    def __hash__(self) -> int:
        return hash((self.baseline_fingerprint, self.candidate_fingerprint))


@dataclass(frozen=True)
class CandidateProvenance:
    """Where a candidate came from -- Discovery search-run metadata, kept
    separate from `CandidateIdentity` because it must never affect "is this
    the same candidate" (see `CandidateIdentity`'s docstring).

    `discovery_revision` is a *project-persistent*, monotonically increasing
    counter (see `Project.validation_history`'s top-level `discovery_revision`
    key, restored on load, never reset by an app restart) -- it advances by
    exactly 1 each time a Perturbation Grid or Fine Scan run *successfully*
    produces a result (never on a failed/cancelled run, never on a Hold-out
    run, never on a mere UI candidate-combo selection; see
    step_verify.VerifyStep._note_discovery_success). `classify_exposure`
    compares it to tell whether Discovery has moved on *since a slice's
    reveal*, even when the newly-produced candidate happens to be numerically
    identical to the one first revealed -- re-running Fine Scan and landing
    on the exact same delta again still counts as "Discovery moved on",
    because tuning happened after the reveal.

    This field used to be a session-local counter that reset to 0 on every
    restart (`discovery_generation`), which made the "has Discovery moved
    on" check silently miss contamination across a restart: a candidate
    revealed at generation 7, then the app restarts (counter resets to 0),
    then Fine Scan runs once more (counter becomes 1) -- `1 > 7` is False,
    so the check found nothing even though Discovery genuinely ran again.
    Making the counter persistent instead of session-local is the fix (see
    `VALIDATION_HISTORY_SCHEMA_VERSION`'s migration-policy comment for how an
    older project's history is treated).
    """

    source: str  # "quick" | "full" | "fine"
    discovery_revision: int


def candidate_identity(baseline_R: np.ndarray, baseline_t: np.ndarray, candidate: CandidateSpec) -> CandidateIdentity:
    """Derive a `CandidateIdentity` from a baseline T and a `CandidateSpec`,
    the same way `evaluate_holdout` itself derives the transform actually
    scored (`perturb_rotation`/`perturb_translation`), so the identity always
    describes the real candidate T, not just its axis/delta label."""
    if candidate.kind == "rotation":
        cand_R, cand_t = perturb_rotation(baseline_R, candidate.axis, candidate.delta), baseline_t
    else:
        cand_R, cand_t = baseline_R, perturb_translation(baseline_t, candidate.axis, candidate.delta)
    return CandidateIdentity(
        baseline_fingerprint=fingerprint_extrinsic(baseline_R, baseline_t),
        candidate_fingerprint=fingerprint_extrinsic(cand_R, cand_t),
        axis=candidate.axis, delta=candidate.delta,
    )


def candidate_provenance(candidate: CandidateSpec, discovery_revision: int) -> CandidateProvenance:
    """Trivial, but kept as a function (not just inline construction) so
    every caller building both halves of a candidate reveal reads the same
    way: `candidate_identity(...)` + `candidate_provenance(...)`.
    `discovery_revision` should be the project-persistent counter's current
    value (see `CandidateProvenance`'s docstring), not a session-local one.
    """
    return CandidateProvenance(source=candidate.source, discovery_revision=discovery_revision)


DEFAULT_EXPOSURE_SLICE_OVERLAP_RATIO = 0.5
# Configurable heuristic default -- see resolve_source_identity/classify_exposure.
# Not a validated cut-off for "how much reuse counts as the same hold-out".


def _signature_canonical_key(signature: BagSignature) -> str:
    canonical = (
        signature.file_size_bytes, round(signature.duration_s, 3), signature.n_topics,
        signature.total_message_count, signature.first_lidar_timestamp_ns, signature.last_lidar_timestamp_ns,
    )
    return "sig:" + hashlib.sha1(repr(canonical).encode("utf-8")).hexdigest()[:16]


def _path_key(bag: str) -> str:
    try:
        resolved = str(Path(bag).resolve())
    except OSError:
        resolved = bag
    return "path:" + resolved


def slice_timestamp_hash(usable_timestamps) -> str:
    ts_bytes = np.asarray(sorted(int(t) for t in usable_timestamps), dtype=np.int64).tobytes()
    return hashlib.sha1(ts_bytes).hexdigest()[:16]


def slice_identity(source_id: str, usable_timestamps) -> str:
    """Identity of one *evaluation slice* within a source: the exact usable
    timestamp set actually evaluated. Two slices with the same
    `source_identity` differing only in which frames were sampled (a
    different Start/End range, a different Frames count) get different
    slice identities here -- but `classify_exposure` below still treats a
    heavily-overlapping slice as exposure, not as a fresh independent one
    (see `DEFAULT_EXPOSURE_SLICE_OVERLAP_RATIO`); this function alone is
    only the *exact*-match half of that.
    """
    return f"{source_id}|{slice_timestamp_hash(usable_timestamps)}"


@dataclass(frozen=True)
class HoldoutSliceExposure:
    """One revealed slice's record -- append-only once written (see
    `record_exposure`): never edited, never removed, even when a later
    reveal is judged to overlap it.

    `candidate` (physical transform identity) and `candidate_provenance`
    (source, persistent discovery revision) are kept as two separate fields
    for exactly the reason `CandidateIdentity`'s docstring gives: "is this
    the same candidate" (`candidate` equality) and "has Discovery moved on
    since this reveal" (`candidate_provenance.discovery_revision`, the
    project-persistent revision this slice was revealed against -- see
    `CandidateProvenance`'s docstring) are different questions, and
    conflating them broke both across an app restart.
    """

    slice_identity: str
    timestamps: tuple             # sorted usable timestamps that were revealed
    candidate: CandidateIdentity
    candidate_provenance: CandidateProvenance
    reveal_margin_s: float        # the guard band in effect at reveal time -- see the module docstring's note


@dataclass
class HoldoutSourceExposure:
    """One Hold-out *source* bag's exposure history -- every slice of it
    ever revealed to a person in this session, oldest first, never deleted
    or overwritten (see the first-reveal note under `record_exposure`)."""

    source_identity: str
    signature: Optional[BagSignature] = None
    revealed_slices: list = field(default_factory=list)   # list[HoldoutSliceExposure]


def resolve_source_identity(registry: dict, bag: str, signature: Optional[BagSignature]) -> str:
    """The `HoldoutSourceExposure` key `bag`/`signature` belongs under --
    preferring an existing entry in `registry` this source is "suspiciously
    similar" to (`signatures_suspiciously_similar`) over minting a new one,
    so a byte-identical copy at a different path ('holdout.bag' vs
    'holdout_copy.bag') folds into the SAME source instead of starting a
    fresh, falsely-independent exposure history (see
    `signatures_suspiciously_similar`'s own tolerances for what counts as
    "the same").

    Falls back to a path-based key when `signature` is None (e.g. a unit
    test, or the bag could not be inspected) -- two different paths are
    never folded together without a signature to compare, since a path
    alone cannot establish they are the same recording.
    """
    if signature is None:
        return _path_key(bag)
    exact = _signature_canonical_key(signature)
    if exact in registry:
        return exact
    for key, source in registry.items():
        if source.signature is not None and signatures_suspiciously_similar(source.signature, signature):
            return key
    return exact


def _slice_overlap(revealed_timestamps, new_timestamps, margin_ns: int) -> "tuple[int, float, float]":
    """(overlap_count, ratio_of_new_covered_by_revealed, ratio_of_revealed_covered_by_new)
    -- "covered" means an exact match OR a near-match within `margin_ns` (see
    `_timestamps_within_margin`), not exact equality alone: a Hold-out
    result already seen at 10.000s should not let 10.033s be sampled later
    and treated as an unseen validation frame."""
    covered_new = _timestamps_within_margin(revealed_timestamps, new_timestamps, margin_ns)
    covered_old = _timestamps_within_margin(new_timestamps, revealed_timestamps, margin_ns)
    ratio_new = len(covered_new) / len(new_timestamps) if len(new_timestamps) else 0.0
    ratio_old = len(covered_old) / len(revealed_timestamps) if len(revealed_timestamps) else 0.0
    return len(covered_new), ratio_new, ratio_old


def classify_exposure(
    source: Optional[HoldoutSourceExposure],
    new_slice_id: str,
    new_timestamps,
    candidate_id: CandidateIdentity,
    candidate_prov: CandidateProvenance,
    reveal_margin_s: float,
    overlap_ratio_threshold: float = DEFAULT_EXPOSURE_SLICE_OVERLAP_RATIO,
) -> "tuple[str, Optional[HoldoutSliceExposure]]":
    """The validation status a Hold-out result about to be shown should
    carry, given `source` (this bag's exposure history, `None`/empty if
    never revealed before) and the slice/candidate identity of the result
    now being shown. Read-only: decides nothing about whether to record
    anything (see `record_exposure`/`determine_and_record_exposure`).
    Returns `(status, matched_slice)` -- `matched_slice` is whichever
    previously-revealed slice this reveal was judged against (`None` for a
    brand-new source or a genuinely new, non-overlapping slice), so the
    caller knows whether a *new* slice record needs appending.

    A new slice counts as the same *exposed* slice as an already-revealed
    one when it matches exactly (`new_slice_id`), or when it overlaps enough
    -- `_slice_overlap`'s ratio (from either side) at or above
    `overlap_ratio_threshold`, where "overlap" also counts near-timestamp
    matches within `reveal_margin_s` (conceptually the same guard-band idea
    as Discovery Exclusion Margin, but purpose-distinct -- see the module
    docstring). All revealed slices are checked; the first sufficiently-
    overlapping one is used as the anchor (there are typically only a
    handful per source in one session).

      - No source / no revealed slices yet -> STATUS_INDEPENDENT.
      - Source known, but this slice doesn't match/overlap any revealed
        slice -> STATUS_INDEPENDENT_NEW_SLICE: independent, but worth
        flagging as reusing an already-touched bag (see the caller's caveat).
      - Matches/overlaps a revealed slice, and the persistent Discovery
        revision has advanced since that slice's reveal
        (`candidate_prov.discovery_revision` newer than the matched slice's
        own) -> STATUS_EXPOSED_DISCOVERY_UPDATED, ALWAYS -- even when
        `candidate_id` is numerically identical to that slice's candidate.
        Re-running Fine Scan after a reveal and landing on the exact same
        delta again is still tuning that happened *after* the reveal, so it
        still counts as exposure; revision-staleness is checked first and
        overrides candidate-equality for exactly this reason (see
        `CandidateProvenance`'s docstring).
      - Matches/overlaps, revision unchanged, `candidate_id` equal to that
        slice's (see `CandidateIdentity`'s fingerprint-only equality -- true
        across an app restart, or between two Discovery search modes that
        happened to land on the same delta) -> STATUS_INDEPENDENT_REPEATED.
      - Matches/overlaps, revision unchanged, different candidate -- i.e.
        picked from Discovery results that already existed at that slice's
        reveal -> STATUS_EXPOSED_CANDIDATE_CHANGED: candidate switching
        after peeking, on the same (or overlapping) hold-out data.
    """
    if source is None or not source.revealed_slices:
        return STATUS_INDEPENDENT, None

    margin_ns = max(int(round(reveal_margin_s * 1e9)), 0)
    matched = next((s for s in source.revealed_slices if s.slice_identity == new_slice_id), None)
    if matched is None:
        for s in source.revealed_slices:
            _, ratio_new, ratio_old = _slice_overlap(s.timestamps, new_timestamps, margin_ns)
            if ratio_new >= overlap_ratio_threshold or ratio_old >= overlap_ratio_threshold:
                matched = s
                break

    if matched is None:
        return STATUS_INDEPENDENT_NEW_SLICE, None
    if candidate_prov.discovery_revision > matched.candidate_provenance.discovery_revision:
        return STATUS_EXPOSED_DISCOVERY_UPDATED, matched
    if candidate_id == matched.candidate:
        return STATUS_INDEPENDENT_REPEATED, matched
    return STATUS_EXPOSED_CANDIDATE_CHANGED, matched


def record_exposure(
    registry: dict, source_id: str, signature: Optional[BagSignature],
    new_slice_id: str, new_timestamps, candidate_id: CandidateIdentity, candidate_prov: CandidateProvenance,
    reveal_margin_s: float, matched: Optional[HoldoutSliceExposure],
) -> None:
    """Append this reveal to `registry[source_id]`'s history -- called only
    once a Hold-out result has actually been shown to a person (see the
    module docstring), never merely because a computation started or
    finished; a cancelled or failed run reveals nothing and must not call
    this.

    A new `HoldoutSliceExposure` is appended only when `matched` is `None`
    (`classify_exposure` found no existing slice this one is the same as, or
    overlaps enough with) -- an exact repeat or an exposed reuse is judged
    against an *existing* slice record, which is never overwritten, moved,
    or duplicated.
    """
    source = registry.get(source_id)
    if source is None:
        source = HoldoutSourceExposure(source_identity=source_id, signature=signature)
        registry[source_id] = source
    elif signature is not None and source.signature is None:
        source.signature = signature  # fills in a signature this source didn't have yet; slices untouched
    if matched is None:
        source.revealed_slices.append(HoldoutSliceExposure(
            slice_identity=new_slice_id, timestamps=tuple(sorted(int(t) for t in new_timestamps)),
            candidate=candidate_id, candidate_provenance=candidate_prov, reveal_margin_s=reveal_margin_s,
        ))


def determine_and_record_exposure(
    registry: dict, bag: str, signature: Optional[BagSignature], usable_timestamps,
    candidate_id: CandidateIdentity, candidate_prov: CandidateProvenance,
    reveal_margin_s: float = DEFAULT_EXCLUSION_MARGIN_S,
    overlap_ratio_threshold: float = DEFAULT_EXPOSURE_SLICE_OVERLAP_RATIO,
) -> str:
    """The single entry point a caller uses once a Hold-out result is
    actually shown to a person: resolves which Source `bag`/`signature`
    belongs to (`resolve_source_identity`, folding in a copied bag),
    classifies this reveal against that Source's slice history
    (`classify_exposure`), records it (`record_exposure`), and returns the
    status to display -- in that order, so classification always compares
    against what was true *before* this reveal.

    `candidate_id`/`candidate_prov` are the two halves of one candidate --
    see `CandidateIdentity`'s docstring for why they are kept separate
    (physical transform identity vs. session-local provenance).

    `reveal_margin_s` is conceptually the same guard-band idea as Discovery
    Exclusion Margin (a moment seen once should not look "new" a few
    milliseconds later), but is tracked as its own value per revealed slice,
    never coupled to whatever the Discovery Exclusion Margin setting happens
    to be at some later time -- callers typically pass the margin that was
    in effect when *this* result was computed (see step_verify.py).

    `registry` is ordinary caller-owned session state (a
    `dict[str, HoldoutSourceExposure]` -- see step_verify.VerifyStep's own
    dict), never anything global. Nothing here writes it to disk either --
    see `serialize_exposure_registry`/`deserialize_exposure_registry` below
    for the (optional, caller-driven) persistence this registry's shape
    supports.
    """
    source_id = resolve_source_identity(registry, bag, signature)
    new_slice_id = slice_identity(source_id, usable_timestamps)
    status, matched = classify_exposure(
        registry.get(source_id), new_slice_id, usable_timestamps, candidate_id, candidate_prov, reveal_margin_s,
        overlap_ratio_threshold,
    )
    record_exposure(registry, source_id, signature, new_slice_id, usable_timestamps, candidate_id, candidate_prov,
                     reveal_margin_s, matched)
    return status


# --------------------------------------------- exposure registry: persistence


VALIDATION_HISTORY_SCHEMA_VERSION = 3
# v1 carried CandidateIdentity's now-removed axis/delta *and* source/
# discovery_generation merged into one "candidate" dict. v2 splits them into
# "candidate_identity" (physical transform) and "candidate_provenance" --
# see CandidateIdentity's docstring for why -- but candidate_provenance's
# "discovery_generation" was still a SESSION-LOCAL counter that reset to 0
# on every app restart. v3 replaces it with a project-PERSISTENT,
# monotonically increasing "discovery_revision", stored both per-slice
# (candidate_provenance's field, now named "discovery_revision") and once at
# the top level of the serialized dict (this module's own current counter
# value, restored on load -- see serialize_exposure_registry). v1 and v2 are
# still READ (see deserialize_exposure_registry's migration below), never
# written; an existing user's history is upgraded silently the next time it
# is saved, not discarded.
#
# MIGRATION POLICY (v1/v2 -> v3), deliberate and worth reading before
# touching this: a v1/v2 project's "discovery_generation" numbers were
# counted by that session's own in-memory counter, which had no
# relationship whatsoever to any other session's counter (both started at
# 0 independently) -- they are simply not on the same scale as the new
# persistent discovery_revision, and comparing them numerically against a
# post-migration revision would be a category error, silently
# reintroducing exactly the false-negative contamination bug this version
# exists to fix. So: those old per-slice generation numbers are DISCARDED
# on migration (not reinterpreted, not carried over as a starting value) --
# every migrated slice's candidate_provenance.discovery_revision is reset to
# 0, and the migrated registry's own top-level discovery_revision starts at
# 0 as well, same as a brand-new project's counter. This is conservative in
# the safe direction: an old project's very first post-migration Discovery
# run already makes discovery_revision 1, which is already ">" every
# migrated slice's 0, so a real re-run is still correctly caught as
# "Discovery moved on" -- migrated history can never look falsely "fresh".
# See _migrate_legacy_candidate_provenance.
SUPPORTED_VALIDATION_HISTORY_VERSIONS = (1, 2, VALIDATION_HISTORY_SCHEMA_VERSION)


def _serialize_candidate_identity(candidate_id: CandidateIdentity) -> dict:
    """Plain, named fields -- never a Python repr/pickle of the dataclass --
    so the schema stays stable and human-readable across versions of this
    codebase and outside it (see the module docstring's project-file note)."""
    return {
        "baseline_fingerprint": candidate_id.baseline_fingerprint,
        "candidate_fingerprint": candidate_id.candidate_fingerprint,
        "axis": candidate_id.axis,
        "delta": candidate_id.delta,
    }


def _deserialize_candidate_identity(data: dict) -> CandidateIdentity:
    return CandidateIdentity(
        baseline_fingerprint=str(data["baseline_fingerprint"]),
        candidate_fingerprint=str(data["candidate_fingerprint"]),
        axis=str(data["axis"]), delta=float(data["delta"]),
    )


def _serialize_candidate_provenance(prov: CandidateProvenance) -> dict:
    return {"source": prov.source, "discovery_revision": prov.discovery_revision}


def _deserialize_candidate_provenance(data: dict) -> CandidateProvenance:
    return CandidateProvenance(source=str(data["source"]), discovery_revision=int(data["discovery_revision"]))


def _migrate_legacy_candidate_provenance(data: dict) -> CandidateProvenance:
    """v1 and v2 both stored a session-local 'discovery_generation' here --
    meaningless once compared against the new persistent discovery_revision
    counter (see VALIDATION_HISTORY_SCHEMA_VERSION's migration-policy
    comment for why it is not safe to carry the old number over), so it is
    deliberately DISCARDED here, never reinterpreted as a revision. Every
    migrated slice starts at discovery_revision=0, same as a brand-new
    project's counter."""
    return CandidateProvenance(source=str(data["source"]), discovery_revision=0)


def _migrate_v1_candidate(merged: dict) -> "tuple[CandidateIdentity, CandidateProvenance]":
    """v1's single 'candidate' dict held both halves together -- split on
    read so an older project's history loads as first-class
    CandidateIdentity/CandidateProvenance objects. See
    _migrate_legacy_candidate_provenance for why the provenance half's old
    'discovery_generation' is discarded rather than carried over."""
    identity = CandidateIdentity(
        baseline_fingerprint=str(merged["baseline_fingerprint"]),
        candidate_fingerprint=str(merged["candidate_fingerprint"]),
        axis=str(merged["axis"]), delta=float(merged["delta"]),
    )
    provenance = _migrate_legacy_candidate_provenance(merged)
    return identity, provenance


def _serialize_bag_signature(signature: Optional[BagSignature]) -> Optional[dict]:
    if signature is None:
        return None
    return {
        "file_size_bytes": signature.file_size_bytes,
        "duration_s": signature.duration_s,
        "n_topics": signature.n_topics,
        "total_message_count": signature.total_message_count,
        "first_lidar_timestamp_ns": signature.first_lidar_timestamp_ns,
        "last_lidar_timestamp_ns": signature.last_lidar_timestamp_ns,
    }


def _deserialize_bag_signature(data: Optional[dict]) -> Optional[BagSignature]:
    if not data:
        return None
    return BagSignature(
        file_size_bytes=int(data["file_size_bytes"]),
        duration_s=float(data["duration_s"]),
        n_topics=int(data["n_topics"]),
        total_message_count=int(data["total_message_count"]),
        first_lidar_timestamp_ns=int(data["first_lidar_timestamp_ns"]),
        last_lidar_timestamp_ns=int(data["last_lidar_timestamp_ns"]),
    )


def serialize_exposure_registry(registry: dict, discovery_revision: int = 0) -> dict:
    """`registry` (`dict[source_identity, HoldoutSourceExposure]`) plus the
    current project-persistent `discovery_revision` counter, as a plain,
    JSON/YAML-safe dict -- ordinary str/int/float/list/dict, no Python
    object anywhere -- suitable for `Project.validation_history` (see
    step_verify.py) or any other plain-data store. Only the LiDAR timestamps
    a slice actually used are kept (as raw nanosecond integers), not
    anything heavier a revealed slice might otherwise seem to need.

    `discovery_revision` is stored at the TOP level, a sibling of
    "holdout_exposure" rather than nested inside it -- it is one counter for
    the whole project, not per-source or per-slice data (see
    `CandidateProvenance`'s docstring for what it means and when it
    advances).
    """
    return {
        "version": VALIDATION_HISTORY_SCHEMA_VERSION,
        "discovery_revision": int(discovery_revision),
        "holdout_exposure": {
            source_id: {
                "signature": _serialize_bag_signature(source.signature),
                "revealed_slices": [
                    {
                        "slice_identity": s.slice_identity,
                        "timestamps_ns": list(s.timestamps),
                        "candidate_identity": _serialize_candidate_identity(s.candidate),
                        "candidate_provenance": _serialize_candidate_provenance(s.candidate_provenance),
                        "reveal_margin_s": s.reveal_margin_s,
                    }
                    for s in source.revealed_slices
                ],
            }
            for source_id, source in registry.items()
        },
    }


def deserialize_exposure_registry(data: Optional[dict]) -> "tuple[dict, int, list]":
    """The inverse of `serialize_exposure_registry` -- returns
    `(registry, discovery_revision, warnings)`. `data` missing, empty, or
    `None` (no `validation_history` in this project file at all -- e.g. a
    project saved before this feature existed) yields an empty registry with
    `discovery_revision=0` and no warnings; that is the expected, ordinary
    case for an older project, not an error.

    Reads `VALIDATION_HISTORY_SCHEMA_VERSION` (current) and every older
    version in `SUPPORTED_VALIDATION_HISTORY_VERSIONS` (`1` via
    `_migrate_v1_candidate`, `2` via `_migrate_legacy_candidate_provenance`
    -- see those functions and `VALIDATION_HISTORY_SCHEMA_VERSION`'s own
    comment for what changed, why, and the migration policy for the old
    per-slice generation numbers). A version outside that set (typically
    newer than this build knows about) is never guessed at: an empty
    registry and `discovery_revision=0` are returned instead, along with a
    warning the caller can surface, rather than risk silently misreading a
    shape this code was never told about. `data` from v1/v2 has no top-level
    "discovery_revision" key at all -- `.get(..., 0)` below naturally yields
    0 for those, exactly the migration policy's starting value, with no
    version-conditional branch needed for that part. One malformed source
    entry is skipped (with its own warning) rather than discarding every
    other entry that parsed fine.
    """
    if not data:
        return {}, 0, []
    version = data.get("version")
    if version not in SUPPORTED_VALIDATION_HISTORY_VERSIONS:
        return {}, 0, [
            f"validation_history schema version {version!r} is not recognised by this build "
            f"(supported: {SUPPORTED_VALIDATION_HISTORY_VERSIONS!r}); Hold-out exposure history was not loaded."
        ]

    discovery_revision = int(data.get("discovery_revision", 0) or 0)
    registry: dict = {}
    warnings: list = []
    for source_id, source_data in (data.get("holdout_exposure") or {}).items():
        try:
            signature = _deserialize_bag_signature(source_data.get("signature"))
            slices = []
            for s in (source_data.get("revealed_slices") or []):
                if version == 1:
                    cand_id, cand_prov = _migrate_v1_candidate(s["candidate"])
                elif version == 2:
                    cand_id = _deserialize_candidate_identity(s["candidate_identity"])
                    cand_prov = _migrate_legacy_candidate_provenance(s["candidate_provenance"])
                else:
                    cand_id = _deserialize_candidate_identity(s["candidate_identity"])
                    cand_prov = _deserialize_candidate_provenance(s["candidate_provenance"])
                slices.append(HoldoutSliceExposure(
                    slice_identity=str(s["slice_identity"]),
                    timestamps=tuple(int(t) for t in s["timestamps_ns"]),
                    candidate=cand_id, candidate_provenance=cand_prov,
                    reveal_margin_s=float(s["reveal_margin_s"]),
                ))
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Could not read Hold-out exposure history for source {source_id!r}: {exc}")
            continue
        registry[source_id] = HoldoutSourceExposure(source_identity=source_id, signature=signature,
                                                      revealed_slices=slices)
    return registry, discovery_revision, warnings


# ------------------------------------------------------- independence: overlap


@dataclass
class OverlapCheck:
    """Whether Discovery and Hold-out draw on the same, or near-same, moment.

    A separate bag file is independent data regardless of what its own
    timestamps happen to be -- two different recordings can share the same
    wall-clock second and that is not contamination. Only when both sides
    resolve to the *same* bag file does a nearby LiDAR timestamp risk being
    the same (or a visually near-identical) physical moment used for both
    discovery and validation -- which is what `exclusion_margin_s` guards
    against: consecutive frames from a moving vehicle can be almost the same
    scene at different exact timestamps, so an exact-equality check alone is
    not a reliable independence test.
    """

    same_source: bool         # True only when both sides are the same bag file
    exclusion_margin_s: float = DEFAULT_EXCLUSION_MARGIN_S
    overlap_timestamps: tuple = ()   # holdout timestamps within the margin of a discovery timestamp

    @property
    def overlapping(self) -> bool:
        return self.same_source and bool(self.overlap_timestamps)

    @property
    def n_overlap(self) -> int:
        return len(self.overlap_timestamps)


def same_bag_identity(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return a == b


def _timestamps_within_margin(discovery_timestamps, candidate_timestamps, margin_ns: int) -> np.ndarray:
    """Every value in `candidate_timestamps` within `margin_ns` of some value
    in `discovery_timestamps` -- O((n+m) log n) via `searchsorted` on sorted
    discovery timestamps, not an O(n*m) brute-force comparison."""
    disc = np.asarray(sorted(int(t) for t in discovery_timestamps), dtype=np.int64)
    cand = np.asarray(sorted(int(t) for t in candidate_timestamps), dtype=np.int64)
    if disc.size == 0 or cand.size == 0:
        return np.array([], dtype=np.int64)
    idx = np.searchsorted(disc, cand)
    idx_lo = np.clip(idx - 1, 0, disc.size - 1)
    idx_hi = np.clip(idx, 0, disc.size - 1)
    dist = np.minimum(np.abs(cand - disc[idx_lo]), np.abs(cand - disc[idx_hi]))
    return cand[dist <= margin_ns]


def check_overlap(
    discovery_bag: str,
    discovery_timestamps,
    holdout_bag: str,
    holdout_timestamps,
    exclusion_margin_s: float = DEFAULT_EXCLUSION_MARGIN_S,
) -> OverlapCheck:
    """Which of `holdout_timestamps` lie within `exclusion_margin_s` seconds
    of some Discovery timestamp -- only meaningful (and only computed) when
    `discovery_bag` and `holdout_bag` resolve to the same file; a different
    bag is independent by file identity alone (see `OverlapCheck`).
    `exclusion_margin_s=0.0` recovers the exact-timestamp-only check.
    """
    same_source = same_bag_identity(discovery_bag, holdout_bag)
    if not same_source:
        return OverlapCheck(same_source=False, exclusion_margin_s=exclusion_margin_s)
    margin_ns = max(int(round(exclusion_margin_s * 1e9)), 0)
    leaked = _timestamps_within_margin(discovery_timestamps, holdout_timestamps, margin_ns)
    return OverlapCheck(
        same_source=True, exclusion_margin_s=exclusion_margin_s,
        overlap_timestamps=tuple(sorted(int(t) for t in leaked)),
    )


@dataclass
class IndependencePartition:
    """Which of a *sampled* hold-out timestamp list are far enough (in time)
    from every Discovery timestamp to actually be scored, and which had to be
    dropped -- computed, and applied, *before* any of them is read from a bag
    (see the module docstring's Discovery/Hold-out separation: a frame
    excluded here is never loaded, never decoded, never projected).

    Only meaningful when both sides are the same bag file (`same_source`); a
    different bag needs no exclusion, since it is independent data by file
    identity alone regardless of its own timestamps.
    """

    same_source: bool
    exclusion_margin_s: float
    requested: int                    # how many timestamps were sampled, before exclusion
    usable_timestamps: tuple = ()
    excluded_timestamps: tuple = ()   # sampled, but within the margin of a Discovery frame

    @property
    def n_usable(self) -> int:
        return len(self.usable_timestamps)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_timestamps)

    @property
    def independence_limited(self) -> bool:
        """Not a pass/fail score -- just a note that few enough independent
        frames remain that a hold-out result here deserves extra caution,
        even if every other number looks clean. DEFAULT_MIN_INDEPENDENT_FRAMES
        is a configurable heuristic default, not a validated minimum."""
        return self.n_usable < DEFAULT_MIN_INDEPENDENT_FRAMES


def partition_independent_frames(
    candidate_timestamps,
    discovery_bag: str,
    discovery_timestamps,
    holdout_bag: str,
    exclusion_margin_s: float = DEFAULT_EXCLUSION_MARGIN_S,
) -> IndependencePartition:
    """Split `candidate_timestamps` (already sampled from wherever Hold-out
    would read frames) into (usable, excluded_near_discovery) -- built on
    `check_overlap`'s margin logic, so the two can never disagree about what
    counts as "too close to Discovery"."""
    candidate_timestamps = list(candidate_timestamps)
    overlap = check_overlap(discovery_bag, discovery_timestamps, holdout_bag, candidate_timestamps, exclusion_margin_s)
    excluded = set(overlap.overlap_timestamps)
    usable = tuple(sorted(int(t) for t in candidate_timestamps if int(t) not in excluded))
    return IndependencePartition(
        same_source=overlap.same_source, exclusion_margin_s=exclusion_margin_s,
        requested=len(candidate_timestamps), usable_timestamps=usable,
        excluded_timestamps=tuple(sorted(excluded)),
    )


# ------------------------------------------------- independence: bag signature


def signatures_suspiciously_similar(
    a: BagSignature,
    b: BagSignature,
    *,
    size_tolerance: float = DEFAULT_SIGNATURE_SIZE_TOLERANCE,
    duration_tolerance_s: float = DEFAULT_SIGNATURE_DURATION_TOLERANCE_S,
) -> bool:
    """Heuristic only -- flags what looks like the same recording saved under
    a different path ('run1.bag' vs 'run1_copy.bag'), never proves it and
    never disproves independence when it returns False. Topic count, total
    message count, and first/last LiDAR timestamp must match *exactly* (a
    genuine copy keeps every message stamp byte-for-byte); duration and file
    size are compared with a small tolerance for container/metadata overhead
    that a bare copy can still introduce.
    """
    if a.n_topics != b.n_topics or a.total_message_count != b.total_message_count:
        return False
    if a.first_lidar_timestamp_ns != b.first_lidar_timestamp_ns or a.last_lidar_timestamp_ns != b.last_lidar_timestamp_ns:
        return False
    if abs(a.duration_s - b.duration_s) > duration_tolerance_s:
        return False
    size_ref = max(a.file_size_bytes, b.file_size_bytes, 1)
    if abs(a.file_size_bytes - b.file_size_bytes) / size_ref > size_tolerance:
        return False
    return True


def cross_bag_time_ranges_overlap(a: BagSignature, b: BagSignature) -> bool:
    """Whether two *different* bags' own LiDAR sensor-timestamp ranges
    overlap -- informational only (see the module docstring's "cross-bag"
    note): a shared sensor clock domain would make this meaningful evidence
    of overlapping recordings, but nothing here can confirm the two bags
    share a clock domain, so this is never used to block a run, only to
    surface a note for a person to judge."""
    return a.first_lidar_timestamp_ns <= b.last_lidar_timestamp_ns and b.first_lidar_timestamp_ns <= a.last_lidar_timestamp_ns


# ------------------------------------------------------------------ provenance


@dataclass(frozen=True)
class DiscoveryProvenance:
    """Which data a candidate was chosen from -- kept alongside every
    HoldoutResult so it stays possible to answer "what found this candidate,
    and what checked it" without cross-referencing separate UI state.

    `signature` (this Discovery bag's own `BagSignature`, best-effort -- see
    step_verify.VerifyStep.run_holdout) fills a gap Candidate A/B Test
    Package Export needs: without it, an export package can only name the
    Discovery bag by path, not by content, so a renamed/moved/copied
    Discovery bag would be unverifiable later. `None` when it could not be
    read (bag inaccessible, or this result predates the field) -- callers
    must treat that as "unknown", never as "no Discovery data", and must
    never fail an export over it (see gui.core.evaluation.ab_export).
    """

    bag: str
    timestamps: tuple            # sorted lidar timestamps ns -- the exact Discovery frame set
    exclusion_margin_s: float
    signature: Optional[BagSignature] = None


@dataclass(frozen=True)
class HoldoutProvenance:
    bag: str
    requested_timestamps: tuple       # sampled before exclusion
    excluded_near_discovery: tuple    # sampled, then dropped for being within the margin -- never read from the bag
    usable_timestamps: tuple          # handed to evaluate_holdout (pre sync-rejection)
    sync_rejected_timestamps: tuple   # subset of usable that failed the configured sync gate


# ------------------------------------------------------------- bootstrap (CI)


# Below this ratio of single-frame blocks, the guard the block bootstrap
# exists to provide (protecting against within-block temporal correlation)
# is only weakly represented -- see BootstrapResult.sparse_sampling.
SPARSE_SINGLE_FRAME_BLOCK_RATIO = 0.5


@dataclass
class BootstrapResult:
    """A confidence interval over paired ΔFrame-P95, computed by the
    **Non-overlapping Block Bootstrap**: comparable frames are grouped, by
    timestamp, into fixed-duration (`block_s`), non-overlapping,
    time-contiguous windows, and every resample draws *whole blocks* --
    never individual frames -- with replacement, `iterations` times. Never a
    p-value, never a pass/fail gate.

    This exists because sequential driving frames are correlated (a bad
    stretch of road affects several consecutive frames at once); resampling
    individual frames as if they were i.i.d. would understate uncertainty --
    see test_holdout.py's block-vs-naive comparison tests. `method` names
    which resampling scheme produced this result
    (`DEFAULT_BOOTSTRAP_METHOD`/`"non_overlapping_block"` today); a future
    moving-block or stationary bootstrap would be a new branch in
    `_run_block_bootstrap` and a new value here, not a change to this field's
    meaning.

    `block_s`, and the 5-block minimum below it, are configurable heuristic
    defaults, not validated correctness thresholds.
    """

    ok: bool
    reason: str = ""
    method: str = ""
    iterations: int = 0
    block_s: float = 0.0
    n_blocks: int = 0

    # Temporal structure -- populated whenever there were enough comparable
    # frames to group at all, even when `ok=False` for having too few
    # *blocks*: seeing "50 frames, but only 3 blocks" is the point.
    n_comparable_frames: int = 0
    median_frame_interval_s: float = float("nan")
    frames_per_block_median: float = float("nan")
    frames_per_block_min: int = 0
    frames_per_block_max: int = 0
    # True when the block duration is doing little work -- most blocks hold
    # only one frame, or frames are typically spaced as far apart as a block
    # is wide. Not a pass/fail signal; see SPARSE_SINGLE_FRAME_BLOCK_RATIO.
    sparse_sampling: bool = False

    median_delta_p95_px: float = float("nan")
    median_delta_ci_low_px: float = float("nan")
    median_delta_ci_high_px: float = float("nan")

    improved_ratio: float = float("nan")
    improved_ratio_ci_low: float = float("nan")
    improved_ratio_ci_high: float = float("nan")

    @property
    def median_ci_excludes_zero(self) -> "bool | None":
        """None when unavailable; True/False are plain facts about where the
        interval sits -- never phrased here as "candidate is/isn't correct"."""
        if not self.ok or not (_finite(self.median_delta_ci_low_px) and _finite(self.median_delta_ci_high_px)):
            return None
        return self.median_delta_ci_high_px < 0.0 or self.median_delta_ci_low_px > 0.0


def _paired_deltas_with_timestamps(frames: list, baseline_point, candidate_point) -> list:
    """(lidar_timestamp_ns, candidate_p95 - baseline_p95) for every frame
    both baseline and candidate scored successfully -- the same "comparable"
    definition `perturbation._pair_frame_metrics` uses internally, just
    carrying each frame's timestamp along so blocks can be built by time."""
    out = []
    for f in frames:
        b = baseline_point.per_frame.get(f.timeline_index)
        c = candidate_point.per_frame.get(f.timeline_index)
        if b is not None and c is not None and b.ok and c.ok:
            out.append((int(f.lidar_timestamp_ns), c.p95_px - b.p95_px))
    return out


def _temporal_block_structure(paired_sorted: list, block_s: float) -> "tuple[list, dict]":
    """Group already-time-sorted (timestamp, delta) pairs into fixed-width,
    non-overlapping blocks, and describe the grouping -- shared by every
    `ok=True` and `ok=False` (too-few-blocks) return of
    `_run_block_bootstrap`, so "50 frames, only 3 blocks" is visible exactly
    when it is the reason no CI was computed."""
    timestamps = [t for t, _ in paired_sorted]
    intervals_s = np.diff(np.asarray(timestamps, dtype=np.int64)) / 1e9 if len(timestamps) > 1 else np.array([])
    median_interval_s = float(np.median(intervals_s)) if intervals_s.size else float("nan")

    t0 = timestamps[0]
    block_ns = max(int(round(block_s * 1e9)), 1)
    blocks: dict = {}
    for t, d in paired_sorted:
        blocks.setdefault((t - t0) // block_ns, []).append(d)
    block_list = list(blocks.values())
    block_sizes = [len(v) for v in block_list]
    n_blocks = len(block_list)
    single_frame_ratio = (sum(1 for s in block_sizes if s == 1) / n_blocks) if n_blocks else 0.0
    sparse = (_finite(median_interval_s) and median_interval_s >= block_s) or single_frame_ratio > SPARSE_SINGLE_FRAME_BLOCK_RATIO

    stats = dict(
        n_comparable_frames=len(paired_sorted), median_frame_interval_s=median_interval_s,
        frames_per_block_median=float(np.median(block_sizes)) if block_sizes else float("nan"),
        frames_per_block_min=min(block_sizes) if block_sizes else 0,
        frames_per_block_max=max(block_sizes) if block_sizes else 0,
        sparse_sampling=sparse,
    )
    return block_list, stats


def _run_block_bootstrap(
    paired: list,
    block_s: float,
    iterations: int,
    min_blocks: int,
    tolerance_px: float,
    rng: Optional[np.random.Generator] = None,
    method: str = DEFAULT_BOOTSTRAP_METHOD,
) -> BootstrapResult:
    """Non-overlapping Block Bootstrap: group `paired` (timestamp, delta)
    values into fixed-duration, time-contiguous, non-overlapping blocks, then
    resample *blocks* (never individual frames) with replacement `iterations`
    times, taking the 2.5th/97.5th percentile of each resample's
    median/improved-ratio as the 95% interval. A block duration long enough
    to swallow every paired frame collapses this to one block that is always
    drawn -- the CI degenerates to zero width, which is the correct
    behaviour for "one block", not a bug (see test_holdout.py's
    block-vs-naive-bootstrap tests for why this matters).

    `method` selects the resampling scheme; only `"non_overlapping_block"` is
    implemented (see `SUPPORTED_BOOTSTRAP_METHODS`) -- passed through so a
    future method can be added as a new branch without changing this
    function's signature or `BootstrapResult`'s shape.
    """
    if method not in SUPPORTED_BOOTSTRAP_METHODS:
        raise ValueError(f"unknown bootstrap method: {method!r} (supported: {SUPPORTED_BOOTSTRAP_METHODS!r})")

    if len(paired) < 2:
        return BootstrapResult(ok=False, method=method, iterations=iterations, block_s=block_s,
                                n_comparable_frames=len(paired),
                                reason="Confidence interval unavailable: fewer than 2 comparable frames.")

    paired_sorted = sorted(paired, key=lambda p: p[0])
    block_list, stats = _temporal_block_structure(paired_sorted, block_s)
    n_blocks = len(block_list)

    if n_blocks < min_blocks:
        return BootstrapResult(
            ok=False, method=method, iterations=iterations, block_s=block_s, n_blocks=n_blocks, **stats,
            reason=f"Confidence interval unavailable: too few independent temporal blocks ({n_blocks} < {min_blocks}).",
        )

    all_deltas = np.array([d for _, d in paired_sorted], dtype=np.float64)
    observed_median = float(np.median(all_deltas))
    observed_improved_ratio = float(np.mean(all_deltas < -tolerance_px))

    rng = rng if rng is not None else np.random.default_rng()
    medians = np.empty(iterations, dtype=np.float64)
    improved_ratios = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        pooled = np.concatenate([np.asarray(block_list[j], dtype=np.float64) for j in chosen])
        medians[i] = np.median(pooled)
        improved_ratios[i] = np.mean(pooled < -tolerance_px)

    ci_lo, ci_hi = np.percentile(medians, [2.5, 97.5])
    ratio_lo, ratio_hi = np.percentile(improved_ratios, [2.5, 97.5])
    return BootstrapResult(
        ok=True, method=method, iterations=iterations, block_s=block_s, n_blocks=n_blocks, **stats,
        median_delta_p95_px=observed_median,
        median_delta_ci_low_px=float(ci_lo), median_delta_ci_high_px=float(ci_hi),
        improved_ratio=observed_improved_ratio,
        improved_ratio_ci_low=float(ratio_lo), improved_ratio_ci_high=float(ratio_hi),
    )


# ------------------------------------------------------------------- result


@dataclass
class SpatialComparisonRow:
    label: str
    baseline_p95_px: float
    candidate_p95_px: float
    delta_p95_px: float          # candidate - baseline; negative = improvement
    baseline_n_points: int
    candidate_n_points: int


def _compare_bins(baseline_dict: dict, candidate_dict: dict, labels: tuple) -> list:
    rows = []
    for label in labels:
        b = baseline_dict.get(label)
        c = candidate_dict.get(label)
        b_p95 = b.p95_px if b is not None else float("nan")
        c_p95 = c.p95_px if c is not None else float("nan")
        delta = c_p95 - b_p95 if (_finite(b_p95) and _finite(c_p95)) else float("nan")
        rows.append(SpatialComparisonRow(
            label=label, baseline_p95_px=b_p95, candidate_p95_px=c_p95, delta_p95_px=delta,
            baseline_n_points=b.n_points if b is not None else 0,
            candidate_n_points=c.n_points if c is not None else 0,
        ))
    return rows


@dataclass(frozen=True)
class HoldoutFrameRecord:
    """One `perturbation.LoadedFrame`'s identity/sync provenance, copied
    verbatim (never recomputed) so a caller -- namely
    `gui.core.evaluation.ab_export` -- can build a per-frame table without
    re-reading the bag or re-deriving sync pairing, which this module's own
    "never re-run evaluation" rule would otherwise make impossible to do
    later. Holds no score of any kind; see `HoldoutResult.axis`'s
    `baseline`/candidate `PerturbationPoint.per_frame` (keyed by the same
    `timeline_index`) for the actual per-frame P95/median."""

    timeline_index: int
    lidar_timestamp_ns: int
    camera_timestamp_ns: int
    signed_sync_offset_ms: float
    abs_sync_offset_ms: float
    sync_rejected: bool


@dataclass
class HoldoutResult:
    ok: bool
    candidate: Optional[CandidateSpec] = None
    reason: str = ""
    cancelled: bool = False

    # Exactly two points -- baseline (delta=0) and the locked candidate --
    # scored and paired by perturbation.summarize_axis, so every paired
    # frame-balanced statistic (n_comparable_frames, n_improved_frames,
    # median_delta_frame_p95_px, ...) is filled in on the candidate point
    # by the same code Perturbation Sensitivity and Fine Scan already use.
    axis: Optional[AxisSensitivity] = None

    n_frames_requested: int = 0
    n_frames_used: int = 0        # usable (non sync-rejected) hold-out frames
    n_sync_rejected: int = 0

    horizontal: list = field(default_factory=list)   # SpatialComparisonRow, HORIZONTAL_REGIONS order
    vertical: list = field(default_factory=list)      # SpatialComparisonRow, VERTICAL_REGIONS order
    depth: list = field(default_factory=list)         # SpatialComparisonRow, DEPTH_BIN_LABELS order

    # Uncertainty over the paired ΔFrame-P95 -- see BootstrapResult. None
    # when there were not even 2 comparable frames to bootstrap at all.
    bootstrap: Optional[BootstrapResult] = None

    # Provenance -- see DiscoveryProvenance/HoldoutProvenance. Either may be
    # None when the caller did not supply that context (e.g. a unit test
    # calling evaluate_holdout directly); the UI always supplies both.
    baseline_fingerprint: str = ""
    discovery: Optional[DiscoveryProvenance] = None
    provenance: Optional[HoldoutProvenance] = None
    signature_warning: str = ""              # non-empty => Hold-out bag looks like a copy of the Discovery bag
    cross_bag_time_overlap_note: str = ""    # non-empty => different bags, but overlapping sensor time ranges

    # Identity -- see candidate_identity()/candidate_provenance()/resolve_source_identity.
    # Used only for session-level exposure tracking
    # (holdout.determine_and_record_exposure), via `provenance.bag`/
    # `provenance.usable_timestamps` (source/slice) and `candidate_identity`/
    # `candidate_provenance` (which candidate, and where it came from -- kept
    # as two separate objects; see CandidateIdentity's docstring for why).
    # `signature` (this Hold-out bag's own BagSignature, distinct from
    # `signature_warning` above, which is a comparison *against the Discovery
    # bag*) lets a copied bag be recognised as the same source even at a
    # different path -- None when the caller could not or did not supply one
    # (e.g. a unit test).
    candidate_identity: Optional[CandidateIdentity] = None
    candidate_provenance: Optional[CandidateProvenance] = None
    signature: Optional[BagSignature] = None

    # Per-frame identity/sync provenance (see HoldoutFrameRecord), captured
    # from the `frames` this evaluation was given -- empty on a failed or
    # cancelled result (there is no frame set worth recording); populated
    # only on the final ok=True return. Exists purely so a later export can
    # build a per-frame table without touching a bag reader again.
    frame_table: tuple = ()

    @property
    def baseline_point(self):
        return self.axis.baseline if self.axis is not None else None

    @property
    def candidate_point(self):
        if self.axis is None:
            return None
        return next((p for p in self.axis.points if p is not self.axis.baseline), None)


def _empty(candidate, n_requested: int, reason: str, cancelled: bool = False,
           n_used: int = 0, n_sync_rejected: int = 0, axis=None,
           baseline_fingerprint: str = "", discovery=None, provenance=None,
           signature_warning: str = "", cross_bag_time_overlap_note: str = "",
           candidate_identity: Optional[CandidateIdentity] = None,
           candidate_provenance: Optional[CandidateProvenance] = None,
           signature: Optional[BagSignature] = None) -> HoldoutResult:
    return HoldoutResult(
        ok=False, candidate=candidate, reason=reason, cancelled=cancelled,
        n_frames_requested=n_requested, n_frames_used=n_used, n_sync_rejected=n_sync_rejected, axis=axis,
        baseline_fingerprint=baseline_fingerprint, discovery=discovery, provenance=provenance,
        signature_warning=signature_warning, cross_bag_time_overlap_note=cross_bag_time_overlap_note,
        candidate_identity=candidate_identity, candidate_provenance=candidate_provenance, signature=signature,
    )


def evaluate_holdout(
    frames: list,
    baseline_R: np.ndarray,
    baseline_t: np.ndarray,
    candidate: CandidateSpec,
    camera,
    min_range: float = 0.0,
    max_range: float = 0.0,
    edge_params: EdgeAlignmentParams = EdgeAlignmentParams(),
    local_min_tolerance_px: float = DEFAULT_LOCAL_MIN_TOLERANCE_PX,
    progress: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    *,
    discovery: Optional[DiscoveryProvenance] = None,
    holdout_bag: str = "",
    holdout_signature: Optional[BagSignature] = None,
    requested_timestamps: tuple = (),
    excluded_near_discovery: tuple = (),
    signature_warning: str = "",
    cross_bag_time_overlap_note: str = "",
    discovery_revision: int = 0,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_block_s: float = DEFAULT_BOOTSTRAP_BLOCK_S,
    min_bootstrap_blocks: int = DEFAULT_MIN_BOOTSTRAP_BLOCKS,
    bootstrap_method: str = DEFAULT_BOOTSTRAP_METHOD,
    bootstrap_rng: Optional[np.random.Generator] = None,
) -> HoldoutResult:
    """Score `baseline_R`/`baseline_t` and `candidate` against the fixed,
    already-prepared `frames` (a `perturbation.prepare_frames` result --
    hold-out frames a caller loaded from a bag/time-range already filtered
    for independence from Discovery, e.g. with `partition_independent_frames`).

    Exactly two trials are ever run here, both through
    `perturbation.run_trial_point` -- the same per-frame projection +
    Edge Alignment scoring Perturbation Sensitivity and Fine Scan use, so a
    hold-out P95 and a discovery P95 are never computed two different ways.
    No grid, no delta range, no candidate search: `candidate` is scored
    exactly as given, never adjusted based on how it turns out here.

    `baseline_R`/`baseline_t` are read only; a fresh `np.eye`/`R_delta @ R`
    array is what actually gets projected (see `perturbation.perturb_rotation`
    / `perturb_translation`), so the caller's arrays are never mutated -- the
    same guarantee `evaluate_perturbation_grid` makes.

    `discovery`/`holdout_bag`/`holdout_signature`/`requested_timestamps`/
    `excluded_near_discovery`/`signature_warning`/`cross_bag_time_overlap_note`
    are pure passthrough provenance -- computed by the caller (the UI, which
    alone knows which bag and sampling produced `frames`) and copied onto the
    returned `HoldoutResult` unexamined. `holdout_signature` (this Hold-out
    bag's own `BagSignature`) is what later lets a copied bag be recognised
    as the same Source even at a different path (see
    `resolve_source_identity`) -- distinct from `signature_warning`, which is
    the result of already comparing it against the *Discovery* bag.
    `discovery_revision` becomes the returned `candidate_provenance`'s
    `discovery_revision` (see `CandidateProvenance`'s docstring for why this
    is provenance, kept separate from `candidate_identity`'s physical
    equality, and why it must be the project-persistent counter, not a
    session-local one) -- it means nothing to this function beyond that.
    `bootstrap_*` control the paired-delta block bootstrap this function
    always computes when there is more than one comparable frame (see
    `BootstrapResult`).
    """
    n_requested = len(frames)
    n_sync_rejected = sum(1 for f in frames if f.sync_rejected)
    usable = [f for f in frames if not f.sync_rejected]
    fingerprint = fingerprint_extrinsic(baseline_R, baseline_t)
    candidate_id = candidate_identity(baseline_R, baseline_t, candidate)
    candidate_prov = candidate_provenance(candidate, discovery_revision)
    common_kwargs = dict(
        baseline_fingerprint=fingerprint, discovery=discovery, signature_warning=signature_warning,
        cross_bag_time_overlap_note=cross_bag_time_overlap_note,
        candidate_identity=candidate_id, candidate_provenance=candidate_prov, signature=holdout_signature,
    )

    if not usable:
        return _empty(
            candidate, n_requested,
            "사용 가능한 hold-out frame이 없습니다 (모두 sync 제외되었거나 비어 있음).",
            n_sync_rejected=n_sync_rejected, **common_kwargs,
        )

    if should_cancel is not None and should_cancel():
        return _empty(candidate, n_requested, "baseline 평가 전에 취소되었습니다.",
                       cancelled=True, n_sync_rejected=n_sync_rejected, **common_kwargs)

    baseline_point = run_trial_point(
        candidate.kind, candidate.axis, 0.0, usable, baseline_R, baseline_t,
        camera, min_range, max_range, edge_params, n_sync_rejected,
    )
    if progress is not None:
        progress(1, 2)

    if should_cancel is not None and should_cancel():
        axis = summarize_axis(candidate.axis, candidate.unit, [baseline_point], baseline_point,
                               local_min_tolerance_px, "median_frame_p95")
        return _empty(candidate, n_requested, "candidate 평가 전에 취소되었습니다.", cancelled=True,
                      n_used=len(usable), n_sync_rejected=n_sync_rejected, axis=axis, **common_kwargs)

    candidate_point = run_trial_point(
        candidate.kind, candidate.axis, candidate.delta, usable, baseline_R, baseline_t,
        camera, min_range, max_range, edge_params, n_sync_rejected,
    )
    if progress is not None:
        progress(2, 2)

    axis = summarize_axis(
        candidate.axis, candidate.unit, [baseline_point, candidate_point], baseline_point,
        local_min_tolerance_px, ranking_metric="median_frame_p95",
    )

    horizontal: list = []
    vertical: list = []
    depth: list = []
    if baseline_point.spatial is not None and candidate_point.spatial is not None:
        horizontal = _compare_bins(baseline_point.spatial.horizontal, candidate_point.spatial.horizontal, HORIZONTAL_REGIONS)
        vertical = _compare_bins(baseline_point.spatial.vertical, candidate_point.spatial.vertical, VERTICAL_REGIONS)
        depth = _compare_bins(baseline_point.spatial.depth_bins, candidate_point.spatial.depth_bins, DEPTH_BIN_LABELS)

    paired = _paired_deltas_with_timestamps(usable, baseline_point, candidate_point)
    bootstrap = _run_block_bootstrap(
        paired, bootstrap_block_s, bootstrap_iterations, min_bootstrap_blocks, local_min_tolerance_px, bootstrap_rng,
        method=bootstrap_method,
    )

    provenance = HoldoutProvenance(
        bag=holdout_bag,
        requested_timestamps=tuple(sorted(int(t) for t in requested_timestamps)),
        excluded_near_discovery=tuple(sorted(int(t) for t in excluded_near_discovery)),
        usable_timestamps=tuple(sorted(f.lidar_timestamp_ns for f in usable)),
        sync_rejected_timestamps=tuple(sorted(f.lidar_timestamp_ns for f in frames if f.sync_rejected)),
    ) if (holdout_bag or requested_timestamps or excluded_near_discovery) else None

    frame_table = tuple(
        HoldoutFrameRecord(
            timeline_index=f.timeline_index, lidar_timestamp_ns=int(f.lidar_timestamp_ns),
            camera_timestamp_ns=int(f.camera_timestamp_ns), signed_sync_offset_ms=float(f.signed_sync_offset_ms),
            abs_sync_offset_ms=float(f.abs_sync_offset_ms), sync_rejected=bool(f.sync_rejected),
        )
        for f in frames
    )

    return HoldoutResult(
        ok=True, candidate=candidate, axis=axis,
        n_frames_requested=n_requested, n_frames_used=len(usable), n_sync_rejected=n_sync_rejected,
        horizontal=horizontal, vertical=vertical, depth=depth, bootstrap=bootstrap,
        provenance=provenance, frame_table=frame_table, **common_kwargs,
    )

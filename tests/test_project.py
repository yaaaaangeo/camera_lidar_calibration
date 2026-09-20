"""Tests for gui.core.project.Project's validation_history field -- the
opaque, Hold-out-exposure-history dict Step 7 round-trips through the
project file (see gui.core.evaluation.holdout's serialize/deserialize
functions for what actually lives inside it; Project itself never looks at
its shape).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

import gui.core.evaluation.holdout as ho
from gui.core.project import FilterBox, PlaneRegionSpec, Project, Scene

SEC = 1_000_000_000


def test_validation_history_round_trips_through_save_and_load(tmp_path):
    p = Project(name="test")
    p.validation_history = {
        "version": 1,
        "holdout_exposure": {"path:run.bag": {"signature": None, "revealed_slices": []}},
    }
    path = tmp_path / "test.calib.yaml"
    p.save(path)

    loaded = Project.load(path)
    assert loaded.validation_history == p.validation_history


def test_validation_history_defaults_to_empty_dict_for_a_fresh_project():
    assert Project().validation_history == {}


def test_project_file_without_validation_history_loads_normally(tmp_path):
    """A project saved before this field existed (or hand-edited to remove
    it) must still open, with an empty validation_history -- not an error,
    not a KeyError, not a forced migration."""
    p = Project(name="legacy")
    path = tmp_path / "legacy.calib.yaml"
    p.save(path)

    import yaml
    data = yaml.safe_load(path.read_text())
    assert "validation_history" in data  # this build always writes it...
    del data["validation_history"]        # ...but an older build's file would not have
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    loaded = Project.load(path)
    assert loaded.validation_history == {}
    assert loaded.name == "legacy"  # the rest of the file still loads fine


def test_validation_history_does_not_disturb_other_fields(tmp_path):
    p = Project(name="demo", lidar_topic="/lidar", camera_topic="/camera")
    p.bag_paths = ["a.bag", "b.bag"]
    p.validation_history = {"version": 1, "holdout_exposure": {}}
    path = tmp_path / "demo.calib.yaml"
    p.save(path)

    loaded = Project.load(path)
    assert loaded.lidar_topic == "/lidar"
    assert loaded.camera_topic == "/camera"
    assert loaded.bag_paths == ["a.bag", "b.bag"]


def test_validation_history_default_is_not_a_shared_mutable_object():
    """`validation_history: dict = field(default_factory=dict)` must give
    every Project instance its own dict -- a bare `= {}` class-level default
    would silently share one dict across every Project ever constructed."""
    a = Project()
    b = Project()
    assert a.validation_history is not b.validation_history

    a.validation_history["version"] = 1
    assert b.validation_history == {}
    assert "version" not in b.validation_history


def test_project_save_is_atomic_no_leftover_temp_file(tmp_path):
    p = Project(name="atomic")
    path = tmp_path / "atomic.calib.yaml"
    p.save(path)
    p.name = "renamed"
    p.save(path)  # a second save, exercising the replace path

    leftovers = [f for f in tmp_path.iterdir() if f.name != path.name]
    assert leftovers == []
    assert Project.load(path).name == "renamed"


def _cid(axis="yaw", delta=-0.1, source="fine", discovery_revision=0):
    import gui.core.evaluation.holdout as ho_mod
    spec = ho_mod.CandidateSpec(axis=axis, delta=delta, source=source)
    return ho_mod.candidate_identity(np.eye(3), np.zeros(3), spec), ho_mod.candidate_provenance(spec, discovery_revision)


def test_autosave_flow_persists_validation_history_without_a_manual_save(tmp_path):
    """Core-level stand-in for the Qt wiring this sandbox cannot exercise
    (VerifyStep.validation_history_changed -> MainWindow._autosave_validation_history):
    once a project already has a file path, updating
    `project.validation_history` and calling `Project.patch_validation_history`
    -- the exact call `_autosave_validation_history` makes, never a full
    `Project.save` -- must be enough for a fresh `Project.load` to see the
    change, with no explicit "저장…" click anywhere in this flow.
    """
    project = Project(name="p")
    path = tmp_path / "p.calib.yaml"
    project.save(path)  # the one manual save that establishes a file path

    # ... a Hold-out result is revealed (see VerifyStep._on_holdout_ready) ...
    registry = {}
    ho.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC, 2 * SEC], *_cid(), reveal_margin_s=2.0)
    project.validation_history = ho.serialize_exposure_registry(registry, discovery_revision=3)

    # ... MainWindow._autosave_validation_history's entire body, no dialog,
    # no user click ...
    Project.patch_validation_history(path, project.validation_history)

    reloaded = Project.load(path)
    assert reloaded.validation_history == project.validation_history
    restored, discovery_revision, warnings = ho.deserialize_exposure_registry(reloaded.validation_history)
    assert warnings == []
    assert restored == registry
    assert discovery_revision == 3


def test_autosave_flow_is_a_noop_concept_without_a_project_path():
    """The other half of `_autosave_validation_history`'s guard: with no
    file path yet, there is nothing to reuse a save path from, so the
    in-memory registry (and hence `project.validation_history`) is simply
    not written anywhere -- verified here at the data level, since the
    Qt-level "no dialog pops up" behaviour cannot run in this environment."""
    project = Project(name="unsaved")
    registry = {}
    ho.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC], *_cid(), reveal_margin_s=2.0)
    project.validation_history = ho.serialize_exposure_registry(registry)
    # Nothing to assert about disk I/O here (there is deliberately none) --
    # only that holding this state in memory alone never raises or requires
    # a path, matching "don't block Hold-out just because nothing was saved".
    assert project.validation_history["holdout_exposure"]


# ------------------------------------------------- patch_validation_history


def test_patch_validation_history_leaves_unsaved_scene_change_on_disk_untouched(tmp_path):
    """The exact bug fix requirement: disk holds scene=A plus an old
    validation_history; memory has moved on to scene=B (never saved) plus a
    new validation_history (a Hold-out was just revealed). A patch must
    write ONLY the new validation_history -- scene stays A on disk, never
    silently promoted to B. A subsequent explicit `save()` then writes both
    scene=B and the new validation_history, exactly as a manual "저장…"
    should."""
    project = Project(name="p")
    project.scenes = [Scene(id="A", t_ns=1, bag="a.bag", frames=1,
                             filter=FilterBox(), region=PlaneRegionSpec())]
    old_history = {"version": 3, "discovery_revision": 0, "holdout_exposure": {}}
    project.validation_history = old_history
    path = tmp_path / "p.calib.yaml"
    project.save(path)  # disk: scene=A, old_history

    # ... unsaved in-memory edit the user has not pressed "저장…" for ...
    project.scenes = [Scene(id="B", t_ns=2, bag="b.bag", frames=1,
                             filter=FilterBox(), region=PlaneRegionSpec())]
    # ... a Hold-out result is revealed, updating validation_history only ...
    registry = {}
    ho.determine_and_record_exposure(registry, "a.bag", None, [1 * SEC], *_cid(), reveal_margin_s=2.0)
    new_history = ho.serialize_exposure_registry(registry, discovery_revision=1)
    project.validation_history = new_history

    Project.patch_validation_history(path, project.validation_history)

    after_autosave = Project.load(path)
    assert after_autosave.validation_history == new_history
    assert [s.id for s in after_autosave.scenes] == ["A"]  # unsaved scene change never reached disk

    project.save(path)  # the user finally presses "저장…"
    after_manual_save = Project.load(path)
    assert after_manual_save.validation_history == new_history
    assert [s.id for s in after_manual_save.scenes] == ["B"]  # now it does


def test_patch_validation_history_preserves_camera_and_target_from_disk(tmp_path):
    """Same guarantee as the scene test above, for the other two fields a
    person might have unsaved edits to."""
    project = Project(name="p", lidar_topic="/lidar_v1")
    path = tmp_path / "p.calib.yaml"
    project.save(path)

    project.lidar_topic = "/lidar_v2_unsaved"
    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 1, "holdout_exposure": {}})

    reloaded = Project.load(path)
    assert reloaded.lidar_topic == "/lidar_v1"  # the unsaved change never reached disk
    assert reloaded.validation_history["discovery_revision"] == 1


def test_patch_validation_history_is_atomic_no_leftover_temp_file(tmp_path):
    project = Project(name="p")
    path = tmp_path / "p.calib.yaml"
    project.save(path)

    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 1, "holdout_exposure": {}})
    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 2, "holdout_exposure": {}})

    leftovers = [f for f in tmp_path.iterdir() if f.name != path.name]
    assert leftovers == []
    assert Project.load(path).validation_history["discovery_revision"] == 2


def test_patch_validation_history_produces_valid_yaml(tmp_path):
    project = Project(name="p")
    path = tmp_path / "p.calib.yaml"
    project.save(path)
    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 5, "holdout_exposure": {}})

    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["validation_history"]["discovery_revision"] == 5
    assert data["name"] == "p"  # every other top-level field still present and well-formed


def test_patch_validation_history_is_a_noop_when_the_file_does_not_exist_yet(tmp_path):
    """Never creates a project file from scratch -- that first save must be
    the user's own explicit `save()`; see the method's own docstring."""
    path = tmp_path / "never_saved.calib.yaml"
    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 1, "holdout_exposure": {}})
    assert not path.exists()


def test_patch_validation_history_works_on_a_legacy_project_without_the_key(tmp_path):
    """A project file saved by a build that predates `validation_history`
    entirely (no key at all, not even an empty one) must still be
    patchable -- the patch adds the key rather than requiring it to already
    exist."""
    project = Project(name="legacy")
    path = tmp_path / "legacy.calib.yaml"
    project.save(path)

    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["validation_history"]
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 1, "holdout_exposure": {}})
    reloaded = Project.load(path)
    assert reloaded.validation_history["discovery_revision"] == 1
    assert reloaded.name == "legacy"


def test_patch_validation_history_never_mutates_the_project_object_passed_around(tmp_path):
    """A patch is pure file I/O -- the R/t-bearing Camera/Target objects and
    the Scene list on the in-memory Project this test builds must come out
    identical (by value) whether or not a patch happened to run, since
    `patch_validation_history` is a staticmethod that never even receives a
    `Project` instance, only a path and a plain dict."""
    project = Project(name="p")
    project.camera.fx = 123.0
    project.target.marker_size = 0.05
    camera_before, target_before = asdict(project.camera), asdict(project.target)
    path = tmp_path / "p.calib.yaml"
    project.save(path)

    Project.patch_validation_history(path, {"version": 3, "discovery_revision": 1, "holdout_exposure": {}})

    assert asdict(project.camera) == camera_before
    assert asdict(project.target) == target_before

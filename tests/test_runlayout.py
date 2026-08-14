"""Where a run's artifacts land.

Attempt numbering is the part worth pinning: a repeat run must not adopt an earlier
attempt's directory, and a run written under the old flat layout must stay findable.
"""

from __future__ import annotations

import json

from research_fleet import runlayout


def test_slug_reduces_a_name_to_one_safe_segment():
    assert runlayout.slug("myc-discovery") == "myc-discovery"
    assert runlayout.slug("MYC discovery / v2") == "MYC-discovery-v2"
    assert "/" not in runlayout.slug("a/b/c")


def test_slug_never_returns_an_empty_segment():
    assert runlayout.slug("") == "run"
    assert runlayout.slug("///") == "run"
    assert runlayout.slug("...") == "run"


def test_repeat_runs_take_the_next_number(tmp_path):
    first = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_a")
    second = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_b")
    third = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_c")
    assert [p.name for p in (first, second, third)] == ["001", "002", "003"]
    assert first.parent == tmp_path / "wf"


def test_different_workflows_number_independently(tmp_path):
    runlayout.allocate_run_dir(tmp_path, "one", run_id="run_a")
    runlayout.allocate_run_dir(tmp_path, "one", run_id="run_b")
    other = runlayout.allocate_run_dir(tmp_path, "two", run_id="run_c")
    assert other.name == "001"


def test_a_new_attempt_never_reuses_an_occupied_directory(tmp_path):
    """A directory left by a crashed run still counts as taken -- adopting it would
    mix two runs' artifacts under one manifest."""
    (tmp_path / "wf" / "007").mkdir(parents=True)
    fresh = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_a")
    assert fresh.name == "008"
    assert not (tmp_path / "wf" / "007" / runlayout.MANIFEST).exists()


def test_the_manifest_describes_the_attempt(tmp_path):
    run_dir = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_a", based_on="run_old")
    manifest = json.loads((run_dir / runlayout.MANIFEST).read_text())
    assert manifest["run_id"] == "run_a"
    assert manifest["workflow"] == "wf"
    assert manifest["attempt"] == 1
    assert manifest["based_on"] == "run_old"
    assert manifest["continued_by"] == []


def test_a_run_is_found_by_its_id(tmp_path):
    runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_a")
    wanted = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_b")
    assert runlayout.run_dir_for(tmp_path, "run_b") == wanted


def test_a_continuation_is_found_under_either_run_id(tmp_path):
    run_dir = runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_a")
    runlayout.note_continuation(run_dir, "run_b")
    assert runlayout.run_dir_for(tmp_path, "run_a") == run_dir
    assert runlayout.run_dir_for(tmp_path, "run_b") == run_dir


def test_directories_from_the_old_flat_layout_are_still_found(tmp_path):
    (tmp_path / "run_legacy").mkdir()
    assert runlayout.run_dir_for(tmp_path, "run_legacy") == tmp_path / "run_legacy"


def test_an_unknown_run_resolves_to_nothing(tmp_path):
    runlayout.allocate_run_dir(tmp_path, "wf", run_id="run_a")
    assert runlayout.run_dir_for(tmp_path, "run_missing") is None
    assert runlayout.run_dir_for(tmp_path / "nope", "run_a") is None


def test_job_directories_are_named_for_their_stage():
    taken: set[str] = set()
    assert runlayout.job_dir_name("research", taken) == "research"
    taken.add("research")
    assert runlayout.job_dir_name("judge", taken) == "judge"


def test_a_repeated_job_name_does_not_overwrite_the_first():
    taken = {"research"}
    assert runlayout.job_dir_name("research", taken) == "research~2"


# ------------------------------------------------------- telling the agent


def _mount(source, target, mode="ro"):
    from research_fleet.spec import Mount
    return Mount(source=source, target=target, mode=mode)


def test_the_brief_names_every_stage_the_job_can_read():
    brief = runlayout.render_environment_brief([
        _mount("/h/judge", "/results", "rw"),
        _mount("/h/research", "/inputs/research"),
        _mount("/h/plan", "/inputs/plan"),
    ])
    assert "`/inputs/research`" in brief
    assert "`/inputs/plan`" in brief


def test_the_brief_explains_that_a_quoted_results_path_is_not_yours():
    """The failure this exists to prevent: a stage reports '/results/findings.md',
    the next stage looks in its own empty /results and calls the file missing."""
    brief = runlayout.render_environment_brief([
        _mount("/h/judge", "/results", "rw"),
        _mount("/h/research", "/inputs/research"),
    ])
    assert "/inputs/<that stage>/" in brief


def test_a_first_stage_is_not_told_about_inputs_it_does_not_have():
    brief = runlayout.render_environment_brief([_mount("/h/research", "/results", "rw")])
    assert "/inputs" not in brief
    assert "/previous-results" not in brief


def test_the_brief_says_whether_the_workspace_is_isolated():
    mounts = [_mount("/h/j", "/results", "rw")]
    assert "your own git worktree" in runlayout.render_environment_brief(mounts, isolated=True)
    assert "shared with every other job" in runlayout.render_environment_brief(mounts, isolated=False)


def test_an_alias_is_not_listed_as_a_second_stage():
    """`research` and `research-0` are one directory; listing both invites the agent
    to go looking for a difference that is not there."""
    stages = runlayout.input_stages([
        _mount("/h/research", "/inputs/research"),
        _mount("/h/research", "/inputs/research-0"),
        _mount("/h/probe1", "/inputs/probe-1"),
    ])
    assert stages == ["probe-1", "research"]


def test_a_chained_stage_is_told_its_tree_is_not_pristine():
    """Sequential isolated stages branch from the previous one's worktree, so files an
    earlier stage wrote are already there. 'You are isolated' alone implies otherwise."""
    mounts = [_mount("/h/j", "/results", "rw")]
    chained = runlayout.render_environment_brief(mounts, isolated=True, chained_from="fleet/run-a-plan")
    assert "already here" in chained
    fresh = runlayout.render_environment_brief(mounts, isolated=True)
    assert "already here" not in fresh
    assert "committed state" in fresh


# ------------------------------------------------------ building on a prior run


def test_stage_directories_are_keyed_by_stage_name(tmp_path):
    run_dir = tmp_path / "001"
    (run_dir / "research").mkdir(parents=True)
    (run_dir / "judge").mkdir()
    (run_dir / "run.json").write_text("{}")
    assert sorted(runlayout.stage_dirs(run_dir)) == ["judge", "research"]


def test_a_legacy_runs_job_ids_are_translated_to_stage_names(tmp_path):
    """`/previous/job_4319620b5f63` tells a later agent nothing about which directory
    holds the findings and which the review."""
    run_dir = tmp_path / "run_old"
    (run_dir / "job_abc").mkdir(parents=True)
    (run_dir / "job_def").mkdir()
    stages = runlayout.stage_dirs(run_dir, {"job_abc": "research", "job_def": "judge"})
    assert sorted(stages) == ["judge", "research"]
    assert stages["research"] == run_dir / "job_abc"


def test_an_unmapped_directory_keeps_its_id_rather_than_vanishing(tmp_path):
    run_dir = tmp_path / "run_old"
    (run_dir / "job_abc").mkdir(parents=True)
    assert list(runlayout.stage_dirs(run_dir, {})) == ["job_abc"]


def test_the_brief_points_at_the_earlier_runs_stages():
    brief = runlayout.render_environment_brief([
        _mount("/h/j", "/results", "rw"),
        _mount("/h/prev", "/previous-results"),
        _mount("/h/prev/research", "/previous/research"),
        _mount("/h/prev/judge", "/previous/judge"),
    ])
    assert "`/previous/research`" in brief and "`/previous/judge`" in brief
    assert "go further" in brief


# ------------------------------------------------------ re-running a stage


def test_a_rerun_stage_reclaims_its_own_directory(tmp_path):
    """`/inputs/<stage>` and `/previous/<stage>` resolve by directory name, so a
    resumed stage must get its name back rather than a suffix beside the failure."""
    stage = tmp_path / "001" / "second"
    stage.mkdir(parents=True)
    (stage / "output.md").write_text("second failing on purpose")

    moved = runlayout.reclaim(stage)
    assert moved is not None
    assert not stage.exists(), "the name was not freed"
    assert (moved / "output.md").read_text() == "second failing on purpose"


def test_the_previous_attempt_is_kept_not_deleted(tmp_path):
    """The trace of the failure is usually why you resumed."""
    stage = tmp_path / "001" / "second"
    stage.mkdir(parents=True)
    (stage / "stream.log").write_text("what went wrong")
    moved = runlayout.reclaim(stage)
    assert moved.parent.name == runlayout.SUPERSEDED
    assert (moved / "stream.log").exists()


def test_reclaiming_an_empty_or_absent_directory_does_nothing(tmp_path):
    empty = tmp_path / "001" / "second"
    empty.mkdir(parents=True)
    assert runlayout.reclaim(empty) is None
    assert runlayout.reclaim(tmp_path / "001" / "never") is None


def test_repeated_reruns_each_keep_their_own_archive(tmp_path):
    stage = tmp_path / "001" / "second"
    for attempt in range(1, 4):
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "output.md").write_text(f"attempt {attempt}")
        runlayout.reclaim(stage)
    archived = sorted(p.name for p in (tmp_path / "001" / runlayout.SUPERSEDED).iterdir())
    assert archived == ["second-1", "second-2", "second-3"]


def test_superseded_attempts_are_not_offered_as_stages(tmp_path):
    """Otherwise a later run building on this one would mount the failure as if it
    were the stage's result."""
    run_dir = tmp_path / "001"
    (run_dir / "second").mkdir(parents=True)
    (run_dir / runlayout.SUPERSEDED / "second-1").mkdir(parents=True)
    assert list(runlayout.stage_dirs(run_dir)) == ["second"]

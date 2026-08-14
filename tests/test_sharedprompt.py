"""The project's standing instructions to every agent.

The file is the user's, so the rules are: find it without being told, never rewrite it,
and never let a missing or oversized one take down a run.
"""

from __future__ import annotations

import pytest

from research_fleet import sharedprompt


def test_a_conventional_file_is_found_without_configuration(tmp_path):
    (tmp_path / "FLEET.md").write_text("always quantify")
    text, path = sharedprompt.load(tmp_path)
    assert text == "always quantify"
    assert path.endswith("FLEET.md")


def test_conventional_locations_are_searched_in_order(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "shared.md").write_text("second choice")
    (tmp_path / "FLEET.md").write_text("first choice")
    text, _ = sharedprompt.load(tmp_path)
    assert text == "first choice"


def test_a_configured_path_wins_over_the_conventions(tmp_path):
    (tmp_path / "FLEET.md").write_text("conventional")
    (tmp_path / "house-rules.md").write_text("configured")
    text, _ = sharedprompt.load(tmp_path, "house-rules.md")
    assert text == "configured"


def test_a_project_without_one_gets_the_packaged_default(tmp_path):
    """The generic rules are worth having everywhere, so they are not opt-in."""
    text, path = sharedprompt.load(tmp_path)
    assert path is None
    assert text == sharedprompt.default_text()
    assert "How a run is organised" in text


def test_a_projects_own_file_replaces_the_default_rather_than_appending(tmp_path):
    """Otherwise deleting a rule would not actually delete it."""
    (tmp_path / "FLEET.md").write_text("Only this.")
    text, _ = sharedprompt.load(tmp_path)
    assert text == "Only this."
    assert "How a run is organised" not in text


def test_a_configured_file_that_is_missing_falls_back(tmp_path):
    """Losing the standing rules should not take down a $13 research run."""
    text, path = sharedprompt.load(tmp_path, "gone.md")
    assert path is None
    assert text == sharedprompt.default_text()


def test_an_implausibly_large_file_is_refused(tmp_path):
    (tmp_path / "FLEET.md").write_text("x" * (sharedprompt.MAX_BYTES + 1))
    text, path = sharedprompt.load(tmp_path)
    assert text == sharedprompt.default_text()
    assert path is not None, "the caller should still learn which file was skipped"


def test_the_rendered_prompt_says_where_it_came_from(tmp_path):
    """Framing matters: unlabelled text reads as part of the task, so a project-wide
    rule gets taken as task-specific or the reverse."""
    rendered = sharedprompt.render("never touch data/")
    assert "Project instructions" in rendered
    assert "never touch data/" in rendered
    assert "task wins" in rendered


def test_a_directory_is_not_mistaken_for_the_prompt(tmp_path):
    (tmp_path / "FLEET.md").mkdir()
    assert sharedprompt.load(tmp_path) == (sharedprompt.default_text(), None)


@pytest.mark.parametrize("name", ["FLEET.md", "prompts/shared.md", ".fleet/shared.md"])
def test_each_documented_location_works(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("rules")
    assert sharedprompt.load(tmp_path)[0] == "rules"


def test_the_default_ships_with_the_package(tmp_path):
    """It must survive installation, not just exist in the checkout."""
    assert sharedprompt.DEFAULT_PATH.is_file()
    text = sharedprompt.default_text()
    assert "How a run is organised" in text
    assert 0 < len(text) < sharedprompt.MAX_BYTES


def test_the_default_says_nothing_domain_specific():
    """It is given to every project, so anything about one problem is a bug."""
    text = sharedprompt.default_text().lower()
    for word in ("myc", "rna-seq", "genome", "doxy", "omomyc", "bigwig"):
        assert word not in text, f"the default mentions {word!r}"


def test_the_default_is_about_the_layout_not_how_to_do_research():
    """It describes the directories fleet generates. What a job should conclude, and to
    what standard, is the task prompt's business -- methodology here applies itself to
    every project whether or not it fits."""
    text = sharedprompt.default_text().lower()
    for word in ("effect size", "test statistic", "falsif", "hypothes", "replicate"):
        assert word not in text, f"the default prescribes methodology: {word!r}"
    for path in ("/results", "/inputs/", "/previous/"):
        assert path in text, f"the default does not explain {path}"


def test_init_writes_the_default_into_a_project(tmp_path):
    path, outcome = sharedprompt.write_default(tmp_path)
    assert outcome == "written"
    assert path == tmp_path / "FLEET.md"
    assert sharedprompt.load(tmp_path)[0] == sharedprompt.default_text()


def test_init_never_silently_reverts_your_edits(tmp_path):
    """A default that quietly undoes an edit is worse than no default."""
    (tmp_path / "FLEET.md").write_text("my own rules")
    path, outcome = sharedprompt.write_default(tmp_path)
    assert outcome == "kept your edits"
    assert path.read_text() == "my own rules"


def test_init_can_be_forced_to_restore_the_default(tmp_path):
    (tmp_path / "FLEET.md").write_text("my own rules")
    _, outcome = sharedprompt.write_default(tmp_path, force=True)
    assert outcome == "written"
    assert sharedprompt.load(tmp_path)[0] == sharedprompt.default_text()


def test_rerunning_init_on_an_untouched_file_is_a_no_op(tmp_path):
    sharedprompt.write_default(tmp_path)
    _, outcome = sharedprompt.write_default(tmp_path)
    assert outcome == "unchanged"

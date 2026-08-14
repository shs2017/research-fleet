"""The project's standing instructions to every agent.

Fleet already tells each agent what it can spend and which directories it has. What it
cannot know is the standing rules of *this* project: how to treat an earlier attempt,
which conventions to follow, what never to touch. Without somewhere to put those, they
end up copied into every task prompt, where they drift apart and go stale.

So a workspace may hold one shared prompt, and fleet prepends it to every agent job.
It is the user's file: fleet reads it, never writes it except when asked to by `fleet
init`, and edits to it are never overwritten.

Most of what needs saying is not project-specific at all -- how stages relate, what to
do with an earlier attempt, what a report should contain -- so fleet ships that as a
default and uses it for any project that has not written its own. A project only needs
a file of its own when it wants to say something more.
"""

from __future__ import annotations

from pathlib import Path

SHARED_PROMPT_NAMES = (
    "FLEET.md",
    "fleet.md",
    "prompts/shared.md",
    ".fleet/shared.md",
)
"""Conventional locations, searched in order when none is configured."""

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "FLEET.md"
"""The instructions every project gets until it writes its own."""

MAX_BYTES = 64_000
"""Beyond this the file is more likely a mistake -- a log, a pasted dataset -- than
instructions, and it would crowd out the task itself."""


def find(workspace: Path, configured: str = "") -> Path | None:
    """Locate the shared prompt, or None.

    A configured path that does not exist returns None rather than raising: the file is
    optional by nature, and a run should not die because one was renamed. The caller
    reports the miss.
    """
    workspace = Path(workspace).expanduser()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        return candidate if candidate.is_file() else None
    for name in SHARED_PROMPT_NAMES:
        candidate = workspace / name
        if candidate.is_file():
            return candidate
    return None


def default_text() -> str:
    """The packaged default instructions, or empty if the install is missing them."""
    try:
        return DEFAULT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def load(workspace: Path, configured: str = "") -> tuple[str, str | None]:
    """Return (text, path) for the shared prompt in force.

    A project that has written its own file gets exactly that -- the default is a
    starting point, not a preamble bolted onto it, so a project that deletes a rule has
    actually deleted it. Everything else falls back to the packaged default.
    """
    path = find(workspace, configured)
    if path is None:
        return default_text(), None
    try:
        if path.stat().st_size > MAX_BYTES:
            return default_text(), str(path)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return default_text(), str(path)
    return text, str(path)


def write_default(workspace: Path, *, force: bool = False) -> tuple[Path, str]:
    """Write the default instructions into `workspace`. Returns (path, what happened).

    An existing file is never silently replaced: the whole point is that a project may
    edit it, and an edit that a later `init` quietly reverts is worse than no default
    at all.
    """
    path = Path(workspace).expanduser() / SHARED_PROMPT_NAMES[0]
    if path.exists() and not force:
        current = path.read_text(encoding="utf-8", errors="replace").strip()
        return path, "unchanged" if current == default_text() else "kept your edits"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_text() + "\n", encoding="utf-8")
    return path, "written"


def render(text: str) -> str:
    """Frame the shared prompt so its authority is unambiguous.

    An agent receives this alongside fleet's own briefs and its task. Saying where it
    came from stops it reading as part of the task -- and stops a task-specific
    instruction being taken as a project-wide rule, or the reverse.
    """
    if not text:
        return ""
    return "\n".join([
        "",
        "## Project instructions",
        "",
        "These are standing instructions for every job in this project, set by its "
        "owner. Your task follows separately; where the two genuinely conflict, the "
        "task wins, and say so in your final message.",
        "",
        text,
    ])

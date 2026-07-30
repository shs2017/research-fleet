"""The bootstrap installer.

Mirrors research-ship's flow: run it from a checkout, it puts the command on PATH under
a prefix, and the checkout becomes disposable. These tests install into a temporary
prefix, so nothing touches the user's real ~/.local.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO / "fleet"

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")


def _run(*args, **kw):
    return subprocess.run([str(BOOTSTRAP), *args], capture_output=True, text=True,
                          timeout=300, **kw)


def test_the_bootstrap_is_executable():
    assert BOOTSTRAP.exists()
    assert BOOTSTRAP.stat().st_mode & 0o111, "must be runnable straight from a checkout"


def test_help_explains_install_and_uninstall():
    out = _run("help").stdout
    assert "install" in out and "uninstall" in out


@pytest.mark.slow
def test_install_then_uninstall_roundtrip(tmp_path):
    prefix = tmp_path / "prefix"

    done = _run("install", str(prefix))
    assert done.returncode == 0, done.stderr
    binary = prefix / "bin" / "fleet"
    assert binary.exists(), "install must put fleet on PATH under the prefix"
    assert "no longer needed" in done.stdout, "should say the checkout is disposable"

    # The installed copy runs on its own, without the checkout on sys.path.
    version = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                             cwd=tmp_path, timeout=120)
    assert version.returncode == 0
    assert "workflow" in version.stdout, "the installed copy should have every command"

    # And it can remove itself, the way `ship uninstall` can.
    removed = subprocess.run([str(binary), "uninstall", str(prefix), "--yes"],
                             capture_output=True, text=True, timeout=120)
    assert removed.returncode == 0, removed.stderr
    assert "Removed" in removed.stdout
    assert "kept" in removed.stdout, "must say state was preserved"
    assert not binary.exists()


@pytest.mark.slow
def test_uninstalling_when_absent_says_so(tmp_path):
    done = _run("uninstall", str(tmp_path / "nowhere"))
    assert done.returncode == 0
    assert "not installed" in done.stdout

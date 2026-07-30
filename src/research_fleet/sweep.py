"""Parameter sweeps: the non-LLM half of the workload.

A sweep expands a grid (or an explicit list of points) into `command` jobs.
Each point's coordinates are recorded in `JobSpec.params`, so the ledger can
answer "which hyperparameters produced this result" without parsing the command
line back apart.

Templating is intentionally minimal: `{name}` placeholders in the command are
substituted with the point's values. Anything more elaborate belongs in the
script being launched, not in the launcher.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Sequence

from .spec import JobSpec, Resources


def expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid, in declaration order."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(list(grid[k]) for k in keys))]


def parse_grid_args(pairs: Iterable[str]) -> dict[str, list[Any]]:
    """Turn CLI `lr=1e-3,3e-4` strings into a grid dict, coercing numerics."""
    out: dict[str, list[Any]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected name=v1,v2, got {pair!r}")
        name, raw = pair.split("=", 1)
        values: list[Any] = []
        for token in raw.split(","):
            token = token.strip()
            for cast in (int, float):
                try:
                    values.append(cast(token))
                    break
                except ValueError:
                    continue
            else:
                values.append(token)
        out[name.strip()] = values
    return out


def _substitute(command: Sequence[str], point: dict[str, Any]) -> list[str]:
    out = []
    for token in command:
        for key, value in point.items():
            token = token.replace("{" + key + "}", str(value))
        out.append(token)
    return out


def build_sweep(
    command: Sequence[str],
    grid: dict[str, Sequence[Any]] | None = None,
    *,
    points: Sequence[dict[str, Any]] | None = None,
    image: str | None = None,
    resources: Resources | None = None,
    name_prefix: str = "sweep",
    timeout_s: int = 3600,
    env: dict[str, str] | None = None,
) -> list[JobSpec]:
    """Expand a sweep into concrete job specs.

    Pass `grid` for a cartesian product, or `points` for an explicit list (an
    agent proposing specific configs to try, for instance).
    """
    all_points = list(points) if points is not None else expand_grid(grid or {})
    specs: list[JobSpec] = []
    for i, point in enumerate(all_points):
        label = "-".join(f"{k}{v}" for k, v in point.items()) or str(i)
        specs.append(
            JobSpec(
                name=f"{name_prefix}-{label}"[:120],
                command=_substitute(command, point),
                params=dict(point),
                image=image or "",
                resources=resources or Resources(),
                timeout_s=timeout_s,
                env=dict(env or {}),
                labels={"sweep": name_prefix},
            )
        )
    return specs

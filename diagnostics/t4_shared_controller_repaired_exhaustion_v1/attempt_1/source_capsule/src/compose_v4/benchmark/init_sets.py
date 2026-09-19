"""The frozen Task 3 initialization sets, with the official ones behind a latch.

`scripts/freeze_molleo_task3_init_sets.py` drew these before any policy existed,
so no development outcome can influence which molecules the official run starts
from.

THE LATCH
---------
`official_init_set()` refuses to hand out an official set unless the caller
passes `official=True` explicitly.  The official five-seed run is spent once and
its number is saved until the policy is frozen; a keyword argument is a small
price for making "I accidentally ran it" impossible to do by mistake.  Ordinary
development calls `development_init_set()` and never sees an official molecule.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INIT_DIR = _REPO_ROOT / "artifacts" / "benchmarks" / "molleo_task3_init_v1"

OFFICIAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEVELOPMENT_SEEDS: tuple[int, ...] = tuple(range(100, 108))
POPULATION = 120


class OfficialRunNotAuthorised(RuntimeError):
    """Raised when the sealed official sets are reached for without saying so."""


@lru_cache(maxsize=4)
def _load(name: str, directory: str) -> dict:
    path = Path(directory) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/freeze_molleo_task3_init_sets.py")
    return json.loads(path.read_text())


def development_init_set(seed: int, directory: Path | None = None) -> list[str]:
    """120 ZINC-250k molecules, disjoint from every official set."""

    data = _load("development_init_sets.json",
                 str(directory or DEFAULT_INIT_DIR))
    if str(seed) not in data["sets"]:
        raise KeyError(f"seed {seed} is not a development seed; the frozen ones "
                       f"are {sorted(int(s) for s in data['sets'])}")
    return list(data["sets"][str(seed)])


def official_init_set(seed: int, *, official: bool = False,
                      directory: Path | None = None) -> list[str]:
    """120 ZINC-250k molecules for one OFFICIAL seed. Read the latch first."""

    if not official:
        raise OfficialRunNotAuthorised(
            "the official Task 3 initialization sets are sealed until the "
            "policy is frozen. Development uses development_init_set(). If this "
            "really is the official run, pass official=True and mean it.")
    data = _load("official_init_sets.json", str(directory or DEFAULT_INIT_DIR))
    if str(seed) not in data["sets"]:
        raise KeyError(f"seed {seed} is not an official seed; they are "
                       f"{sorted(int(s) for s in data['sets'])}")
    return list(data["sets"][str(seed)])


def init_provenance(directory: Path | None = None) -> dict:
    """Pool file, its sha256, and the per-set digests -- for a run manifest."""

    directory = str(directory or DEFAULT_INIT_DIR)
    official = _load("official_init_sets.json", directory)
    development = _load("development_init_sets.json", directory)
    return {
        "pool": official["pool"],
        "pool_sha256": official["pool_sha256"],
        "population": official["population"],
        "official_sets_sha256": official["sets_sha256"],
        "development_seeds": development["seeds"],
        "official_status": official["status"],
    }

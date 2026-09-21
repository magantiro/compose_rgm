#!/usr/bin/env python3
"""Fetch and hash-verify the OFFICIAL InVirtuoGen fragment-constrained evaluator.

The fragment-constrained benchmark metrics (validity / uniqueness / quality /
diversity) are defined by ``in_virtuo_gen.train_utils.metrics.evaluate_smiles``
in the InVirtuoGen results repository.  This repo deliberately does NOT vendor
that source: InVirtuoGen is released under CC BY-NC-SA 4.0 and copying it into
this tree would propagate that licence.  Instead this script downloads the exact
pinned blobs, verifies each SHA-256 against the value recorded here, and
assembles a minimal importable package in a local cache directory.

Only two import targets are stubbed, and NEITHER is reachable on the code path
this repo uses (``evaluate_smiles(..., already_smiles=True)``):

* ``preprocess.preprocess_tokenize.custom_decode_sequence`` -- only called when
  ``already_smiles`` is False.
* ``utils.fragments.bridge_smiles_fragments{,_fix}`` -- likewise.

Every function that actually computes a reported number (``is_valid_smiles``,
``compute_properties``, ``is_drug_like_and_synthesizable``,
``calculate_average_tanimoto``) comes from the verified upstream bytes.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
REPO = "invirtuolabs/InVirtuoGen_results"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"

# Upstream path -> (destination inside the assembled package, pinned sha256).
OFFICIAL_BLOBS: dict[str, tuple[str, str]] = {
    "in_virtuo_gen/train_utils/metrics.py": (
        "in_virtuo_gen/train_utils/metrics.py",
        "3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099",
    ),
    "in_virtuo_gen/utils/mol.py": (
        "in_virtuo_gen/utils/mol.py",
        "ed5609c72a3effe2547f4b264e35be8134affccc206978605307d1365a028145",
    ),
    "in_virtuo_gen/evaluation/downstream.py": (
        "in_virtuo_gen/evaluation/downstream.py",
        "ab34f586b41f1ad6e21cd933b08e7ff86eb44711c9c9a4c31c29f894cc9c7b50",
    ),
    "references/frags_downstream.csv": (
        "references/frags_downstream.csv",
        "dceeec39f389dda928a210694449948645be73d12da0ad93bc843ba1f2bb5975",
    ),
    "references/reference_metrics.csv": (
        "references/reference_metrics.csv",
        "e75676431628987d3d0bacbafc3028ffd7ebc8d797ecb03fb386b13a453823f9",
    ),
}

_STUB_TOKENIZE = '''"""Stub. Unreachable when evaluate_smiles(already_smiles=True)."""


def custom_decode_sequence(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "custom_decode_sequence is not part of the already_smiles=True metric path"
    )
'''

_STUB_FRAGMENTS = '''"""Stub. Unreachable when evaluate_smiles(already_smiles=True)."""


def bridge_smiles_fragments(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "bridge_smiles_fragments is not part of the already_smiles=True metric path"
    )


def bridge_smiles_fragments_fix(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "bridge_smiles_fragments_fix is not part of the already_smiles=True metric path"
    )
'''


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / ".official_eval_cache"


def fetch(cache_dir: Path | None = None, *, offline_ok: bool = True) -> Path:
    """Materialise the verified package. Returns the directory to put on sys.path."""
    cache_dir = cache_dir or default_cache_dir()
    pkg_root = cache_dir / "pkg"
    for upstream, (dest, expected) in OFFICIAL_BLOBS.items():
        out = pkg_root / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and _sha256(out) == expected:
            continue
        if not offline_ok and out.exists():
            raise RuntimeError(f"hash drift for cached {dest}")
        data = urllib.request.urlopen(f"{RAW}/{upstream}", timeout=60).read()
        got = hashlib.sha256(data).hexdigest()
        if got != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {upstream}: expected {expected}, got {got}. "
                "Refusing to use an unverified official evaluator."
            )
        out.write_bytes(data)

    # Package scaffolding + the two unreachable stubs.
    (pkg_root / "in_virtuo_gen" / "__init__.py").write_text("")
    for sub in ("train_utils", "utils", "evaluation", "preprocess", "models"):
        d = pkg_root / "in_virtuo_gen" / sub
        d.mkdir(parents=True, exist_ok=True)
        init = d / "__init__.py"
        if not init.exists():
            init.write_text("")
    (pkg_root / "in_virtuo_gen" / "preprocess" / "preprocess_tokenize.py").write_text(
        _STUB_TOKENIZE
    )
    (pkg_root / "in_virtuo_gen" / "utils" / "fragments.py").write_text(_STUB_FRAGMENTS)
    return pkg_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_only(cache_dir: Path | None = None) -> dict[str, str]:
    """Re-verify every cached official blob; raise on drift."""
    cache_dir = cache_dir or default_cache_dir()
    pkg_root = cache_dir / "pkg"
    seen: dict[str, str] = {}
    for dest, expected in OFFICIAL_BLOBS.values():
        out = pkg_root / dest
        if not out.exists():
            raise FileNotFoundError(f"official blob not fetched: {dest}")
        got = _sha256(out)
        if got != expected:
            raise RuntimeError(f"SHA-256 drift for {dest}: {got} != {expected}")
        seen[dest] = got
    return seen


if __name__ == "__main__":
    root = fetch()
    for dest, digest in verify_only().items():
        print(f"{digest}  {dest}")
    print(f"package root: {root}", file=sys.stderr)

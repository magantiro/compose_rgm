"""Enforcement: molecular occupancy must go through the authoritative real-slot predicate.

States are SLOT-STABLE. A deleted atom leaves a NULL slot mid-array, and deletion can also leave a SCAR
(``SCAR_IDX``), an inert marker that is occupied but is NOT a real element. So the natural-looking
expressions are all wrong:

    atom_types >= 0            NULL_IDX == 0, so padding counts as an atom
    atom_types != NULL_IDX     a SCAR counts as an atom
    array[:n_real_atoms]       silently drops trailing real atoms after a mid-array delete
    count_nonzero(atom_types)  same failure as `>= 0`

The correct predicates are ``is_element`` (real chemical element -- valence, identity, aromaticity) and
``is_occupied`` (element OR scar -- connectivity, components).

This is enforced rather than documented because the trap has now been hit twice by someone who knew about
it: the unsafe representation makes the wrong code look natural. A guard is cheaper than the next silent
mis-measurement.

Escape hatch: put ``# slot-safe: <reason>`` on the line. Inlining the full
``(x != NULL_IDX) & (x != SCAR_IDX)`` is also accepted, since torch call sites cannot use the numpy helper.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Modules where the rule is enforced strictly: the experiment infrastructure built for the paper.
_ENFORCED = (
    "src/compose_v4/experiments",
    "scripts/run_e6_sizing_sweep.py",
)

# Known occurrences elsewhere, frozen as a ratchet: this set may SHRINK but never grow. Production training
# code is not edited mid-run, so pre-existing sites are recorded rather than rewritten.
_RATCHET: frozenset[str] = frozenset()

_UNSAFE = (
    ("atom_types_ge_zero", re.compile(r"atom_types\s*>=\s*0")),
    ("atom_types_gt_zero", re.compile(r"atom_types\s*>\s*0")),
    ("slice_n_real_atoms", re.compile(r"\[\s*:\s*(?:self\.)?n_real_atoms\s*\]")),
    ("count_nonzero_atom_types", re.compile(r"count_nonzero\(\s*[\w.]*atom_types")),
    ("null_only_comparison", re.compile(r"atom_types\s*!=\s*(?:NULL_IDX|0)\b")),
)

_SCAR_GUARD = re.compile(r"SCAR_IDX")
_EXEMPTION = re.compile(r"#\s*slot-safe:")


def _scan(paths) -> list[str]:
    findings = []
    for path in paths:
        text = path.read_text(errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if _EXEMPTION.search(line):
                continue
            for label, pattern in _UNSAFE:
                if not pattern.search(line):
                    continue
                # A NULL comparison paired with a SCAR exclusion is the full is_element semantics inlined,
                # which torch call sites legitimately need.
                if label == "null_only_comparison" and _SCAR_GUARD.search(line):
                    continue
                try:
                    shown = path.relative_to(_ROOT)
                except ValueError:
                    shown = path  # synthetic fixture outside the repo
                findings.append(f"{shown}:{number} [{label}] {line.strip()[:90]}")
    return findings


def _enforced_paths():
    paths = []
    for entry in _ENFORCED:
        target = _ROOT / entry
        if target.is_dir():
            paths.extend(sorted(target.rglob("*.py")))
        elif target.is_file():
            paths.append(target)
    return paths


def test_experiment_infrastructure_is_slot_safe():
    """The modules built for the paper must never hand-roll molecular occupancy."""
    findings = _scan(_enforced_paths())
    assert not findings, (
        "slot-unsafe occupancy test in experiment infrastructure; use is_element/is_occupied (or inline the "
        "full NULL+SCAR exclusion), or annotate with '# slot-safe: <reason>':\n  " + "\n  ".join(findings)
    )


def test_the_rest_of_the_repo_does_not_regress():
    """Ratchet: pre-existing sites are frozen, new ones fail. Production is not edited mid-run."""
    everywhere = []
    for directory in ("src", "scripts", "modal_apps"):
        base = _ROOT / directory
        if base.is_dir():
            everywhere.extend(sorted(base.rglob("*.py")))
    enforced = set(_enforced_paths())
    findings = _scan([path for path in everywhere if path not in enforced])
    locations = {finding.split(" [")[0] for finding in findings}
    new = sorted(locations - _RATCHET)
    assert not new, (
        "new slot-unsafe occupancy test outside the enforced set:\n  " + "\n  ".join(new)
    )


def test_the_scan_actually_detects_each_unsafe_pattern(tmp_path):
    """A guard that cannot fire is worthless -- exercise every pattern against a synthetic offender."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "real = atom_types >= 0\n"
        "n = np.count_nonzero(atom_types)\n"
        "head = charges[:n_real_atoms]\n"
        "mask = atom_types != NULL_IDX\n"
        "positive = atom_types > 0\n"
    )
    findings = _scan([offender])
    labels = {finding.split("[")[1].split("]")[0] for finding in findings}
    assert labels == {label for label, _ in _UNSAFE}, labels


def test_the_scan_accepts_the_safe_forms(tmp_path):
    safe = tmp_path / "safe.py"
    safe.write_text(
        "real = is_element(state.atom_types)\n"
        "occupied = is_occupied(state.atom_types)\n"
        "torch_real = (atom_types != NULL_IDX) & (atom_types != SCAR_IDX)\n"
        "legacy = atom_types != NULL_IDX  # slot-safe: scars impossible in this codepath\n"
    )
    assert _scan([safe]) == []

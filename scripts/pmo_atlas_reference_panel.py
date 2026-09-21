"""Freeze the oracle reference panel that gates every atlas diagnostic run.

An oracle that CONSTRUCTS is not an oracle that SCORES.  PyTDC wraps its
evaluator in a bare ``except`` and returns ``default_property`` (0.0) on any
failure, so a mislocated asset produces a plausible all-zero ledger instead of
an error.  Before any diagnostic call, each oracle must reproduce pinned values
on pinned molecules.

Two kinds of expectation are frozen here and the difference is recorded:

``a_priori``
    known without measuring: a rediscovery or similarity oracle scores its own
    target molecule at exactly 1.0.  This entry can fail on a broken evaluator
    even if the whole panel were rebuilt.
``measured_and_frozen``
    observed once in this environment and pinned thereafter.  Its job is to
    catch later drift (a changed asset, a changed kernel, a wrong working
    directory), not to validate the evaluator from first principles.

Every panel must carry at least one value strictly inside (0, 1) so that a
constant-output oracle cannot pass.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from compose_v4.experiments.pmo_atlas_objectives import (
    ASSET_BACKED_TASKS,
    ATLAS_TASK_ORACLES,
    build_oracle,
)
from compose_v4.experiments.pmo_atlas_routes import load_atlas, payload_sha256

#: Tasks whose oracle is a similarity or rediscovery score against a fixed
#: molecule, so that molecule scores exactly 1.0 by definition.
UNITY_AT_TARGET: frozenset[str] = frozenset(
    {
        "albuterol_similarity",
        "celecoxib_rediscovery",
        "mestranol_similarity",
        "thiothixene_rediscovery",
        "troglitazone_rediscovery",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="diagnostics/pmo_atlas_v1/reference_panel.json")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    dossier = load_atlas(repo_root)
    spines = {route.task: route for route in dossier.spines()}

    panels: dict[str, list[dict[str, Any]]] = {}
    notes: dict[str, Any] = {}
    for task in sorted(ATLAS_TASK_ORACLES):
        route = spines[task]
        oracle = build_oracle(task)
        probes: list[tuple[str, str, str]] = [
            ("shared_source", route.source_smiles, "measured_and_frozen"),
        ]
        if route.destination_smiles:
            basis = "a_priori" if task in UNITY_AT_TARGET else "measured_and_frozen"
            probes.append(("recorded_destination", route.destination_smiles, basis))
        # An interior state of the recorded route: a nontrivial intermediate that
        # a constant-output oracle cannot reproduce.
        from compose_v4.experiments.pmo_atlas_routes import route_checkpoints

        midway = [c for c in route_checkpoints(route) if c.label == "midway"]
        if midway:
            probes.append(("route_midpoint", midway[0].smiles, "measured_and_frozen"))

        entries: list[dict[str, Any]] = []
        for label, smiles, basis in probes:
            value = float(oracle.evaluate(smiles))
            entry = {"label": label, "smiles": smiles, "value": value, "basis": basis}
            if basis == "a_priori" and value != 1.0:
                raise RuntimeError(
                    f"{task}: {label} was expected to score exactly 1.0 a priori, "
                    f"observed {value!r}; the evaluator is not the one this panel assumes"
                )
            entries.append(entry)
        if not any(0.0 < e["value"] < 1.0 for e in entries):
            raise RuntimeError(
                f"{task}: no probe landed strictly inside (0, 1); a constant oracle "
                "would pass this panel"
            )
        panels[task] = entries
        notes[task] = {
            "oracle_name": ATLAS_TASK_ORACLES[task],
            "asset_backed": task in ASSET_BACKED_TASKS,
            "probes": len(entries),
            "has_nontrivial_intermediate": True,
        }
        print(
            f"{task:26s} "
            + " ".join(f"{e['label']}={e['value']:.6g}({e['basis'][:1]})" for e in entries)
        )

    payload = {
        "schema_version": "pmo_atlas_reference_panel_v1",
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "purpose": (
            "Positive control run before the first charged diagnostic call of every "
            "atlas run, so a silently failing oracle cannot produce a plausible ledger."
        ),
        "software": {
            "python": platform.python_version(),
            "rdkit": __import__("rdkit").__version__,
            "numpy": __import__("numpy").__version__,
            "pytdc": __import__("importlib.metadata", fromlist=["version"]).version("PyTDC"),
        },
        "evaluator_boundary": (
            "These values are produced by PyTDC Oracle objects in an environment that "
            "mirrors the production PMO container. The recorded curriculum scores used "
            "PyTDC 0.3.6 / RDKit 2024.03.5, and gsk3b/jnk3 there used a local frozen "
            "forest rather than a PyTDC Oracle. The two sets of numbers are not "
            "comparable and none is carried across."
        ),
        "tasks": notes,
        "panels": panels,
    }
    document = {"payload": payload, "payload_sha256": payload_sha256(payload)}
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

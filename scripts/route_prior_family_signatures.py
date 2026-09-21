"""MEASURE which executor rules each production module family emits.

The teacher corpus speaks in primitive executor rules; the production
synthesizer speaks in the thirteen generic module families.  The map between
those two vocabularies is a property of the production code, so it is MEASURED
by compiling every family on real drug-like molecules and recording what comes
out -- never assumed from a family's name.

The substrate is deliberately GENERIC unlabeled chemistry (the ZINC-derived
h_phi pool), disjoint from every molecule the prior is fitted on or evaluated
against, so this statistic cannot carry a held-out source into the fit.  It uses
no teacher label of any kind; it is an instrument reading, not supervision.

    PYTHONPATH=src python scripts/route_prior_family_signatures.py \
        --out diagnostics/route_program_prior/emission_profile_v1.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import (
    GENERIC_MODULES,
    compile_generic_module,
)
from compose_v4.control.route_program_prior import RULE_VOCABULARY

SCHEMA_VERSION = "route_prior_emission_profile_v1"

#: The refusal surface of `compile_generic_module`, stated explicitly rather
#: than caught blind. `InvalidRewrite` subclasses `ValueError`, so it is named
#: only for documentation; the other four are the remainder of the set
#: `t4_fiber_campaign.expand` swallows per draw. A refusal is a real refusal of
#: the module on this molecule, which is why it is COUNTED and reported rather
#: than dropped.
_REFUSALS = (ValueError, RuntimeError, KeyError, IndexError, TypeError)

#: Slot capacity of the T4 proposal path. `t4_fiber_campaign` builds every
#: parent as `pad_molecular_graph(..., 48)`, and `whole_ring_plan` refuses any
#: other width, so 48 -- not the editing corpus's 40 -- is the number here.
#: 40 is a separate quantity: the heavy-atom ceiling the fiber applies to
#: ENDPOINTS. Confusing the two has invalidated measurements before.
PROPOSAL_SLOTS = 48


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"


def measure(smiles_list, *, replicates: int, seed: int) -> dict:
    counts: dict[str, collections.Counter] = {
        family: collections.Counter() for family in GENERIC_MODULES
    }
    compiled = collections.Counter()
    refused = collections.Counter()
    unpaddable = 0
    for index, smiles in enumerate(smiles_list):
        try:
            graph = pad_molecular_graph(
                smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS
            )
        except (ValueError, KeyError):
            unpaddable += 1
            continue
        for family in GENERIC_MODULES:
            for replicate in range(replicates):
                rng = np.random.default_rng(seed + 1_000_003 * index + 101 * replicate)
                try:
                    _product, stage = compile_generic_module(graph, rng, family)
                except _REFUSALS as exc:  # the synthesizer's own refusal surface
                    refused[f"{family}:{type(exc).__name__}"] += 1
                    continue
                compiled[family] += 1
                for action in stage["actions"]:
                    rule = str(action["executor_rule"])
                    if rule not in RULE_VOCABULARY:
                        raise ValueError(
                            f"{family} emitted {rule!r}, which is outside the declared "
                            "rule vocabulary; the vocabulary is stale"
                        )
                    counts[family][rule] += 1
    return {
        "counts": {f: dict(sorted(c.items())) for f, c in sorted(counts.items())},
        "compiled": dict(sorted(compiled.items())),
        "refused": dict(sorted(refused.items())),
        "unpaddable_sources": unpaddable,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="data/jin/hphi_train_1024.txt")
    parser.add_argument("--sources", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    pool_path = Path(args.pool)
    pool_bytes = pool_path.read_bytes()
    smiles = [line.strip() for line in pool_bytes.decode().splitlines() if line.strip()]
    # A deterministic, content-addressed stride: no RNG, reproducible by anyone.
    ordered = sorted(smiles, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    chosen = ordered[: args.sources]

    started = time.monotonic()
    measured = measure(chosen, replicates=args.replicates, seed=args.seed)
    elapsed = time.monotonic() - started

    families_without_emissions = sorted(
        family for family, counts in measured["counts"].items() if not counts
    )
    if families_without_emissions:
        raise SystemExit(
            "no emission measured for "
            f"{families_without_emissions}; the profile would silently drop them"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "emission_profile": measured["counts"],
        "compiled_per_family": measured["compiled"],
        "refusals": measured["refused"],
        "unpaddable_sources": measured["unpaddable_sources"],
        "substrate": {
            "path": str(pool_path),
            "sha256": hashlib.sha256(pool_bytes).hexdigest(),
            "selection": "sha256-lexicographic prefix; deterministic, no RNG",
            "sources_used": len(chosen),
            "replicates_per_family": args.replicates,
            "seed": args.seed,
            "carries_teacher_labels": False,
        },
        "proposal_slots": PROPOSAL_SLOTS,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "implementation": {
            "code_revision": _revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
            "wall_seconds": round(elapsed, 2),
            "compiles_attempted": len(chosen) * len(GENERIC_MODULES) * args.replicates,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True)
    out.write_text(body + "\n")
    print(json.dumps({
        "wrote": str(out),
        "payload_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "wall_seconds": round(elapsed, 2),
        "compiled_per_family": measured["compiled"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

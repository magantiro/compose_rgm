"""Does the learned structural law recover what the TASK-GATED law achieves?

`free_gate_margin_v1` is the measured T4 region repair: it tilts the region draw
by the child's free-gate margin -- similarity to the benchmark source, QED, SA,
heavy-atom capacity.  It works, and it is not available to PMO, because PMO
declares no similarity reference and no delta.  The learned route law is built
from structure alone, so if it reproduces the gated law's shift in EXCISION
SCALE then a PMO-usable substitute exists; if it does not, the gated law's
advantage is its task knowledge and there is nothing to transfer.

Three arms on the T4 held-out parents, production entry point, zero oracle:

    production_v1              region_law=None, the shipped default
    production_route_law       the learned structural law (leave-source-out)
    production_free_gate_v1    the engineered task-gated law, delta=0.6

The learned arm is leave-source-out; the gated arm has nothing to hold out
because it is not fitted -- which is itself worth stating beside any comparison
of the two.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from compose_v4.control.bridge_region_law import FreeFeasibilityGate, free_gate_margin_law
from compose_v4.control.route_program_prior import RouteProgramPrior
from compose_v4.experiments.route_program_prior_eval import parent_state, run_production_arm

SCHEMA_VERSION = "route_prior_gated_law_reference_v1"

#: The T4 delta this reference arm is configured for. It is a TASK input, which
#: is precisely why the gated law cannot serve a benchmark that declares none.
T4_DELTA = 0.6


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--draws", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    fit_raw = Path(args.fit).read_bytes()
    fit = json.loads(fit_raw.decode())
    priors = {k: RouteProgramPrior.from_payload(v) for k, v in fit["priors"].items()}
    fold_of = {s: name for name, held in fit["folds"].items() for s in held}

    sources = {}
    for path in sorted(glob.glob("diagnostics/ivg_winner_paths/pairs/*.json.gz")):
        record = json.loads(gzip.decompress(Path(path).read_bytes()).decode())
        if record["payload"]["path"]["status"] != "witness_found":
            continue
        smiles = str(record["pair"]["source"])
        sources[hashlib.sha256(smiles.encode()).hexdigest()] = smiles

    started = time.monotonic()
    rows = []
    for source_id, smiles in sorted(sources.items()):
        graph = parent_state(smiles)
        seed = args.seed + int(source_id[:8], 16) % 1_000_003
        arms = {
            "production_v1": None,
            "production_route_law": priors[fold_of[source_id]],
            "production_free_gate_v1": free_gate_margin_law(
                FreeFeasibilityGate(delta=T4_DELTA), smiles
            ),
        }
        rows.append(
            {
                "source_id": source_id,
                "heavy_atoms": int(graph.n_real_atoms),
                "arms": {
                    name: run_production_arm(
                        graph, draws=args.draws, seed=seed, region_law=law
                    ).summary()
                    for name, law in arms.items()
                },
            }
        )

    def pooled(name: str) -> dict:
        got = [row["arms"][name] for row in rows]
        return {
            "complete_program_yield": float(
                np.mean([a["complete_program_yield"] for a in got])
            ),
            "released_region_size_mean": float(
                np.mean([a["released_region_size"]["mean"] for a in got])
            ),
            "released_region_size_max": float(
                np.mean([a["released_region_size"]["max"] for a in got])
            ),
            "primitive_count_mean": float(
                np.mean([a["primitive_count"]["mean"] for a in got])
            ),
            "changed_fraction_mean": float(
                np.mean([a["changed_fraction"]["mean"] for a in got])
            ),
            "distinct_endpoints_per_parent": float(
                np.mean([a["distinct_endpoints"] for a in got])
            ),
            "teacher_region_recall": float(
                np.mean([a["teacher_region_recall"] for a in got])
            ),
            "module_compiles_attempted": int(
                sum(a["work"]["module_compiles_attempted"] for a in got)
            ),
            "wall_seconds": float(sum(a["wall_seconds"] for a in got)),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "headline": {name: pooled(name) for name in rows[0]["arms"]},
        "per_parent": rows,
        "regime": {
            "production_route_law": "leave-source-out",
            "production_free_gate_v1": "not fitted; it consumes the task's own "
            f"declared delta={T4_DELTA} and the benchmark source as similarity "
            "reference, so it has nothing to hold out and is NOT available to a "
            "benchmark that declares neither",
        },
        "settings": {"draws_per_parent_per_arm": args.draws, "seed": args.seed},
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "inputs": {
            "fit": {"path": args.fit, "sha256": hashlib.sha256(fit_raw).hexdigest()}
        },
        "implementation": {
            "code_revision": _revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
            "wall_seconds": round(time.monotonic() - started, 2),
            "wall_clock_shares_this_machine": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload["headline"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

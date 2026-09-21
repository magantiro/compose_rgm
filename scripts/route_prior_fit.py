"""Build the teacher-route corpus and fit the offline structural prior.

CORPUS, AND THE EXCLUSION THIS SCRIPT CERTIFIES
-----------------------------------------------
The only admitted corpus is ``diagnostics/ivg_winner_paths/pairs/*.json.gz`` --
exact compiled witnesses carrying a T4 lead-optimization benchmark source to a
reported T4 endpoint.  These are T4 routes: docking-benchmark structure, with no
PMO oracle, no PMO task identity and no PMO selection anywhere in them.

Two route corpora on disk are DELIBERATELY NOT READ, and the script refuses to
run if it finds itself pointed at them:

* ``diagnostics/pmo_route_distillation/`` -- 184 routes whose own result file
  records "All PMO supervision is answer-known, panel-informed or
  winner-informed development evidence".  A winner route carries oracle
  information even with its score deleted, because the SELECTION of that
  molecule as a winner is itself the oracle's output.
* ``diagnostics/pmo_teacher_route_gap_v1.json`` -- routes indexed BY PMO
  OBJECTIVE (gsk3b, celecoxib_rediscovery, albuterol_similarity, ...).

This matters beyond hygiene.  The existing
``t4_compositional_structural_subgoal_generator`` fold reports
``training_trace_counts = {"pmo_dependency_region": 85, "t4_complete": 50}`` --
63% of its training routes are PMO routes -- so that generator, whatever its
merits as a T4 result, is NOT admissible as a no-prescreen PMO prior.  This
script exists partly to produce one that is.

SPLITS
------
Leave-source-out by SOURCE MOLECULE, which is the grouping the transfer question
is about.  All routes from one source stay in one side of the split; there are
fifteen source molecules and 160 routes, so a row-level split would put two
routes from the same molecule on opposite sides and measure memorization.
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
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.control.route_program_prior import (
    ProgramShape,
    RegionChoice,
    fit_route_program_prior,
    route_events,
)

SCHEMA_VERSION = "route_program_prior_fit_v1"

ADMITTED_CORPUS_GLOB = "diagnostics/ivg_winner_paths/pairs/*.json.gz"

#: Route corpora that carry PMO oracle information, directly or through the
#: selection of a winner. Named so the refusal is explicit and auditable.
FORBIDDEN_CORPORA = (
    "diagnostics/pmo_route_distillation",
    "diagnostics/pmo_teacher_route_gap_v1.json",
    "diagnostics/pmo_winner_program_curriculum",
    "diagnostics/pmo_public_winner_recovery",
)


class CorpusExclusionError(ValueError):
    """The requested corpus carries information the prior must not see."""


def assert_admissible(pattern: str) -> None:
    for forbidden in FORBIDDEN_CORPORA:
        if forbidden in pattern:
            raise CorpusExclusionError(
                f"{pattern!r} resolves inside {forbidden!r}, which carries PMO "
                "winner or panel information; a prior fitted on it cannot "
                "support a no-prescreen claim"
            )


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "unknown"


def load_corpus(pattern: str = ADMITTED_CORPUS_GLOB) -> dict:
    """Every exact witness route, with its source identity and file hash."""

    assert_admissible(pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no teacher routes matched {pattern!r}")
    routes, inputs, statuses = [], [], {}
    for path in files:
        raw = Path(path).read_bytes()
        record = json.loads(gzip.decompress(raw).decode())
        route = record["payload"]["path"]
        status = str(route["status"])
        statuses[status] = statuses.get(status, 0) + 1
        if status != "witness_found":
            continue
        source = str(record["pair"]["source"])
        routes.append(
            {
                "source_id": hashlib.sha256(source.encode()).hexdigest(),
                "pair_id": str(record["pair"]["pair_id"]),
                "states": tuple(route["states"]),
                "actions": tuple(route["actions"]),
            }
        )
        inputs.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest()})
    return {"routes": routes, "inputs": inputs, "statuses": statuses}


def extract(routes) -> tuple[list[RegionChoice], list[ProgramShape]]:
    choices, shapes = [], []
    for route in routes:
        route_choices, shape = route_events(
            route["states"], route["actions"], source_id=route["source_id"]
        )
        choices.extend(route_choices)
        shapes.append(shape)
    return choices, shapes


def source_folds(shapes, *, folds: int) -> list[tuple[str, ...]]:
    """Held-out source groups, assigned by a content hash, never by order.

    Iteration order here is file order, which is a sha256 of the pair id and so
    is not random with respect to source; assigning folds by a hash of the
    source identity makes the split reproducible by anyone and independent of
    how the directory happens to list.
    """

    sources = sorted({shape.source_id for shape in shapes})
    buckets: list[list[str]] = [[] for _ in range(folds)]
    for source in sources:
        index = int(hashlib.sha256(("fold:" + source).encode()).hexdigest(), 16) % folds
        buckets[index].append(source)
    return [tuple(sorted(bucket)) for bucket in buckets]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=ADMITTED_CORPUS_GLOB)
    parser.add_argument("--emission-profile", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    started = time.monotonic()
    emission_raw = Path(args.emission_profile).read_bytes()
    emission = json.loads(emission_raw.decode())["emission_profile"]

    corpus = load_corpus(args.corpus)
    choices, shapes = extract(corpus["routes"])
    folds = source_folds(shapes, folds=args.folds)

    all_sources = sorted({shape.source_id for shape in shapes})
    fitted = {
        "train_on_all": fit_route_program_prior(
            choices,
            shapes,
            training_identity="train_on_all",
            emission_profile=emission,
        ).to_payload()
    }
    for index, held in enumerate(folds):
        held_set = set(held)
        fold_choices = [c for c in choices if c.source_id not in held_set]
        fold_shapes = [s for s in shapes if s.source_id not in held_set]
        prior = fit_route_program_prior(
            fold_choices,
            fold_shapes,
            training_identity=f"leave_source_out_fold_{index}",
            emission_profile=emission,
        )
        payload = prior.to_payload()
        leaked = held_set & set(payload["training_sources"])
        if leaked:
            raise RuntimeError(
                f"fold {index} fitted on its own held-out sources {sorted(leaked)}"
            )
        fitted[f"fold_{index}"] = payload

    census = {
        "routes": len(corpus["routes"]),
        "statuses": corpus["statuses"],
        "distinct_sources": len(all_sources),
        "region_choices": len(choices),
        "program_shapes": len(shapes),
        "module_count_histogram": dict(
            sorted(Counter(s.module_count for s in shapes).items())
        ),
        "primitive_transitions": int(sum(s.primitive_count for s in shapes)),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "priors": fitted,
        "folds": {f"fold_{i}": list(h) for i, h in enumerate(folds)},
        "census": census,
        "exclusion_certificate": {
            "admitted_corpus": args.corpus,
            "admitted_evidence": "T4 lead-optimization witness routes (IVG endpoints)",
            "forbidden_corpora_not_read": list(FORBIDDEN_CORPORA),
            "pmo_oracle_values_read": 0,
            "pmo_task_identities_read": 0,
            "docking_scores_read": 0,
            "note": (
                "A PMO winner route carries oracle information even with its "
                "score deleted, because selecting that molecule as a winner is "
                "itself an oracle output."
            ),
        },
        "inputs": {
            "emission_profile": {
                "path": args.emission_profile,
                "sha256": hashlib.sha256(emission_raw).hexdigest(),
            },
            "teacher_routes": corpus["inputs"][:5],
            "teacher_route_count": len(corpus["inputs"]),
            "teacher_route_manifest_sha256": hashlib.sha256(
                json.dumps(corpus["inputs"], sort_keys=True).encode()
            ).hexdigest(),
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "implementation": {
            "code_revision": _revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
            "wall_seconds": round(time.monotonic() - started, 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True)
    out.write_text(body + "\n")
    print(json.dumps({"wrote": str(out), "census": census}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

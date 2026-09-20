"""The multi-target constructive-decision corpus, built from archived controller traces.

The 77 sealed teacher routes yield 97 constructive decisions. That is enough to fit the
pooled terms of the coupled law -- held-out site rank moves from 13 of 27 to 4.5 -- and
demonstrably not enough to fit the interaction: 216 ridged coefficients on ~78 training
decisions per fold made every held-out number equal or worse than switching the
interaction off. The limit measured there is POWER, not the shape of the problem, so the
response is more decisions rather than a smaller model.

Every archived controller candidate carries `trace.states` and `trace.actions` in the
same shape as a sealed route, so one decomposition serves both and a decision cannot
mean two things depending on its provenance. Across the frozen `t4_strategy_reset`
checkpoints that is roughly 15,000 constructive decisions over all five targets.

Three things this module refuses to do, each for a reason already paid for here:

- It DEDUPLICATES on the decision itself. `full146` supplies 45 of 71 checkpoints, and
  the same construction is serialised once per replicate and per arm that reached it.
  Duplicate serialisation is not repeated independent observation; left alone it would
  teach the controller's own sampling frequency as if it were chemistry.
- It carries the TARGET on every record, so leave-one-target-out folds survive into the
  fit. A protein's own archive must never inform its own held-out score.
- It records docking scores as metadata only. They are what `h_phi` is fitted against
  later; the proposal law is fitted on structure, and no score enters `q_theta`.

Reads frozen artifacts. Writes no oracle calls.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from compose_v4.control.constructive_features import mode_identity
from compose_v4.experiments.t4_constructive_prior import trace_decisions

SCHEMA_VERSION = "t4_construction_corpus_v1"

# full146_5ht1b_0_r0_checkpoint.json.gz -> arm, cell, replicate
CHECKPOINT_NAME = re.compile(
    r"^(?P<arm>[a-z0-9]+)_(?P<cell>[a-z0-9]+_\d+)_r(?P<replicate>\d+)_checkpoint"
)


def checkpoint_identity(path: Path) -> dict | None:
    match = CHECKPOINT_NAME.match(Path(path).name)
    if not match:
        return None
    cell = match.group("cell")
    return {
        "arm": match.group("arm"),
        "cell": cell,
        "target": cell.rsplit("_", 1)[0],
        "replicate": int(match.group("replicate")),
        "checkpoint": Path(path).name,
    }


def decision_key(record: dict) -> str:
    """Identity of the DECISION, not of the trace that happened to contain it."""
    payload = json.dumps(
        {
            "state": record["state"],
            "site": record["site"],
            "mode": mode_identity(record["mode"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_decisions(path: Path) -> tuple[list[dict], list[dict]]:
    """Constructive decisions from one archived controller run, plus refused traces."""
    identity = checkpoint_identity(path)
    if identity is None:
        return [], []
    payload = json.loads(gzip.decompress(Path(path).read_bytes())).get("payload") or {}
    search = payload.get("search") or {}
    observations = {
        key: value.get("score")
        for key, value in (search.get("observations") or {}).items()
        if isinstance(value, dict)
    }

    found, refused = [], []
    for entry_id, entry in (search.get("entries") or {}).items():
        trace = entry.get("trace") or {}
        if not trace.get("states") or not trace.get("actions"):
            continue
        provenance = entry.get("provenance") or {}
        try:
            decisions = trace_decisions(trace["states"], trace["actions"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            # A trace whose decomposition refuses is dropped and counted, never
            # patched into the corpus with a guessed structure.
            refused.append({"entry_id": entry_id, "reason": type(exc).__name__})
            continue
        for decision in decisions:
            found.append(
                {
                    **identity,
                    **decision,
                    "entry_id": entry_id,
                    "endpoint": entry.get("endpoint"),
                    "static_score": entry.get("static_score"),
                    "inherited_static_score": entry.get("inherited_static_score"),
                    "measured_score": observations.get(entry.get("endpoint")),
                    "parent_measured_score": provenance.get("parent_measured_score"),
                    "channel": provenance.get("channel"),
                    "source_group": entry.get("source_group"),
                    # Lineage identity. 34,073 rows are not 34,073 independent
                    # observations: they descend from 133 lineage components, so both
                    # the weighting and the split have to be lineage-aware.
                    "ancestry": [
                        step.get("entry_id")
                        for step in (entry.get("construction_ancestry") or [])
                        if isinstance(step, dict)
                    ],
                    "ancestral_primitive_edits": entry.get("ancestral_primitive_edits"),
                }
            )
    return found, refused


def collapse(records) -> tuple[list[dict], dict]:
    """One row per distinct decision, with its provenance folded in as metadata."""
    collapsed: dict[str, dict] = {}
    for record in records:
        key = decision_key(record)
        held = collapsed.get(key)
        if held is None:
            collapsed[key] = {
                **record,
                "decision_id": key,
                "serialisations": 1,
                "arms": [record["arm"]],
                "cells": [record["cell"]],
            }
            continue
        held["serialisations"] += 1
        if record["arm"] not in held["arms"]:
            held["arms"].append(record["arm"])
        if record["cell"] not in held["cells"]:
            held["cells"].append(record["cell"])
    rows = list(collapsed.values())
    for row in rows:
        row["arms"] = sorted(row["arms"])
        row["cells"] = sorted(row["cells"])
    census = {
        "raw_decisions": len(records),
        "distinct_decisions": len(rows),
        "duplicate_fraction": 1 - len(rows) / max(len(records), 1),
        "decisions_seen_in_more_than_one_cell": sum(1 for r in rows if len(r["cells"]) > 1),
    }
    return rows, census


def assert_folds_disjoint(rows) -> None:
    """A decision appearing under two targets would leak across the folds that grade it."""
    targets: dict[str, set] = {}
    for row in rows:
        targets.setdefault(row["decision_id"], set()).add(row["target"])
    crossing = sorted(key for key, seen in targets.items() if len(seen) > 1)
    if crossing:
        raise ValueError(f"{len(crossing)} decisions appear under more than one target fold")


def census(rows) -> dict:
    """What the corpus holds, per target and per construction mode."""
    return {
        "schema_version": SCHEMA_VERSION,
        "decisions": len(rows),
        "per_target": dict(sorted(Counter(r["target"] for r in rows).items())),
        "per_arm": dict(sorted(Counter(next(iter(r["arms"])) for r in rows).items())),
        "distinct_modes": len({mode_identity(r["mode"]) for r in rows}),
        "distinct_source_states": len({r["decision_id"][:32] for r in rows}),
        "modes_present_in_every_target": sum(
            1
            for mode in {mode_identity(r["mode"]) for r in rows}
            if len({r["target"] for r in rows if mode_identity(r["mode"]) == mode}) == 5
        ),
        "decisions_with_a_measured_score": sum(
            1 for r in rows if r.get("measured_score") is not None
        ),
        "new_oracle_calls": 0,
    }


def build(raw_root: Path, destination: Path, *, progress=None) -> dict:
    """Extract, collapse and write the corpus. Returns its census."""
    checkpoints = sorted(Path(raw_root).glob("*_checkpoint.json.gz"))
    records, refused = [], []
    for index, path in enumerate(checkpoints, start=1):
        found, rejects = checkpoint_decisions(path)
        records.extend(found)
        refused.extend(rejects)
        if progress is not None:
            progress(index, len(checkpoints), len(records), path.name)
    rows, collapse_census = collapse(records)
    assert_folds_disjoint(rows)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        **census(rows),
        **collapse_census,
        "checkpoints": len(checkpoints),
        "destination": str(destination),
    }

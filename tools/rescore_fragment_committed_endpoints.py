#!/usr/bin/env python3
"""Re-score the secondary fragment metrics over the COMMITTED endpoints.

Uniqueness, quality and diversity were previously scored only over the EMITTED
set -- the one the task filter had already censored.  For motif extension that
set excluded 51% of what the generator committed, and for scaffold decoration
96%, so those numbers described the survivors rather than the generator, and
they could not be repaired because only counters and five example SMILES had
been kept.

The runner now persists every committed endpoint and every emitted sample, so
both denominators are recoverable from the artifact without re-running the
sampler.  This tool reports them side by side:

    over EMITTED    the published protocol's denominator: attempts, with a
                    censored endpoint counted as an invalid sample.
    over COMMITTED  what the generator actually produced, before the task
                    filter. Validity here is 100% by construction, so
                    uniqueness / quality / diversity describe the generator.

Both are computed by the same verified upstream ``evaluate_smiles``; only the
list handed to it differs, and each block records its own denominator.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from compose_v4.benchmark.fragment_official_metrics import (
    official_prompt_metrics,
    official_unique_valid,
)

KEYS = ("validity", "uniqueness", "quality", "diversity")


def canonicalise(samples: list[str]) -> list[str]:
    """Official uniqueness de-duplicates the RAW string, so spell them one way.

    Committed endpoints come from ``molecular_graph_to_smiles`` while emitted
    samples come from the constraint checker's isomeric canonicalisation. Two
    spellings of one molecule would inflate uniqueness, so both lists are put
    through the same canonical form before scoring.
    """
    out = []
    for sample in samples:
        mol = Chem.MolFromSmiles(sample) if sample else None
        out.append(Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else "")
    return out


def score(samples: list[str], *, denominator: int) -> dict[str, float] | None:
    if denominator <= 0:
        return None
    padded = samples + [""] * (denominator - len(samples))
    return official_prompt_metrics(padded[:denominator], expected_samples=denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    per_task: dict[str, list[dict]] = defaultdict(list)
    missing_molecules = []
    for path in sorted(args.shards.glob("*.json")):
        shard = json.loads(path.read_text())
        for task, result in shard["results"].items():
            for drug, entries in result["per_drug"].items():
                for row in entries:
                    committed = row.get("committed_endpoint_smiles")
                    emitted = row.get("emitted_samples")
                    if committed is None or emitted is None:
                        missing_molecules.append(f"{task}/{drug}/seed{row['seed']}")
                        continue
                    committed = canonicalise(list(committed))
                    emitted = canonicalise(list(emitted))
                    per_task[task].append(
                        {
                            "drug": drug,
                            "seed": int(row["seed"]),
                            "attempts": int(row["attempts"]),
                            "committed": len(committed),
                            "over_emitted": score(
                                emitted, denominator=int(row["attempts"])
                            ),
                            "over_committed": score(
                                committed, denominator=len(committed)
                            ),
                            "unique_committed": len(official_unique_valid(committed)),
                        }
                    )

    if missing_molecules:
        raise SystemExit(
            "these shards predate molecule persistence and cannot be re-scored: "
            f"{missing_molecules[:5]} ({len(missing_molecules)} total)"
        )

    tasks = {}
    for task, rows in per_task.items():
        block = {}
        for scope in ("over_emitted", "over_committed"):
            usable = [r[scope] for r in rows if r[scope] is not None]
            block[scope] = {
                key: {
                    "mean": statistics.fmean([m[key] for m in usable]),
                    "std": statistics.pstdev([m[key] for m in usable]),
                }
                for key in KEYS
            } if usable else None
            block[f"{scope}_rows"] = len(usable)
        block["committed_endpoints"] = sum(r["committed"] for r in rows)
        block["unique_committed_endpoints"] = sum(r["unique_committed"] for r in rows)
        tasks[task] = block

    payload = {
        "schema": "compose_fragment_secondary_rescore_v1",
        "denominators": {
            "over_emitted": (
                "attempts; a censored or failed attempt counts as an invalid "
                "sample. This is the published protocol's denominator."
            ),
            "over_committed": (
                "endpoints the generator actually committed. Validity is 100% "
                "by construction, so the other three describe the generator "
                "rather than the subset that survived the task filter."
            ),
        },
        "metric_source": (
            "official in_virtuo_gen.train_utils.metrics.evaluate_smiles @ b50bb3ae, "
            "already_smiles=True; only the list handed to it differs"
        ),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"{'task':26s} {'scope':16s} " + " ".join(f"{k:>10s}" for k in KEYS))
    for task, block in tasks.items():
        for scope in ("over_emitted", "over_committed"):
            stats = block[scope]
            if stats is None:
                continue
            print(
                f"{task:26s} {scope:16s} "
                + " ".join(f"{stats[k]['mean']:>10.2f}" for k in KEYS)
            )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

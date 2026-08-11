"""Census: are there enough held-out real 4-6 step transformations to run on?

READ-ONLY. No model, no checkpoint, no rollouts, no Modal. This measures whether
the experiment is well-posed; it selects nothing and freezes nothing.

WHAT IT ANSWERS
---------------
1. A strict holdout FUNNEL over verified 4-6 step `real_endpoint_multistep_path`
   traces: source held out, both endpoints absent from the training-source
   universe, no transition of the trace used in training. Intermediate overlap
   is reported descriptively and never excludes.
2. Reference-path SHAPE: does similarity-to-target rise at every step, or dip?
3. Transformation DIVERSITY: heavy-atom change, ring/topology change, operator
   mix, and the 4/5/6 split.
4. Panel SUFFICIENCY against the 50-100 pair threshold.

TRACES ARE RECONSTRUCTED, NOT READ
----------------------------------
The consolidated corpus dropped `trace_id`/`step_index` -- it keeps only what
the training path reads. But every entry carries `source_state_sha256` and
`target_state_sha256`, and within a trace consecutive transitions chain as
target(t) == source(t+1). So traces are rebuilt by hash-joining transitions.

A state can appear in more than one trace with a different recorded successor,
which would silently merge two traces into one wrong chain. Every such fork is
counted and the chain is abandoned rather than guessed at, so an ambiguous join
shows up as a dropped trace rather than as a fabricated transformation.

NAMING
------
`REFERENCE_DIP` means THIS verified path is non-monotone. It does NOT mean no
monotone path exists -- proving that needs an exhaustive monotone search which
is not run here. The stronger label is deliberately not claimed.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The provenance lane holding real-endpoint multi-step transformations.
REAL_LANE = "real_endpoint_multistep_path"
#: Verified executable path length -- NOT a claim of minimality. No shortest-path
#: search is run, and none is needed: the experiment wants verified reachable
#: transformations at a matched budget.
MIN_STEPS = 4
MAX_STEPS = 6
#: Panel size at which the strict set carries the main experiment on its own.
SUFFICIENT = 50


def load_records(path: Path) -> list[dict[str, Any]]:
    consolidated = path / "LIBRARY.jsonl.gz"
    if not consolidated.exists():
        raise SystemExit(f"no consolidated library at {consolidated}")
    with gzip.open(consolidated, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_chains(records: list[dict[str, Any]]) -> tuple[list[list[dict]], dict[str, int]]:
    """Rebuild traces by chaining transitions on state hashes."""

    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    targets: set[str] = set()
    for record in records:
        by_source[record["source_state_sha256"]].append(record)
        targets.add(record["target_state_sha256"])

    forks = sum(1 for rows in by_source.values() if len(rows) > 1)
    heads = [rows[0] for source, rows in by_source.items()
             if source not in targets and len(rows) == 1]

    chains: list[list[dict]] = []
    abandoned = 0
    for head in heads:
        chain = [head]
        seen = {head["source_state_sha256"]}
        while True:
            nxt = by_source.get(chain[-1]["target_state_sha256"], [])
            if not nxt:
                break
            if len(nxt) > 1:
                # Ambiguous continuation: abandon rather than guess, so a fork
                # becomes a dropped trace instead of a fabricated one.
                abandoned += 1
                break
            step = nxt[0]
            if step["source_state_sha256"] in seen:
                break
            seen.add(step["source_state_sha256"])
            chain.append(step)
        chains.append(chain)

    stats = {"forked_states": forks, "abandoned_at_fork": abandoned,
             "chain_heads": len(heads), "chains": len(chains)}
    return chains, stats


def shape_of(keys: list[str], target: str) -> tuple[str, list[float]]:
    """REFERENCE_MONOTONE if similarity-to-target rises at every step."""

    from compose_v4.drd2_oracle import tanimoto_to

    trajectory = [tanimoto_to(target, key) for key in keys]
    rising = all(b > a for a, b in zip(trajectory, trajectory[1:]))
    return ("REFERENCE_MONOTONE" if rising else "REFERENCE_DIP"), trajectory


def describe(source: str, target: str) -> dict[str, Any]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    a, b = Chem.MolFromSmiles(source), Chem.MolFromSmiles(target)
    if a is None or b is None:
        return {"parsable": False}
    ring_a = Chem.GetSSSR(a)
    ring_b = Chem.GetSSSR(b)
    return {
        "parsable": True,
        "heavy_atoms_source": a.GetNumHeavyAtoms(),
        "heavy_atoms_target": b.GetNumHeavyAtoms(),
        "heavy_atom_delta": b.GetNumHeavyAtoms() - a.GetNumHeavyAtoms(),
        "ring_delta": int(ring_b) - int(ring_a),
        "topology_change": int(ring_b) != int(ring_a),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--reserve", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = load_records(args.corpus)
    print(f"{len(records):,} corpus transitions")

    provenance = json.load(gzip.open(args.provenance, "rt"))
    lane_by_entry = provenance["lane_by_entry_id"]
    reserve = json.load(gzip.open(args.reserve, "rt"))
    reserve_sources = set(reserve["reserve_source_keys"])
    training_sources = set(reserve["training_source_keys"])
    training_entries = set(reserve["training_entry_ids"])
    print(f"  reserve sources {len(reserve_sources):,} / training-source universe "
          f"{len(training_sources):,} / training entries {len(training_entries):,}")

    chains, join_stats = build_chains(records)
    print(f"  reconstructed {join_stats['chains']:,} chains from "
          f"{join_stats['chain_heads']:,} heads "
          f"({join_stats['forked_states']:,} forked states, "
          f"{join_stats['abandoned_at_fork']:,} abandoned at a fork)")

    lengths = collections.Counter(len(c) for c in chains)
    print(f"  chain length distribution (top): "
          f"{dict(sorted(lengths.items())[:10])}")

    # ---- funnel ---------------------------------------------------------
    funnel = collections.OrderedDict()

    def stage(name: str, rows: list) -> list:
        funnel[name] = len(rows)
        print(f"    {name:52} {len(rows):>7,}")
        return rows

    print(f"\n  HOLDOUT FUNNEL (verified {MIN_STEPS}-{MAX_STEPS} step traces)")
    banded = stage("verified 4-6 step chains",
                   [c for c in chains if MIN_STEPS <= len(c) <= MAX_STEPS])
    real = stage(f"every transition in lane {REAL_LANE}",
                 [c for c in banded
                  if all(lane_by_entry.get(e["p50_entry_sha256"]) == REAL_LANE
                         for e in c)])
    source_held = stage("source key in the matched reserve",
                        [c for c in real
                         if c[0]["teacher_successor_fiber"]["source_key"]
                         in reserve_sources])
    endpoints_clean = stage("BOTH endpoints absent from training-source universe",
                            [c for c in source_held
                             if c[0]["teacher_successor_fiber"]["source_key"]
                             not in training_sources
                             and c[-1]["teacher_successor_fiber"]["target_key"]
                             not in training_sources])
    strict = stage("no transition of the trace used in training",
                   [c for c in endpoints_clean
                    if not any(e["p50_entry_sha256"] in training_entries
                               for e in c)])

    if not strict:
        print("\n  STRICT PANEL IS EMPTY -- report the funnel, select nothing")
        args.out.write_text(json.dumps(
            {"schema": "compose.editing_v2.real_endpoint_panel_census",
             "status": "READ_ONLY_CENSUS_SELECTS_NOTHING",
             "join": join_stats, "funnel": funnel}, indent=2) + "\n")
        return 0

    # ---- shape and diversity -------------------------------------------
    print(f"\n  SHAPE AND DIVERSITY over {len(strict):,} strict traces")
    rows: list[dict[str, Any]] = []
    for chain in strict:
        source = chain[0]["teacher_successor_fiber"]["source_key"]
        target = chain[-1]["teacher_successor_fiber"]["target_key"]
        keys = [source] + [e["teacher_successor_fiber"]["target_key"] for e in chain]
        shape, trajectory = shape_of(keys, target)
        # Descriptive only: an intermediate seen in training never excludes.
        intermediates = keys[1:-1]
        rows.append({
            "source": source, "target": target, "steps": len(chain),
            "shape": shape,
            "similarity_trajectory": [round(v, 4) for v in trajectory],
            "operators": [e["model_family"] for e in chain],
            "intermediates_in_training_sources":
                sum(1 for k in intermediates if k in training_sources),
            **describe(source, target),
        })

    shapes = collections.Counter(r["shape"] for r in rows)
    by_len = collections.Counter(r["steps"] for r in rows)
    operators = collections.Counter(op for r in rows for op in r["operators"])
    topology = sum(1 for r in rows if r.get("topology_change"))
    grew = sum(1 for r in rows if r.get("heavy_atom_delta", 0) > 0)
    shrank = sum(1 for r in rows if r.get("heavy_atom_delta", 0) < 0)
    dips_per_len = {
        n: collections.Counter(r["shape"] for r in rows if r["steps"] == n)
        for n in sorted(by_len)
    }

    print(f"    REFERENCE_MONOTONE {shapes['REFERENCE_MONOTONE']:>5,}")
    print(f"    REFERENCE_DIP      {shapes['REFERENCE_DIP']:>5,}")
    print(f"    steps 4/5/6        {by_len.get(4,0):,} / {by_len.get(5,0):,} / "
          f"{by_len.get(6,0):,}")
    for n, counter in dips_per_len.items():
        print(f"      {n} steps: monotone {counter['REFERENCE_MONOTONE']:>4,}  "
              f"dip {counter['REFERENCE_DIP']:>4,}")
    print(f"    topology (ring count) change  {topology:,}")
    print(f"    heavy atoms grew / shrank     {grew:,} / {shrank:,}")
    print(f"    operator mix                  {dict(operators.most_common(8))}")
    seen_intermediates = sum(1 for r in rows
                             if r["intermediates_in_training_sources"] > 0)
    print(f"    traces with an intermediate seen in training (DESCRIPTIVE, not "
          f"excluded): {seen_intermediates:,}")

    verdict = (
        f"SUFFICIENT -- {len(strict):,} strict traces carries the main experiment"
        if len(strict) >= SUFFICIENT else
        f"THIN -- {len(strict):,} strict traces; keep strict claim-bearing and "
        f"use broader unseen-transition pairs only as secondary")
    print(f"\n  {verdict}")

    args.out.write_text(json.dumps({
        "schema": "compose.editing_v2.real_endpoint_panel_census",
        "status": "READ_ONLY_CENSUS_SELECTS_NOTHING",
        "naming": (
            "REFERENCE_DIP means THIS verified path is non-monotone. It does NOT "
            "mean no monotone path exists; no exhaustive monotone search was run, "
            "so the stronger label is not claimed."),
        "path_length_note": (
            "Verified executable path length, not shortest edit distance. No "
            "minimality is claimed or needed."),
        "join": join_stats,
        "funnel": funnel,
        "shape": dict(shapes),
        "by_steps": {str(k): v for k, v in sorted(by_len.items())},
        "shape_by_steps": {str(n): dict(c) for n, c in dips_per_len.items()},
        "topology_change": topology,
        "heavy_atoms_grew": grew, "heavy_atoms_shrank": shrank,
        "operator_mix": dict(operators),
        "traces_with_intermediate_seen_in_training": seen_intermediates,
        "sufficiency_threshold": SUFFICIENT,
        "verdict": verdict,
        "traces": rows,
    }, indent=2) + "\n")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

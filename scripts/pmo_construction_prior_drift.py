"""Does routing the chemical prior into the CONSTRUCTION draw reduce manifold drift?

`diagnostics/pmo_atlas_v1/manifold_drift_v1.json` measures the outcome that
matters: blind PMO search leaves the drug-like manifold on 10 of 11 tasks, and
celecoxib loses 0.130 median QED between the first and last quartile of its
charged calls.  %SA<=4 and mean dSA are MECHANISM metrics; drift is the outcome.

Two arms, both spending ZERO oracle calls.

ARM `paired` -- open-loop, high power.
    For every candidate the real 250-call celecoxib run charged, re-run the
    production proposal lane from that candidate's OWN parent state with
    matched seeds, once with the prior and once without.  Same parent, same
    RNG, same candidate set; only the selection rule differs, so parent and
    round variance cancel.  It answers "is the chemistry the proposal law
    produces better at each point along the real trajectory", and it CANNOT
    answer whether the drift would have been prevented, because the parents are
    the drifted run's own.

ARM `lineage` -- closed-loop, the ratchet.
    Start from the real run's 16 initialization entries and iterate the
    proposal lane for `generations` rounds, choosing parents uniformly at
    random from the population built so far.  Parent choice reads NO score and
    NO oracle, so this is a pure test of whether the proposal law alone walks
    the population off the manifold.  The two arms share a seed schedule, so
    the parent INDEX sequence is matched even though the molecules diverge --
    which is the whole point of a closed-loop arm.

The checkpoint is PROVISIONAL (`configs/ringcore_v1_checkpoint_selection.json`
forbids it for frozen results); every number here is a mechanism probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.learned_successor_prior import LearnedSuccessorPrior
from compose_v4.rewrite.trace_shard import decode_state

RDLogger.DisableLog("rdApp.*")

REPLAY = Path("/Users/rmaganti/compose_pmo_replay_data")
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
CHECKPOINT_BACKUP = Path("/Users/rmaganti/compose_ckpt_backup/ringcore_a7546e2_best.pt")
SCOPE = "3721d69851110fdd"


def _chem(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return {
        "smiles": smiles,
        "qed": float(QED.qed(mol)),
        "sa": float(sascorer.calculateScore(mol)),
        "heavy": int(mol.GetNumHeavyAtoms()),
    }


def _seed(*parts) -> np.random.SeedSequence:
    raw = hashlib.blake2b(
        "|".join(str(p) for p in parts).encode(), digest_size=8
    ).digest()
    return np.random.SeedSequence(int.from_bytes(raw, "big"))


def _propose(state, seed_sequence, prior):
    """One production proposal; returns the endpoint chemistry or None."""

    try:
        _s, _p, _b, trace, metadata = synthesize_dynamic_program(
            state,
            np.random.default_rng(seed_sequence),
            max_modules=3,
            successor_prior=prior,
        )
    except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
        return None
    chem = _chem(trace["endpoint"])
    if chem is None:
        return None
    chem["state"] = trace["states"][-1]
    chem["tagged"] = "construction_law" in metadata
    return chem


def _rounds(task: str):
    campaign = REPLAY / task / "campaign"
    return sorted(campaign.glob("round_*"))


def _charged_sequence(task: str):
    """Every entry the real run charged, with its own parent, in round order."""

    seen, sequence = set(), []
    for index, round_dir in enumerate(_rounds(task)):
        path = round_dir / "complete.json"
        if not path.exists():
            continue
        entries = json.loads(path.read_text())["snapshot"]["entries"]
        for key in sorted(entries):
            if key in seen:
                continue
            seen.add(key)
            entry = entries[key]
            sequence.append(
                {
                    "round": index,
                    "entry_id": key,
                    "endpoint": entry["endpoint"],
                    "score": entry.get("score"),
                    "source_state": entry["source_state"],
                }
            )
    return sequence


def _quartile_table(rows, key, n_quartiles=4):
    if not rows:
        return {}
    size = max(1, len(rows) // n_quartiles)
    table = {}
    for q in range(n_quartiles):
        lo = q * size
        hi = len(rows) if q == n_quartiles - 1 else (q + 1) * size
        chunk = [row[key] for row in rows[lo:hi] if row.get(key) is not None]
        table[f"Q{q + 1}"] = {
            "n": len(chunk),
            "median": round(statistics.median(chunk), 4) if chunk else None,
            "mean": round(statistics.fmean(chunk), 4) if chunk else None,
        }
    return table


def run_paired(task, prior, *, draws, limit, seed):
    sequence = _charged_sequence(task)[:limit]
    rows = []
    for index, record in enumerate(sequence):
        source = decode_state(record["source_state"])
        parent = _chem(molecular_graph_to_smiles(source))
        if parent is None:
            continue
        for draw in range(draws):
            sequence_seed = _seed(seed, task, "paired", record["entry_id"], draw)
            off = _propose(source, sequence_seed, None)
            on = _propose(source, sequence_seed, prior)
            if off is None or on is None:
                continue
            rows.append(
                {
                    "order": index,
                    "round": record["round"],
                    "parent_qed": parent["qed"],
                    "parent_sa": parent["sa"],
                    "parent_heavy": parent["heavy"],
                    "off_qed": off["qed"],
                    "on_qed": on["qed"],
                    "off_sa": off["sa"],
                    "on_sa": on["sa"],
                    "off_heavy": off["heavy"],
                    "on_heavy": on["heavy"],
                    "same": off["smiles"] == on["smiles"],
                    "on_tagged": on["tagged"],
                }
            )
    return rows


def run_lineage(task, prior, *, generations, per_generation, seed):
    """One closed-loop, score-free arm.  Returns per-generation chemistry."""

    round_zero = _rounds(task)[0] / "complete.json"
    entries = json.loads(round_zero.read_text())["snapshot"]["entries"]
    seeds_state = [entries[key]["trace"]["states"][-1] for key in sorted(entries)]
    arms = {}
    for label, active in (("off", None), ("on", prior)):
        population = [dict(row) for row in
                      [{"state": s, **(_chem(molecular_graph_to_smiles(decode_state(s))) or {})}
                       for s in seeds_state] if row.get("qed") is not None]
        history = [
            {
                "generation": 0,
                "n": len(population),
                "median_qed": round(statistics.median([r["qed"] for r in population]), 4),
                "median_sa": round(statistics.median([r["sa"] for r in population]), 4),
                "median_heavy": statistics.median([r["heavy"] for r in population]),
                "frac_qed_above_0.6": round(
                    sum(r["qed"] >= 0.6 for r in population) / len(population), 4
                ),
                "frac_sa_at_most_4": round(
                    sum(r["sa"] <= 4.0 for r in population) / len(population), 4
                ),
            }
        ]
        for generation in range(1, generations + 1):
            picker = np.random.default_rng(_seed(seed, task, "lineage", generation))
            produced = []
            for slot in range(per_generation):
                index = int(picker.integers(len(population)))
                parent = population[index]
                child = _propose(
                    decode_state(parent["state"]),
                    _seed(seed, task, "lineage", generation, slot),
                    active,
                )
                if child is not None:
                    produced.append(child)
            population = population + produced
            history.append(
                {
                    "generation": generation,
                    "produced": len(produced),
                    "n": len(population),
                    "median_qed": round(
                        statistics.median([r["qed"] for r in population]), 4
                    ),
                    "median_sa": round(
                        statistics.median([r["sa"] for r in population]), 4
                    ),
                    "median_heavy": statistics.median([r["heavy"] for r in population]),
                    "frac_qed_above_0.6": round(
                        sum(r["qed"] >= 0.6 for r in population) / len(population), 4
                    ),
                    "frac_sa_at_most_4": round(
                        sum(r["sa"] <= 4.0 for r in population) / len(population), 4
                    ),
                    "produced_median_qed": (
                        round(statistics.median([r["qed"] for r in produced]), 4)
                        if produced
                        else None
                    ),
                    "produced_median_sa": (
                        round(statistics.median([r["sa"] for r in produced]), 4)
                        if produced
                        else None
                    ),
                }
            )
        arms[label] = history
    return arms


def _paired_summary(rows):
    if not rows:
        return {}

    def delta(field):
        values = [row[f"on_{field}"] - row[f"off_{field}"] for row in rows]
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        se = sd / (len(values) ** 0.5) if values else 0.0
        return {
            "paired_mean_delta": round(mean, 4),
            "paired_se": round(se, 4),
            "sigma": round(mean / se, 2) if se else None,
            "n": len(values),
        }

    quartiles = {}
    for arm in ("off", "on"):
        quartiles[arm] = {
            "qed": _quartile_table(rows, f"{arm}_qed"),
            "sa": _quartile_table(rows, f"{arm}_sa"),
        }
    return {
        "n_paired_decisions": len(rows),
        "disagreement_rate": round(
            sum(not row["same"] for row in rows) / len(rows), 4
        ),
        "tagged_on_proposals": sum(row["on_tagged"] for row in rows),
        "delta_qed": delta("qed"),
        "delta_sa": delta("sa"),
        "delta_heavy": delta("heavy"),
        "off_frac_qed_above_0.6": round(
            sum(row["off_qed"] >= 0.6 for row in rows) / len(rows), 4
        ),
        "on_frac_qed_above_0.6": round(
            sum(row["on_qed"] >= 0.6 for row in rows) / len(rows), 4
        ),
        "off_frac_sa_at_most_4": round(
            sum(row["off_sa"] <= 4.0 for row in rows) / len(rows), 4
        ),
        "on_frac_sa_at_most_4": round(
            sum(row["on_sa"] <= 4.0 for row in rows) / len(rows), 4
        ),
        "quartiles": quartiles,
        "drift_off_Q1_to_Q4": _drift(quartiles["off"]["qed"]),
        "drift_on_Q1_to_Q4": _drift(quartiles["on"]["qed"]),
    }


def _drift(table):
    first, last = table.get("Q1", {}).get("median"), table.get("Q4", {}).get("median")
    if first is None or last is None:
        return None
    return round(last - first, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="celecoxib_rediscovery")
    ap.add_argument("--arm", default="both", choices=("paired", "lineage", "both"))
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--per-generation", type=int, default=16)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    checkpoint = CHECKPOINT if CHECKPOINT.exists() else CHECKPOINT_BACKUP
    model, _ = load_factorized_rollout_checkpoint(checkpoint, expected_scope_hash=SCOPE)
    model.eval()
    prior = LearnedSuccessorPrior(
        model,
        floor=args.floor,
        temperature=args.temperature,
        cache_entries=4096,
    )

    began = perf_counter()
    payload = {
        "schema_version": "pmo_construction_prior_drift_v1",
        "oracle_calls_spent": 0,
        "task": args.task,
        "kernel": "rdkit 2023.09.6 (PMO production)",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "status": "PROVISIONAL_EDITING_CHECKPOINT -- forbidden for frozen results",
        },
        "settings": vars(args) | {"out": str(args.out)},
        "what_this_measures": (
            "QED/SA of the production proposal stream with and without the "
            "construction-lane chemical prior.  Zero oracle calls: no score is "
            "read anywhere, and the lineage arm chooses parents uniformly."
        ),
    }
    if args.arm in ("paired", "both"):
        rows = run_paired(
            args.task, prior, draws=args.draws, limit=args.limit, seed=args.seed
        )
        payload["paired"] = _paired_summary(rows)
        payload["paired_rows"] = rows
    if args.arm in ("lineage", "both"):
        payload["lineage"] = run_lineage(
            args.task,
            prior,
            generations=args.generations,
            per_generation=args.per_generation,
            seed=args.seed,
        )
        for arm, history in payload["lineage"].items():
            payload.setdefault("lineage_drift", {})[arm] = {
                "median_qed_generation_0": history[0]["median_qed"],
                "median_qed_final": history[-1]["median_qed"],
                "drift": round(history[-1]["median_qed"] - history[0]["median_qed"], 4),
            }
    payload["elapsed_seconds"] = round(perf_counter() - began, 1)
    payload["prior_statistics"] = prior.statistics
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    reduced = {k: v for k, v in payload.items() if k != "paired_rows"}
    print(json.dumps(reduced, indent=1, sort_keys=True, default=str)[:4000])
    print("wrote", args.out)


if __name__ == "__main__":
    main()

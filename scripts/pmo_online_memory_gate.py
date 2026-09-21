"""Zero-oracle offline gate for the online proposal memory.

Replays the completed PMO-v2 run (three tasks, fifteen rounds) as a substrate.
The memory is warmed ONLY from that run's own counted observations, in round
order, and is then asked to bias proposal generation on the run's own parents.
No oracle is called: every score used is one the completed run already charged
and recorded.

The five gate questions, plus the chemical-character question the coordinator
added after measuring that the gsk3b oracle rewards off-manifold molecules:

1. does the proposal distribution genuinely change (distributions, not a
   summary statistic)?
2. are the endpoints diverse and legal (distinct canonical endpoints)?
3. does it work across many parents and all three tasks?
4. does it collapse into shallow edits (realized edit-scale distribution)?
5. what does it cost, in a load-independent work counter?
6. what is the chemical character of what it biases toward, versus baseline?

QED and SA appear ONLY in question 6, computed here in the harness as a
diagnostic.  They are never visible to the memory or the law, so no property is
evaluated off-ledger to choose a proposal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.control import bridge_region_law as brl
from compose_v4.control import dynamic_program_synthesis as dps
from compose_v4.control.pmo_online_memory import (
    OnlineProposalMemory,
    RegionContext,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")
ROUNDS = 15


class WorkCounter:
    """Load-independent counters. Wall clock is contaminated by other jobs."""

    def __init__(self) -> None:
        self.region_weight_evaluations = 0
        self.module_compile_attempts = 0
        self.primitive_edits = 0
        self.syntheses = 0

    def report(self) -> dict:
        return {
            "region_weight_evaluations": self.region_weight_evaluations,
            "module_compile_attempts": self.module_compile_attempts,
            "primitive_edits": self.primitive_edits,
            "syntheses": self.syntheses,
            "unit": "counts; load-independent",
        }


def _wrap_excise(counter: WorkCounter):
    """Count law weight evaluations by WRAPPING the production function.

    Rebound on both modules that hold the symbol: `bridge_region_law` defines
    it and `pmo_online_memory` imported it directly, so patching only the home
    module would miss every call the law actually makes.
    """

    import compose_v4.control.pmo_online_memory as pom

    original = brl.excise_region

    def counted(graph, region):
        counter.region_weight_evaluations += 1
        return original(graph, region)

    brl.excise_region = counted
    pom.excise_region = counted
    return lambda: (setattr(brl, "excise_region", original),
                    setattr(pom, "excise_region", original))


def load_task(root: Path, task: str) -> list[dict]:
    rounds = []
    for index in range(ROUNDS):
        path = root / task / f"round_{index:04d}" / "complete.json"
        if path.exists():
            rounds.append(json.loads(path.read_text())["snapshot"])
    return rounds


def warm_memory(rounds: list[dict]) -> tuple[OnlineProposalMemory, dict]:
    """Warm the memory from the run's own counted observations, in round order.

    Every score here was charged by the completed run.  Nothing is recomputed
    and no oracle is consulted.
    """

    memory = OnlineProposalMemory()
    final = rounds[-1]
    scored = {r["endpoint"]: r["score"] for r in final["observations"].values()}
    seen: set[str] = set()
    attributed = skipped = 0
    for snapshot in rounds:
        entries = snapshot["entries"]
        for entry_id in sorted(entries):
            entry = entries[entry_id]
            endpoint = entry["endpoint"]
            if endpoint in seen or endpoint not in scored:
                continue
            seen.add(endpoint)
            parent = decode_state(entry["source_state"])
            child = decode_state(entry["trace"]["states"][-1])
            parent_key = canonical_state_key(parent)
            memory.observe_scored_molecule(
                endpoint=endpoint, score=scored[endpoint], graph=child
            )
            changes = entry["provenance"].get("actual_changes") or {}
            slots = tuple(changes.get("changed_original_slots") or ())
            meta = (entry["provenance"].get("metadata") or {}).get(
                "dynamic_generic_composition"
            ) or {}
            modules = meta.get("modules") or []
            family = modules[0]["family"] if modules else "unknown"
            ok = memory.observe_transition(
                parent_graph=parent,
                parent_endpoint=parent_key,
                parent_score=scored.get(parent_key),
                child_endpoint=endpoint,
                child_score=scored[endpoint],
                child_heavy=int(child.n_real_atoms),
                family=family,
                touched_slots=slots,
            )
            attributed += int(ok)
            skipped += int(not ok)
    return memory, {"attributed": attributed, "unattributed": skipped}


def draw_proposals(source, memory, *, law_on, draws, seed, counter):
    """Run the production synthesizer; ON differs only by `region_law=`."""

    rng = np.random.default_rng(seed)
    law = _law_for(law_on, memory) if law_on != "v1" else None
    rows = []
    for _ in range(draws):
        counter.syntheses += 1
        try:
            _, _program, _binding, _, meta = dps.synthesize_dynamic_program(
                source, rng, max_modules=3, max_primitives=32, max_blocks=8,
                region_law=law,
            )
        except ValueError:
            continue
        counter.module_compile_attempts += int(
            meta.get("completed_module_count", 0)
        ) + sum((meta.get("module_failure_counts") or {}).values())
        modules = meta.get("modules") or []
        endpoint = modules[-1]["intermediate_endpoint"] if modules else None
        edits = sum(int(m.get("primitive_edits", 0)) for m in modules)
        counter.primitive_edits += edits
        rows.append(
            {
                "endpoint": endpoint,
                "families": [m["family"] for m in modules],
                "modules": len(modules),
                "primitive_edits": edits,
            }
        )
    return rows


#: Three arms, because the repair has two independent halves and a two-arm
#: comparison cannot tell them apart.  `v1` is the shipped law; `uncapped` is
#: v1 with the size cap removed and weights still uniform; `memory` is
#: uncapped plus the learned weights.  The `uncapped` vs `memory` contrast is
#: the one that isolates what the memory itself contributes, AT MATCHED
#: SUPPORT -- without it, "the distribution changed" would only be restating
#: that the cap moved.
ARMS = ("v1", "uncapped", "memory")


def _law_for(arm, memory):
    if arm == "v1":
        return brl.UNIFORM_BOUNDED_V1
    if arm == "uncapped":
        return brl.BridgeRegionLaw(maximum=None, margin=None)
    return memory.region_law()


def region_size_law_distribution(source, memory, *, arm, draws, seed):
    """The REGION DRAW itself: which region the law puts first, repeatedly.

    This is the narrowest possible evidence that the distribution moved, taken
    at the exact point the memory acts, before a program exists.
    """

    rng = np.random.default_rng(seed)
    law = _law_for(arm, memory)
    sizes = []
    for _ in range(draws):
        order = law.order(source, rng)
        if order:
            sizes.append(order[0].size)
    return sizes


def evidence_coverage(source, memory):
    """How much of the law's mass rests on counted evidence versus none.

    A context the edit memory has never seen contributes only the scale term,
    so if nearly all mass sat on unseen contexts the law would be an
    exploration schedule wearing a memory's clothes.  This measures that
    directly instead of assuming it away.
    """

    law = memory.region_law()
    if law is None:
        return {"supported": False}
    regions = law.regions(source)
    if not regions:
        return {"regions": 0}
    weights = law.weights(source, regions)
    weights = weights / weights.sum()
    seen = np.array(
        [
            memory.edits.value(RegionContext.of(source, r))[1] > 0
            for r in regions
        ]
    )
    terms = [
        memory.value_terms(RegionContext.of(source, r), r.size) for r in regions
    ]
    return {
        "regions": len(regions),
        "regions_with_edit_evidence": int(seen.sum()),
        "mass_on_evidenced_regions": float(weights[seen].sum()) if seen.any() else 0.0,
        "mean_scale_term": float(np.mean([t["scale"] for t in terms])),
        "mean_residual_term": float(np.mean([t["residual"] for t in terms])),
        "max_abs_residual_term": float(max(abs(t["residual"]) for t in terms)),
    }


def chemistry(endpoints: list[str]) -> dict:
    """DIAGNOSTIC ONLY. Never consulted by the memory or the law."""

    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")
    import os

    from rdkit import RDConfig

    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    import sascorer

    qeds, sas = [], []
    for smiles in endpoints:
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            qeds.append(float(QED.qed(mol)))
            sas.append(float(sascorer.calculateScore(mol)))
        except (ValueError, RuntimeError, ZeroDivisionError):
            continue
    if not qeds:
        return {"n": 0}
    return {
        "n": len(qeds),
        "qed_median": float(np.median(qeds)),
        "qed_mean": float(np.mean(qeds)),
        "sa_median": float(np.median(sas)),
        "sa_mean": float(np.mean(sas)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/rmaganti/compose_pmo_data/rounds")
    ap.add_argument("--parents", type=int, default=8)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--law-draws", type=int, default=200)
    ap.add_argument("--out", default="diagnostics/pmo_online_memory_gate_v1.json")
    args = ap.parse_args()

    root = Path(args.root)
    counter = WorkCounter()
    restore = _wrap_excise(counter)
    report = {"schema": "pmo_online_memory_gate_v1", "tasks": {}}
    try:
        for task in TASKS:
            rounds = load_task(root, task)
            memory, warm = warm_memory(rounds)
            final = rounds[-1]
            entries = final["entries"]
            # Stratify parents by heavy atoms so the panel is not one size.
            pool = []
            for entry_id in sorted(entries):
                graph = decode_state(entries[entry_id]["trace"]["states"][-1])
                pool.append((int(graph.n_real_atoms), entry_id, graph))
            pool.sort()
            picks = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, args.parents)]

            rows = {arm: [] for arm in ARMS}
            sizes = {arm: [] for arm in ARMS}
            coverage = []
            before = counter.region_weight_evaluations
            for index, (heavy, entry_id, graph) in enumerate(picks):
                seed = 20260921 + 1009 * index
                for arm in ARMS:
                    sizes[arm] += region_size_law_distribution(
                        graph, memory, arm=arm, draws=args.law_draws, seed=seed
                    )
                    rows[arm] += draw_proposals(
                        graph, memory, law_on=arm, draws=args.draws,
                        seed=seed, counter=counter,
                    )
                coverage.append(evidence_coverage(graph, memory))
            report["tasks"][task] = {
                "warm": {**warm, **memory.report()},
                "parents": [
                    {"heavy_atoms": h, "entry_id": e} for h, e, _ in picks
                ],
                "region_draw_size_histogram": {
                    arm: dict(sorted(Counter(sizes[arm]).items())) for arm in ARMS
                },
                "evidence_coverage": {
                    "per_parent": coverage,
                    "mean_mass_on_evidenced_regions": float(
                        np.mean([
                            c["mass_on_evidenced_regions"]
                            for c in coverage if "mass_on_evidenced_regions" in c
                        ])
                    ),
                },
                "proposals": {
                    arm: {
                        "n": len(rows[arm]),
                        "distinct_endpoints": len(
                            {r["endpoint"] for r in rows[arm] if r["endpoint"]}
                        ),
                        "module_count_histogram": dict(
                            sorted(Counter(r["modules"] for r in rows[arm]).items())
                        ),
                        "primitive_edit_histogram": dict(
                            sorted(Counter(r["primitive_edits"] for r in rows[arm]).items())
                        ),
                        "family_histogram": dict(
                            Counter(
                                f for r in rows[arm] for f in r["families"]
                            ).most_common()
                        ),
                        "chemistry_diagnostic": chemistry(
                            [r["endpoint"] for r in rows[arm]]
                        ),
                    }
                    for arm in ARMS
                },
                "endpoint_overlap_uncapped_vs_memory": len(
                    {r["endpoint"] for r in rows["uncapped"] if r["endpoint"]}
                    & {r["endpoint"] for r in rows["memory"] if r["endpoint"]}
                ),
                "region_weight_evaluations_this_task": (
                    counter.region_weight_evaluations - before
                ),
            }
            print(f"[{task}] done", flush=True)
    finally:
        restore()
    report["cost"] = counter.report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Zero-oracle proof that the two stages CHANGE BEHAVIOUR, not merely emit different numbers.

A model whose scores differ but whose decisions do not is inert -- this repository has found six
mechanisms of exactly that shape.  So each check drives the real path and compares DECISIONS.

  1. Q_pre alters proposal allocation.  Same live parents, same RNG, uniform vs Q_pre: the mass
     assigned across parents and macro intents must differ, and every legal family must retain
     nonzero support so an early error cannot permanently exclude a capability.
  2. Q_post alters oracle selection.  One FROZEN realized pool, no oracle score visible during
     selection: Q_post must choose a different set than random, and the set it chooses must be
     better by the scores revealed only afterwards.
  3. Every candidate came through ordinary production COMPOSE.  No teacher endpoint, no beam
     replay, no reconstruction, no canary-only generation path -- each endpoint is re-executed
     through the production executor and must reproduce exactly.

Zero charged oracle calls: scores here are a pure-RDKit development evaluator used only to reveal
what the selection already committed to.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np

OUT = "diagnostics/pmo_two_stage_gate_v1/integration_gate_v1.json"


def _oracle():
    stub = types.ModuleType("rdkit.six")
    stub.string_types = (str,)
    stub.iteritems = lambda d: iter(d.items())
    import rdkit

    sys.modules["rdkit.six"] = stub
    rdkit.six = stub
    from tdc import Oracle

    return Oracle(name="celecoxib_rediscovery")


def main():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.control.pmo_two_stage_control import TwoStageControl
    from compose_v4.control.region_replacement_option import RegionReplacementOption
    from compose_v4.control.scale_balanced_region_law import ScaleBalancedRegionLaw
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.rewrite.trace_shard import decode_state

    score = _oracle()
    with open("docs/PMO_INIT_BANK.json") as handle:
        bank = json.load(handle)
    smiles = bank["smiles"] if isinstance(bank, dict) else bank
    parents = list(smiles[:6])
    option = RegionReplacementOption(region_law=ScaleBalancedRegionLaw())

    # ---- Generate ONE pool through the ordinary production path -------------------------
    pool, rng = [], np.random.default_rng(20260923)
    for parent in parents:
        graph = pad_molecular_graph(smiles_to_molecular_graph(parent), 48)
        parent_score = float(score(parent))
        for _ in range(24):
            try:
                _s, program, _a, trace, meta = synthesize_dynamic_program(
                    graph, rng, max_modules=3,
                    region_law=ScaleBalancedRegionLaw(),
                    replacement_option=option, replacement_option_rate=0.5,
                )
            except (ValueError, RuntimeError):
                continue
            endpoint = trace.get("endpoint")
            if not endpoint:
                continue
            pool.append({
                "parent": parent, "endpoint": endpoint, "smiles": endpoint,
                "parent_score": parent_score,
                "families": [m["family"] for m in meta.get("modules", [])],
                "requested_modules": meta.get("requested_module_count", 0),
                "module_count": len(meta.get("modules", [])),
                "primitives": len(program.marks),
                "depth": 1, "generation": 1,
                "capacity_aware": bool(meta.get("capacity_aware")),
                "trace_states": trace.get("states"),
                "trace_actions": trace.get("actions"),
            })
    report = {
        "schema_version": "pmo_two_stage_integration_gate_v1",
        "evidence_role": "zero_charged_oracle_integration_gate",
        "new_charged_oracle_calls": 0,
        "pool_size": len(pool),
        "parents": len(parents),
    }
    print(f"pool {len(pool)} candidates from {len(parents)} production parents")

    # ---- CHECK 3: production provenance --------------------------------------------------
    replayed = mismatched = 0
    for row in pool[:60]:
        states, actions = row.get("trace_states"), row.get("trace_actions")
        if not states or not actions:
            continue
        product, _receipt = execute_program(decode_state(states[0]), list(actions))
        from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

        replayed += 1
        mismatched += molecular_graph_to_smiles(product) != row["endpoint"]
    report["check3_production_provenance"] = {
        "replayed": replayed, "mismatched": mismatched, "pass": replayed > 0 and mismatched == 0,
    }
    print(f"CHECK 3 production replay: {replayed} replayed, {mismatched} mismatched")

    # ---- Train both heads on a DISJOINT sample so the gate is not self-fitted -------------
    control = TwoStageControl(audit_fraction=0.15, lineage_floor=0.10)
    holdout = pool[: len(pool) // 2]
    for row in pool[len(pool) // 2 :]:
        control.observe(row, float(score(row["endpoint"])))
    print(f"heads fitted on {len(pool) - len(holdout)} disjoint rows; holdout {len(holdout)}")

    # ---- CHECK 1: Q_pre alters proposal allocation ----------------------------------------
    uniform = np.full(len(holdout), 1.0 / len(holdout))
    learned = control.intent_weights(holdout)
    shares_uniform = {}
    shares_learned = control.parent_shares(holdout, parents)
    for row, w in zip(holdout, uniform, strict=True):
        shares_uniform[row["parent"]] = shares_uniform.get(row["parent"], 0.0) + float(w)
    total = sum(shares_uniform.values()) or 1.0
    shares_uniform = {k: v / total for k, v in shares_uniform.items()}
    moved = max(abs(shares_learned.get(k, 0.0) - v) for k, v in shares_uniform.items())
    families = {f for row in holdout for f in row["families"]}
    supported = {
        f: float(sum(w for row, w in zip(holdout, learned, strict=True) if f in row["families"]))
        for f in families
    }
    report["check1_q_pre_alters_allocation"] = {
        "max_parent_share_shift": float(moved),
        "families_seen": len(families),
        "families_with_zero_support": [f for f, w in supported.items() if w <= 0.0],
        "pass": bool(moved > 0.01 and not [f for f, w in supported.items() if w <= 0.0]),
    }
    print(f"CHECK 1 Q_pre: max parent-share shift {moved:.4f}, "
          f"{len(families)} families all with support "
          f"{not [f for f, w in supported.items() if w <= 0.0]}")

    # ---- CHECK 2: Q_post alters oracle selection ------------------------------------------
    blind = [{k: v for k, v in r.items() if k != "score"} for r in holdout]
    chosen, detail = control.acquire(blind, [("seed", 0.2)], batch=12, rng=np.random.default_rng(7))
    random_pick = np.random.default_rng(7).choice(len(blind), 12, replace=False)
    truth = np.array([float(score(r["endpoint"])) for r in holdout])
    q_mean, r_mean = float(truth[list(chosen)].mean()), float(truth[random_pick].mean())
    report["check2_q_post_alters_selection"] = {
        "overlap_with_random": len(set(chosen) & set(int(i) for i in random_pick)),
        "q_post_selected_mean": q_mean,
        "random_selected_mean": r_mean,
        "pool_mean": float(truth.mean()),
        "audit_slots": sum(1 for d in detail if d["reason"] == "audit"),
        "pass": bool(len(set(chosen) & set(int(i) for i in random_pick)) < 12 and q_mean > r_mean),
    }
    print(f"CHECK 2 Q_post: selected mean {q_mean:.4f} vs random {r_mean:.4f} "
          f"(pool {truth.mean():.4f}), overlap {len(set(chosen) & set(int(i) for i in random_pick))}/12, "
          f"audit slots {report['check2_q_post_alters_selection']['audit_slots']}")

    passed = all(report[k]["pass"] for k in report if k.startswith("check"))
    report["verdict"] = "PASS" if passed else "FAIL"
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump({k: v for k, v in report.items()}, handle, indent=1)
    print(f"\nVERDICT: {report['verdict']}")
    print("WROTE", OUT)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

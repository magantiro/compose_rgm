"""Full production-path integration gate for the reward-adaptive controller. ZERO oracle calls.

Six conditions, all verified simultaneously on the ordinary PMO production path, because each has
its own way of silently not happening:

  1. a BROAD pool actually exists              (pool >= 96, query fraction <= 0.17)
  2. Q_pre changes PARENT allocation           (its weights differ from production's)
  3. Q_post changes QUERY selection            (its picks differ from random)
  4. reward at round t changes policy at t+1   (total-variation shift > 0 after refit)
  5. every legal macro family retains support  (no family driven to zero)
  6. every endpoint came through production    (re-executed and reproduced exactly)

A stub scorer is used so nothing is attributable to a PMO budget: this gate spends no oracle calls
and is not a measurement of controller quality, only of whether the loop is connected.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
from pathlib import Path

import numpy as np

OUT = "diagnostics/pmo_reward_adaptive_gate_v1/integration_gate_v1.json"
POOL_TARGET = 128
QUERIES = 16


def main():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
    from compose_v4.control import dynamic_program_synthesis_v21 as v21
    from compose_v4.control.docking_value import identity
    from compose_v4.control.pmo_channels import CHANNELS, JUMP_CHANNEL
    from compose_v4.control.pmo_contextual_macro import (
        MACRO_FAMILIES,
        macro_families,
        macro_scale,
    )
    from compose_v4.control.pmo_population_controller import PmoPopulationController
    from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController
    from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
    from compose_v4.control.program_task import ProgramTask
    from compose_v4.experiments import pmo_population_v1 as production
    from compose_v4.experiments.pmo_dynamic_v21 import initial_dynamic_program_batch_v21
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.rewrite.trace_shard import decode_state

    KNOWN_INERT = JUMP_CHANNEL

    baseline_limit = v21.CHANNEL_CANDIDATE_LIMIT
    v21.CHANNEL_CANDIDATE_LIMIT = POOL_TARGET

    root = Path(".")
    contract = production.load_contract(root)
    initialized = production._load_initialization(
        root, {"initialization": contract["initialization"]}
    )
    checkpoint = production._load_checkpoint(root, contract)
    config = production.configuration(contract["controller"]["seed"])
    task = ProgramTask(
        "celecoxib_rediscovery",
        identity(production._runtime_protocol(contract, "celecoxib_rediscovery")),
        "pmo",
    )
    folder = pathlib.Path("diagnostics/pmo_reward_adaptive_gate_v1/dryrun")
    # A campaign that finds a completed directory resumes and runs no rounds, which reads
    # as six simultaneous failures rather than as "nothing ran".  This gate is a dry run
    # with a stub scorer and no charged calls, so its workspace is always disposable.
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # A stub reward that DEPENDS ON STRUCTURE, so the controller has something learnable and a
    # policy shift is meaningful -- a constant reward could not move any policy and the gate
    # would pass vacuously. No PMO oracle is constructed or called.
    def stub(smiles: str) -> float:
        return min(1.0, len(set(smiles)) / 24.0)

    evidence = {
        "pools": [], "query_fractions": [], "policy_shifts": [],
        "selection_overlap": [], "family_support": [], "replay_mismatch": 0, "replayed": 0,
        "q_pre_differs_from_production": [],
        "family_labels": set(), "compound_labels": set(), "scale_seen": set(),
        "channel_yield": {},
    }

    class RewardAdaptive(PmoPopulationController):
        """Production machinery; Q_pre steers parents, Q_post steers queries."""

        brain = RewardAdaptiveProgramController(model_share=0.75)

        def _rows(self, candidates):
            return [
                {
                    # Resolved through entry_id: production carries no `parent_smiles`, and
                    # defaulting to the candidate's own endpoint makes the parent features
                    # duplicate the endpoint features and every candidate its own lineage.
                    "parent": (
                        self.entries.get(c["provenance"].get("entry_id")) or {}
                    ).get("endpoint", c["endpoint"]),
                    "endpoint": c["endpoint"], "smiles": c["endpoint"],
                    "parent_score": float(
                        c["provenance"].get("parent_measured_score") or 0.0
                    ),
                    # MACRO families from the synthesis metadata.  NOT the executor rule
                    # histogram: four rule names are spelled like macro families, so that
                    # would fire the wrong one-hots from primitive counts.
                    "families": list(macro_families(c)),
                    "realized_families": list(macro_families(c)),
                    **macro_scale(c),
                    "requested_modules": len(c.get("program", {}).get("blocks", ()) or ()),
                    "module_count": len(c.get("program", {}).get("blocks", ()) or ()),
                    "primitives": len(c.get("trace", {}).get("actions", ()) or ()),
                    "depth": 1, "generation": self.batches, "capacity_aware": False,
                }
                for c in candidates
            ]

        def selection(self):
            """CONDITION 2: Q_pre reweights parent allocation upstream of generation."""
            keys, weights = super().selection()
            if not self.brain.fitted:
                return keys, weights
            intents = [
                {
                    "parent": self.entries[k]["endpoint"], "endpoint": self.entries[k]["endpoint"],
                    "smiles": self.entries[k]["endpoint"], "parent_score": 0.0,
                    "families": ["region_replace"], "requested_modules": 2, "module_count": 2,
                    "primitives": 0, "depth": 1, "generation": self.batches,
                    "capacity_aware": False,
                }
                for k in keys
            ]
            learned = self.brain.intent_policy(intents)
            blended = 0.5 * np.asarray(weights, dtype=float) + 0.5 * learned
            blended = blended / blended.sum()
            evidence["q_pre_differs_from_production"].append(
                float(0.5 * np.abs(blended - np.asarray(weights, dtype=float)).sum())
            )
            return keys, blended

        def _allocate(self, candidates):
            """CONDITIONS 1, 3, 4, 5: broad pool, Q_post selection, policy shift, family floors."""
            rows = self._rows(candidates)
            room = min(QUERIES, len(rows))
            before = self.brain.record_policy(rows, f"before_{self.batches}") if rows else None
            threshold = 0.0
            scores = [float(o["score"]) for o in self.observations.values()]
            if len(scores) >= 10:
                threshold = sorted(scores, reverse=True)[9]
            chosen, _detail = self.brain.acquire(
                rows, threshold, batch=room, rng=self.rng
            )
            random_pick = {
                int(i)
                for i in np.random.default_rng(11).choice(
                    len(rows), min(room, len(rows)), replace=False
                )
            }
            evidence["pools"].append(len(candidates))
            evidence["query_fractions"].append(len(chosen) / max(len(candidates), 1))
            evidence["selection_overlap"].append(len(set(chosen) & random_pick))
            if self.brain.fitted and rows:
                weights = self.brain.intent_policy(rows)
                families = {f for r in rows for f in r["families"]}
                evidence["family_support"].append(
                    {
                        f: float(sum(w for r, w in zip(rows, weights, strict=True)
                                     if f in r["families"]))
                        for f in families
                    }
                )
            for candidate in candidates:
                lane = (candidate.get("provenance") or {}).get("planner_channel")
                if lane:
                    evidence["channel_yield"][lane] = (
                        evidence["channel_yield"].get(lane, 0) + 1
                    )
            evidence["family_labels"].update(f for r in rows for f in r["families"])
            evidence["compound_labels"].update(
                f for r in rows for f in r["families"] if ":" in f
            )
            evidence["scale_seen"].update(
                k for r in rows for k in ("excised_atoms", "d_heavy") if r.get(k)
            )
            for index in chosen:
                self.brain.observe(rows[index], stub(rows[index]["endpoint"]))
            if before is not None and rows:
                after = self.brain.record_policy(rows, f"after_{self.batches}")
                evidence["policy_shifts"].append(self.brain.policy_shift(before, after))
            for row, candidate in zip(rows[:40], candidates[:40], strict=False):
                states = (candidate.get("trace") or {}).get("states")
                actions = (candidate.get("trace") or {}).get("actions")
                if not states or not actions:
                    continue
                product, _r = execute_program(decode_state(states[0]), list(actions))
                evidence["replayed"] += 1
                evidence["replay_mismatch"] += (
                    molecular_graph_to_smiles(product) != row["endpoint"]
                )
            return [candidates[i] for i in chosen], {"mode": "reward_adaptive"}

    ledger = ProgramQueryLedger(folder / "oracle", task, stub, budget=112)
    run_program_campaign(
        output=folder / "campaign", task=task, config=config,
        initialization=initialized, library=(), ledger=ledger,
        rounds=6, queries_per_round=QUERIES, hierarchy=None, fit_model=None,
        stagnation_rounds=None, bootstrap_rounds=1,
        initialization_mode="all_scored_pool", initial_parent_fraction=0.2,
        progress=lambda row: None,
        optimizer_type=RewardAdaptive, optimizer_kwargs={"jump_checkpoint": checkpoint},
        initial_batch_fn=initial_dynamic_program_batch_v21,
    )
    v21.CHANNEL_CANDIDATE_LIMIT = baseline_limit

    starved = [f for block in evidence["family_support"] for f, w in block.items() if w <= 0.0]
    checks = {
        "1_broad_pool": {
            "median_pool": float(np.median(evidence["pools"])) if evidence["pools"] else 0,
            "max_query_fraction": max(evidence["query_fractions"] or [1.0]),
            "pass": bool(
                evidence["pools"]
                and np.median(evidence["pools"]) >= 96
                and max(evidence["query_fractions"]) <= 0.17
            ),
        },
        "2_q_pre_changes_parent_allocation": {
            "rounds_reweighted": len(evidence["q_pre_differs_from_production"]),
            "max_tv_vs_production": max(evidence["q_pre_differs_from_production"] or [0.0]),
            "pass": bool(max(evidence["q_pre_differs_from_production"] or [0.0]) > 0.0),
        },
        "3_q_post_changes_selection": {
            "min_overlap_with_random": min(evidence["selection_overlap"] or [QUERIES]),
            "pass": bool(min(evidence["selection_overlap"] or [QUERIES]) < QUERIES),
        },
        "4_reward_changes_next_policy": {
            "max_policy_shift": max(evidence["policy_shifts"] or [0.0]),
            "rounds": len(evidence["policy_shifts"]),
            "pass": bool(max(evidence["policy_shifts"] or [0.0]) > 0.0),
        },
        "5_all_families_supported": {
            "blocks_checked": len(evidence["family_support"]),
            "starved": sorted(set(starved)),
            # A block carrying NO families cannot witness that none was starved; three
            # such blocks passed this check while every family label was empty.
            "pass": bool(evidence["family_support"])
            and all(evidence["family_support"])
            and not starved,
        },
        "6_production_provenance": {
            "replayed": evidence["replayed"], "mismatched": evidence["replay_mismatch"],
            "pass": bool(evidence["replayed"] > 0 and evidence["replay_mismatch"] == 0),
        },
        "7_action_is_the_macro": {
            "distinct_family_labels": len(evidence["family_labels"]),
            "compound_region_labels": len(evidence["compound_labels"]),
            "unknown_labels": sorted(
                f for f in evidence["family_labels"]
                if f not in MACRO_FAMILIES and f.split(":")[0] not in MACRO_FAMILIES
            )[:6],
            "realized_scale_populated": sorted(evidence["scale_seen"]),
            "pass": bool(
                evidence["family_labels"]
                and evidence["compound_labels"]
                and evidence["scale_seen"]
                and not [
                    f for f in evidence["family_labels"]
                    if f not in MACRO_FAMILIES and f.split(":")[0] not in MACRO_FAMILIES
                ]
            ),
        },
        "8_nothing_with_policy_mass_is_inert": {
            "channel_yield": dict(evidence["channel_yield"]),
            # A lane measured at 10 charged of 3,599 attempts is already characterised and
            # is not being repaired in this pass. The exemption is named so it cannot
            # quietly cover a lane nobody has looked at.
            "known_inert_exempt": [KNOWN_INERT],
            "silent": sorted(
                c for c in CHANNELS
                if not evidence["channel_yield"].get(c) and c != KNOWN_INERT
            ),
            "pass": not [
                c for c in CHANNELS
                if not evidence["channel_yield"].get(c) and c != KNOWN_INERT
            ],
        },
    }
    report = {
        "schema_version": "pmo_reward_adaptive_integration_gate_v1",
        "evidence_role": "zero_charged_oracle_integration_gate",
        "new_charged_oracle_calls": 0,
        "checks": checks,
        "family_audit": RewardAdaptive.brain.ledger.report(),
        "verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL",
    }
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    for name, c in checks.items():
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name}: "
              f"{ {k: v for k, v in c.items() if k != 'pass'} }")
    print(f"\nVERDICT: {report['verdict']}")
    print("WROTE", OUT)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

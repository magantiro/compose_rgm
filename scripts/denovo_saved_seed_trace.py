"""Replay one saved C1 seed and locate when qualitative defects enter a trajectory.

Diagnostic only: the original model, prior, RNG and production sampler are
unchanged. A temporary observation wrapper records committed states, then is
restored. Endpoint identity must match the frozen shard before interpretation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import rdkit
import torch
from denovo_postring_quality_audit import characterize, sha256
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.eval.denovo_ring_marginal import RingSystemPlanPrior
from compose_v4.experiments import tracelet_conditional
from compose_v4.experiments.denovo_ring_plan import (
    CatalogSignatureIndex,
    sample_denovo_arm,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system

SCHEMA = "denovo_saved_seed_trace_v1"
EXPECTED_CHECKPOINT = "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c"


def replay(records_path: Path, checkpoint: Path, prior_path: Path, index: int) -> dict:
    if sha256(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    records = json.loads(records_path.read_text())
    matches = [row for row in records if row["arm"] == "C1" and row["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"expected one saved C1 record for index {index}; got {len(matches)}")
    saved = matches[0]
    torch.set_num_threads(1)
    model, payload = load_factorized_rollout_checkpoint(str(checkpoint))
    model.eval()
    prior = RingSystemPlanPrior.read(prior_path)
    catalog_index = CatalogSignatureIndex.build(model.ring_system_templates)
    states: list[dict] = []
    original = tracelet_conditional.compact_state_observation

    def observe(state):
        observation = original(state)
        smiles = molecular_graph_to_smiles(state)
        pseudo_record = {
            "smiles": smiles,
            "index": saved["index"],
            "trajectory_seed": saved["trajectory_seed"],
            "events": 0,
            "event_rules": [],
            "valid_state": True,
            "connected": True,
        }
        states.append(characterize(pseudo_record, "C1_trace", records_path))
        return observation

    tracelet_conditional.compact_state_observation = observe
    try:
        result = sample_denovo_arm(
            model,
            arm="C1",
            rng=np.random.default_rng(saved["trajectory_seed"]),
            source_prior=payload["tree_source_prior"],
            index=catalog_index,
            plan_prior=prior,
            n_slots=40,
            operational_horizon=16.0,
            time_step=0.1,
            max_events=128,
            runtime=de_novo_rewrite_system(),
        )
    finally:
        tracelet_conditional.compact_state_observation = original
    if result["smiles"] != saved["smiles"]:
        raise RuntimeError(
            f"replay endpoint mismatch for C1 index {index}: "
            f"{result['smiles']} != {saved['smiles']}"
        )
    if len(states) != len(result["event_rules"]) + 1:
        raise RuntimeError("observed states are not aligned with committed event rules")
    if states[-1]["canonical_smiles"] != saved["canonical_smiles"]:
        raise RuntimeError("final captured state differs from saved endpoint")

    deltas = Counter()
    events = []
    for step, (before, after, rule) in enumerate(
        zip(states, states[1:], result["event_rules"]), start=1
    ):
        delta_f = after["elements"].get("F", 0) - before["elements"].get("F", 0)
        delta_odd_ring = (
            after["nonaromatic_unsaturated_5_6_rings"] - before["nonaromatic_unsaturated_5_6_rings"]
        )
        delta_aromatic = after["aromatic_rings"] - before["aromatic_rings"]
        if delta_f > 0:
            deltas[f"F_birth:{rule}"] += delta_f
        if delta_odd_ring > 0:
            deltas[f"nonaromatic_unsaturated_ring_birth:{rule}"] += delta_odd_ring
        if delta_aromatic < 0:
            deltas[f"aromatic_ring_loss:{rule}"] += -delta_aromatic
        events.append(
            {
                "step": step,
                "rule": rule,
                "smiles": after["canonical_smiles"],
                "qed": after["qed"],
                "sa": after["sa"]["score"],
                "sa_fragment_score": after["sa"]["fragment_score"],
                "F_atoms": after["elements"].get("F", 0),
                "delta_F_atoms": delta_f,
                "aromatic_rings": after["aromatic_rings"],
                "delta_aromatic_rings": delta_aromatic,
                "nonaromatic_unsaturated_5_6_rings": after["nonaromatic_unsaturated_5_6_rings"],
                "delta_nonaromatic_unsaturated_5_6_rings": delta_odd_ring,
            }
        )
    return {
        "schema_version": SCHEMA,
        "evidence_class": "exact_saved_seed_replay_exploratory_local_rdkit",
        "source": {"records": str(records_path), "sha256": sha256(records_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "ring_prior": {"path": str(prior_path), "sha256": sha256(prior_path)},
        "implementation": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "script_sha256": sha256(Path(__file__)),
            "descriptor_helper_sha256": sha256(Path(characterize.__code__.co_filename)),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
        },
        "configuration": {
            "arm": "C1",
            "n_slots": 40,
            "horizon": 16.0,
            "time_step": 0.1,
            "max_events": 128,
        },
        "saved": {
            "index": index,
            "seed": saved["trajectory_seed"],
            "group": saved["group"],
            "smiles": saved["canonical_smiles"],
        },
        "replay_endpoint_exact": True,
        "initial_state": states[0]["canonical_smiles"],
        "plan": result["ring_plan"],
        "rule_event_counts": dict(sorted(Counter(result["event_rules"]).items())),
        "qualitative_event_counts": dict(sorted(deltas.items())),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.records, args.checkpoint, args.prior, args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "index": args.index,
                "events": len(result["events"]),
                "qualitative_event_counts": result["qualitative_event_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

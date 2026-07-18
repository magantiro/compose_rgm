"""Replay and render every visible state from a preserved COMPOSE checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import exp
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from scripts.evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


def trajectory_seeds(seed: int, samples: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence(seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(samples)
    )


@torch.no_grad()
def replay(
    model,
    source_prior,
    *,
    seed: int,
    n_slots: int,
    operational_horizon: float,
    time_step: float,
    max_events: int,
):
    rng = np.random.default_rng(seed)
    runtime = de_novo_rewrite_system()
    state = source_prior.sample(rng, n_slots=n_slots)
    states = [state]
    event_times: list[float] = []
    event_rules: list[str] = []
    event_actions: list[object] = []
    operational_time = 0.0
    while operational_time < operational_horizon and len(event_times) < max_events:
        interval_end = min(operational_time + time_step, operational_horizon)
        frozen_time = 1.0 - exp(-(operational_time + interval_end) / 2.0)
        while operational_time < interval_end and len(event_times) < max_events:
            sampled = model.sample_rewrite_mark(state, frozen_time, rng)
            total_hazard = float(sampled.total_hazard)
            if total_hazard <= 1e-12:
                operational_time = interval_end
                break
            waiting_time = float(rng.exponential(1.0 / total_hazard))
            if waiting_time >= interval_end - operational_time:
                operational_time = interval_end
                break
            operational_time += waiting_time
            state = runtime.apply(state, sampled.rule_name, sampled.action)
            if not is_valid_state(state) or not is_connected_or_null(state):
                raise RuntimeError("the validity-closed runtime emitted an invalid state")
            states.append(state)
            event_times.append(operational_time)
            event_rules.append(sampled.rule_name)
            event_actions.append(sampled.action)
    return states, event_times, event_rules, event_actions


def molecule_and_smiles(state):
    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise RuntimeError("a replay state cannot be converted to SMILES")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("a replay state has invalid SMILES")
    return molecule, smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cache = torch.load(args.rollout_cache, map_location="cpu", weights_only=False)
    model, checkpoint = load_factorized_rollout_checkpoint(args.checkpoint)
    source_prior = checkpoint["tree_source_prior"]
    seeds = trajectory_seeds(int(cache["seed"]), max(args.indices) + 1)
    selected_step = checkpoint.get("selected_validation", {}).get("selected_step")
    if selected_step is None:
        selected_step = checkpoint.get("best_metrics", {}).get("selected_step")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index in args.indices:
        states, event_times, event_rules, event_actions = replay(
            model,
            source_prior,
            seed=seeds[index],
            n_slots=int(cache["n_slots"]),
            operational_horizon=float(cache["operational_horizon"]),
            time_step=float(cache["time_step"]),
            max_events=int(cache["max_events"]),
        )
        state_keys = [canonical_state_key(state) for state in states]
        self_transitions = {
            state_index
            for state_index in range(1, len(states))
            if state_keys[state_index] == state_keys[state_index - 1]
        }
        # A strict two-cycle is A -> B -> A with A and B molecularly
        # distinct.  Without the inequality below, canonical self-transitions
        # were incorrectly labeled as immediate reversals.
        reversals = {
            event_index
            for event_index in range(2, len(states))
            if state_keys[event_index] == state_keys[event_index - 2]
            and state_keys[event_index] != state_keys[event_index - 1]
        }
        revisits = len(state_keys) - len(set(state_keys))
        molecules = []
        legends = []
        rows = []
        for state_index, state in enumerate(states):
            molecule, smiles = molecule_and_smiles(state)
            molecules.append(molecule)
            if state_index == 0:
                legend = (
                    f"STATE 0 | SOURCE\n"
                    f"atoms={molecule.GetNumHeavyAtoms()}, rings={molecule.GetRingInfo().NumRings()}"
                )
                event_time = 0.0
                rule = "<SOURCE>"
                action = None
            else:
                rule = event_rules[state_index - 1]
                event_time = event_times[state_index - 1]
                action = event_actions[state_index - 1]
                event_note = ""
                if state_index in self_transitions:
                    event_note = " | molecular self-transition"
                elif state_index in reversals:
                    event_note = " | strict two-cycle"
                legend = (
                    f"STATE {state_index} | {rule}{event_note}\n"
                    f"t={event_time:.3f}, atoms={molecule.GetNumHeavyAtoms()}, "
                    f"rings={molecule.GetRingInfo().NumRings()}"
                )
            legends.append(legend)
            rows.append(
                {
                    "state_index": state_index,
                    "event_time": event_time,
                    "rule": rule,
                    "action": None if action is None else repr(action),
                    "smiles": smiles,
                    "canonical_state_key": state_keys[state_index],
                    "heavy_atoms": molecule.GetNumHeavyAtoms(),
                    "rings": molecule.GetRingInfo().NumRings(),
                    "canonical_self_transition": state_index in self_transitions,
                    "immediate_reversal": state_index in reversals,
                }
            )

        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=5,
            subImgSize=(360, 270),
            legends=legends,
            useSVG=False,
        )
        png_path = args.output_dir / f"full_trajectory_step{int(selected_step)}_index{index}.png"
        image.save(png_path)
        page_paths = []
        page_size = 30
        for page_index, start in enumerate(range(0, len(molecules), page_size), start=1):
            page_image = Draw.MolsToGridImage(
                molecules[start : start + page_size],
                molsPerRow=5,
                subImgSize=(420, 315),
                legends=legends[start : start + page_size],
                useSVG=False,
            )
            page_path = args.output_dir / (
                f"full_trajectory_step{int(selected_step)}_index{index}_page{page_index:02d}.png"
            )
            page_image.save(page_path)
            page_paths.append(page_path.name)
        summary = {
            "trajectory_index": index,
            "trajectory_seed": seeds[index],
            "selected_checkpoint_step": selected_step,
            "events": len(event_rules),
            "event_counts": dict(sorted(Counter(event_rules).items())),
            "unique_states": len(set(state_keys)),
            "state_revisits": revisits,
            "canonical_self_transitions": len(self_transitions),
            "immediate_reversals": len(reversals),
            "exhausted_event_budget": len(event_rules) >= int(cache["max_events"]),
            "png": png_path.name,
            "png_pages": page_paths,
            "states": rows,
        }
        json_path = png_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2))
        summaries.append({key: value for key, value in summary.items() if key != "states"})
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)

    (args.output_dir / "full_trajectory_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )


if __name__ == "__main__":
    main()

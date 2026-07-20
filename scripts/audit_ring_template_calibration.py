"""Localize small-ring amplification in a cached ancestral rollout.

The audit reconstructs states immediately before ``ring_system_grow`` events
and compares small-ring mass under four state-conditional distributions:

1. uniform over exact executable templates;
2. the empirical catalog prior restricted to exact support;
3. learned residual logits without the catalog prior; and
4. the production learned-plus-prior template distribution.

No checkpoint, cache, or sampler behavior is mutated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from rdkit import Chem
import torch

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.eval.ring_calibration import (
    masked_category_mass,
    undesirable_small_ring_mask,
    uniform_category_mass,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    prepare_factorized_mark_batch,
)
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint


def _ring_size_counts(smiles: str) -> Counter[int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid cached molecular state: {smiles!r}")
    return Counter(len(tuple(ring)) for ring in Chem.GetSymmSSSR(molecule))


def _new_ring_sizes(before: str, after: str) -> tuple[int, ...]:
    difference = _ring_size_counts(after) - _ring_size_counts(before)
    return tuple(
        size
        for size, count in sorted(difference.items())
        for _ in range(int(count))
    )


def _ring_event_rows(rollouts: tuple[object, ...]) -> tuple[tuple[str, str, float], ...]:
    rows: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str, float]] = set()
    for rollout in rollouts:
        diagnostics = rollout.diagnostics
        for index, rule_name in enumerate(rollout.event_rules):
            if rule_name != "ring_system_grow":
                continue
            row = (
                diagnostics.canonical_state_keys[index],
                diagnostics.canonical_state_keys[index + 1],
                float(rollout.event_times[index]),
            )
            if row not in seen:
                seen.add(row)
                rows.append(row)
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=32)
    parser.add_argument("--small-ring-max", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    if args.max_states <= 0:
        raise ValueError("--max-states must be positive")
    torch.set_num_threads(args.torch_threads)

    model, checkpoint = load_factorized_rollout_checkpoint(args.checkpoint)
    payload = torch.load(args.rollout_cache, map_location="cpu", weights_only=False)
    rollouts = tuple(payload["rollouts"])
    n_slots = int(payload["n_slots"])
    event_rows = _ring_event_rows(rollouts)
    selected_rows = event_rows[: args.max_states]
    category = undesirable_small_ring_mask(
        model.ring_system_templates,
        maximum_size=args.small_ring_max,
    )
    prior_logits = model.ring_system_template_log_prior.detach().cpu()

    results: list[dict[str, object]] = []
    for row_index, (before, after, event_time) in enumerate(selected_rows):
        state = pad_molecular_graph(smiles_to_molecular_graph(before), n_slots)
        support = torch.tensor(model._ring_grow_support(state), dtype=torch.bool)
        batch = prepare_factorized_mark_batch(
            (state,),
            (event_time,),
            (None,),
            (None,),
            (0.0,),
            ring_catalog=None,
            compute_ring_grow_support=False,
        ).to(model.device)
        with torch.no_grad():
            _, global_state, _ = model._encode_batch(batch)
            query = model.ring_system_template_query(global_state)
            keys = model.ring_system_template_key.weight[
                : len(model.ring_system_templates)
            ]
            residual_logits = (
                torch.einsum("br,tr->bt", query, keys)
                / float(query.shape[-1]) ** 0.5
            )[0].detach().cpu()
        production_logits = residual_logits + prior_logits
        new_sizes = _new_ring_sizes(before, after)
        item = {
            "row": row_index,
            "time": event_time,
            "before": before,
            "after": after,
            "new_ring_sizes": new_sizes,
            "observed_small_ring": any(size <= args.small_ring_max for size in new_sizes),
            "legal_template_count": int(support.sum()),
            "legal_small_template_count": int((support & category).sum()),
            "small_mass_uniform_support": uniform_category_mass(support, category),
            "small_mass_prior_given_support": masked_category_mass(
                prior_logits, support, category
            ),
            "small_mass_residual_given_support": masked_category_mass(
                residual_logits, support, category
            ),
            "small_mass_production": masked_category_mass(
                production_logits, support, category
            ),
        }
        results.append(item)
        print(json.dumps({"phase": "row", **item}, sort_keys=True), flush=True)

    mass_keys = (
        "small_mass_uniform_support",
        "small_mass_prior_given_support",
        "small_mass_residual_given_support",
        "small_mass_production",
    )
    summary = {
        "checkpoint_selected_validation": checkpoint.get("selected_validation"),
        "rollout_count": len(rollouts),
        "ring_event_count": sum(
            rule == "ring_system_grow"
            for rollout in rollouts
            for rule in rollout.event_rules
        ),
        "unique_ring_event_state_count": len(event_rows),
        "audited_state_count": len(results),
        "small_template_count": int(category.sum()),
        "template_count": len(category),
        "unconditional_catalog_small_mass": float(
            prior_logits.exp()[category].sum()
        ),
        "observed_small_event_fraction": (
            sum(bool(row["observed_small_ring"]) for row in results) / len(results)
            if results
            else None
        ),
        "mean_masses": {
            key: (
                sum(float(row[key]) for row in results) / len(results)
                if results
                else None
            )
            for key in mass_keys
        },
    }
    output = {"summary": summary, "rows": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"phase": "summary", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Audit dense Generator-Matching teachers for zero-probability marks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.experiments.factorized_mark_conditional import (
    FactorizedMarkCollator,
    FactorizedMarkDataset,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.ring_system_fiber import (
    semantic_ring_categories_for_action,
    semantic_ring_next_category_mask,
    semantic_ring_prefix_is_completable,
    matching_ring_system_template_indices,
    ring_system_placement,
    ring_system_placement_key,
    warm_ring_system_candidate_indices,
)
from compose_v4.rewrite.tracelets import RingSystemGrow
from scripts.train_tracelet_cnof_gate import (
    _filter_records_by_finalized_manifest,
)


def _ring_failure_diagnostics(
    model: FactorizedTraceletRateModel,
    state,
    action: object,
) -> dict[str, object] | None:
    if not isinstance(action, RingSystemGrow):
        return None
    teacher_placement = ring_system_placement(action)
    teacher_key = ring_system_placement_key(teacher_placement)
    template_indices = matching_ring_system_template_indices(
        action,
        model.ring_system_templates,
    )
    support = model._ring_grow_support(state)
    template_rows = []
    for template_index in template_indices:
        placements = model._ring_template_placements(state, template_index)
        matching = tuple(
            placement
            for placement in placements
            if ring_system_placement_key(placement) == teacher_key
        )
        match_rows = []
        for placement in matching:
            decoder = model._ring_semantic_decoder(state, placement)
            try:
                categories = semantic_ring_categories_for_action(decoder, action)
            except ValueError as exc:
                match_rows.append(
                    {
                        "root_completable": semantic_ring_prefix_is_completable(
                            decoder,
                            (),
                        ),
                        "teacher_mapping_error": str(exc),
                    }
                )
                continue
            prefix: tuple[int, ...] = ()
            label_rows = []
            for slot, category in zip(placement.system_atoms, categories):
                category_mask = semantic_ring_next_category_mask(decoder, prefix)
                label_rows.append(
                    {
                        "slot": int(slot),
                        "teacher_category": int(category),
                        "teacher_allowed": bool(category_mask[int(category)]),
                        "allowed_categories": [
                            index
                            for index, enabled in enumerate(category_mask)
                            if enabled
                        ],
                    }
                )
                prefix = (*prefix, int(category))
            match_rows.append(
                {
                    "root_completable": semantic_ring_prefix_is_completable(
                        decoder,
                        (),
                    ),
                    "teacher_completable": semantic_ring_prefix_is_completable(
                        decoder,
                        categories,
                    ),
                    "labels": label_rows,
                }
            )
        template_rows.append(
            {
                "template_index": int(template_index),
                "support": bool(support[template_index]),
                "placements": len(placements),
                "matching_placements": match_rows,
            }
        )
    return {
        "matching_template_indices": list(template_indices),
        "templates": template_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--couplings-per-target", type=int, default=2)
    parser.add_argument("--late-time-fraction", type=float, default=0.5)
    parser.add_argument("--operational-horizon", type=float, default=16.0)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ring-electronic-mode",
        choices=("factorized_local", "catalog_exact"),
        default="factorized_local",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")

    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)
    shard = torch.load(args.shard, map_location="cpu", weights_only=False)
    raw_records = tuple(shard["records"])
    records = _filter_records_by_finalized_manifest(
        raw_records,
        tuple(manifest["supported_smiles"][args.partition]),
        n_slots=args.max_atoms,
        couplings_per_target=args.couplings_per_target,
        partition=args.partition,
        supported_record_keys=tuple(manifest["supported_record_keys"][args.partition]),
    )
    dataset = FactorizedMarkDataset(
        records,
        start_index=0,
        length=args.batch_size,
        seed=args.seed,
        late_time_fraction=args.late_time_fraction,
        operational_horizon=args.operational_horizon,
        progress_stratification_fraction=0.0,
        ring_catalog=manifest["ring_catalog"],
        ring_electronic_mode=args.ring_electronic_mode,
    )
    collator = FactorizedMarkCollator(
        use_aromatic_bond_view=True,
        ring_catalog=manifest["ring_catalog"],
    )
    warm_ring_system_candidate_indices(manifest["ring_catalog"])
    loader_options: dict[str, object] = {}
    if args.workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        dataset,
        batch_size=args.chunk_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collator,
        drop_last=False,
        **loader_options,
    )
    device = torch.device(args.device)
    model = FactorizedTraceletRateModel(
        manifest["ring_catalog"],
        hidden_dim=32,
        message_passing_steps=1,
        ring_electronic_mode=args.ring_electronic_mode,
    ).to(device)
    model.eval()

    failures: list[dict[str, object]] = []
    family_counts: dict[str, int] = {}
    family_failures: dict[str, int] = {}
    with torch.no_grad():
        start = 0
        for batch in loader:
            batch = batch.to(device)
            stop = start + len(batch.states)
            prediction = model.forward_mark_batch(batch)
            finite = torch.isfinite(prediction.selected_mark_log_probability)
            for offset, rule_name in enumerate(batch.teacher_rule_names):
                family = rule_name or "terminal"
                family_counts[family] = family_counts.get(family, 0) + 1
                if bool(finite[offset]):
                    continue
                family_failures[family] = family_failures.get(family, 0) + 1
                if len(failures) >= args.max_failures:
                    continue
                action = batch.teacher_actions[offset]
                failures.append(
                    {
                        "index": start + offset,
                        "rule_name": rule_name,
                        "action_type": (None if action is None else type(action).__name__),
                        "action": repr(action)[:2000],
                        "state_smiles": molecular_graph_to_smiles(batch.states[offset]),
                        "ring_diagnostics": _ring_failure_diagnostics(
                            model,
                            batch.states[offset],
                            action,
                        ),
                        "enabled_families": prediction.enabled_families[offset]
                        .detach()
                        .cpu()
                        .tolist(),
                    }
                )
            print(
                json.dumps(
                    {
                        "phase": "audit_progress",
                        "complete": stop,
                        "total": args.batch_size,
                        "failures": sum(family_failures.values()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            start = stop

    report = {
        "phase": "audit_complete",
        "examples": args.batch_size,
        "ring_electronic_mode": args.ring_electronic_mode,
        "family_counts": family_counts,
        "family_failures": family_failures,
        "failures": failures,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

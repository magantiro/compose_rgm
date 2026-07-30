#!/usr/bin/env python3
"""Run the bounded true-successor capacity gate on all seven core edit families.

This is a development diagnostic, not a training run and not paper evidence.
It constructs exact one-step targets from production legal actions and
executor-replayed cycle traces, compiles their complete canonical-successor
fibers, and asks whether a fresh configured model can memorize each family
under the productive embedded editing kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    ORGANIC_VOCABULARY,
    MolecularGraphError,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.experiments.cycle_op_prior import (  # noqa: E402
    build_cycle_op_records,
)
from compose_v4.experiments.successor_micro_overfit import (  # noqa: E402
    CORE_EDITING_FAMILIES,
    SuccessorSupervisionExample,
    examples_from_traces,
    prepare_successor_panel,
    train_successor_micro_panel,
)
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.factorized_fiber import (  # noqa: E402
    _factorized_candidates,
    enumerate_pendant_graft_actions,
)
from compose_v4.rewrite.kernel import (  # noqa: E402
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.source_corruption import _cycle_edges  # noqa: E402
from compose_v4.rewrite.typed_ring_catalog import (  # noqa: E402
    build_typed_ring_catalog,
)


def _load_smiles(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("SMILES source must contain a JSON list")
    rows: list[str] = []
    for item in payload:
        if isinstance(item, str):
            rows.append(item)
        elif isinstance(item, list) and len(item) >= 2 and isinstance(item[1], str):
            rows.append(item[1])
        elif isinstance(item, dict):
            value = item.get("smiles")
            if isinstance(value, str):
                rows.append(value)
    if not rows:
        raise ValueError("SMILES source yielded no molecules")
    return tuple(dict.fromkeys(rows))


def _direct_local_examples(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
    per_family: int,
) -> tuple[SuccessorSupervisionExample, ...]:
    """Select one exact legal successor per source/family without random walks."""

    system = de_novo_rewrite_system()
    families = CORE_EDITING_FAMILIES[:5]
    examples: list[SuccessorSupervisionExample] = []
    counts = {family: 0 for family in families}
    seen = {family: set() for family in families}
    for text in smiles:
        try:
            graph = smiles_to_molecular_graph(text)
            state = pad_molecular_graph(graph, n_slots)
        except (MolecularGraphError, ValueError):
            continue
        source_key = canonical_state_key(state)
        candidates: dict[str, list[Any]] = {
            family: []
            for family in families
            if counts[family] < per_family and source_key not in seen[family]
        }
        if not candidates:
            continue
        cycle_edges = _cycle_edges(state)
        for rule_name, action in _factorized_candidates(
            state,
            allow_bond_reroute=False,
            vocabulary=ORGANIC_VOCABULARY,
        ):
            if rule_name not in candidates or rule_name == "bond_reroute":
                continue
            if (
                rule_name == "bond_reorder"
                and frozenset((int(action.a), int(action.b))) in cycle_edges
            ):
                continue
            candidates[rule_name].append(action)
        if "bond_reroute" in candidates:
            candidates["bond_reroute"].extend(
                enumerate_pendant_graft_actions(state)
            )
        for family in families:
            if family not in candidates:
                continue
            for action in candidates[family]:
                try:
                    successor = system.apply(state, family, action)
                except Exception:
                    continue
                if canonical_state_key(successor) == source_key:
                    continue
                examples.append(
                    SuccessorSupervisionExample(
                        family_name=family,
                        state=state,
                        target=successor,
                        teacher_rule_name=family,
                        teacher_action=action,
                        data_lane="production_support_capacity_probe",
                    )
                )
                counts[family] += 1
                seen[family].add(source_key)
                break
        if all(count >= per_family for count in counts.values()):
            break
    return tuple(examples)


def _collect_examples(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
    seed: int,
    per_family: int,
) -> tuple:
    local_examples = _direct_local_examples(
        smiles,
        n_slots=n_slots,
        per_family=per_family,
    )
    cycle_examples: list[SuccessorSupervisionExample] = []
    cycle_counts = {family: 0 for family in CORE_EDITING_FAMILIES[5:]}
    cycle_seen = {family: set() for family in CORE_EDITING_FAMILIES[5:]}
    for molecule_index, text in enumerate(smiles):
        records, _ = build_cycle_op_records(
            (text,),
            n_slots=n_slots,
            seed=seed + molecule_index,
            max_bonds_per_molecule=1,
        )
        new_examples = examples_from_traces(
            (record.path.trace for record in records),
            families=CORE_EDITING_FAMILIES[5:],
            unique_source_molecules=False,
            data_lane="real_cycle_bond_roundtrip",
        )
        for row in new_examples:
            family = row.family_name
            if (
                cycle_counts[family] >= per_family
                or row.source_key in cycle_seen[family]
            ):
                continue
            cycle_examples.append(row)
            cycle_counts[family] += 1
            cycle_seen[family].add(row.source_key)
        if all(count >= per_family for count in cycle_counts.values()):
            break
    examples = (*local_examples, *cycle_examples)
    counts = {
        family: sum(row.family_name == family for row in examples)
        for family in CORE_EDITING_FAMILIES
    }
    missing = {
        family: count
        for family, count in counts.items()
        if count < per_family
    }
    if missing:
        raise RuntimeError(
            "development source did not fill every successor panel: "
            f"{missing}; requested {per_family} per family"
        )
    return tuple(examples)


def _model(
    *,
    hidden_dim: int,
    message_passing_steps: int,
    seed: int,
) -> FactorizedTraceletRateModel:
    torch.manual_seed(int(seed))
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog(()),
        hidden_dim=hidden_dim,
        message_passing_steps=message_passing_steps,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.smiles_json)
    smiles = _load_smiles(source_path)
    examples = _collect_examples(
        smiles,
        n_slots=args.n_slots,
        seed=args.seed,
        per_family=args.per_family,
    )
    panel_counts = {
        family: sum(row.family_name == family for row in examples)
        for family in CORE_EDITING_FAMILIES
    }
    report: dict[str, Any] = {
        "schema": "compose.successor_micro_overfit_gate",
        "schema_version": 1,
        "status": "DEVELOPMENT_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "unique_smiles": len(smiles),
        },
        "panel": {
            "n_slots": args.n_slots,
            "per_family": args.per_family,
            "counts": panel_counts,
            "unique_source_molecules_required": True,
            "target_definition": "production_executor_then_canonical_state_key",
        },
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "message_passing_steps": args.message_passing_steps,
            "atom_vocabulary_classes": len(ORGANIC_VOCABULARY),
            "cycle_ops": True,
            "ring_system_grow": False,
        },
        "optimization": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scopes": args.scopes,
            "seed": args.seed,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(args.device),
        },
        "reports": {},
    }
    if args.collect_only:
        return report

    scopes = tuple(
        part.strip()
        for part in args.scopes.split(",")
        if part.strip()
    )
    for family_index, family in enumerate(CORE_EDITING_FAMILIES):
        family_rows = tuple(
            row for row in examples if row.family_name == family
        )
        report["reports"][family] = {}
        for scope_index, scope in enumerate(scopes):
            model_seed = args.seed + 1000 * family_index + 100 * scope_index
            model = _model(
                hidden_dim=args.hidden_dim,
                message_passing_steps=args.message_passing_steps,
                seed=model_seed,
            ).to(torch.device(args.device))
            panel = prepare_successor_panel(model, family_rows)
            report["reports"][family][scope] = train_successor_micro_panel(
                model,
                panel,
                steps=args.steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                scope=scope,
                seed=model_seed,
                report_points=(
                    min(10, args.steps),
                    min(50, args.steps),
                    min(100, args.steps),
                    min(250, args.steps),
                ),
            )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles-json",
        default=str(ROOT / "configs" / "benchmarks" / "cnof_leads.json"),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--n-slots", type=int, default=40)
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--message-passing-steps", type=int, default=6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scopes", default="heads_only,all")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

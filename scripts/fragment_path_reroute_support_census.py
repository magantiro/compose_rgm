#!/usr/bin/env python3
"""What DOES the prior propose at the state where v3 exchanges the bridge?

The composite scoring probe finds the ``bond_reroute`` constituent unrecovered
at its decisive budget and again at a 16,384-draw cap, while the
``bond_reroute`` FAMILY fires at 1.5-14% of draws at the very same state.  That
combination is the one that must not be turned into a vocabulary claim without
looking: a family that fires is not a boundary, and a probe elsewhere in this
tree nearly published one because ``atom_type`` is a vocabulary INDEX rather
than an atomic number.

So this census does not ask whether the target is missing.  It asks what the
prior offers instead, by classifying every ``bond_reroute`` the model draws at
the post-insert state:

* which atom it MOVES -- the newly inserted atom, the path atom, the far-core
  anchor, some other core atom, or something else;
* where it moves it TO;
* how many distinct successors the family reaches at all;
* and whether the exact coordinate v3 executes ever appears.

``BondReroute(a, b, u, v)`` removes ``(a, b)`` and inserts ``(u, v)``.  v3
executes ``a=path_atom, b=far_anchor, u=new_atom, v=far_anchor``: the removed
edge's endpoints and the inserted edge's endpoints share only ``far_anchor``,
so ``u`` is NOT an endpoint of the cut.  The model's graft head is a RESTRICTED
reroute in which the moved atom is itself a cut endpoint, which is a structural
reason the coordinate could be absent -- but a reason is not a measurement, and
the successor is what matters anyway, since two actions reaching the same
molecule are the same transition to everything downstream.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.benchmark import fragment_conditioned_sampler as sampler_module
from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    SamplerConfig,
    SamplingReceipt,
    build_prompt_context,
    sample_completion,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.chem.molecular_graph import molecular_graph_to_smiles

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
SEED = 20260921


class _RecordingSystem:
    def __init__(self, system):
        self._system = system
        self.calls: list[tuple] = []

    def apply(self, state, rule_name, action):
        successor = self._system.apply(state, rule_name, action)
        self.calls.append((state, rule_name, action, successor))
        return successor

    def __getattr__(self, name):
        return getattr(self._system, name)


def _role(slot: int, *, inserted: int, path_atom: int, far_anchor: int, cores) -> str:
    if slot == inserted:
        return "inserted_atom"
    if slot == path_atom:
        return "path_atom"
    if slot == far_anchor:
        return "far_core_anchor"
    if slot in cores:
        return "other_core_atom"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts-per-drug", type=int, default=200)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.operators import BondReroute

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    config = SamplerConfig()
    control = AttachmentControlConfig(enabled=True, path_program=True)
    prompts = [
        p for p in load_genmol_prompts(MANIFEST) if p.task is FragmentTask.LINKER_DESIGN
    ]

    import rdkit

    rows = []
    started = time.time()
    original = sampler_module._attempt_path_transaction
    for prompt in prompts:
        captured: list[list[tuple]] = []

        def wrapped(m_, s_, st_, c_, l_, r_, rec_, _sink=captured, **kw):
            recorder = _RecordingSystem(s_)
            out = original(m_, recorder, st_, c_, l_, r_, rec_, **kw)
            if out is not None:
                _sink.append(list(recorder.calls))
            return out

        sampler_module._attempt_path_transaction = wrapped
        try:
            context = build_prompt_context(
                prompt, config=config, control=control, linker_bridge_atoms=1
            )
            controller = sampler_module.AttachmentController(
                context.attachment, context.locked_slots, control
            )
            sites = controller.path_transaction_sites(context.start_state)
            rng = np.random.default_rng(SEED)
            receipt = SamplingReceipt()
            for _ in range(args.attempts_per_drug):
                sample_completion(
                    model, system, context, rng,
                    config=config, receipt=receipt, control=control,
                )
                if captured:
                    break
        finally:
            sampler_module._attempt_path_transaction = original

        if not captured or sites is None:
            rows.append({"drug": prompt.drug_name, "captured": False})
            print(f"{prompt.drug_name:14s} no transaction captured", flush=True)
            continue

        path_atom, far_anchor, inserted = sites
        calls = captured[0]
        # The reroute constituent's SOURCE state and its target successor.
        source, _rule, v3_action, successor = calls[-1]
        target_smiles = molecular_graph_to_smiles(successor)
        cores = frozenset().union(*context.attachment.lock_groups)

        moved_roles: Counter[str] = Counter()
        target_roles: Counter[str] = Counter()
        coordinates: Counter[tuple[int, int]] = Counter()
        successors: set[str] = set()
        family_draws = 0
        exact_coordinate_seen = 0
        draw_rng = np.random.default_rng(SEED + 7)
        for _ in range(args.draws):
            try:
                mark = model.sample_rewrite_mark(source, 0.0, draw_rng)
            except Exception:  # noqa: BLE001, S112 -- a refused draw is not a proposal
                continue
            if mark.rule_name != "bond_reroute" or not isinstance(
                mark.action, BondReroute
            ):
                continue
            family_draws += 1
            act = mark.action
            moved_roles[
                _role(int(act.u), inserted=inserted, path_atom=path_atom,
                      far_anchor=far_anchor, cores=cores)
            ] += 1
            target_roles[
                _role(int(act.v), inserted=inserted, path_atom=path_atom,
                      far_anchor=far_anchor, cores=cores)
            ] += 1
            coordinates[(int(act.u), int(act.v))] += 1
            if (int(act.u), int(act.v)) == (
                int(v3_action.u), int(v3_action.v)
            ):
                exact_coordinate_seen += 1
            try:
                smiles = molecular_graph_to_smiles(
                    system.apply(source, mark.rule_name, mark.action)
                )
            except Exception:  # noqa: BLE001, S112
                continue
            if smiles:
                successors.add(smiles)

        row = {
            "drug": prompt.drug_name,
            "captured": True,
            "draws": args.draws,
            "bond_reroute_draws": family_draws,
            "bond_reroute_family_rate": family_draws / args.draws,
            "distinct_bond_reroute_successors": len(successors),
            "target_successor_reached": target_smiles in successors,
            "v3_coordinate": [int(v3_action.u), int(v3_action.v)],
            "v3_removed_edge": [int(v3_action.a), int(v3_action.b)],
            "v3_moved_atom_is_a_cut_endpoint": int(v3_action.u) in (
                int(v3_action.a), int(v3_action.b)
            ),
            "exact_v3_coordinate_drawn": exact_coordinate_seen,
            "moved_atom_roles": dict(moved_roles.most_common()),
            "new_partner_roles": dict(target_roles.most_common()),
            "distinct_coordinates": len(coordinates),
        }
        rows.append(row)
        print(
            f"{prompt.drug_name:14s} reroute_draws={family_draws:5d} "
            f"distinct_succ={len(successors):4d} target={row['target_successor_reached']} "
            f"moved={dict(moved_roles.most_common(3))}",
            flush=True,
        )

    payload = {
        "schema": "compose_fragment_path_reroute_support_census_v1",
        "question": (
            "at the state where v3 exchanges the bridge, what bond_reroute "
            "transitions does the prior actually propose"
        ),
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
        },
        "draws_per_state": args.draws,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

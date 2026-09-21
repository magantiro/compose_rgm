#!/usr/bin/env python3
"""Can the close-then-open path transaction execute at all?

v1 of the two-interface path program was falsified: a per-event monotone path
predicate commits nothing, because no single legal event lengthens the
core-to-core path.  The measured reason is that both bonds of the seeded bridge
are BRIDGES in the graph sense, so deleting either disconnects the molecule and
the executor refuses it.  Lengthening therefore requires a coordinated
transaction -- insert an atom on the seed, ring-close it to the far core, then
open the original bond -- and with the program off, 1,440 events across 120
rollouts never once produced a longer path.

Before writing a composite primitive, this probe asks whether the transaction is
executable at all.  If the hand-constructed sequence cannot run, a composite has
nothing to compose and v2 dies as cleanly as v1 did.

It also answers the question that separates a legitimate macro from a capability
extension: is each constituent something the PROPOSAL LAW can rank?  A composite
bundling moves the prior could already make, in an order it does not make them,
is an acceleration.  A composite needing a constituent the prior cannot rank is
an extension and has to be declared as one.

The standard for calling a zero hard was raised by the LOVASTATIN result: that
drug read exactly 0.0000 at 256 draws AND at 2048, then recovered to 0.80 at
8192.  A zero surviving ONE budget increase is therefore not a hard zero, and
this probe sweeps a wide ladder before calling any constituent unrankable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from compose_v4.benchmark.fragment_attachment_control import (
    AttachmentControlConfig,
    AttachmentController,
)
from compose_v4.benchmark.fragment_conditioned_sampler import (
    RegionLock,
    SamplerConfig,
    build_prompt_context,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.chem.molecular_graph import is_element, molecular_graph_to_smiles
from compose_v4.rewrite.operators import AtomInsert, BondDelete, BondInsert

MANIFEST = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
CARBON = 6


def _graph(state):
    real = is_element(state.atom_types)
    g = nx.Graph()
    for i in range(state.n_atoms):
        if not real[i]:
            continue
        g.add_node(i)
        for j in range(i + 1, state.n_atoms):
            if real[j] and int(state.bonds[i][j]) > 0:
                g.add_edge(i, j)
    return g


def _free_slot(state) -> int | None:
    real = is_element(state.atom_types)
    for i in range(state.n_atoms):
        if not real[i]:
            return i
    return None


def _successor_key(system, state, rule, action):
    try:
        successor = system.apply(state, rule, action)
    except Exception:  # noqa: BLE001
        return None, None
    return successor, molecular_graph_to_smiles(successor) or None


def _rankable(model, system, state, target_key, draws, rng_seed=7):
    """Does the model propose a transition reaching ``target_key``?"""
    rng = np.random.default_rng(rng_seed)
    for _ in range(draws):
        try:
            mark = model.sample_rewrite_mark(state, 0.0, rng)
        except Exception:  # noqa: BLE001, S112 -- a refused draw is not a match
            continue
        _, key = _successor_key(system, state, mark.rule_name, mark.action)
        if key == target_key:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--drug", action="append", default=None)
    parser.add_argument(
        "--ladder",
        default="64,256,1024,4096,16384",
        help="draw budgets swept before a zero may be called hard",
    )
    args = parser.parse_args()

    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    from compose_v4.rewrite.kernel import de_novo_rewrite_system

    model, _meta = load_factorized_rollout_checkpoint(args.checkpoint)
    system = de_novo_rewrite_system()
    config = SamplerConfig()
    control = AttachmentControlConfig(enabled=True)
    ladder = [int(x) for x in args.ladder.split(",") if x.strip()]

    prompts = [p for p in load_genmol_prompts(MANIFEST) if p.task is FragmentTask.LINKER_DESIGN]
    if args.drug:
        wanted = {d.upper() for d in args.drug}
        prompts = [p for p in prompts if p.drug_name.upper() in wanted]

    rows = []
    for prompt in prompts:
        context = build_prompt_context(
            prompt, config=config, control=control, linker_bridge_atoms=1
        )
        controller = AttachmentController(
            context.attachment, context.locked_slots, control
        )
        lock = RegionLock(
            context.start_state,
            context.locked_slots,
            released_pairs=context.attachment.released_pairs,
        )
        state = context.start_state
        cores = context.attachment.lock_groups
        graph = _graph(state)

        # The seed atom: the one real atom outside every retained core.
        seeds = [n for n in graph.nodes if all(n not in g for g in cores)]
        if len(seeds) != 1:
            rows.append({"drug": prompt.drug_name, "status": "unexpected seed count"})
            continue
        seed = seeds[0]
        # Its two neighbours are the two cores' declared attachment atoms.
        near = [n for n in graph.neighbors(seed) if n in cores[0]]
        far = [n for n in graph.neighbors(seed) if n in cores[1]]
        if not near or not far:
            rows.append({"drug": prompt.drug_name, "status": "seed not spanning both cores"})
            continue
        anchor_far = far[0]

        record = {
            "drug": prompt.drug_name,
            "start_path_length": controller.realized_linker_length(state),
            "seed_slot": seed,
            "far_core_anchor": anchor_far,
            "steps": [],
        }

        # ---- step 1: insert a carbon onto the seed ----
        slot = _free_slot(state)
        step1 = AtomInsert(
            slot=slot, atom_type=CARBON, formal_charge=0,
            implicit_h_count=2, neighbors=((seed, 1),),
        )
        s1, key1 = _successor_key(system, state, "atom_insert", step1)
        record["steps"].append({
            "step": "insert carbon on the seed",
            "rule": "atom_insert",
            "executes": s1 is not None,
            "lock_permits": bool(s1 is not None and lock.permits(s1)),
            "path_length_after": None if s1 is None else controller.realized_linker_length(s1),
        })
        if s1 is None:
            rows.append(record | {"status": "TRANSACTION NOT EXECUTABLE at step 1"})
            continue

        # ---- step 2: ring-close the new atom to the far core ----
        step2 = BondInsert(a=slot, b=anchor_far, order=1)
        s2, key2 = _successor_key(system, s1, "bond_insert", step2)
        record["steps"].append({
            "step": "ring-close the new atom to the far core",
            "rule": "bond_insert",
            "executes": s2 is not None,
            "lock_permits": bool(s2 is not None and lock.permits(s2)),
            "path_length_after": None if s2 is None else controller.realized_linker_length(s2),
        })
        if s2 is None:
            rows.append(record | {"status": "TRANSACTION NOT EXECUTABLE at step 2"})
            continue

        # ---- step 3: open the original seed-to-far-core bond ----
        step3 = BondDelete(a=seed, b=anchor_far)
        s3, key3 = _successor_key(system, s2, "bond_delete", step3)
        record["steps"].append({
            "step": "open the original seed-to-far-core bond",
            "rule": "bond_delete",
            "executes": s3 is not None,
            "lock_permits": bool(s3 is not None and lock.permits(s3)),
            "path_length_after": None if s3 is None else controller.realized_linker_length(s3),
        })
        if s3 is None:
            rows.append(record | {"status": "TRANSACTION NOT EXECUTABLE at step 3"})
            continue

        final_length = controller.realized_linker_length(s3)
        separated = controller.cores_are_separated(s3)
        record["final_path_length"] = final_length
        record["cores_separated"] = separated
        record["final_smiles"] = key3
        record["status"] = (
            "TRANSACTION EXECUTES and lengthens the path"
            if final_length is not None and final_length > 1 and separated
            else "TRANSACTION EXECUTES but does not lengthen the path"
        )

        # ---- is each constituent RANKABLE by the proposal law? ----
        # This is the macro-versus-extension question, and it is asked per step.
        constituents = [
            ("insert carbon on the seed", state, key1),
            ("ring-close the new atom to the far core", s1, key2),
            ("open the original seed-to-far-core bond", s2, key3),
        ]
        for name, from_state, target_key in constituents:
            curve = {}
            rankable_at = None
            for draws in ladder:
                hit = _rankable(model, system, from_state, target_key, draws)
                curve[str(draws)] = hit
                if hit:
                    rankable_at = draws
                    break
            record.setdefault("constituent_rankability", []).append({
                "constituent": name,
                "ladder": curve,
                "first_rankable_at_draws": rankable_at,
                "verdict": "IN_SUPPORT" if rankable_at else "NO_HIT_UP_TO_" + str(ladder[-1]),
            })
        rows.append(record)
        print(
            f"{prompt.drug_name:14s} {record['status']}  "
            f"length {record['start_path_length']} -> {final_length}  "
            f"separated={separated}",
            flush=True,
        )
        for c in record["constituent_rankability"]:
            print(f"    {c['constituent']:42s} {c['verdict']}", flush=True)

    executable = [r for r in rows if str(r.get("status", "")).startswith("TRANSACTION EXECUTES and")]
    payload = {
        "schema": "compose_fragment_path_transaction_probe_v1",
        "question": (
            "is the close-then-open path transaction executable, and is each of "
            "its constituents rankable by the proposal law?"
        ),
        "why": (
            "A composite whose constituents are each individually in support is an "
            "ACCELERATION -- it bundles moves the prior could make but does not make "
            "in sequence. A composite needing a constituent the prior cannot rank is "
            "a capability EXTENSION and must be declared as one rather than wrapped."
        ),
        "hard_zero_standard": (
            "A zero surviving ONE budget increase is not a hard zero: LOVASTATIN motif "
            "read 0.0000 at 256 AND at 2048 draws, then 0.80 at 8192. This probe sweeps "
            f"{ladder} before calling any constituent unrankable."
        ),
        "drugs_where_the_transaction_executes": [r["drug"] for r in executable],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

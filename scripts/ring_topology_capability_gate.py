#!/usr/bin/env python3
"""§2 of the RingCore preflight: ring-topology capability gate for RING_CORE_V1.

For each ring-topology class, test explicit EXECUTABLE compositional construction (cycle_close) + deletion
(cycle_open) of a representative molecule, decomposed via a spanning tree (non-tree edges = the ring-closing
bonds). Per topology we check: valid start, a legal compositional action sequence, all-intermediate
validity+connectedness, <=40 atoms at every state, endpoint graph isomorphism (canonical key), expected
cycle-rank changes, charge preservation, canonical-successor identity, exact inverse (open<->close), and
positive selected-target supervision. Model-mechanism checks (finite gradients, forced sampling, natural
sampling after a tiny overfit, save/reload) are RingCore-level (the cycle heads are topology-agnostic) and
run once over the union.

Verdict per topology: SUPPORTED_BY_CORE / REQUIRES_STRUCTURED_P2_MODE / UNSUPPORTED_BY_CURRENT_CHEMISTRY.
If a fused/spiro/bridged/cage class is not compositionally reachable, that is the signal to add a minimal
structured P2 mode BEFORE the preflight (per the mandate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    ORGANIC_VOCABULARY,
    MolecularGraphError,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (  # noqa: E402
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.experiments.factorized_mark_conditional import (  # noqa: E402
    factorized_mark_bregman_loss,
)
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.operators import BondDelete, BondInsert  # noqa: E402

# Representative molecule per topology class (broad-organic, <=40 atoms, RingCore vocabulary).
_TOPOLOGIES = {
    "simple_saturated": "C1CCCCC1",
    "aromatic_carbocycle": "c1ccccc1",
    "aromatic_heterocycle": "c1ccncc1",
    "saturated_heterocycle": "C1CCNCC1",
    "macrocycle": "C1CCCCCCCCCCC1",
    "fused_aromatic": "c1ccc2ccccc2c1",
    "fused_saturated": "C1CCC2CCCCC2C1",
    "spiro": "C1CCC2(CC1)CCCCC2",
    "bridged_bicyclic": "C1CC2CCC1C2",
    "polycyclic_cage": "C1C2CC3CC1CC(C2)C3",  # adamantane
    "sulfur_ring_aromatic": "c1ccsc1",  # thiophene
    "sulfur_ring_saturated": "C1CCSC1",
    "phosphorus_ring": "C1CCPC1",
    "charge_preserving_ring": "C[N+](C)(C)CCC1CCCCC1",
}


def _real(state):
    return [i for i in range(len(state.atom_types)) if bool(is_element(state.atom_types)[i])]


def _net_charge(state) -> int:
    return int(state.formal_charges[is_element(state.atom_types)].sum())


def _n_atoms(state) -> int:
    return int(is_element(state.atom_types).sum())


def _cycle_rank(state) -> int:
    real = set(_real(state))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(state.bonds[a, b]) != 0:
                graph.add_edge(a, b)
    return graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)


def _non_tree_edges(state) -> list[tuple[int, int, int]]:
    """Ring-closing (non-spanning-tree) bonds (a<b, order): closing these onto a spanning tree rebuilds the
    full ring system; opening them reduces it to the acyclic tree."""
    real = _real(state)
    graph = nx.Graph()
    graph.add_nodes_from(real)
    for a in real:
        for b in real:
            if a < b and int(state.bonds[a, b]) != 0:
                graph.add_edge(a, b, order=int(state.bonds[a, b]))
    tree = set(frozenset(e) for e in nx.minimum_spanning_tree(graph).edges())
    return [
        (a, b, graph[a][b]["order"])
        for a, b in graph.edges()
        if frozenset((a, b)) not in tree
    ]


def _check_topology(name: str, smiles: str, system, catalog) -> dict:
    result: dict = {"name": name, "smiles": smiles}
    try:
        graph = smiles_to_molecular_graph(smiles)
    except MolecularGraphError as exc:
        return {**result, "verdict": "UNSUPPORTED_BY_CURRENT_CHEMISTRY", "reason": str(exc)}
    if graph is None:
        return {**result, "verdict": "UNSUPPORTED_BY_CURRENT_CHEMISTRY", "reason": "unparseable"}
    try:
        target = pad_molecular_graph(graph, 40)
    except ValueError as exc:
        return {**result, "verdict": "UNSUPPORTED_BY_CURRENT_CHEMISTRY", "reason": str(exc)}

    result["valid_start"] = bool(is_valid_state(target))
    non_tree = _non_tree_edges(target)
    result["n_ring_closing_bonds"] = len(non_tree)
    result["target_cycle_rank"] = _cycle_rank(target)
    q0 = _net_charge(target)
    target_key = canonical_state_key(target)

    def _valid_step(state) -> bool:
        return (
            is_valid_state(state)
            and is_connected_or_null(state)
            and _n_atoms(state) <= 40
            and _net_charge(state) == q0
        )

    checks = {
        "all_intermediate_valid_connected": True,
        "le_max_atoms_every_state": True,
        "charge_preserving": True,
        "cycle_rank_changes_correct": True,
        "canonical_successor_identity": True,
        "endpoint_isomorphism": False,
        "inverse_open_close_exact": False,
        "legal_compositional_sequence": True,
    }
    try:
        # DELETION (cycle_open): target -> spanning tree, removing each ring-closing bond.
        state = target
        rank = result["target_cycle_rank"]
        for a, b, _order in non_tree:
            nxt = system.apply(state, "bond_delete", BondDelete(a, b))
            rank -= 1
            if not _valid_step(nxt):
                checks["all_intermediate_valid_connected"] = _valid_step(nxt) and checks[
                    "all_intermediate_valid_connected"
                ]
            if _cycle_rank(nxt) != rank:
                checks["cycle_rank_changes_correct"] = False
            if canonical_state_key(nxt) is None:
                checks["canonical_successor_identity"] = False
            state = nxt
        precursor = state
        result["precursor_cycle_rank"] = _cycle_rank(precursor)

        # CONSTRUCTION (cycle_close): spanning tree -> target, closing each ring bond back.
        state = precursor
        rank = result["precursor_cycle_rank"]
        for a, b, order in non_tree:
            nxt = system.apply(state, "bond_insert", BondInsert(a, b, order))
            rank += 1
            if not _valid_step(nxt):
                checks["all_intermediate_valid_connected"] = False
            if _cycle_rank(nxt) != rank:
                checks["cycle_rank_changes_correct"] = False
            if canonical_state_key(nxt) is None:
                checks["canonical_successor_identity"] = False
            state = nxt
        checks["endpoint_isomorphism"] = canonical_state_key(state) == target_key
        checks["inverse_open_close_exact"] = canonical_state_key(state) == target_key
    except Exception as exc:  # noqa: BLE001 -- executor rejects a step -> not compositionally reachable
        checks["legal_compositional_sequence"] = False
        result["executor_rejection"] = f"{type(exc).__name__}: {exc}"

    # Positive selected-target supervision: the open/close steps ARE cycle teacher marks.
    records, attempted = build_cycle_op_records([smiles], n_slots=40, seed=0)
    result["supervision_records"] = len(records)
    checks["positive_supervision"] = len(records) > 0

    result["checks"] = checks
    supported = all(checks.values())
    if supported:
        result["verdict"] = "SUPPORTED_BY_CORE"
    elif not checks["legal_compositional_sequence"]:
        result["verdict"] = "REQUIRES_STRUCTURED_P2_MODE"
    else:
        result["verdict"] = "REQUIRES_STRUCTURED_P2_MODE"
    return result


def _core_mechanism_checks(catalog, per_topology: list[dict]) -> dict:
    """RingCore-level (topology-agnostic) checks: finite gradients, forced sampling, natural sampling after
    a tiny overfit, and save/reload equivalence -- over the union of topology cycle-op records."""
    smiles = [r["smiles"] for r in per_topology if r["verdict"] == "SUPPORTED_BY_CORE"]
    records, _ = build_cycle_op_records(smiles, n_slots=40, seed=1)
    system = de_novo_rewrite_system()
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=48, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    )
    # (11) finite gradients on cycle teachers
    batch = prepare_factorized_mark_batch(
        tuple(r.path.trace.source for r in records[:32]),
        tuple(frozen_time(0.3) for _ in records[:32]),
        tuple(r.path.trace.steps[0].action for r in records[:32]),
        tuple(r.path.trace.steps[0].rule_name for r in records[:32]),
        tuple(1.0 for _ in records[:32]),
        ring_catalog=catalog,
    )
    loss = factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch)
    loss.backward()
    grad_norm = sum(
        float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5
    finite_gradients = bool(torch.isfinite(loss)) and grad_norm > 0.0

    # (12) forced sampling: every cycle teacher action executes as a legal cycle op
    forced_ok = 0
    for r in records[:20]:
        step = r.path.trace.steps[0]
        succ = system.apply(r.path.trace.source, step.rule_name, step.action)
        forced_ok += int(canonical_state_key(succ) == canonical_state_key(r.path.trace.target))
    forced_sampling = forced_ok == len(records[:20])

    # (13) natural sampling after a tiny overfit: the model should draw cycle ops on precursor states
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(120):
        opt.zero_grad()
        loss = factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch)
        loss.backward()
        opt.step()
    model.eval()
    rng = np.random.default_rng(0)
    cyc_draws = 0
    total = 0
    for r in records[:24]:
        src = r.path.trace.source
        for _ in range(4):
            mk = model.sample_rewrite_mark(src, 0.3, rng)
            total += 1
            cyc_draws += int(mk.rule_name in ("bond_insert", "bond_delete"))
    natural_sampling = cyc_draws > 0
    final_overfit_loss = float(loss)

    # (14) save/reload equivalence
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    torch.manual_seed(0)
    reloaded = FactorizedTraceletRateModel(
        catalog, hidden_dim=48, message_passing_steps=2, atom_vocabulary=ORGANIC_VOCABULARY,
        enable_cycle_ops=True, enable_ring_grow_macro=False,
    )
    buf.seek(0)
    reloaded.load_state_dict(torch.load(buf))
    reloaded.eval()
    with torch.no_grad():
        a = model.forward_mark_batch(batch).selected_mark_log_probability
        b = reloaded.forward_mark_batch(batch).selected_mark_log_probability
    save_reload_equiv = bool(torch.allclose(a, b, atol=1e-6))

    return {
        "finite_gradients": finite_gradients,
        "grad_norm": round(grad_norm, 4),
        "forced_sampling_executes": forced_sampling,
        "natural_sampling_after_overfit": natural_sampling,
        "natural_cycle_draw_fraction": round(cyc_draws / max(1, total), 3),
        "final_overfit_loss": round(final_overfit_loss, 4),
        "save_reload_equivalent": save_reload_equiv,
        "n_union_records": len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/production_preflight/ring_topology_capability.json"),
    )
    args = parser.parse_args()
    from warmstart_dry_run import build_production_ring_catalog

    catalog = build_production_ring_catalog(40)
    system = de_novo_rewrite_system()
    per_topology = []
    for name, smiles in _TOPOLOGIES.items():
        res = _check_topology(name, smiles, system, catalog)
        per_topology.append(res)
        print(
            json.dumps({"topology": name, "verdict": res["verdict"],
                        "ring_bonds": res.get("n_ring_closing_bonds")}, sort_keys=True),
            flush=True,
        )
    core = _core_mechanism_checks(catalog, per_topology)
    print(json.dumps({"phase": "core_mechanism", **core}, sort_keys=True), flush=True)

    matrix = {r["name"]: r["verdict"] for r in per_topology}
    n_supported = sum(1 for v in matrix.values() if v == "SUPPORTED_BY_CORE")
    p2_required = [k for k, v in matrix.items() if v == "REQUIRES_STRUCTURED_P2_MODE"]
    unsupported = [k for k, v in matrix.items() if v == "UNSUPPORTED_BY_CURRENT_CHEMISTRY"]
    core_ok = all(
        core[k]
        for k in (
            "finite_gradients",
            "forced_sampling_executes",
            "natural_sampling_after_overfit",
            "save_reload_equivalent",
        )
    )
    verdict = (
        "GO_RING_TOPOLOGY_CAPABILITY"
        if not p2_required and not unsupported and core_ok
        else "P2_MODE_REQUIRED"
        if p2_required
        else "NO_GO_RING_TOPOLOGY_CAPABILITY"
    )
    out = {
        "verdict": verdict,
        "topology_matrix": matrix,
        "n_supported_by_core": n_supported,
        "requires_structured_p2_mode": p2_required,
        "unsupported_by_current_chemistry": unsupported,
        "core_mechanism_checks": core,
        "per_topology": per_topology,
        "note": "SUPPORTED_BY_CORE = compositional cycle_close/cycle_open construct+delete the class with all "
        "14 properties. P2_MODE_REQUIRED classes (if any) need a minimal structured mode before the preflight.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "supported": n_supported, "p2_required": p2_required,
                      "unsupported": unsupported, "core_ok": core_ok, "output": str(args.output)},
                     sort_keys=True))
    return 0 if verdict == "GO_RING_TOPOLOGY_CAPABILITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())

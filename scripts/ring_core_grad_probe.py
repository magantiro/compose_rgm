#!/usr/bin/env python3
"""Offline per-family grad-norm probe (§3/§4 update evidence for the RingCore schedule check).

For a checkpoint, build ONE fixed edit batch (corruption + cycle-op teachers over a fixed lead set, seeded so
the batch is identical across checkpoints), forward+backward the GM loss, and report the grad norm each
family's HEAD receives -> the "are lower-frequency families still getting gradient signal / being updated"
signal. A shrinking grad with finite, non-decreasing mass = converging (not dying); a family with zero grad,
zero mass, AND absent from rollouts is effectively dead. Pure offline diagnostic -- never touches training.

Usage:
  python scripts/ring_core_grad_probe.py <checkpoint.pt> [current_state_dict_source|best] [label]
The 2nd arg, when a path, overrides the loaded state_dict with that file's ``current_state_dict`` (so a
recovery checkpoint's step-N CURRENT model is probed, not its best_state_dict); pass "best" or omit to probe
the checkpoint's own state_dict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1  # noqa: E402
from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records  # noqa: E402
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.experiments.factorized_mark_conditional import sample_factorized_mark_batch  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import factorized_mark_bregman_loss  # noqa: E402
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint  # noqa: E402

# drug-like leads with rings + heteroatoms so corruption yields restate/reroute/ring teachers
LEADS = [
    "O=C1NC(=O)c2ccccc21", "Cc1ccc(cc1)C(=O)Nc1ccccc1", "C1CCNCC1C(=O)O",
    "c1ccc2c(c1)ccc1ccccc12", "COc1ccc(cc1)CCN", "O=C(O)c1ccc(cc1)S(=O)(=O)N",
    "c1ccc(nc1)C1CCNCC1", "CC(=O)Nc1ccc(O)cc1", "Clc1ccccc1C(=O)Nc1ccncc1", "OCC1CCC(O)CC1",
]
HEAD_TO_FAMILY = {
    "restate_head": "atom_restate", "reorder_head": "bond_reorder", "graft_head": "bond_reroute",
    "cycle_close_head": "cycle_insert", "cycle_open_head": "cycle_attach",
    "ring_restate_head": "ring_system_restate", "ring_system_delete_head": "ring_system_delete",
    "ring_system_grow_head": "ring_system_grow_DEAD", "delete_head": "atom_delete",
    "grow_root_head": "atom_insert", "family_head": "FAMILY_SELECTOR",
}


def _head_grad_norm(model, head_name: str):
    head = getattr(model, head_name, None)
    if head is None:
        return None
    grads = [p.grad.detach() for p in head.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads)))


def probe(checkpoint_path: str, current_state_dict_from: str | None, label: str, seed: int = 4242) -> dict:
    model, _meta = load_factorized_rollout_checkpoint(
        Path(checkpoint_path), expected_scope_hash=BROAD_ORGANIC_V1.scope_hash()
    )
    if current_state_dict_from is not None:
        rec = torch.load(current_state_dict_from, map_location="cpu", weights_only=False)
        model.load_state_dict(rec["current_state_dict"])  # override best -> the step's CURRENT model
    model.train()
    catalog = model.ring_catalog
    records, _log = build_corrupted_prior_records(
        LEADS, n_slots=40, depth_max=5, seed=seed, catalog=catalog,
        vocabulary=ORGANIC_VOCABULARY, couplings_per_target=2,
    )
    cyc_records, _c = build_cycle_op_records(tuple(LEADS), n_slots=40, seed=seed)
    records = tuple(records) + tuple(cyc_records)  # add cycle teachers so cycle heads get gradient too
    batch = sample_factorized_mark_batch(
        records, batch_size=128, seed=seed, late_time_fraction=0.5, operational_horizon=16.0,
        progress_stratification_fraction=0.5, ring_catalog=catalog,
        capabilities=model.operator_capabilities,
    )
    model.zero_grad(set_to_none=True)
    loss = factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch)
    loss.backward()
    out = {"label": label, "gm_loss": round(float(loss.detach()), 4)}
    for head_name, fam in HEAD_TO_FAMILY.items():
        gn = _head_grad_norm(model, head_name)
        out[fam] = round(gn, 5) if gn is not None else None
    return out


def main() -> int:
    ckpt = sys.argv[1]
    src = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "best" else None
    label = sys.argv[3] if len(sys.argv) > 3 else "probe"
    print(json.dumps(probe(ckpt, src, label), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

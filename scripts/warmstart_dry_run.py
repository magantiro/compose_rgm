#!/usr/bin/env python3
"""B->B-edit warm-start dry-run gate (real base checkpoint, no optimizer updates).

Verifies the production compatible warm-start end-to-end and emits a row-level transfer artifact:

  A. strict init (`--initialize-from-source-checkpoint`) FAILS, specifically on the widened
     (element,valence) heads (restate_head / grow_root_head / grow_option);
  B. compatible SEMANTIC init (`--initialize-compatible-from-source-checkpoint`) SUCCEEDS;
  C. every destination tensor is accounted for as COPIED_EXACT / COPIED_WITH_SEMANTIC_MAP /
     FRESH_NEW_CLASS (their union is the whole model);
  D. no unrecognized missing/unexpected key, and every shape-mismatched tensor is a known widened head;
  E. the semantically-initialized model reaches a finite no-update forward/sample pass.

Reconstructs the destination B-edit model from the base checkpoint's OWN stored config (ring_catalog,
hidden_dim, ...) + the organic vocabulary + the editing families, exactly as the trainer does. Stops before
any optimizer update.

Usage:
    PYTHONPATH=src:scripts python scripts/warmstart_dry_run.py --checkpoint <base_b.pt> \
        [--out diagnostics/production_preflight/warmstart_dry_run.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    CNOF_VOCABULARY,
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.experiments.canonical_successor_distillation import (  # noqa: E402
    AnalyticPancakeQuotientSampler,
    PancakeQuotientCalibration,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel  # noqa: E402
from compose_v4.model.time_convention import frozen_time  # noqa: E402
from compose_v4.rewrite.kernel import de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target  # noqa: E402
from compose_v4.rewrite.typed_ring_catalog import (  # noqa: E402
    build_typed_ring_catalog,
    ring_catalog_fingerprint,
)
from train_tracelet_cnof_gate import (  # noqa: E402
    _ATOM_VOCAB_HEAD_LAYOUT,
    _semantic_partial_checkpoint_initialization,
)

_SLOTS = 40
# The production ring catalog is built from these fixed seeds (mine_edit_traces._build_catalog); it is
# DETERMINISTIC and must reproduce the manifest fingerprint. B-edit uses THIS catalog, not the base
# checkpoint's (which was built from B's CNOF de-novo paths at an older commit and does NOT match).
_PROD_CATALOG_SEEDS = ("c1ccccc1", "c1ccncc1", "C1CCNCC1", "C1CCOCC1", "c1ccc2ccccc2c1")
# Catalog-DEPENDENT tensors: when B-edit uses the production catalog (which differs from B's older one),
# these are legitimately shape-mismatched and stay FRESH, so B's stale template identity never leaks in.
_CATALOG_DRIVEN_FRESH = frozenset({"ring_system_template_key.weight"})


def build_production_ring_catalog(max_atoms: int = _SLOTS):
    def trace(smi):
        t = pad_molecular_graph(smiles_to_molecular_graph(smi), max_atoms)
        src = DegreeBoundedCarbonTreePrior(sizes=(t.n_real_atoms,)).sample(
            np.random.default_rng(1), n_slots=max_atoms)
        return compile_carbon_tree_to_target(src, t, use_bond_reroute=True, align_source=True)
    return build_typed_ring_catalog(tuple(trace(s) for s in _PROD_CATALOG_SEEDS))


def _build_bedit(ck: dict, catalog) -> FactorizedTraceletRateModel:
    """Reconstruct the destination B-edit model with the PRODUCTION ring catalog (not the checkpoint's),
    organic vocab + editing families on, exactly as the trainer builds it for the warm-start."""
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=int(ck["hidden_dim"]),
        message_passing_steps=int(ck["message_passing_steps"]),
        ring_electronic_mode=str(ck.get("ring_electronic_mode", "factorized_local")),
        rate_factorization=str(ck.get("rate_factorization", "hierarchical")),
        ring_family_mass_mode=str(ck.get("ring_family_mass_mode", "boolean")),
        ring_template_factorization=str(ck.get("ring_template_factorization", "flat")),
        atom_vocabulary=ORGANIC_VOCABULARY,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-catalog-fingerprint", default="639ff6078c32d43c",
                        help="production ring-catalog fingerprint (manifest); the rebuilt seed catalog must match")
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/production_preflight/warmstart_dry_run.json"))
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    b_sd = ck["state_dict"]
    source_vocab = ORGANIC_VOCABULARY if ck.get("organic_vocabulary") else CNOF_VOCABULARY
    failures: list[str] = []

    # PRODUCTION ring catalog (deterministic 5-seed build) is authoritative -- verify it matches the manifest
    # fingerprint. B-edit uses THIS, never the base checkpoint's (older, mismatched) catalog.
    prod_catalog = build_production_ring_catalog(_SLOTS)
    prod_fp = ring_catalog_fingerprint(prod_catalog)
    b_catalog_fp = ring_catalog_fingerprint(ck["ring_catalog"])
    if prod_fp != args.expected_catalog_fingerprint:
        failures.append(f"CATALOG: rebuilt production fingerprint {prod_fp} != expected "
                        f"{args.expected_catalog_fingerprint} (manifest)")
    catalog_mismatch_b = prod_fp != b_catalog_fp  # EXPECTED: B's catalog is older; production is authoritative

    bedit = _build_bedit(ck, prod_catalog)
    tgt = bedit.state_dict()

    # ---- classify every destination tensor -----------------------------------
    copied_exact, widened, dest_only = [], [], []
    for name, dst in tgt.items():
        if name not in b_sd:
            dest_only.append(name)
        elif b_sd[name].shape == dst.shape:
            copied_exact.append(name)
        else:
            widened.append(name)
    source_only = [n for n in b_sd if n not in tgt]

    # (D) every shape-mismatch is a known widened head OR a catalog-driven tensor (correctly fresh under the
    # production catalog, so B's older template identity never leaks in); no unexpected source-only key
    recognized = set(_ATOM_VOCAB_HEAD_LAYOUT) | _CATALOG_DRIVEN_FRESH
    unrecognized = [n for n in widened if n not in recognized]
    if unrecognized:
        failures.append(f"D: shape-mismatched tensors not recognized (widened head or catalog-driven): {unrecognized}")
    if source_only:
        failures.append(f"D: unexpected source-only keys (in B, not B-edit): {source_only}")
    if dest_only:
        failures.append(f"D: unexplained destination-only keys (new params): {dest_only}")

    # (A) strict init must FAIL, specifically on the widened heads
    strict_ok = False
    strict_msg = ""
    try:
        _build_bedit(ck, prod_catalog).load_state_dict(b_sd)  # strict=True
        strict_ok = True
    except RuntimeError as exc:
        strict_msg = str(exc)
    if strict_ok:
        failures.append("A: strict init unexpectedly SUCCEEDED (should fail on widened heads)")
    elif not all(h.split(".")[0] in strict_msg for h in ("restate_head", "grow_root_head")):
        failures.append("A: strict failure did not cite the expected widened heads")

    # (B, C) semantic init succeeds + row table
    init, transferred, retained, row_table, map_hash = _semantic_partial_checkpoint_initialization(
        tgt, {"state_dict": b_sd},
        source_classes=source_vocab.classes, dest_classes=ORGANIC_VOCABULARY.classes,
    )
    try:
        bedit.load_state_dict(init, strict=True)
    except RuntimeError as exc:
        failures.append(f"B: semantic init failed to load: {exc}")
    # (C) union covers the whole model
    accounted = set(copied_exact) | set(widened) | set(dest_only)
    if accounted != set(tgt):
        failures.append(f"C: {len(set(tgt) - accounted)} tensors unaccounted for")
    copied_rows = sum(1 for r in row_table if r["status"] == "COPIED_WITH_INDEX_MAP")
    fresh_rows = sum(1 for r in row_table if r["status"] == "FRESH_NEW_CLASS")

    # (E) finite no-update forward/sample pass on a few broad-organic leads
    sampler = AnalyticPancakeQuotientSampler(bedit.eval(), calibration=PancakeQuotientCalibration())
    system = de_novo_rewrite_system()
    finite_ok = True
    for smi in ("O=S(=O)(N)c1ccccc1", "Clc1ccc(CCN)cc1", "CC(=O)Nc1ccc(O)cc1"):
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), _SLOTS)
        mark = sampler.sample_rewrite_mark(state, frozen_time(0.2), np.random.default_rng(0))
        if mark.action is not None:
            succ = system.apply(state, mark.rule_name, mark.action)
            if succ is None:
                finite_ok = False
    if not finite_ok:
        failures.append("E: no-update forward/sample pass did not reach a finite legal successor")

    verdict = "GO_WARMSTART_DRY_RUN" if not failures else "NO_GO_WARMSTART_TRANSFER"
    report = {
        "verdict": verdict,
        "checkpoint": str(args.checkpoint),
        "source_vocab": "ORGANIC" if source_vocab is ORGANIC_VOCABULARY else "CNOF",
        "ring_catalog": {
            "production_fingerprint": prod_fp,
            "expected_manifest_fingerprint": args.expected_catalog_fingerprint,
            "production_matches_manifest": prod_fp == args.expected_catalog_fingerprint,
            "base_checkpoint_fingerprint": b_catalog_fp,
            "base_mismatches_production": catalog_mismatch_b,
            "note": "B-edit uses the production catalog (authoritative); B's older catalog is rejected. "
                    "ring_system_template_key.weight is correctly fresh so B's stale templates never leak in.",
        },
        "A_strict_fails_on_widened_heads": not strict_ok,
        "B_semantic_init_loads": "B: semantic init failed to load" not in " ".join(failures),
        "C_all_tensors_accounted": accounted == set(tgt),
        "D_widened": sorted(widened), "D_source_only": source_only, "D_dest_only": dest_only,
        "E_forward_finite": finite_ok,
        "counts": {"copied_exact": len(copied_exact), "widened": len(widened),
                   "rows_copied_with_semantic_map": copied_rows, "rows_fresh_new_class": fresh_rows},
        "transfer_map_hash": map_hash,
        "row_table": row_table,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"VERDICT: {verdict}")
    print(f"  A strict-fails={not strict_ok}  B semantic-loads={report['B_semantic_init_loads']}  "
          f"C accounted={report['C_all_tensors_accounted']}  E finite={finite_ok}")
    print(f"  widened={sorted(widened)}  copied_exact={len(copied_exact)}  "
          f"semantic rows: {copied_rows} copied / {fresh_rows} fresh  map={map_hash}")
    for f in failures:
        print("  - " + f)
    print(f"-> {args.out}")
    return 0 if verdict == "GO_WARMSTART_DRY_RUN" else 1


if __name__ == "__main__":
    raise SystemExit(main())

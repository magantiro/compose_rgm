#!/usr/bin/env python3
"""Production preflight §10 stage: operator-SUBTYPE supervision over the ACTUAL B-edit record builders.

Runs the exact production record generators -- ``build_corrupted_prior_records`` (micro edits + cyclic
graft + clean ring-opening + de-aromatization) and ``build_cycle_op_records`` (compositional
cycle_close/cycle_open) -- on a real held-out GuacaMol sample scoped to broad-organic, then applies the
operator-subtype supervision gate. Unlike the atom-class ``cold_vocab_audit`` this counts positive selected
targets per operator FAMILY (subtype), so a production-enabled family with zero supervision fails loudly.

The cycle ops record EXECUTOR rule_names (``bond_insert``/``bond_delete``) that the rate model scores under
families ``cycle_insert``/``cycle_attach`` (``_CYCLE_OP_EXECUTOR_TO_FAMILY``); the gate aliases them so a
fully-supervised cycle op is not misread as unsupervised.

Two enabled-set verdicts are reported:
  * ``compositional_support`` -- the families that MUST be supervised now that P1 (compositional cycle ops)
    + P3 (real-molecule cycle supervision) have landed. Expected GO.
  * ``with_grow_macro`` -- adds ``ring_system_grow``, whose data-derived K-macro supervision is deferred to
    P4. Expected NO_GO until P4 lands; recorded so the gate provably catches the still-open macro gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import CNOF_VOCABULARY, ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, scan_corpus  # noqa: E402
from compose_v4.experiments.corrupted_source_prior import (  # noqa: E402
    build_corrupted_prior_records,
)
from compose_v4.experiments.cycle_op_prior import build_cycle_op_records  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
)
from operator_subtype_supervision_gate import (  # noqa: E402
    check_operator_subtype_supervision,
    family_sequences_from_records,
    subtype_target_counts,
)
from warmstart_dry_run import build_production_ring_catalog  # noqa: E402

# The families the compositional-support B-edit config enables and therefore MUST supervise (P1 + P3 done).
# bond_insert/bond_delete are the executor names the cycle ops record; the alias folds them into the
# cycle_insert/cycle_attach model families before the check.
_ENABLED_COMPOSITIONAL_SUPPORT = (
    "atom_insert", "atom_delete", "atom_restate",  # micro edits
    "bond_reorder",                                 # bond-order
    "bond_reroute",                                 # cyclic graft
    "ring_system_delete",                           # clean ring-opening (decoration-preserving)
    "ring_system_restate",                          # de-aromatization
    "cycle_insert", "cycle_attach",                 # compositional ring close / open (support-complete)
)
# ring_system_grow's data-derived K-macro supervision is P4; the gate must fail if it is enabled before then.
_ENABLED_WITH_GROW_MACRO = _ENABLED_COMPOSITIONAL_SUPPORT + ("ring_system_grow",)

_DEFAULT_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")
_DEFAULT_OUT = Path("diagnostics/production_preflight/subtype_supervision.json")


def _sample(corpus: Path, n: int, seed: int) -> tuple[list[str], dict]:
    """Deterministically sample ``n`` broad-organic-eligible molecules from a held-out GuacaMol file, using
    the production membership predicate (scan_corpus under BROAD_ORGANIC_V1)."""
    with corpus.open() as handle:
        texts = [line.strip().split()[0] for line in handle if line.strip()]
    accepted, _census = scan_corpus(texts, BROAD_ORGANIC_V1, workers=0)
    rng = np.random.default_rng(seed)
    molecules = np.asarray(accepted, dtype=object)
    rng.shuffle(molecules)
    picked = [str(m) for m in molecules[: min(n, len(molecules))]]
    return picked, {"corpus_eligible": len(accepted), "corpus_scanned": len(texts),
                    "corpus_scope_hash": BROAD_ORGANIC_V1.scope_hash()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--organic-vocabulary", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    sample, corpus_meta = _sample(args.corpus, args.sample_size, args.seed)
    print(json.dumps({"phase": "sample", **corpus_meta, "sampled": len(sample)}, sort_keys=True), flush=True)

    catalog = build_production_ring_catalog(args.max_atoms)
    vocabulary = ORGANIC_VOCABULARY if args.organic_vocabulary else CNOF_VOCABULARY

    edit_records, edit_attempted = build_corrupted_prior_records(
        sample, n_slots=args.max_atoms, depth_max=5, seed=args.seed + 7,
        catalog=catalog, vocabulary=vocabulary,
    )
    print(json.dumps({"phase": "corrupted_prior", "records": len(edit_records),
                      "attempted": edit_attempted}, sort_keys=True), flush=True)

    cycle_records, cycle_attempted = build_cycle_op_records(
        sample, n_slots=args.max_atoms, seed=args.seed + 8,
    )
    print(json.dumps({"phase": "cycle_op", "records": len(cycle_records),
                      "attempted": cycle_attempted}, sort_keys=True), flush=True)

    records = tuple(edit_records) + tuple(cycle_records)
    sequences = family_sequences_from_records(records)
    counts = subtype_target_counts(sequences, _CYCLE_OP_EXECUTOR_TO_FAMILY)

    verdicts = {}
    for label, enabled in (
        ("compositional_support", _ENABLED_COMPOSITIONAL_SUPPORT),
        ("with_grow_macro", _ENABLED_WITH_GROW_MACRO),
    ):
        ok, report = check_operator_subtype_supervision(
            sequences, enabled, _CYCLE_OP_EXECUTOR_TO_FAMILY
        )
        verdicts[label] = {
            "ok": ok,
            "verdict": "GO" if ok else "NO_GO_OPERATOR_SUBTYPE_SUPERVISION",
            "unsupervised_enabled_families": report["unsupervised_enabled_families"],
        }

    support_ok = verdicts["compositional_support"]["ok"]
    result = {
        "verdict": "GO_SUBTYPE_SUPERVISION" if support_ok else "NO_GO_SUBTYPE_SUPERVISION",
        "sample": corpus_meta | {"sampled": len(sample)},
        "records": {"corrupted_prior": len(edit_records), "cycle_op": len(cycle_records),
                    "total": len(records)},
        "positive_targets_by_family": dict(counts),
        "cycle_op_family_alias": _CYCLE_OP_EXECUTOR_TO_FAMILY,
        "enabled_sets": {
            "compositional_support": list(_ENABLED_COMPOSITIONAL_SUPPORT),
            "with_grow_macro": list(_ENABLED_WITH_GROW_MACRO),
        },
        "verdicts": verdicts,
        "note": (
            "compositional_support MUST pass now (P1+P3). with_grow_macro is EXPECTED to NO_GO until P4 "
            "supplies data-derived ring_system_grow macro supervision; the failure proves the gate catches "
            "the still-open macro gap rather than silently passing it."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "compositional_support": verdicts["compositional_support"]["verdict"],
                      "with_grow_macro": verdicts["with_grow_macro"]["verdict"],
                      "output": str(args.output)}, sort_keys=True))
    return 0 if support_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

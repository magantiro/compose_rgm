#!/usr/bin/env python3
"""RingCore-V1 pre-launch finalization §1-§3: version + characterize the teacher-representability filter and
the post-filter training distribution.

The teacher-in-exact-candidates invariant drops corruption traces whose selected teacher is outside the
model's exact dynamic candidate set (notably inverse/grow ring_system_restate on fused ring systems). This
removal is part of the DECLARED production data contract, not an undocumented runtime skip. This script
measures it at scale on a broad-organic sample and emits a versioned artifact with:
  * filter version + capability + operator hashes (§1);
  * removed records attributed by layer/family/subtype/direction/topology/charge/element/path-bin/reason,
    with the measured removal rate (§2);
  * the realized post-filter family/subtype selected-target distribution + a positive-target check for every
    production-enabled family (§3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    ORGANIC_VOCABULARY,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, scan_corpus  # noqa: E402
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    OperatorCapabilities,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.source_corruption import (  # noqa: E402
    TEACHER_REPRESENTABILITY_FILTER_VERSION,
    make_edit_pair,
    trace_teacher_representability_detail,
)

import ring_core_identity as _rci  # noqa: E402
from warmstart_dry_run import build_production_ring_catalog  # noqa: E402

_CORPUS = Path("results/tree_fcd_transfer_stage1_factorized_v1/guacamol_heldout_val_5000_seed0.smiles")
# The production-enabled families that MUST retain positive selected targets after filtering (RingCore-V1:
# compositional cycle ops, no legacy grow). cycle ops are recorded under executor names bond_insert/bond_delete
# but the corruption doesn't emit them (cycle supervision is a separate build_cycle_op_records path) -- so the
# corruption-side production-enabled set is the editing families it generates.
_CORRUPTION_PRODUCTION_FAMILIES = (
    "atom_insert", "atom_delete", "atom_restate", "bond_reorder", "bond_reroute",
    "ring_system_delete", "ring_system_restate",
)


def _topology(state) -> str:
    try:
        mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    except Exception:  # noqa: BLE001
        return "unparsed"
    if mol is None:
        return "unparsed"
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if not rings:
        return "acyclic"
    fused = any(len(rings[i] & rings[j]) >= 2 for i in range(len(rings)) for j in range(i + 1, len(rings)))
    aromatic = any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings)
    return f"{'fused' if fused else 'mono'}/{'aromatic' if aromatic else 'saturated'}"


def _charge_stratum(state) -> str:
    q = state.formal_charges[is_element(state.atom_types)]
    net = int(q.sum())
    if net == 0 and not bool((q != 0).any()):
        return "neutral"
    if net == 0:
        return "zwitterion"
    return "cation" if net > 0 else "anion"


def _path_bin(n: int) -> str:
    return "1" if n <= 1 else ("2-3" if n <= 3 else ("4-5" if n <= 5 else "6+"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--depth-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--couplings", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/production_preflight/teacher_filter_characterization.json")
    )
    args = parser.parse_args()

    with _CORPUS.open() as handle:
        texts = [line.strip().split()[0] for line in handle if line.strip()]
    accepted, _ = scan_corpus(texts, BROAD_ORGANIC_V1, workers=0)
    rng = np.random.default_rng(args.seed)
    mols = np.asarray(accepted, dtype=object)
    rng.shuffle(mols)
    sample = [str(m) for m in mols[: args.sample_size]]
    catalog = build_production_ring_catalog(40)
    system = de_novo_rewrite_system()

    pre = accepted_n = removed_n = 0
    removed_by = {k: Counter() for k in
                  ("family", "subtype_reason", "direction", "topology", "charge", "element", "path_bin")}
    accepted_family_targets: Counter = Counter()
    accepted_by_direction: Counter = Counter()
    accepted_by_layer: Counter = Counter()  # corruption traces only here (MMP is a separate pool)
    accepted_path_bins: Counter = Counter()
    removed_examples = []

    draw = np.random.default_rng(args.seed + 1)
    for smi in sample:
        try:
            target = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        except Exception:  # noqa: BLE001
            continue
        for _ in range(args.couplings):
            depth = int(draw.integers(1, args.depth_max + 1))
            trim, grow = make_edit_pair(target, depth, system=system, rng=draw, catalog=catalog,
                                        vocabulary=ORGANIC_VOCABULARY)
            for trace in (trim, grow):
                if trace is None:
                    continue
                pre += 1
                ok, detail = trace_teacher_representability_detail(trace, system=system, catalog=catalog)
                if ok:
                    accepted_n += 1
                    accepted_by_direction[str((trace.metadata or {}).get("prior", "?"))] += 1
                    accepted_by_layer["corruption"] += 1
                    accepted_path_bins[_path_bin(len(trace.steps))] += 1
                    for step in trace.steps:
                        fam = _CYCLE_OP_EXECUTOR_TO_FAMILY.get(step.rule_name, step.rule_name)
                        accepted_family_targets[fam] += 1
                else:
                    removed_n += 1
                    # attribute against the state the failing step applies to
                    st = trace.source
                    for _k in range(detail["step"]):
                        st = system.apply(st, trace.steps[_k].rule_name, trace.steps[_k].action)
                    removed_by["family"][detail["rule_name"]] += 1
                    removed_by["subtype_reason"][detail["reason"]] += 1
                    removed_by["direction"][detail["direction"]] += 1
                    removed_by["topology"][_topology(st)] += 1
                    removed_by["charge"][_charge_stratum(st)] += 1
                    els = sorted({int(e) for e in st.atom_types[is_element(st.atom_types)]})
                    removed_by["element"]["organic" if els else "?"] += 1
                    removed_by["path_bin"][_path_bin(len(trace.steps))] += 1
                    if len(removed_examples) < 8:
                        removed_examples.append({
                            "smiles": molecular_graph_to_smiles(st), **detail, "depth": len(trace.steps)})

    removed_fraction = round(removed_n / max(1, pre), 4)
    unsupervised = [f for f in _CORRUPTION_PRODUCTION_FAMILIES if accepted_family_targets.get(f, 0) == 0]

    caps = OperatorCapabilities(
        compute_ring_grow_support=False, compute_ring_restates=True,
        compute_cyclic_graft=True, compute_ring_opening=True,
    )
    filter_policy = {
        "filter_version": TEACHER_REPRESENTABILITY_FILTER_VERSION,
        "capability_fingerprint": caps.fingerprint(),
        "operator_registry_hash": _rci.recompute_operator_registry_hash(),
        "cycle_op_semantic_hash": _rci.recompute_cycle_op_semantic_hash(),
        "invariant": "every selected teacher action belongs to the exact dynamic candidate set for its state",
    }
    filter_hash = hashlib.sha256(json.dumps(filter_policy, sort_keys=True).encode()).hexdigest()[:16]

    result = {
        "verdict": "GO_TEACHER_FILTER_CHARACTERIZED" if not unsupervised else "NO_GO_UNSUPERVISED_FAMILY",
        "filter_policy": filter_policy,
        "teacher_filter_hash": filter_hash,
        "counts": {
            "pre_filter_traces": pre,
            "accepted_traces": accepted_n,
            "removed_traces": removed_n,
            "removed_fraction": removed_fraction,
        },
        "removed_attribution": {k: dict(v) for k, v in removed_by.items()},
        "removed_examples": removed_examples,
        "post_filter_distribution": {
            "family_selected_targets": dict(accepted_family_targets),
            "by_direction": dict(accepted_by_direction),
            "by_layer": dict(accepted_by_layer),
            "path_length_bins": dict(accepted_path_bins),
            "cycle_note": "cycle_insert/cycle_attach come from build_cycle_op_records (separate path), not the "
            "corruption; this characterization covers the corruption layer's editing families.",
        },
        "production_enabled_families_checked": list(_CORRUPTION_PRODUCTION_FAMILIES),
        "unsupervised_families_after_filter": unsupervised,
        "sample": {"size": len(sample), "couplings": args.couplings, "depth_max": args.depth_max,
                   "corpus": str(_CORPUS)},
        "note": "NON_SCIENTIFIC_PREFLIGHT. Removal rate is measured on a broad-organic held-out sample; the "
        "production corruption draws from split.train under the same policy, so this rate is representative.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": result["verdict"],
        "removed_fraction": removed_fraction,
        "removed_by_family": dict(removed_by["family"]),
        "removed_by_direction": dict(removed_by["direction"]),
        "unsupervised_after_filter": unsupervised,
        "teacher_filter_hash": filter_hash,
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if not unsupervised else 1


if __name__ == "__main__":
    raise SystemExit(main())

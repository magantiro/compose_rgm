#!/usr/bin/env python3
"""Cold-vocabulary audit for broad-organic B-edit.

The base B was trained through the CNOF-neutral loader with 4 (element,valence) classes; B-edit widens the
class heads to the 15 ORGANIC classes, so the 11 non-CNOF slots (S/P/Cl/Br/I/B) are NEWLY_INITIALIZED --
COLD until the broad corpus supervises them. This audit answers, per class, whether that supervision
actually arrives: a molecule merely CONTAINING sulfur does not prove the sulfur output classes receive a
positive edit target. We measure, from real broad-corpus corruption (both directions):

  - source-state occurrences: atoms of the class present in source states (context exposure);
  - positive edit targets:    teacher marks (atom_insert / atom_restate) that PRODUCE an atom of the class
                              -- the only signal that warms an output-head slot;

and combine with the structural status (does the head slot exist) to assign each class a status:
INHERITED_TRAINED (CNOF, carried from B) / BEDIT_SUPERVISED (non-CNOF, positive targets present) /
INHERITED_COLD (non-CNOF, no positive target -> stays at init) / DISABLED / NO_GO.

Run: PYTHONPATH=src:scripts python scripts/cold_vocab_audit.py --corpus <smiles> --limit 1500
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import (
    CNOF_VOCABULARY,
    IDX_TO_ELEMENT,
    ORGANIC_VOCABULARY,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1, classify_smiles
from compose_v4.model.factorized_tracelet_rate_model import BOND_CLASS_TO_H_CHANGE
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomInsert, AtomRestate
from compose_v4.rewrite.source_corruption import make_edit_pair

REPO = Path(__file__).resolve().parent.parent
_VOCAB = ORGANIC_VOCABULARY
_CNOF_ELEMS = {e for e, _ in CNOF_VOCABULARY.classes}


def _target_class(action) -> int | None:
    """The (element,valence) class an atom_insert / atom_restate teacher mark PRODUCES."""
    if isinstance(action, AtomInsert):
        bond_sum = sum(int(BOND_CLASS_TO_H_CHANGE[int(o)]) for _, o in action.neighbors)
        return _VOCAB.class_index(int(action.atom_type), bond_sum, int(action.implicit_h_count),
                                  formal_charge=int(action.formal_charge))
    if isinstance(action, AtomRestate):
        return None  # restate valence depends on the bonded state; counted separately below by element
    return None


def _restate_class_from_state(state, action) -> int | None:
    v = int(action.v)
    bond_valence = sum(int(BOND_CLASS_TO_H_CHANGE[int(o)]) for o in state.bonds[v])
    return _VOCAB.class_index(int(action.atom_type), bond_valence, int(action.implicit_h_count),
                              formal_charge=int(action.formal_charge))


def audit(smiles: list[str], *, n_slots: int, depth_max: int, seed: int) -> dict:
    system = de_novo_rewrite_system()
    rng = np.random.default_rng(seed)
    source_occurrences: Counter = Counter()   # class -> atoms present in source states
    positive_targets: Counter = Counter()      # class -> teacher marks producing that class
    target_by_optype: dict[str, Counter] = {"atom_insert": Counter(), "atom_restate": Counter()}
    traces_seen = 0
    for smi in smiles:
        if not classify_smiles(smi, BROAD_ORGANIC_V1)[0]:
            continue
        target = pad_molecular_graph(smiles_to_molecular_graph(smi), n_slots)
        real = is_element(target.atom_types)
        for v in np.flatnonzero(real):
            bond_sum = sum(int(BOND_CLASS_TO_H_CHANGE[int(o)]) for o in target.bonds[v])
            ci = _VOCAB.class_index(int(target.atom_types[v]), bond_sum,
                                    int(target.implicit_h_counts[v]), int(target.formal_charges[v]))
            if ci is not None:
                source_occurrences[ci] += 1
        depth = int(rng.integers(1, depth_max + 1))
        trim, grow = make_edit_pair(target, depth, system=system, rng=rng,
                                    vocabulary=ORGANIC_VOCABULARY)
        # walk each trace's steps against its running state to resolve restate target classes.
        for trace in (trim, grow):
            if trace is None:
                continue
            traces_seen += 1
            state = trace.source
            for step in trace.steps:
                if isinstance(step.action, AtomInsert):
                    ci = _target_class(step.action)
                    if ci is not None:
                        positive_targets[ci] += 1
                        target_by_optype["atom_insert"][ci] += 1
                elif isinstance(step.action, AtomRestate):
                    ci = _restate_class_from_state(state, step.action)
                    if ci is not None:
                        positive_targets[ci] += 1
                        target_by_optype["atom_restate"][ci] += 1
                try:
                    state = system.apply(state, step.rule_name, step.action)
                except Exception:  # noqa: BLE001
                    break

    rows = []
    for ci, (elem, val) in enumerate(_VOCAB.classes):
        symbol = IDX_TO_ELEMENT[elem]
        is_cnof = elem in _CNOF_ELEMS
        occ = source_occurrences.get(ci, 0)
        pos = positive_targets.get(ci, 0)
        if is_cnof:
            status = "INHERITED_TRAINED"
        elif pos > 0:
            status = "BEDIT_SUPERVISED"
        elif occ > 0:
            status = "INHERITED_COLD"  # present in context but no positive target -> head slot stays cold
        else:
            status = "ABSENT"
        rows.append({
            "class_index": ci, "element": symbol, "valence": val,
            "cnof": is_cnof,
            "source_state_occurrences": occ,
            "positive_edit_targets": pos,
            "target_atom_insert": target_by_optype["atom_insert"].get(ci, 0),
            "target_atom_restate": target_by_optype["atom_restate"].get(ci, 0),
            "output_head_slot_exists": True,  # grow_root_head + restate_head are 15-wide (verified)
            "status": status,
        })
    return {
        "vocabulary": "ORGANIC_VOCABULARY",
        "n_classes": len(_VOCAB.classes),
        "cnof_classes": len(CNOF_VOCABULARY.classes),
        "traces_seen": traces_seen,
        "molecules_audited": len(smiles),
        "rows": rows,
        "cold_non_cnof_classes": [r["element"] + f"(v{r['valence']})" for r in rows
                                  if not r["cnof"] and r["status"] in ("INHERITED_COLD", "ABSENT")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="results/tree_fcd_transfer_stage1_factorized_v1/"
                                            "guacamol_heldout_val_5000_seed0.smiles")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--n-slots", type=int, default=48)
    parser.add_argument("--depth-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="diagnostics/composition/cold_vocab_audit.json")
    args = parser.parse_args()

    smiles = []
    with (REPO / args.corpus).open() as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                smiles.append(token)
            if len(smiles) >= args.limit:
                break
    result = audit(smiles, n_slots=args.n_slots, depth_max=args.depth_max, seed=args.seed)

    print(f"{'cls':>3} {'elem':>4} {'val':>3} {'cnof':>5} {'src_occ':>8} {'pos_tgt':>8} "
          f"{'ins':>5} {'rst':>5}  status")
    for r in result["rows"]:
        print(f"{r['class_index']:>3} {r['element']:>4} {r['valence']:>3} {str(r['cnof']):>5} "
              f"{r['source_state_occurrences']:>8} {r['positive_edit_targets']:>8} "
              f"{r['target_atom_insert']:>5} {r['target_atom_restate']:>5}  {r['status']}")
    print(f"\ncold/absent non-CNOF classes: {result['cold_non_cnof_classes']}")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

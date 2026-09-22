"""What does the PMO construction lane actually BUILD, prior ON vs OFF?

The drift measurement is the outcome; this is the mechanism.  For the same real
parents and the same seeds, every realized ``_grow_actions`` insertion is
recorded as (inserted element, attachment element) and the two arms compared.

Two things worth knowing before reading the table.  First, the v1 element draw is
uniform over ``("C", "N", "O")`` but its OUTCOME is not: an oxygen tip has one
free hydrogen after a single bond and cannot extend, so the chains that survive
to full length are nitrogen-enriched, and the uniform draw's most common
insertion is nitrogen onto nitrogen.  Second, both arms are counted over the same
number of completed chains, so the comparison carries no survivorship
difference.

Zero oracle calls.  Nothing here reads a score, a task identity or an objective.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT
from compose_v4.control import dynamic_program_synthesis as dps
from compose_v4.control.learned_successor_prior import LearnedSuccessorPrior
from compose_v4.rewrite.trace_shard import decode_state

CKPT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
model, _ = load_factorized_rollout_checkpoint(CKPT, expected_scope_hash="3721d69851110fdd")
model.eval()
prior = LearnedSuccessorPrior(model, cache_entries=4096)

payload = json.loads(Path("/Users/rmaganti/compose_pmo_replay_data/production_parents_v1.json").read_text())
rows = sorted(
    payload["celecoxib_rediscovery"]["parents"], key=lambda x: (-x["uses"], x["key"])
)[:40]

def census(active):
    pairs = collections.Counter()
    for row in rows:
        src = decode_state(row["state"])
        if src.n_real_atoms < 4:
            continue
        for seed in range(12):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
            try:
                actions, _at, _chosen = dps._grow_actions(
                    src, rng, length=3, elements=("C", "N", "O"), successor_prior=active
                )
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
                continue
            current = src
            for record in actions:
                from compose_v4.rewrite.action_codec_v4 import decode_action
                _fam, action = decode_action(record)
                anchor = int(action.neighbors[0][0])
                host = IDX_TO_ELEMENT[int(current.atom_types[anchor])]
                new = IDX_TO_ELEMENT[int(action.atom_type)]
                pairs[(new, host)] += 1
                current, _ = dps.execute_program(current, [record])
    return pairs

out = {}
for label, active in (("off", None), ("on", prior)):
    pairs = census(active)
    total = sum(pairs.values())
    hetero_on_hetero = sum(v for (n, h), v in pairs.items() if n != "C" and h != "C")
    c_on_c = sum(v for (n, h), v in pairs.items() if n == "C" and h == "C")
    elem = collections.Counter()
    for (n, _h), v in pairs.items():
        elem[n] += v
    out[label] = {
        "insertions": total,
        "element_mix": {k: round(v / total, 4) for k, v in sorted(elem.items())},
        "heteroatom_onto_heteroatom_rate": round(hetero_on_hetero / total, 4),
        "carbon_onto_carbon_rate": round(c_on_c / total, 4),
        "top_pairs": [[f"{n}_onto_{h}", v, round(v / total, 4)]
                      for (n, h), v in pairs.most_common(8)],
    }
if out["on"]["heteroatom_onto_heteroatom_rate"]:
    out["hetero_on_hetero_suppression_x"] = round(
        out["off"]["heteroatom_onto_heteroatom_rate"]
        / out["on"]["heteroatom_onto_heteroatom_rate"], 2)
out["schema_version"] = "pmo_construction_placement_census_v1"
out["oracle_calls_spent"] = 0
out["kernel"] = "rdkit 2023.09.6 (PMO production)"
out["checkpoint_status"] = "PROVISIONAL_EDITING_CHECKPOINT -- forbidden for frozen results"
print(json.dumps(out, indent=1))
Path("diagnostics/pmo_construction_prior/placement_census_v1.json").write_text(
    json.dumps(out, indent=2, sort_keys=True))

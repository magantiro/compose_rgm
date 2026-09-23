"""The proposal policy must not depend on how Python happens to order a set.

Both floors MUTATE the weight vector as they are applied, so the result depends on the
order of application. Iterating a set of strings makes that order depend on
PYTHONHASHSEED, which differs between processes -- two identical runs then produce
different parent probabilities, and over a long campaign that eventually flips a
selection. Measured before the fix: 3.0e-3 between two identical uncached runs.

This drives the real policy in two subprocesses under different hash seeds, because the
ordering is a property of the interpreter and cannot be provoked inside one process.
"""
from __future__ import annotations

import json
import subprocess
import sys

PROBE = """
import json
import numpy as np
from compose_v4.control.pmo_reward_adaptive import RewardAdaptiveProgramController

def row(option, parent):
    return {
        "parent": parent, "endpoint": "Oc1ccccc1", "smiles": "Oc1ccccc1",
        "parent_score": 0.4,
        "families": ["region_replace", f"region_replace:{option}", option],
        "realized_families": ["region_replace", f"region_replace:{option}", option],
        "requested_modules": 2, "module_count": 2, "primitives": 3,
        "depth": 1, "generation": 0, "capacity_aware": False,
    }

# Many families and many lineages, so most of them fall BELOW the floors once the policy
# concentrates. A floor that never binds cannot expose an ordering dependence, which is
# exactly how the first version of this test passed against the unfixed code.
options = ["fuse_ring", "regrow", "append_ring", "segment_grow", "cycle_close",
           "carbonyl_insert", "heteroatom_substitute", "segment_shrink",
           "substituent_delete", "bond_reroute", "cycle_open", "functionalize",
           "ring_system_restate", "segment_replace", "ring_then_grow"]
parents = ["CC(=O)Oc1ccccc1C(=O)O", "Cc1ccccc1", "CCO", "c1ccccc1", "CCN",
           "CC(C)O", "c1ccncc1", "CCCC"]
rows = [row(o, p) for o in options for p in parents]

brain = RewardAdaptiveProgramController()
# Strongly differentiated reward so the softmax CONCENTRATES: most families then sit under
# the 0.02 family floor and the floors actually apply.
for index, r in enumerate(rows):
    favoured = "fuse_ring" in r["families"] and r["parent"] == parents[0]
    brain.observe(r, 0.98 if favoured else 0.02)
weights = brain.intent_policy(rows)
floored = sum(1 for w in weights if w > 0)
print(json.dumps({"weights": [float(w) for w in weights], "nonzero": floored}))
"""


def _weights(hash_seed: str) -> list[float]:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": "src", "PYTHONHASHSEED": hash_seed, "OMP_NUM_THREADS": "1"},
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])["weights"]


def test_policy_is_identical_under_different_hash_seeds():
    """Different PYTHONHASHSEED must not change a single proposal probability."""
    baseline = _weights("0")
    assert len(baseline) > 1
    for seed in ("1", "12345"):
        other = _weights(seed)
        assert len(other) == len(baseline)
        worst = max(abs(a - b) for a, b in zip(baseline, other, strict=True))
        assert worst == 0.0, f"PYTHONHASHSEED={seed} moved the policy by {worst:.3e}"

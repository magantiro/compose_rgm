#!/usr/bin/env python3
"""Standard molecular-optimization objectives for the value-guided SMC controller.

Provides the oracles used by the benchmarks that graph flows / diffusion /
GFlowNets / RL baselines report, so the same controller runs the whole table:

- QED (constrained-QED optimization; success = QED >= target at Tanimoto >= floor).
- Penalized logP (the JT-VAE normalized objective used by GraphAF, GraphDF,
  MoFlow, GCPN, JT-VAE; reported as constrained *improvement* over the lead).

Penalized logP uses the standard ZINC-normalized definition
    plogp(m) = z(logP) - z(SA) - z(largest_ring_penalty)
with the training-set means/stds used across the constrained-optimization
literature, so our numbers are comparable to reported ones.
"""

from __future__ import annotations

import os
import sys

from rdkit import Chem
from rdkit.Chem import Crippen, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles  # noqa: E402

# JT-VAE / GCPN / GraphAF penalized-logP normalization constants (ZINC train set).
_LOGP_MEAN, _LOGP_STD = 2.4570953396190123, 1.434324384576046
_SA_MEAN, _SA_STD = 3.0525811293166134, 0.8335207024513095
_CYCLE_MEAN, _CYCLE_STD = 0.0485696876403779, 0.2860212110245455


def penalized_logp_mol(mol) -> float:
    log_p = Crippen.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    ring_info = mol.GetRingInfo()
    largest_ring = max((len(r) for r in ring_info.AtomRings()), default=0)
    cycle_penalty = max(0, largest_ring - 6)
    return (
        (log_p - _LOGP_MEAN) / _LOGP_STD
        - (sa - _SA_MEAN) / _SA_STD
        - (cycle_penalty - _CYCLE_MEAN) / _CYCLE_STD
    )


def penalized_logp_state_oracle(state) -> float:
    mol = Chem.MolFromSmiles(molecular_graph_to_smiles(state))
    if mol is None:
        return -100.0
    return penalized_logp_mol(mol)


def get_objective_oracle(name: str):
    """Return a state -> float oracle for the named objective."""
    if name == "qed":
        # Imported lazily so this module has no hard dependency on the analytic stack.
        if __package__:
            from scripts.run_griddd_analytic_zero_sidecar_smoke import qed_state_oracle
        else:
            from run_griddd_analytic_zero_sidecar_smoke import qed_state_oracle
        return qed_state_oracle
    if name == "penalized_logp":
        return penalized_logp_state_oracle
    raise ValueError(f"unknown objective: {name}")


# Metric conventions per objective, for reporting against the literature.
OBJECTIVE_METRIC = {
    "qed": "success_rate_at_target",  # fraction reaching >= target QED at sim >= floor
    "penalized_logp": "mean_improvement",  # mean (best_feasible - lead) over leads
}

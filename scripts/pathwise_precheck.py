#!/usr/bin/env python3
"""Pathwise-safety pre-check (the deciding measurement).

Pathwise-constrained generation flips the quantifier from endpoint to EVERY
committed state. Tier-1 (pathwise safety) is a spotlight result ONLY IF endpoint-
only generation actually traverses forbidden (toxicophore) states en route to a
clean endpoint. This measures exactly that, before building any figure.

Endpoint-only baseline = the base model's own unconstrained ancestral editing
trajectory from a lead (what any endpoint-constrained optimizer rides on). We log
every committed intermediate molecule and check a set of CNOF-relevant structural
alerts. Headline number: P(some dirty intermediate | the endpoint is clean).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from griddd_value_guided_smc_controller import _load_base_sampler

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
LEADS = str(_ROOT / "configs" / "benchmarks" / "cnof_leads.json")
CKPT = "/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"

# Defensible, unambiguous reactive/unstable groups only (CNOF). Deliberately NOT
# including "any N-N single bond" (benign in pyrazoles/hydrazides) or bare N=O
# (matches N-oxides/oximes) -- those over-count. These are genuine structural
# alerts a medicinal chemist would flag as reactive or unstable.
ALERTS = {
    "nitro": "[$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[OX1-])]",
    "aldehyde": "[CX3H1](=O)[#6]",                          # aldehyde on carbon
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",           # a,b-unsaturated carbonyl
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",                     # strained 3-ring ether
    "aziridine": "[NX3r3]1[#6r3][#6r3]1",                   # strained 3-ring amine
    "peroxide": "[OX2][OX2]",                                # O-O bond
    "acyl_halide_like_anhydride": "[CX3](=[OX1])[OX2][CX3]=[OX1]",  # anhydride
    "geminal_N_O_aminal": "[NX3][CX4][OX2H]",              # hemiaminal (labile)
}
ALERT_MOLS = {k: Chem.MolFromSmarts(v) for k, v in ALERTS.items()}


def dirty(mol):
    for a in ALERT_MOLS.values():
        if a is not None and mol.HasSubstructMatch(a):
            return True
    return False


def rollout_states(sampler, rewrite, start, rng, max_events=48, time_step=0.1, horizon=16.0):
    node, t = start, 0.0
    out = []
    for _ in range(max_events):
        if t >= horizon:
            break
        mark = sampler.sample_rewrite_mark(node, frozen_time(t), rng)
        if mark.action is None:
            break
        try:
            node = rewrite.apply(node, mark.rule_name, mark.action)
        except Exception:  # noqa: BLE001
            break
        smi = molecular_graph_to_smiles(node)
        if smi:
            out.append(smi)
        t += time_step
    return out


def main():
    sampler, _ = _load_base_sampler(CKPT, 0.0)
    rewrite = de_novo_rewrite_system()
    leads = json.load(open(LEADS))[:5]
    N = 40
    rng = np.random.default_rng(11)

    n_traj = endpoint_clean = clean_end_dirty_path = any_dirty = 0
    total_states = dirty_states = 0
    which_alerts = {k: 0 for k in ALERTS}
    for _, smi, _ in leads:
        start = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        for _ in range(N):
            states = rollout_states(sampler, rewrite, start, rng)
            mols = [(s, Chem.MolFromSmiles(s)) for s in states]
            mols = [(s, m) for s, m in mols if m is not None]
            if not mols:
                continue
            n_traj += 1
            flags = [dirty(m) for _, m in mols]
            total_states += len(flags)
            dirty_states += sum(flags)
            for _, m in mols:
                for k, a in ALERT_MOLS.items():
                    if a is not None and m.HasSubstructMatch(a):
                        which_alerts[k] += 1
            end_clean = not flags[-1]
            path_dirty = any(flags[:-1])
            if end_clean:
                endpoint_clean += 1
                if path_dirty:
                    clean_end_dirty_path += 1
            if any(flags):
                any_dirty += 1

    print(f"trajectories: {n_traj} (5 leads x {N} rollouts)")
    print(f"committed states audited: {total_states}")
    print(f"state-level toxicophore rate: {dirty_states/max(total_states,1)*100:.1f}%  ({dirty_states}/{total_states})")
    print(f"trajectories with ANY dirty state: {any_dirty/max(n_traj,1)*100:.1f}%")
    print(f"endpoint-clean trajectories: {endpoint_clean/max(n_traj,1)*100:.1f}%  ({endpoint_clean}/{n_traj})")
    print(f"*** DECIDING NUMBER: P(dirty intermediate | endpoint clean) = "
          f"{clean_end_dirty_path/max(endpoint_clean,1)*100:.1f}%  ({clean_end_dirty_path}/{endpoint_clean}) ***")
    print("alert incidences (state-level):", {k: v for k, v in which_alerts.items() if v})
    verdict = "SPOTLIGHT (real exposed failure mode)" if clean_end_dirty_path / max(endpoint_clean, 1) > 0.15 \
        else "PIVOT to Tier 2 (trajectory-as-rationale; endpoint-only paths are mostly clean)"
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()

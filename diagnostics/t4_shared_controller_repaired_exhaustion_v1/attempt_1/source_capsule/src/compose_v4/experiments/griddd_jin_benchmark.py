"""Jin et al. ZINC-250k constrained-editing benchmark: sources and scoring.

Protocol frozen in `docs/GRIDDD_JIN_PROTOCOL.md`. Tier-1: COMPOSE runs alone
and GrIDDD's NeurIPS 2025 table is cited as reported.

THE SOURCE SETS ARE THE AUTHORS' OWN FILES, NOT A RECONSTRUCTION
----------------------------------------------------------------
Fetched from `wengong-jin/iclr19-graph2graph`, 800 molecules each, and verified
on arrival rather than assumed:

    qed/test.txt     800/800 with QED in [0.700, 0.800]  <- exactly as published
    logp06/test.txt  800 molecules, mean logP 0.454, min -4.178
    logp04/test.txt  BYTE-IDENTICAL to logp06

That last point matters: there is ONE logP source set evaluated at TWO
similarity thresholds, not two different source populations.

Reconstructing "the 800 lowest penalized logP" ourselves would have been a
guess -- papers differ on whether the pool is the full dataset or the test
split -- and a wrong guess silently breaks tier-1 alignment while still
looking like a head-to-head. The authors' files remove that risk entirely.

METRIC DEFINITIONS, VERBATIM FROM THE PROTOCOL
----------------------------------------------
LogP task: best improvement in PENALIZED logP over the source, among candidates
satisfying the similarity floor. Penalized logP is the standard
Jin/Kusner objective, `logP - SA - ring_penalty`, each term normalized by the
ZINC-250k training statistics.

QED task: SUCCESS RATE -- the fraction of sources for which at least one
candidate reaches QED in [0.9, 1.0] while satisfying Tanimoto >= 0.4.

Similarity is Morgan radius 2, 2048 bits, Tanimoto -- the benchmark convention.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parents[3]
JIN_DIR = REPO / "data/jin"   #: committed; small and load-bearing
ZINC_CSV = REPO / "local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv"

#: Published by GrIDDD (NeurIPS 2025). Cited, never recomputed.
PUBLISHED = {
    "JT-VAE": {"logp04": 1.03, "logp06": 0.28, "qed": 0.088},
    "CG-VAE": {"logp04": 0.61, "logp06": 0.25, "qed": 0.048},
    "GCPN":   {"logp04": 2.49, "logp06": 0.79, "qed": 0.094},
    "GrIDDD": {"logp04": 2.70, "logp06": 1.33, "qed": 0.451},
}

#: Normalization constants for penalized logP, from the ZINC-250k training set.
#: Computed once by `calibrate_penalized_logp()` and frozen into the run record
#: so the objective cannot drift between tasks or reruns.
PLOGP_STATS_FILE = REPO / "diagnostics/zinc250k_plogp_normalizers.json"


def load_sources(task: str) -> list[str]:
    """The authors' own 800-molecule test set. `task` in {logp, qed}."""
    name = {"logp": "logp06_test.txt", "qed": "qed_test.txt"}[task]
    smis = [ln.strip() for ln in (JIN_DIR / name).read_text().splitlines() if ln.strip()]
    if len(smis) != 800:
        raise ValueError(f"{name}: expected 800 sources, found {len(smis)}")
    return smis


def morgan_fp(mol):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def tanimoto(fp_a, fp_b) -> float:
    from rdkit import DataStructs
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _ring_penalty(mol) -> float:
    """Largest ring beyond size 6, the standard Jin/Kusner cycle term."""
    import networkx as nx
    from rdkit import Chem
    # `nx.Graph(list_of_lists)` is parsed as an EDGE LIST, not an adjacency
    # matrix. `from_numpy_array` preserves the standard Jin/Kusner semantics.
    cycles = nx.cycle_basis(nx.from_numpy_array(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largest = max((len(c) for c in cycles), default=0)
    return float(max(0, largest - 6))


def penalized_logp(mol, stats: dict[str, float]) -> float:
    """logP - SA - ring_penalty, each z-normalized by ZINC-250k statistics."""
    import sys

    from rdkit.Chem import Crippen, RDConfig
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer  # type: ignore

    lp = (float(Crippen.MolLogP(mol)) - stats["logp_mean"]) / stats["logp_std"]
    sa = (float(sascorer.calculateScore(mol)) - stats["sa_mean"]) / stats["sa_std"]
    cy = (_ring_penalty(mol) - stats["cycle_mean"]) / stats["cycle_std"]
    return lp - sa - cy


def logp_task_score(
    source: str, candidates: Sequence[str], stats: dict[str, float],
    threshold: float,
) -> float | None:
    """Best penalized-logP IMPROVEMENT among candidates meeting the floor.

    Returns None when no candidate satisfies the similarity constraint -- the
    benchmark's own convention for a failed source, which must be carried as a
    failure rather than silently dropped from the mean.
    """
    from rdkit import Chem

    ms = Chem.MolFromSmiles(source)
    if ms is None:
        return None
    fs, base = morgan_fp(ms), penalized_logp(ms, stats)
    best = None
    for c in candidates:
        mc = Chem.MolFromSmiles(c)
        if mc is None or tanimoto(fs, morgan_fp(mc)) < threshold:
            continue
        gain = penalized_logp(mc, stats) - base
        if best is None or gain > best:
            best = gain
    return best


def qed_task_success(source: str, candidates: Sequence[str],
                     threshold: float = 0.4) -> bool:
    """True if ANY candidate reaches QED in [0.9, 1.0] at Tanimoto >= threshold."""
    from rdkit import Chem
    from rdkit.Chem import QED

    ms = Chem.MolFromSmiles(source)
    if ms is None:
        return False
    fs = morgan_fp(ms)
    for c in candidates:
        mc = Chem.MolFromSmiles(c)
        if mc is None or tanimoto(fs, morgan_fp(mc)) < threshold:
            continue
        if 0.9 <= float(QED.qed(mc)) <= 1.0:
            return True
    return False


def calibrate_penalized_logp(limit: int | None = None) -> dict[str, float]:
    """Compute the ZINC-250k normalizers ONCE, from the canonical dataset."""
    import csv
    import sys

    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, RDConfig

    RDLogger.DisableLog("rdApp.*")
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer  # type: ignore

    lp, sa, cy = [], [], []
    with open(ZINC_CSV, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                break
            m = Chem.MolFromSmiles(row["smiles"].strip())
            if m is None:
                continue
            lp.append(float(Crippen.MolLogP(m)))
            sa.append(float(sascorer.calculateScore(m)))
            cy.append(_ring_penalty(m))
    return {
        "logp_mean": float(np.mean(lp)), "logp_std": float(np.std(lp)),
        "sa_mean": float(np.mean(sa)), "sa_std": float(np.std(sa)),
        "cycle_mean": float(np.mean(cy)), "cycle_std": float(np.std(cy)),
        "n_molecules": len(lp),
    }

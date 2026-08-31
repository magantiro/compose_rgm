"""Medicinal-chemistry validity gate for RETURNED molecules.

Sits entirely outside the frozen process: it reads a SMILES and returns reasons.
It does NOT touch production_successor_kernel.py, does not change
process_identity_sha256, and requires no retraining of R_theta.

Motivation: QED, SA and Tanimoto jointly fail to reject hypervalent iodine,
phosphines, and 8-10 membered rings. QuickVina scores such structures happily,
so they can win cells. See docs/AMENDMENT_T4_TRAJECTORY_CONTROLLER.md.

Calibration requirement (tests/test_med_chem_gate.py): the gate must ACCEPT all
15 T4 seeds and all 25 published InVirtuoGen winners, and REJECT the 8 known-bad
COMPOSE molecules. A gate that rejects a real drug-like molecule is a worse bug
than the one it fixes.
"""
from __future__ import annotations
from rdkit import Chem

# Elements in the T4 seeds and IVG winners, plus S/Br/P which are ordinary in drugs.
ALLOWED_ELEMENTS = frozenset({"C", "N", "O", "F", "S", "Cl", "Br", "I", "P"})
HALOGENS = frozenset({"F", "Cl", "Br", "I"})
# SSSR reports "envelope" rings spanning fused systems: the parp1 seed's tricyclic
# core reports [5,6,7] and the jak2 s14 seed reports [5,6,6,6,8]. Ring SIZE alone
# therefore cannot separate a fused drug scaffold from a real macrocycle. We flag a
# large ring only when it is genuinely isolated -- most of its atoms belong to no
# other ring. Measured over the 15 seeds + 25 IVG winners, the largest such
# unshared span is MAX_ISOLATED_RING.
MAX_ISOLATED_RING = 7
MAX_FORMAL_CHARGE = 1      # protonated amines [NH2+] appear in the 5ht1b seeds


def validity_reasons(smiles: str) -> list[str]:
    """Return a list of reasons the molecule should be rejected. Empty == accept."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ["unparseable"]
    out: list[str] = []

    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s not in ALLOWED_ELEMENTS:
            out.append(f"element:{s}")
            continue
        if a.GetNumRadicalElectrons():
            out.append(f"radical:{s}{a.GetIdx()}")
        if abs(a.GetFormalCharge()) > MAX_FORMAL_CHARGE:
            out.append(f"charge:{s}{a.GetFormalCharge():+d}")
        # A halogen in a drug is a terminal substituent: one bond, no hydrogens.
        # [IH2] (2 bonds + 2 H) and [IH4] (1 bond + 4 H) are hypervalent iodine.
        if s in HALOGENS and (a.GetTotalNumHs() or a.GetDegree() != 1):
            out.append(f"hypervalent_halogen:{s}H{a.GetTotalNumHs()}d{a.GetDegree()}")
        # Sulfur: thioether/thiol/sulfoxide/sulfone/thiophene are all fine, but
        # S-H inside a ring or with >2 heavy neighbours is not drug chemistry.
        # Missed on the first pass, which let C1#[SH2]CNC=N1 -- a triple bond to
        # a hypervalent sulfur -- through the gate and into macro endpoints.
        if s == "S" and a.GetTotalNumHs() and a.GetDegree() >= 2:
            out.append(f"hypervalent_S:H{a.GetTotalNumHs()}d{a.GetDegree()}")
        # Drug phosphorus is phosphate/phosphonate: P(=O) with O neighbours, no P-H.
        if s == "P":
            nb = [n.GetSymbol() for n in a.GetNeighbors()]
            if a.GetTotalNumHs() or nb.count("O") < 2:
                out.append(f"non_phosphate_P:H{a.GetTotalNumHs()}:{''.join(sorted(nb))}")

    ri = m.GetRingInfo()
    rings = ri.AtomRings()
    for ring in rings:
        if len(ring) > MAX_ISOLATED_RING:
            unshared = sum(1 for a in ring
                           if sum(1 for r2 in rings if a in r2) == 1)
            if unshared > MAX_ISOLATED_RING:
                out.append(f"isolated_ring:{len(ring)}")
        # diazirine / diazetine: N-N inside a small strained ring
        if len(ring) <= 4 and sum(1 for i in ring
                                  if m.GetAtomWithIdx(i).GetSymbol() == "N") >= 2:
            out.append(f"strained_NN_ring:{len(ring)}")
        # Saturated polyaza rings: hexazine (N1NNNNN1) and its relatives are not
        # synthesizable, and the size<=4 rule above let every one of them
        # through -- BUILD_RING_SYSTEM with composition="hetero_rich" built a
        # hexazine pendant and BOTH gates passed it.
        #
        # Counted on NON-AROMATIC N-N bonds only. Aromatic polyaza rings are
        # legitimate and common: tetrazole (c1nnnn1) has four contiguous
        # nitrogens and appears in marketed drugs, so an aromatic-blind rule
        # would reject real chemistry to catch this.
        nn = sum(1 for i in ring for j in ring
                 if i < j
                 and m.GetAtomWithIdx(i).GetSymbol() == "N"
                 and m.GetAtomWithIdx(j).GetSymbol() == "N"
                 and (b := m.GetBondBetweenAtoms(i, j)) is not None
                 and not b.GetIsAromatic())
        if nn >= 2:
            out.append(f"polyaza_ring:{len(ring)}:NN{nn}")

    # cumulated double bonds (C=C=C) are not stable drug motifs
    for a in m.GetAtoms():
        if a.GetSymbol() == "C" and sum(
            1 for b in a.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
        ) >= 2 and not a.GetIsAromatic():
            out.append(f"cumulene:C{a.GetIdx()}")

    return sorted(set(out))


def is_valid(smiles: str) -> bool:
    return not validity_reasons(smiles)


# ---------------------------------------------------------------------------
# TWO GATES, TWO PLACES.
#
# Applying the full med-chem gate to EVERY primitive state strangles the search:
# a particle cannot pass THROUGH a temporarily unattractive molecule to reach a
# good one. Measured on the first three GATED30 cells, where the gate ran inside
# the state profiler: -9.7 / -8.2 / -7.8 against an ungated archive's
# -9.9 / -10.2 / -8.8, with endpoints barely displaced from their seeds.
#
#   pathwise   is this an EXECUTABLE MOLECULE?   sanitisation, impossible
#              valence, hypervalent I/P/S. Cheap, and never blocks a legal
#              trajectory that could still lead somewhere.
#   endpoint   is this an ACCEPTABLE ANSWER?     the above PLUS QED/SA/similarity
#              and the isolated-macrocycle preference. Applied to macro
#              endpoints, archive admission and pre-docking selection only.
#
# T4 itself imposes QED/SA/similarity on the RETURNED molecule only, so this
# split matches the benchmark's own semantics rather than tightening them.
# ---------------------------------------------------------------------------

#: reasons that describe an unexecutable / impossible molecule
_PATHWISE_PREFIXES = ("unparseable", "element:", "radical:", "charge:",
                      "hypervalent_halogen", "hypervalent_S", "non_phosphate_P")


def pathwise_reasons(smiles: str) -> list[str]:
    """Only the reasons a state is not a real molecule at all."""
    return [r for r in validity_reasons(smiles)
            if r.startswith(_PATHWISE_PREFIXES)]


def is_executable(smiles: str) -> bool:
    """Pathwise gate: may a search PASS THROUGH this state?"""
    return not pathwise_reasons(smiles)


def is_acceptable_endpoint(smiles: str, qed_min: float = 0.6,
                           sa_max: float = 4.0) -> bool:
    """Endpoint gate: is this a molecule we would RETURN or dock?"""
    if not is_valid(smiles):
        return False
    from rdkit import Chem
    from rdkit.Chem import QED
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    try:
        import os, sys
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        sa = float(sascorer.calculateScore(m))
    except Exception:
        return False
    return float(QED.qed(m)) >= qed_min and sa <= sa_max


def t4_feasible(smiles: str, seed_smiles: str, delta: float,
                qed_min: float = 0.6, sa_max: float = 4.0) -> bool:
    """THE T4 constraint. All four, or it is not feasible.

        QED >= 0.6  AND  SA <= 4  AND  sim(x, seed) >= delta  AND  gate-clean

    This exists because every macro experiment on 2026-08-25 measured only
    QED and SA and reported the result as "feasible yield" -- 3.1% -> 22.5% ->
    35% -- while the true rate was 1.7%, because SIMILARITY is the binding
    constraint and was never checked. 118 of 120 carbon-rich endpoints failed
    it (median 0.143 against a required 0.40). Never call anything feasible
    without going through this function.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator
    m = Chem.MolFromSmiles(smiles)
    s = Chem.MolFromSmiles(seed_smiles)
    if m is None or s is None:
        return False
    if not is_valid(smiles):
        return False
    if float(QED.qed(m)) < qed_min:
        return False
    try:
        import os, sys
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        if float(sascorer.calculateScore(m)) > sa_max:
            return False
    except Exception:
        return False
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sim = DataStructs.TanimotoSimilarity(gm.GetFingerprint(s), gm.GetFingerprint(m))
    return sim >= float(delta)

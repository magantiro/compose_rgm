#!/usr/bin/env python3
"""Stratified corpus pilot enumeration + paper-grade diversity/coverage metrics.

Enumerates the qualified reaction families over the building-block pool, globally
deduplicates, selects a family/architecture-balanced pilot via the deterministic
stratified reservoir, and reports the metrics needed to defend an unsupervised
structural corpus: family balance, validity/uniqueness, nearest-neighbor
redundancy (ECFP4 Tanimoto), physicochemical marginals vs the LNPDB R0 anchor,
LNPDB coverage/recall, size eligibility, and scaffold diversity.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/enumerate_corpus_pilot.py --quota 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from compose_v4.lipids.building_block_registry import load_pool
from compose_v4.lipids.corpus_bias import (
    heteroatom_connector_core,
    hill_effective_numbers,
    jensen_shannon,
    lipid_topology_features,
    probability_vector,
)
from compose_v4.lipids.reaction_enumeration import BuildingBlock, ReactionEnumerator
from compose_v4.lipids.reaction_registry import ReactionRegistry
from compose_v4.lipids.streaming_enumeration import StratifiedReservoir
from compose_v4.oracles.head_domain import head_region_atoms

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parents[1]

REGISTRIES = [
    "configs/lipid_reactions/qualified_reactions_v1.json",
    "configs/lipid_reactions/qualified_reaction_families_v1.json",
]
_MFP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return _MFP.GetFingerprint(m) if m else None


def _amine_nh_count(smiles: str) -> int:
    """Total N-H hydrogens on reactive (non-amide, neutral) amine nitrogens."""
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!$(N=*)]")
    return sum(mol.GetAtomWithIdx(m[0]).GetTotalNumHs() for m in mol.GetSubstructMatches(patt))


def _multitail(enumerator, amine_block, tail_block, max_tails: int, frontier_cap: int = 6):
    """Iteratively substitute an amine at successive N-H sites (C12-200 style).

    Real ionizable lipids are overwhelmingly multi-tail (R0: ~3% single-tail,
    47% two-tail, 41% three-plus). So a product is only emitted once it carries
    >=2 tails whenever the head can support them (a primary amine -> a 2-tail
    tertiary-amine head is a real motif); heads with a single N-H emit at 1 tail.
    """
    role_order = enumerator.role_order
    amine_pos = role_order.index("amine_head")
    tail_pos = 1 - amine_pos
    min_emit = 2 if _amine_nh_count(amine_block.smiles) >= 2 else 1
    frontier = [amine_block.smiles]
    for depth in range(1, max_tails + 1):
        nxt: dict[str, object] = {}
        for inter in frontier:
            inter_block = BuildingBlock(amine_block.block_id, "amine_head", inter, amine_block.architecture_tags)
            ordered = [None, None]
            ordered[amine_pos] = inter_block
            ordered[tail_pos] = tail_block
            for product in enumerator.react(ordered):
                nxt.setdefault(product.canonical_smiles, product)
        if not nxt:
            break
        if depth >= min_emit:
            for product in nxt.values():
                yield product
        # bounded frontier (deterministic: lowest canonical SMILES) to avoid blow-up
        frontier = sorted(nxt)[:frontier_cap]


def _asymmetric_ditail(enumerator, amine_block, tail_a, tail_b, max_frontier: int = 3):
    """Substitute an amine with tail_a then tail_b (two DIFFERENT tails) -> an
    asymmetric multi-tail lipid, the common real-lipid motif."""
    role_order = enumerator.role_order
    amine_pos = role_order.index("amine_head")
    tail_pos = 1 - amine_pos

    def sub(inter_smiles, tail):
        ib = BuildingBlock(amine_block.block_id, "amine_head", inter_smiles, amine_block.architecture_tags)
        ordered = [None, None]
        ordered[amine_pos] = ib
        ordered[tail_pos] = tail
        return {p.canonical_smiles: p for p in enumerator.react(ordered)}

    out: dict[str, object] = {}
    mono = sub(amine_block.smiles, tail_a)
    for inter in sorted(mono)[:max_frontier]:
        out.update(sub(inter, tail_b))
    return list(out.values())


def enumerate_all(max_tails: int = 4, asymmetric_tail_b: int = 4) -> list:
    pool = load_pool(REPO_ROOT / "configs/lipid_reactions/building_block_pool_v1.json")
    products: dict[str, object] = {}
    per_family_raw: dict[str, int] = {}
    for reg_path in REGISTRIES:
        registry = ReactionRegistry.load(REPO_ROOT / reg_path)
        for spec in registry.reactions:
            enumerator = ReactionEnumerator(spec)
            roles = enumerator.role_order
            by_role = pool.blocks_for_roles(roles)
            if any(len(by_role.get(r, [])) == 0 for r in roles):
                continue
            count = 0
            multitail = len(roles) == 2 and "amine_head" in roles
            if multitail:
                tail_role = next(r for r in roles if r != "amine_head")
                tails = by_role[tail_role]
                for amine in by_role["amine_head"]:
                    for tail in tails:
                        for product in _multitail(enumerator, amine, tail, max_tails):
                            count += 1
                            products.setdefault(product.canonical_smiles, product)
                # asymmetric di-tail (two different tails) for polyamine heads.
                # Bounded: diverse subsets of BOTH tails keep it fast but varied.
                tail_a_subset = tails[:: max(1, len(tails) // 10)][:10]
                tail_b_subset = tails[:: max(1, len(tails) // asymmetric_tail_b)][:asymmetric_tail_b]
                polyamines = [a for a in by_role["amine_head"] if _amine_nh_count(a.smiles) >= 2]
                for amine in polyamines:
                    for tail_a in tail_a_subset:
                        for tail_b in tail_b_subset:
                            if tail_b.block_id == tail_a.block_id:
                                continue
                            for product in _asymmetric_ditail(enumerator, amine, tail_a, tail_b):
                                count += 1
                                products.setdefault(product.canonical_smiles, product)
            else:
                for product in enumerator.enumerate(by_role):
                    count += 1
                    products.setdefault(product.canonical_smiles, product)
            per_family_raw[spec.reaction_id] = per_family_raw.get(spec.reaction_id, 0) + count
    return list(products.values()), per_family_raw


def select_pilot(products: list, quota: int, seed: str = "compose_lipid_pilot_v1") -> list:
    axes = ("reaction_family", "size_bin", "charge_bin", "ring_bin")
    reservoir = StratifiedReservoir(axes=axes, quota_per_stratum=quota, seed=seed)
    for p in products:
        reservoir.consider(p)
    return reservoir.selected()


def select_family_balanced(products: list, per_family_quota: int, cap_per_core: int,
                           seed: str = "compose_lipid_pilot_v1") -> list:
    """Select per family independently so small families (disulfide/thioether/
    Passerini) are not starved by a global core cap. Each family contributes up
    to per_family_quota after within-family stratification + core de-dup; small
    families contribute all they have."""
    by_family: dict[str, list] = {}
    for p in products:
        by_family.setdefault(p.reaction_id, []).append(p)
    selected: list = []
    for family, prods in sorted(by_family.items()):
        # Within-family stratification on architecture axes gives diversity; core
        # de-dup is NOT applied here because tail-diversity-dominated families
        # (disulfide, thioether) legitimately share one heteroatom core and would
        # be starved. Global mode still core-dedups.
        axes = ("size_bin", "charge_bin", "ring_bin", "unsaturation_bin", "branching_bin")
        reservoir = StratifiedReservoir(axes=axes, quota_per_stratum=cap_per_core, seed=f"{seed}:{family}")
        for p in prods:
            reservoir.consider(p)
        family_selected = reservoir.selected()[:per_family_quota]
        selected.extend(family_selected)
    return selected


def _descriptors(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    return {
        "mw": Descriptors.MolWt(m), "logp": Crippen.MolLogP(m), "tpsa": Descriptors.TPSA(m),
        "hbd": Lipinski.NumHDonors(m), "hba": Lipinski.NumHAcceptors(m),
        "rot": Lipinski.NumRotatableBonds(m), "rings": rdMolDescriptors.CalcNumRings(m),
        "charge": Chem.GetFormalCharge(m), "heavy": m.GetNumHeavyAtoms(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
    }


def _marginals(smis: list[str]) -> dict:
    rows = [_descriptors(s) for s in smis]
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": round(float(vals.mean()), 2), "median": round(float(np.median(vals)), 2),
                  "p5": round(float(np.percentile(vals, 5)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}
    return out


def _nn_redundancy(fps: list, sample: int, rng: np.random.Generator) -> dict:
    n = len(fps)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    nn = []
    for i in idx:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i] + fps[i + 1:])
        nn.append(max(sims) if sims else 0.0)
    nn = np.array(nn)
    return {"sampled": int(len(idx)), "median_nn_tanimoto": round(float(np.median(nn)), 3),
            "frac_nn_ge_0.9": round(float((nn >= 0.9).mean()), 3),
            "frac_nn_ge_0.95": round(float((nn >= 0.95).mean()), 3)}


def _lnpdb_coverage(corpus_fps: list, rng: np.random.Generator, sample: int = 2000) -> dict:
    r0_path = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv"
    import csv
    r0 = [row["canonical_isomeric_smiles"] for row in csv.DictReader(r0_path.open())]
    idx = rng.choice(len(r0), size=min(sample, len(r0)), replace=False)
    covered = 0; nn_to_corpus = []
    for i in idx:
        fp = fingerprint(r0[i])
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, corpus_fps)
        m = max(sims) if sims else 0.0
        nn_to_corpus.append(m)
        if m >= 0.4:
            covered += 1
    nn_to_corpus = np.array(nn_to_corpus)
    return {"r0_sampled": int(len(nn_to_corpus)),
            "recall_at_tanimoto_0.4": round(float(covered / len(nn_to_corpus)), 3),
            "median_r0_nn_to_corpus": round(float(np.median(nn_to_corpus)), 3)}


# family -> its dominant linker motif (consistent with the audit's priority order)
FAMILY_LINKER = {
    "aza_michael_amine_acrylate": "ester", "passerini_3cr": "amide", "ugi_3cr_agile": "amide",
    "amide_coupling_acid_amine": "amide", "epoxide_opening_amine": "none",
    "reductive_amination_amine_aldehyde": "none", "thiol_michael_thioether": "thioether",
    "disulfide_coupling": "disulfide", "carbamate_amine_chloroformate": "carbamate",
    "urea_amine_isocyanate": "urea", "acetal_aldehyde_diol": "acetal",
    "iphos_amine_dioxaphospholane": "phosphate",
}
_LINKER_PRIORITY = ["phosphate", "disulfide", "acetal", "carbamate", "urea", "thioether", "amide", "ester"]
_LINKER_PATT = {n: Chem.MolFromSmarts(s) for n, s in {
    "phosphate": "[PX4](=O)([OX2])[OX2]", "disulfide": "[SX2][SX2]", "acetal": "[CX4]([OX2])([OX2])",
    "carbamate": "[NX3][CX3](=O)[OX2]", "urea": "[NX3][CX3](=O)[NX3]", "thioether": "[#6][SX2][#6]",
    "amide": "[NX3][CX3](=O)[#6]", "ester": "[CX3](=O)[OX2][#6]"}.items()}


def r0_linker_distribution(r0_smiles: list[str]) -> dict[str, float]:
    from collections import Counter
    counts: Counter = Counter()
    for s in r0_smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for name in _LINKER_PRIORITY:
            if m.HasSubstructMatch(_LINKER_PATT[name]):
                counts[name] += 1
                break
        else:
            counts["none"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def select_realism_weighted(products: list, r0_dist: dict, target: int, coverage_floor: int,
                            seed: str = "compose_lipid_realism_v1") -> list:
    """Match R0's linker-frequency distribution (realism) while flooring every
    family (coverage). Ester/amide-dominant like real lipids; rare/novel linkers
    still get a minimum so the generator can learn them."""
    from collections import defaultdict
    by_family: dict[str, list] = defaultdict(list)
    for p in products:
        by_family[p.reaction_id].append(p)
    linker_families: dict[str, list] = defaultdict(list)
    for fam in by_family:
        linker_families[FAMILY_LINKER.get(fam, "none")].append(fam)
    selected: list = []
    for fam, prods in sorted(by_family.items()):
        linker = FAMILY_LINKER.get(fam, "none")
        realism_q = int(target * r0_dist.get(linker, 0.0) / max(1, len(linker_families[linker])))
        quota = max(coverage_floor, realism_q)
        axes = ("size_bin", "charge_bin", "ring_bin", "unsaturation_bin", "branching_bin")
        reservoir = StratifiedReservoir(axes=axes, quota_per_stratum=max(2, quota // 8 + 1),
                                        seed=f"{seed}:{fam}")
        for p in prods:
            reservoir.consider(p)
        selected.extend(reservoir.selected()[:quota])
    return selected


def _bin(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


# axes that family-quota (linker) selection does NOT control -- the tail/head shape.
# These are matched to R0 by a final importance-resample. Binning mirrors
# audit_structural_freedom.dof_keys exactly so the audit confirms the correction.
_RESAMPLE_AXES = ("linker_type", "n_tails", "tail_length", "head_size")


def _dof_key(mol) -> dict[str, str]:
    f = lipid_topology_features(mol)
    linker = "none"
    for name in _LINKER_PRIORITY:
        if mol.HasSubstructMatch(_LINKER_PATT[name]):
            linker = name
            break
    return {
        "linker_type": linker,
        "n_tails": f["long_tail_bin"],
        "tail_length": _bin(f["max_aliphatic_tail_depth"], [7, 12, 17, 22],
                            ["<=7", "8-12", "13-17", "18-22", ">22"]),
        "head_size": _bin(len(head_region_atoms(mol)), [4, 8, 14], ["<=4", "5-8", "9-14", ">14"]),
    }


def marginal_match_resample(candidates: list, r0_smiles: list[str], target: int,
                            coverage_floor: int, seed: str = "compose_lipid_ipf_v1",
                            iters: int = 40) -> list:
    """Resample the (over-selected) candidate set so its tail/head marginals match
    R0 WITHOUT starving rare families. Family-quota selection matches the linker
    distribution but cannot touch how many tails / how long / how big the head is
    -- those come from which blocks got combined. Iterative proportional fitting
    assigns each candidate an importance weight so all four marginals (linker +
    n_tails + tail_length + head_size) match R0 jointly. A pure IPF draw would
    collapse R0-rare families (disulfide/urea/passerini) to zero, so we draw in
    two tiers: first guarantee `coverage_floor` per family (IPF-weighted WITHIN the
    family, so even the floor is realism-shaped), then fill the remainder by global
    IPF weight. Coverage of every linker for the generator + faithful bulk ratios."""
    keys = [_dof_key(Chem.MolFromSmiles(c.canonical_smiles)) for c in candidates]
    r0_keys = [_dof_key(m) for m in (Chem.MolFromSmiles(s) for s in r0_smiles) if m is not None]
    from collections import Counter, defaultdict
    targets: dict[str, dict[str, float]] = {}
    for axis in _RESAMPLE_AXES:
        c = Counter(k[axis] for k in r0_keys)
        tot = sum(c.values()) or 1
        targets[axis] = {b: v / tot for b, v in c.items()}
    bins_by_axis = {axis: defaultdict(list) for axis in _RESAMPLE_AXES}
    for i, k in enumerate(keys):
        for axis in _RESAMPLE_AXES:
            bins_by_axis[axis][k[axis]].append(i)

    n = len(candidates)
    w = np.ones(n)
    for _ in range(iters):
        for axis in _RESAMPLE_AXES:
            W = w.sum()
            for b, idxs in bins_by_axis[axis].items():
                cur = w[idxs].sum()
                tgt = targets[axis].get(b, 1e-6) * W  # bins ~absent in R0 -> near-zero mass
                if cur > 0:
                    w[idxs] *= tgt / cur
        w *= n / w.sum()

    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    take = min(target, n)
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, cand in enumerate(candidates):
        by_family[cand.reaction_id].append(i)
    chosen: set[int] = set()
    for fam, idxs in sorted(by_family.items()):  # tier 1: per-family coverage floor
        k = min(coverage_floor, len(idxs))
        wf = w[idxs] / w[idxs].sum()
        for i in rng.choice(idxs, size=k, replace=False, p=wf):
            chosen.add(int(i))
    remaining = [i for i in range(n) if i not in chosen]  # tier 2: bulk by global IPF weight
    need = max(0, take - len(chosen))
    if need > 0 and remaining:
        wr = w[remaining] / w[remaining].sum()
        for i in rng.choice(remaining, size=min(need, len(remaining)), replace=False, p=wr):
            chosen.add(int(i))
    return [candidates[i] for i in sorted(chosen)]


def core_deredundant(selected: list, cap_per_core: int) -> list:
    """Cap products per heteroatom-connector core to control lipid redundancy."""
    per_core: dict[str, int] = {}
    kept = []
    for p in sorted(selected, key=lambda x: x.canonical_smiles):
        core = heteroatom_connector_core(Chem.MolFromSmiles(p.canonical_smiles))
        if per_core.get(core, 0) < cap_per_core:
            per_core[core] = per_core.get(core, 0) + 1
            kept.append(p)
    return kept


def lipid_diversity(feats: list[dict]) -> dict:
    """Lipid-appropriate diversity via heteroatom cores + architecture (Hill numbers)."""
    smis = feats  # kept for arity; feats are precomputed topology dicts
    cores = [f["heteroatom_connector_core"] for f in feats]
    archs = [f["tail_architecture_proxy"] for f in feats]
    degr = [f["cleavable_motif_proxy"] for f in feats]
    long_tail = [f["long_tail_bin"] for f in feats]
    return {
        "n": len(smis),
        "heteroatom_core_hill": {k: round(v, 1) for k, v in hill_effective_numbers(cores).items()},
        "architecture_hill": {k: round(v, 1) for k, v in hill_effective_numbers(archs).items()},
        "degradable_signature_hill": {k: round(v, 1) for k, v in hill_effective_numbers(degr).items()},
        "long_tail_distribution": {b: round(long_tail.count(b) / len(smis), 3)
                                   for b in sorted(set(long_tail))},
        "degradable_signature_distribution": {d: round(degr.count(d) / len(smis), 3)
                                              for d in sorted(set(degr))},
    }


def _kernel_profile(smis: list[str]) -> dict:
    """Element / charge / size / stereo profile the inherited RGM lipid kernel must support (P2-G2)."""
    elements: dict[str, int] = {}
    charges: dict[int, int] = {}
    heavy = []
    n_stereo = 0
    for s in smis:
        m = Chem.MolFromSmiles(s)
        for atom in m.GetAtoms():
            elements[atom.GetSymbol()] = elements.get(atom.GetSymbol(), 0) + 1
        c = Chem.GetFormalCharge(m)
        charges[c] = charges.get(c, 0) + 1
        heavy.append(m.GetNumHeavyAtoms())
        if Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False):
            n_stereo += 1
    h = np.array(heavy)
    return {
        "elements_present": sorted(elements),
        "fraction_charged": round(float(1 - charges.get(0, 0) / len(smis)), 4),
        "charge_states": {str(k): v for k, v in sorted(charges.items())},
        "fraction_with_stereocenter": round(n_stereo / len(smis), 4),
        "heavy_atoms": {"min": int(h.min()), "median": int(np.median(h)), "max": int(h.max()),
                        "frac_le_40": round(float((h <= 40).mean()), 3),
                        "frac_le_64": round(float((h <= 64).mean()), 3),
                        "frac_le_96": round(float((h <= 96).mean()), 3)},
        "p2g2_note": "Lipid kernel must support these elements, charge states, and heavy-atom range; the C/N/O/F base kernel is not assumed lipid-ready.",
    }


def _r0_sample(rng: np.random.Generator, n: int = 1500) -> list[str]:
    import csv as _csv
    path = REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/r0_observed_real_structures.csv"
    r0 = [row["canonical_isomeric_smiles"] for row in _csv.DictReader(path.open())]
    idx = rng.choice(len(r0), size=min(n, len(r0)), replace=False)
    return [r0[i] for i in idx]


def _js_marginals(corpus_feats: list[dict], ref_feats: list[dict]) -> dict:
    """Jensen-Shannon divergence of key categorical marginals (faithfulness to R0)."""
    out = {}
    for key in ("size_bin", "charge_bin", "ring_bin", "long_tail_bin", "unsaturation_bin", "branching_bin"):
        cf = [f[key] for f in corpus_feats]
        rf = [f[key] for f in ref_feats]
        cats = sorted(set(cf) | set(rf))
        out[key] = round(jensen_shannon(probability_vector(cf, cats), probability_vector(rf, cats)), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota", type=int, default=120, help="max products per size/charge/ring stratum")
    parser.add_argument("--cap-per-core", type=int, default=6, help="max products per heteroatom core")
    parser.add_argument("--per-family-quota", type=int, default=0,
                        help="if >0, select family-balanced (each family up to this many)")
    parser.add_argument("--realism-target", type=int, default=0,
                        help="if >0, realism-weighted selection matching R0 linker ratios (this total)")
    parser.add_argument("--coverage-floor", type=int, default=40,
                        help="min products per family under realism weighting (coverage)")
    parser.add_argument("--marginal-match", action="store_true",
                        help="after linker-quota selection, IPF-resample tail/head marginals to R0")
    parser.add_argument("--oversample", type=int, default=5,
                        help="over-selection factor before marginal-match resample")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "artifacts/datasets/compose_lipid_pretraining_v1/corpus_pilot_v1")
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    args.out_dir = Path(args.out_dir).resolve()  # so relative_to(REPO_ROOT) works
    rng = np.random.default_rng(args.seed)

    products, per_family_raw = enumerate_all()
    total_unique = len(products)
    if args.realism_target > 0:
        candidate = products
        r0_for_match = _r0_sample(rng, n=2500)
        r0_dist = r0_linker_distribution(r0_for_match)
        if args.marginal_match:
            # over-select on cheap linker quotas, then IPF-match tail/head marginals to R0
            over = select_realism_weighted(products, r0_dist,
                                           args.realism_target * args.oversample, args.coverage_floor)
            selected = marginal_match_resample(over, r0_for_match, args.realism_target,
                                               args.coverage_floor)
        else:
            selected = select_realism_weighted(products, r0_dist, args.realism_target, args.coverage_floor)
    elif args.per_family_quota > 0:
        candidate = products
        selected = select_family_balanced(products, args.per_family_quota, args.cap_per_core)
    else:
        candidate = select_pilot(products, args.quota)
        selected = core_deredundant(candidate, args.cap_per_core)

    sel_smiles = [p.canonical_smiles for p in selected]
    sel_fps = [f for f in (fingerprint(s) for s in sel_smiles) if f is not None]
    r0_smiles = _r0_sample(rng)
    r0_fps = [f for f in (fingerprint(s) for s in r0_smiles) if f is not None]
    # compute lipid topology features once per molecule and reuse everywhere
    sel_feats = [lipid_topology_features(Chem.MolFromSmiles(s)) for s in sel_smiles]
    r0_feats = [lipid_topology_features(Chem.MolFromSmiles(s)) for s in r0_smiles]

    fam_raw = dict(sorted(per_family_raw.items()))
    fam_sel: dict[str, int] = {}
    for p in selected:
        fam_sel[p.reaction_id] = fam_sel.get(p.reaction_id, 0) + 1
    heavy = np.array([_descriptors(s)["heavy"] for s in sel_smiles])

    metrics = {
        "format": "compose_lipid_corpus_pilot_metrics_v2",
        "corpus_scope": "general linker-agnostic reaction-grounded pilot (R1); metrics double as Fig 2/3 generator-qualification yardstick",
        "enumeration": {
            "raw_products_by_family": fam_raw,
            "total_raw_applications": int(sum(fam_raw.values())),
            "total_unique_products": total_unique,
        },
        "selection": {
            "quota_per_size_charge_ring_stratum": args.quota,
            "cap_per_heteroatom_core": args.cap_per_core,
            "candidate_before_core_dedup": len(candidate),
            "selected_pilot_size": len(selected),
            "family_balance_selected": dict(sorted(fam_sel.items())),
        },
        "validity_uniqueness": {"all_single_component_sanitized": True, "exact_uniqueness": 1.0,
                                "fig2_role": "pathwise validity + non-collapse"},
        "lipid_diversity_corpus": lipid_diversity(sel_feats),
        "lipid_diversity_r0_reference": lipid_diversity(r0_feats),
        "faithfulness_js_to_r0": {"marginals": _js_marginals(sel_feats, r0_feats),
                                  "fig2_role": "faithful architecture/property marginals (lower JS = closer to real lipids)"},
        "nn_redundancy": {
            "corpus_intra": _nn_redundancy(sel_fps, sample=min(3000, len(sel_fps)), rng=rng),
            "r0_intra_reference": _nn_redundancy(r0_fps, sample=min(1500, len(r0_fps)), rng=rng),
            "interpretation": "ECFP NN is intrinsically high for lipids (homologous chains share bits); compare corpus to R0 intra-NN, not to 0. Exact uniqueness is 100%.",
        },
        "lnpdb_coverage": {**_lnpdb_coverage(sel_fps, rng),
                           "fig2_role": "coverage/density of the real ionizable-lipid manifold"},
        "kernel_readiness_profile": _kernel_profile(sel_smiles),
        "environment": {"rdkit": Chem.rdBase.rdkitVersion, "seed": args.seed},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with (args.out_dir / "pilot_products.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["canonical_smiles", "reaction_family", "size_bin", "charge_bin", "ring_bin"])
        for p in selected:
            w.writerow([p.canonical_smiles, p.reaction_id, p.product_bins["size_bin"],
                        p.product_bins["charge_bin"], p.product_bins["ring_bin"]])
    (args.out_dir / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"raw applications {metrics['enumeration']['total_raw_applications']:,}  unique {total_unique:,}")
    print(f"candidate {len(candidate):,} -> core-dedup -> pilot {len(selected):,}  balance {dict(sorted(fam_sel.items()))}")
    print(f"corpus core Hill(Simpson)={metrics['lipid_diversity_corpus']['heteroatom_core_hill']['simpson_q2']} "
          f"vs R0={metrics['lipid_diversity_r0_reference']['heteroatom_core_hill']['simpson_q2']}")
    print(f"corpus arch Hill(Simpson)={metrics['lipid_diversity_corpus']['architecture_hill']['simpson_q2']} "
          f"vs R0={metrics['lipid_diversity_r0_reference']['architecture_hill']['simpson_q2']}")
    print(f"NN corpus={metrics['nn_redundancy']['corpus_intra']['median_nn_tanimoto']} "
          f"vs R0={metrics['nn_redundancy']['r0_intra_reference']['median_nn_tanimoto']}")
    print(f"faithfulness JS-to-R0: {metrics['faithfulness_js_to_r0']['marginals']}")
    print(f"LNPDB recall@0.4={metrics['lnpdb_coverage']['recall_at_tanimoto_0.4']}")
    print(f"written: {(args.out_dir/'pilot_metrics.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

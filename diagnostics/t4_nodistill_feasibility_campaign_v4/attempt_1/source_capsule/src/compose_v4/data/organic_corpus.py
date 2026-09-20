"""Shared, versioned BROAD-ORGANIC corpus scope for B-edit (LOCKED 2026-07-26).

The single source of truth for "which molecules are in scope" across mining, MMP + scaffold-pair
construction, corruption, edit-prior training, validation, held-out rollout evaluation, and benchmark
ingestion. It replaces the CNOF-neutral ``load_cnof_corpus_split`` in every production B-edit path.

Scope = broad organic: an atom is in scope iff (a) its element is in the ACTIVE ``ORGANIC_VOCABULARY``
(never a restated list) and (b) its exact ``(element, valence, formal_charge)`` is a representable class
under the SAME derivation the model's teacher path uses (``AtomVocabulary.class_index`` with
``bond_order_sum = Σ BOND_CLASS_TO_H_CHANGE``). Charged states are RETAINED (not required neutral); the
representability check is what enforces "represented charges satisfy the current state representation".

Charge is PRESERVED, not optimized: this scope decides membership only. It never authorizes a
charge-changing edit -- that is the corruption/operator contract's responsibility (source->target pairs
needing an unsupported charge change are rejected+counted there, not here).

Every acceptance/rejection is counted with an explicit reason so the corpus census is fully auditable. The
scope is immutable and carries a deterministic ``scope_hash`` stored in manifests / training config /
checkpoint metadata / sampler metadata so a cross-scope load fails loudly.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    IDX_TO_ELEMENT,
    ORGANIC_VOCABULARY,
    AtomVocabulary,
    MolecularGraphError,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null

RDLogger.DisableLog("rdApp.*")
_CNOF_SYMBOLS = frozenset({"C", "N", "O", "F"})

# ---- rejection reasons (counted; never collapsed) -----------------------------------------------------
REJECT_UNPARSEABLE = "unparseable"
REJECT_EMPTY = "empty"
REJECT_TOO_BIG = "too_big"
REJECT_DISCONNECTED = "disconnected_multicomponent"
REJECT_UNSUPPORTED_ELEMENT = "unsupported_element"
REJECT_UNSUPPORTED_CLASS = "unsupported_class"  # (element, valence, charge) not representable
REJECT_CHARGED_NOT_ALLOWED = "charged_not_allowed"  # only under a neutral-only scope


@dataclass(frozen=True)
class CorpusScope:
    """Immutable corpus-scope contract. The element set + representable classes come from the ACTIVE
    registry vocabulary (referenced by name), never from a hardcoded list."""

    name: str
    allow_charges: bool
    max_atoms: int
    standardization: str
    disconnected_policy: str
    isotope_stereo_radical_policy: str
    vocabulary_name: str = "ORGANIC_VOCABULARY"

    @property
    def vocabulary(self) -> AtomVocabulary:
        if self.vocabulary_name != "ORGANIC_VOCABULARY":
            raise ValueError(f"unknown vocabulary {self.vocabulary_name!r}")
        return ORGANIC_VOCABULARY  # the active registry vocab -- single source of truth

    def element_indices(self) -> frozenset[int]:
        return frozenset(self.vocabulary.element_index)

    def element_symbols(self) -> tuple[str, ...]:
        return tuple(sorted({IDX_TO_ELEMENT[e] for e in self.element_indices()}))

    def classify(self, graph) -> tuple[bool, str | None]:
        """(accepted, rejection_reason). Reason is None iff accepted. Pure function of the graph + scope."""
        n = int(graph.n_real_atoms)
        if n == 0:
            return False, REJECT_EMPTY
        if n > self.max_atoms:
            return False, REJECT_TOO_BIG
        if not is_connected_or_null(graph):
            return False, REJECT_DISCONNECTED
        real = is_element(graph.atom_types)
        allowed = self.element_indices()
        elems = {int(t) for t in graph.atom_types[real]}
        if elems - allowed:
            return False, REJECT_UNSUPPORTED_ELEMENT
        if not self.allow_charges and bool(np.any(graph.formal_charges[real] != 0)):
            return False, REJECT_CHARGED_NOT_ALLOWED
        vocab = self.vocabulary
        for v in range(n):
            bond_sum = sum(int(BOND_CLASS_TO_H_CHANGE[int(o)]) for o in graph.bonds[v])
            if vocab.class_index(
                int(graph.atom_types[v]), bond_sum,
                int(graph.implicit_h_counts[v]), int(graph.formal_charges[v]),
            ) is None:
                return False, REJECT_UNSUPPORTED_CLASS
        return True, None

    def scope_hash(self) -> str:
        payload = {
            "name": self.name,
            "allow_charges": self.allow_charges,
            "max_atoms": self.max_atoms,
            "standardization": self.standardization,
            "disconnected_policy": self.disconnected_policy,
            "isotope_stereo_radical_policy": self.isotope_stereo_radical_policy,
            "vocabulary_name": self.vocabulary_name,
            "vocabulary_classes": list(self.vocabulary.classes),  # binds to the exact active vocab
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def descriptor(self) -> dict:
        """Startup-printable + manifest-storable scope record (includes the hash)."""
        return {
            "scope_name": self.name,
            "scope_hash": self.scope_hash(),
            "vocabulary_name": self.vocabulary_name,
            "allowed_elements": list(self.element_symbols()),
            "charge_policy": "retain_representable_charges" if self.allow_charges else "neutral_only",
            "max_atoms": self.max_atoms,
            "standardization": self.standardization,
            "disconnected_policy": self.disconnected_policy,
            "isotope_stereo_radical_policy": self.isotope_stereo_radical_policy,
        }


# The LOCKED production scope. allow_charges=True retains charged contexts; the class_index check keeps
# only charges the model can actually represent. Charge is preserved by the operator contract, not edited.
BROAD_ORGANIC_V1 = CorpusScope(
    name="broad_organic_v1",
    allow_charges=True,
    max_atoms=40,
    standardization="molecular_graph_canonical_smiles",
    disconnected_policy="reject_multicomponent",
    isotope_stereo_radical_policy="graph_drops_isotope_stereo; radical_or_bad_valence->unsupported_class",
)

# A neutral-only sibling kept ONLY as a later ablation (Option 2); NOT the production scope.
BROAD_ORGANIC_NEUTRAL_V1 = CorpusScope(
    name="broad_organic_neutral_v1",
    allow_charges=False,
    max_atoms=40,
    standardization="molecular_graph_canonical_smiles",
    disconnected_policy="reject_multicomponent",
    isotope_stereo_radical_policy="graph_drops_isotope_stereo; radical_or_bad_valence->unsupported_class",
)


def classify_smiles(text: str, scope: CorpusScope) -> tuple[bool, str | None]:
    """Parse then classify a SMILES string; parse failure is a counted rejection, never an exception. A
    multi-component SMILES (contains '.') is rejected as a salt/multicomponent before parsing so the count
    is not conflated with true parse failures."""
    if not text:
        return False, REJECT_EMPTY
    if "." in text:
        return False, REJECT_DISCONNECTED
    try:
        graph = smiles_to_molecular_graph(text)
    except (MolecularGraphError, ValueError):
        return False, REJECT_UNPARSEABLE
    if graph is None:
        return False, REJECT_UNPARSEABLE
    return scope.classify(graph)


# ---- single-pass corpus scan + rich census ------------------------------------------------------------
def _scan_one(task: tuple[str, CorpusScope]) -> dict:
    """One molecule -> a census record. RDKit categorizes rejections finely (unparseable vs salt vs
    unsupported-element, all visible even for out-of-vocab atoms RDKit still parses); the model's own graph
    gate then decides representability (size / charge-class) and the canonical key for accepted molecules."""
    text, scope = task
    rec: dict = {"canonical": None, "reason": None, "accepted": False,
                 "elements": (), "net_charge": 0, "contains_charged": False}
    if not text:
        rec["reason"] = REJECT_EMPTY
        return rec
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        rec["reason"] = REJECT_UNPARSEABLE
        return rec
    rec["elements"] = tuple(sorted({atom.GetSymbol() for atom in mol.GetAtoms()}))
    rec["net_charge"] = int(Chem.GetFormalCharge(mol))
    rec["contains_charged"] = any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
    if "." in text:
        rec["reason"] = REJECT_DISCONNECTED
        return rec
    if set(rec["elements"]) - set(scope.element_symbols()):
        rec["reason"] = REJECT_UNSUPPORTED_ELEMENT
        return rec
    try:
        graph = smiles_to_molecular_graph(text)
    except (MolecularGraphError, ValueError):
        rec["reason"] = REJECT_UNPARSEABLE
        return rec
    ok, reason = scope.classify(graph)
    if not ok:
        rec["reason"] = reason
        return rec
    rec["accepted"] = True
    rec["canonical"] = molecular_graph_to_smiles(graph)
    return rec


def _charge_category(net_charge: int, contains_charged: bool) -> str:
    if not contains_charged:
        return "neutral"
    if net_charge == 0:
        return "zwitterion_net_zero"
    return "nonzero_net_charge"


def scan_corpus(smiles, scope: CorpusScope, *, workers: int = 0) -> tuple[list[str], dict]:
    """Scan a SMILES iterable ONCE. Return (accepted canonical SMILES in first-seen order, deduped) and a
    rich census with every rejection reason + per-element / per-charge-category / element-combination
    tallies over accepted molecules, plus the canonical-duplicate count. Parallelizable via workers>1."""
    texts = list(smiles)
    tasks = ((text, scope) for text in texts)
    if workers and workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                records = list(pool.map(_scan_one, tasks, chunksize=256))
        except (OSError, PermissionError):
            records = [_scan_one(t) for t in ((text, scope) for text in texts)]
    else:
        records = [_scan_one(t) for t in tasks]

    accepted: dict[str, None] = {}
    reasons: Counter = Counter()
    element_molecule_count: Counter = Counter()
    non_cnof_combo_count: Counter = Counter()
    charge_category: Counter = Counter()
    net_charge_dist: Counter = Counter()
    contains_charged_atom = 0
    duplicates = 0
    for rec in records:
        if not rec["accepted"]:
            reasons[rec["reason"]] += 1
            continue
        canonical = rec["canonical"]
        if canonical in accepted:
            duplicates += 1
            continue
        accepted[canonical] = None
        for symbol in rec["elements"]:
            element_molecule_count[symbol] += 1
        non_cnof = tuple(sorted(set(rec["elements"]) - _CNOF_SYMBOLS))
        non_cnof_combo_count["+".join(non_cnof) if non_cnof else "(cnof_only)"] += 1
        charge_category[_charge_category(rec["net_charge"], rec["contains_charged"])] += 1
        net_charge_dist[str(rec["net_charge"])] += 1
        contains_charged_atom += int(rec["contains_charged"])

    total = len(texts)
    retained = len(accepted)
    census = {
        "scope": scope.descriptor(),
        "total": total,
        "retained": retained,
        "rejected": total - retained - duplicates,
        "canonical_duplicates": duplicates,
        "retained_fraction": round(retained / total, 4) if total else 0.0,
        "rejection_reasons": dict(reasons),
        "element_molecule_count": dict(element_molecule_count.most_common()),
        "non_cnof_element_combination_count": dict(non_cnof_combo_count.most_common(30)),
        "charge_category": dict(charge_category),
        "contains_charged_atom": contains_charged_atom,
        "net_charge_distribution": dict(sorted(net_charge_dist.items(), key=lambda kv: int(kv[0]))),
    }
    return list(accepted), census


@dataclass
class OrganicCorpusSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    scope: dict
    census: dict


def load_organic_corpus_split(
    path: Path,
    *,
    scope: CorpusScope = BROAD_ORGANIC_V1,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int = 20260714,
    scan_all: bool = True,
    workers: int = 0,
) -> OrganicCorpusSplit:
    """Load, canonicalize+dedup under the broad-organic scope, and deterministically split (same shuffle
    convention as the CNOF loader). Returns the split + the full census. Fails loudly if fewer eligible
    molecules than requested. Mines/trains the TRAIN partition only -> no val/test leakage."""
    requested = train_size + validation_size + test_size
    if requested <= 0 or min(train_size, validation_size, test_size) < 0:
        raise ValueError("split sizes must be non-negative with a positive total")

    with Path(path).open() as handle:
        texts = [line.strip().split()[0] for line in handle if line.strip()]
    if not scan_all:
        # bounded scan: enough to fill the request (census is then partial, flagged below)
        texts = texts[: max(requested * 3, requested + 1000)]
    accepted, census = scan_corpus(texts, scope, workers=workers)
    census["scan_all"] = bool(scan_all)

    if len(accepted) < requested:
        raise ValueError(
            f"broad-organic scope '{scope.name}' found {len(accepted)} eligible molecules; "
            f"requested {requested}. Corpus census: {json.dumps(census['rejection_reasons'])}"
        )

    rng = np.random.default_rng(seed)
    molecules = np.asarray(accepted, dtype=object)
    rng.shuffle(molecules)
    molecules = molecules[:requested]
    train = tuple(str(m) for m in molecules[:train_size])
    validation = tuple(str(m) for m in molecules[train_size:train_size + validation_size])
    test = tuple(str(m) for m in molecules[train_size + validation_size:requested])
    return OrganicCorpusSplit(train=train, validation=validation, test=test,
                              scope=scope.descriptor(), census=census)

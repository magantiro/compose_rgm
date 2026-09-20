"""BUILD_RING_SYSTEM as a STRUCTURED, PARAMETERIZED semantic action.

The controller does not choose from a vocabulary of ring types. It chooses
VALUES FOR PARAMETERS:

    BUILD_RING_SYSTEM(topology, size, stoichiometry, state)

and those parameters are learned as FACTORIZED distributions from reward. There
is no list of allowable heterocycles anywhere in this module, and no ring is
ever named. "Piperidine" is not a concept here -- it is whatever falls out of
(linked, 6, {N:1, C:5}, saturated) when the compiler realizes it.

WHY IT IS PARAMETERIZED RATHER THAN ENUMERATED
----------------------------------------------
An enumerated vocabulary bakes today's evidence into the interface. The frozen
T4 space did exactly that with three hard-coded ring actions, and on 5HT1B
seed 7 that became a measured support ceiling: the controller plateaus at
-11.3 kcal/mol from ~90 oracle calls onward while every InVirtuoGen winner on
the cell caps its aryl arms with a saturated N-heterocycle. Our best molecule
matches the winners on heavy atoms, similarity, aromatic count and topology,
and is short exactly the amine-bearing ring. A vocabulary cannot express what
was not foreseen; parameters can.

WHAT IS FIXED, AND ON WHAT EVIDENCE
-----------------------------------
Topology stays {linked, fused}. Across all 25 IVG winners RDKit reports
CalcNumSpiroAtoms = 0 and CalcNumBridgeheadAtoms = 0. That is measurement, not
taste, so it is a domain restriction rather than a learned parameter.

Everything else is a controller variable over an EXECUTOR-QUALIFIED domain,
discovered by probing the compiler rather than hand-picked. Qualification means
the executor builds the ring AND the requested electronic state is actually
realized on a ring of the requested size -- RDKit merely parsing the product is
not enough, since e.g. size-8 all-carbon parses and is not aromatic.

FOUR INVARIANTS
---------------
1. STABLE SEMANTIC TUPLES, never positional indices. A checkpoint written under
   one domain and read under another degrades (unknown keys dropped, absent
   keys fall back to prior) instead of silently remapping credit onto whatever
   now occupies an index -- the failure that once produced fictitious "elite
   programs" in this project.
2. FIXED PARENT MASS. BUILD_RING_SYSTEM holds a constant share of the prior no
   matter how many parameter tuples happen to be executable, so a richer state
   cannot sample rings more often and manufacture a macro effect. Six ring
   actions at 0.04 vs one at 0.05 was once read here as a -9.6 -> -11.4 win
   that was purely sampling rate.
3. FACTORIZED CREDIT. Reward updates each parameter's marginal, so evidence for
   "saturated" transfers across sizes, topologies and stoichiometries. A
   combinatorial tuple space is far too sparse to learn per-tuple.
4. REALIZATIONS STAY UNDERNEATH. Attachment site and heteroatom POSITION are
   realizations of a semantic request, ranked by frozen R_theta. A mass-matched
   ablation already refuted exposing per-site tokens to the controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# NARROW PARITY IS STRUCTURAL, NOT BEHAVIOURAL. READ BEFORE CITING IT.
#
# Restricted to the original three T4 semantics, this path reproduces the
# legacy candidate sets EXACTLY -- but only after the legacy ENDPOINT gate is
# applied to intermediates. Measured on 5HT1B seed 7:
#
#     spec                        legacy   hard-support   +endpoint gate
#     ring:linked/6/C6/aromatic        7              7                7
#     ring:fused/6/C6/aromatic         2              4                2
#     ring:fused/6/C6/saturated        3              4                3
#
# Under hard support alone the corrected narrow controller exposes MORE
# intermediate realizations than the legacy one did. That is an INTENDED
# CORRECTION -- the legacy path was applying endpoint feasibility to
# intermediate states, i.e. silently enforcing a pathwise constraint on an
# endpoint-constrained benchmark -- and it is NOT exact behavioural parity.
#
# Because of that, any 5HT1B gain must be attributed across THREE arms, never
# two:
#     A  legacy narrow      3 semantics, endpoint gate on intermediates
#     B  corrected narrow   3 semantics, hard-support-only intermediates
#     C  corrected broad    full parameterized semantics, hard-support-only
#
#     B - A  = removing the accidental pathwise endpoint restriction
#     C - B  = learning broader ring parameters
#
# Reporting C - A alone would conflate the two and credit the parameterization
# with a gain that may belong to the gate correction.
# ---------------------------------------------------------------------------

RING_FAMILY = "BUILD_RING_SYSTEM"

# Measured domain restriction, not a learned parameter. See module docstring.
TOPOLOGIES: tuple[str, ...] = ("linked", "fused")
STATES: tuple[str, ...] = ("aromatic", "saturated")

# Ring elements are taken from what the EXECUTOR DECLARES, not from a curated
# list. ORGANIC_RING_ELEMENTS is the declared ring vocabulary; an element is
# usable as a ring MEMBER only if it also has a declared valence class of at
# least 2, since a ring atom needs two ring bonds. That excludes F (declared
# valence 1) by the state space's own rule rather than by our preference, and
# it admits P, which a hand-picked {C,N,O,S} would have silently dropped.
def _declared_ring_elements() -> tuple[str, ...]:
    from compose_v4.chem.molecular_graph import (ATOM_VALENCE_CLASSES,
                                                 IDX_TO_ELEMENT,
                                                 ORGANIC_RING_ELEMENTS)
    max_val: dict[int, int] = {}
    for z, val in ATOM_VALENCE_CLASSES:
        max_val[z] = max(max_val.get(z, 0), int(val))
    out = [IDX_TO_ELEMENT[z] for z in ORGANIC_RING_ELEMENTS
           if max_val.get(z, 0) >= 2]
    return tuple(out)


RING_ELEMENTS: tuple[str, ...] = _declared_ring_elements()
ELEMENT_CODE = {"C": 2, "N": 3, "O": 4, "F": 5, "P": 6, "S": 7}


@dataclass(frozen=True)
class RingRequest:
    """One semantic request. `stoich` is a count vector, sum == size."""
    topology: str
    size: int
    stoich: tuple[tuple[str, int], ...]
    state: str

    @property
    def key(self) -> str:
        s = "".join(f"{e}{n}" for e, n in self.stoich)
        return f"ring:{self.topology}/{self.size}/{s}/{self.state}"

    @property
    def hetero(self) -> tuple[tuple[str, int], ...]:
        return tuple((e, n) for e, n in self.stoich if e != "C" and n)

    def axes(self) -> dict[str, str]:
        """Marginals credit is factorized over. `n_hetero` and the identity of
        each heteroatom are separate axes so that 'one N' generalizes across
        sizes, and 'N rather than O' generalizes across counts."""
        ax = {"topology": self.topology, "size": str(self.size),
              "state": self.state,
              # the exact vector the masked sampler draws, so credit reaches
              # the axis that was actually sampled...
              "stoich": "".join(f"{e}{n}" for e, n in self.stoich),
              # ...and its factorized marginals, so evidence still generalizes
              # across sizes and across counts.
              "n_hetero": str(sum(n for _, n in self.hetero))}
        for e, n in self.hetero:
            ax[f"elem:{e}"] = str(n)
        return ax

    def hetero_slots(self) -> list[str]:
        out: list[str] = []
        for e, n in self.hetero:
            out.extend([e] * n)
        return out


def parse_stoich_label(label: str, size: int) -> tuple[tuple[str, int], ...]:
    """Inverse of the key's stoichiometry field: "C5N1" -> (("C",5),("N",1)).

    Accepts the legacy set-labels too, mapping them to all-carbon, so a legacy
    macro string can be replayed through the parameterized path unchanged.
    """
    import re
    if label in ("carbon_rich", "C", ""):
        return normalize_stoich({}, size)
    counts: dict[str, int] = {}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(label)):
        if not elem:
            continue
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    counts.pop("C", None)          # carbon is the remainder by construction
    return normalize_stoich(counts, size)


def normalize_stoich(counts: dict[str, int], size: int) -> tuple[tuple[str, int], ...]:
    """Canonical count vector: carbon takes the remainder, elements sorted."""
    het = {e: int(n) for e, n in counts.items() if e != "C" and int(n) > 0}
    used = sum(het.values())
    if used > size:
        raise ValueError(f"stoichiometry {counts} exceeds size {size}")
    out = [("C", size - used)] + sorted(het.items())
    return tuple((e, n) for e, n in out if n > 0)


# ---------------------------------------------------------------------------
# EXECUTOR-QUALIFIED DOMAIN. Probed from the compiler, never hand-picked.
# ---------------------------------------------------------------------------

def _state_realized(smi: str, size: int, want_aromatic: bool) -> bool:
    """The requested electronic state must actually hold on a ring of the
    requested size. Parsing is not qualification: size-8 all-carbon parses
    cleanly and is not aromatic."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) != size:
            continue
        if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in ring) == want_aromatic:
            return True
    return False


def qualify_domain(states, predict_fn, sizes: Iterable[int] = range(3, 11),
                   max_anchors: int = 14) -> dict:
    """Discover which (state, size) the executor can actually deliver.

    Returns {"size": {state: [sizes]}, "probe": {...}}. Run once per frozen
    executor and cached in provenance -- this is a property of the compiler,
    not of the molecule being edited.
    """
    out: dict[str, dict[str, list[int]]] = {"size": {s: [] for s in STATES}}
    for size in sizes:
        for st in STATES:
            aro = (st == "aromatic")
            ok = False
            for state in states:
                n_real = int(getattr(state, "n_real_atoms", 0) or 0)
                for elem in RING_ELEMENTS:
                    het = None if elem == "C" else {size // 2: ELEMENT_CODE[elem]}
                    for a in range(min(n_real, max_anchors)):
                        try:
                            smi = predict_fn(state, a, size=size, aromatic=aro,
                                             hetero=het)
                        except Exception:
                            continue
                        if smi and _state_realized(smi, size, aro):
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    break
            if ok:
                out["size"][st].append(size)
    return out


# ---------------------------------------------------------------------------
# FACTORIZED POLICY over the parameters. No vocabulary; no named ring types.
# ---------------------------------------------------------------------------

class FactorizedRingPolicy:
    """Independent learned distributions per parameter, composed into requests.

    Sampling draws topology, state, size, a heteroatom COUNT and then that many
    element identities; `normalize_stoich` turns them into a count vector with
    carbon taking the remainder. Any stoichiometry the elements admit is
    therefore reachable, including ones nobody enumerated.
    """

    def __init__(self, rng, domain: dict, max_hetero: int = 3,
                 smoothing: float = 0.3, temperature: float = 1.0):
        self.rng = rng
        self.domain = domain
        self.max_hetero = int(max_hetero)
        self.smoothing = float(smoothing)
        self.temperature = float(temperature)
        self.logit: dict[str, dict[str, float]] = {}

    # -- distributions -----------------------------------------------------
    def _p(self, axis: str, values: Sequence[str]) -> list[float]:
        lg = self.logit.get(axis, {})
        sc = [lg.get(v, 0.0) / max(1e-6, self.temperature) for v in values]
        mx = max(sc) if sc else 0.0
        ex = [math.exp(s - mx) for s in sc]
        t = sum(ex) or 1.0
        return [e / t for e in ex]

    def _draw(self, axis: str, values: Sequence[str]) -> str:
        if not values:
            raise ValueError(f"no values for axis {axis}")
        p = self._p(axis, list(values))
        return list(values)[int(self.rng.choice(len(values), p=p))]

    def sample_request(self) -> RingRequest:
        topo = self._draw("topology", TOPOLOGIES)
        state = self._draw("state", STATES)
        sizes = [str(s) for s in (self.domain.get("size", {}).get(state) or [])]
        if not sizes:
            raise ValueError(f"no qualified sizes for state={state}")
        size = int(self._draw("size", sizes))
        nmax = min(self.max_hetero, size - 1)
        n_het = int(self._draw("n_hetero", [str(i) for i in range(nmax + 1)]))
        counts: dict[str, int] = {}
        for _ in range(n_het):
            e = self._draw("element", [e for e in RING_ELEMENTS if e != "C"])
            counts[e] = counts.get(e, 0) + 1
        return RingRequest(topo, size, normalize_stoich(counts, size), state)

    def _score_tilt(self, req: "RingRequest") -> float:
        """Docking-learned multiplicative tilt for one spec, in log space.

        This is W_dock only -- the R_theta prior is applied separately by
        `sample_prior_tilted`, so plausibility and purpose stay separable.
        """
        return sum(self.logit.get(axis, {}).get(val, 0.0)
                   for axis, val in req.axes().items())

    # -- factorized credit -------------------------------------------------
    def update(self, requests: Sequence[RingRequest], scores: Sequence[float],
               elite_frac: float = 0.25) -> None:
        """Elites are the LOWEST scores (docking: more negative is better)."""
        if not requests:
            return
        order = sorted(range(len(requests)), key=lambda i: scores[i])
        k = max(1, int(round(elite_frac * len(requests))))
        elite = [requests[i] for i in order[:k]]
        tally: dict[str, dict[str, float]] = {}
        for r in elite:
            for axis, val in r.axes().items():
                tally.setdefault(axis, {})[val] = tally.setdefault(axis, {}).get(val, 0.0) + 1.0 / len(elite)
        for axis, vals in tally.items():
            d = self.logit.setdefault(axis, {})
            for v, share in vals.items():
                d[v] = (1 - self.smoothing) * d.get(v, 0.0) + self.smoothing * share * 4.0

    def note_infeasible(self, request: "RingRequest", weight: float = 1.0) -> None:
        """Condition the distribution on the EXECUTABLE domain.

        Reward alone cannot teach the policy that a region is unreachable: an
        unrealizable request never produces a score at all, so it is invisible
        to `update`. Without this the sampler keeps paying for draws the
        compiler always rejects -- measured 15% survival under the med-chem
        gate on 5HT1B, i.e. ~6 of 7 draws wasted. Down-weighting the axes of a
        request that failed to realize is how the policy learns the shape of
        its own support.

        This carries NO reward signal, so it cannot substitute for `update`:
        it moves mass off the infeasible, not onto the good.
        """
        for axis, val in request.axes().items():
            d = self.logit.setdefault(axis, {})
            d[val] = d.get(val, 0.0) - self.smoothing * float(weight)

    # -- checkpointing by stable keys -------------------------------------
    def state_dict(self) -> dict:
        return {"domain": self.domain, "max_hetero": self.max_hetero,
                "logit": {a: dict(v) for a, v in self.logit.items()},
                "domain_sha": hashlib.sha256(
                    json.dumps(self.domain, sort_keys=True).encode()).hexdigest()[:16]}

    def load_state_dict(self, d: dict) -> dict:
        """Axis/value keys are self-describing, so a domain change degrades
        rather than corrupts: values no longer in the domain are dropped."""
        cur = json.dumps(self.domain, sort_keys=True).encode()
        same = hashlib.sha256(cur).hexdigest()[:16] == d.get("domain_sha")
        dropped = []
        for axis, vals in (d.get("logit") or {}).items():
            keep = {}
            for v, x in vals.items():
                if axis == "size":
                    live = {str(s) for ss in self.domain.get("size", {}).values() for s in ss}
                    if v not in live:
                        dropped.append(f"{axis}={v}")
                        continue
                keep[v] = float(x)
            self.logit.setdefault(axis, {}).update(keep)
        return {"domain_match": same, "dropped": dropped}


# ---------------------------------------------------------------------------
# COMPILER: validate a SAMPLED tuple at this state and enumerate realizations.
# ---------------------------------------------------------------------------

# --- UNSAT taxonomy. These must never be collapsed into each other. --------
EXECUTOR_UNSAT = "executor_unsat"                # the process genuinely cannot
CHEMISTRY_UNSAT = "chemistry_unsat"              # not a valid molecular state
COMPILER_UNIMPLEMENTED = "compiler_unimplemented"  # WE have not written it yet


def _hard_valid(smi: str) -> bool:
    """HARD support: would the PRODUCTION executor commit this state?

    Calls `is_valid_state` -- the predicate the docstring of which says it is
    "shared by rewrite validators and the runtime" -- rather than any
    reconstruction of it. An audit of 6 edge cases found a hand-written
    ATOM_VALENCE_CLASSES predicate agreed 6/6, but agreement today is not a
    guarantee against drift, and the executor is the authority on its own
    support.

    Audited consequence, recorded deliberately: hypervalent iodine (I with 3
    bonds) and triaryl phosphines ARE executor-legal -- the declared state
    space contains (I,3) and (P,3). They therefore remain REACHABLE. Whether
    the declared state space should contain them is a question about the state
    space, not something to repair with a downstream medicinal-chemistry
    filter.

    NOT used here: QED / SA / similarity / docking (T4 endpoint criteria --
    intermediates may leave that region and recover) and med-chem heuristics
    (large-ring, sulfur-rich or heteroatom-count bans). Those belong to
    endpoint acceptance and to learned reward.
    """
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import is_connected_or_null, is_valid_state
    try:
        mg = smiles_to_molecular_graph(smi)
        return bool(is_valid_state(mg) and is_connected_or_null(mg))
    except Exception:
        return False


def _annotate(smi, request, fref, gen):
    """Build a realization record. Endpoint properties are ATTACHED for
    downstream scoring/archive/return decisions -- they are never used to mask
    the semantic support."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED
    if not smi or not _state_realized(smi, request.size,
                                      request.state == "aromatic"):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None or not _hard_valid(smi):
        return None
    rec = {"smiles": smi,
           "sim": round(DataStructs.TanimotoSimilarity(fref, gen.GetFingerprint(m)), 4),
           "qed": round(QED.qed(m), 4)}
    try:
        import os
        import sys

        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        rec["sa"] = round(float(sascorer.calculateScore(m)), 3)
    except Exception:
        rec["sa"] = None
    return rec


def realize(request: RingRequest, state, seed_smiles: str,
            predict_fn=None, max_anchors: int = 24,
            max_realizations: int = 64, fused_fn=None,
            edges_fn=None) -> tuple[list[dict], str | None]:
    """Hard-executable realizations of one semantic request at this state.

    Returns (realizations, unsat_reason). A non-empty list means the request is
    in the support here. Note there is NO delta/qed/sa argument: endpoint
    feasibility is not a support criterion (see `_hard_valid`).
    """
    from itertools import combinations, permutations

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as _rfg

    gen = _rfg.GetMorganGenerator(radius=2, fpSize=2048)
    ref = Chem.MolFromSmiles(seed_smiles)
    if ref is None:
        return [], CHEMISTRY_UNSAT
    fref = gen.GetFingerprint(ref)
    slots = request.hetero_slots()
    n_real = int(getattr(state, "n_real_atoms", 0) or 0)

    if request.topology == "fused":
        if fused_fn is None or edges_fn is None:
            return [], COMPILER_UNIMPLEMENTED
        # The fused predictor now takes the same position->element map as the
        # pendant one, so exact stoichiometry is expressible on both paths.
        # Ring positions 0 and size-1 are the SHARED EDGE: those atoms are
        # inherited from the scaffold and cannot be transmuted by fusing, so
        # heteroatoms are placed only among the size-2 NEW atoms. That is a
        # property of fusion itself, not a compiler limitation.
        slots_f = request.hetero_slots()
        if len(slots_f) > request.size - 2:
            return [], EXECUTOR_UNSAT      # more heteroatoms than new atoms
        placements_f: list[dict | None]
        if not slots_f:
            placements_f = [None]
        else:
            placements_f, seen_f = [], set()
            for combo in combinations(range(1, request.size - 1), len(slots_f)):
                for perm in set(permutations(slots_f)):
                    mp = {p: ELEMENT_CODE[e] for p, e in zip(combo, perm)}
                    sg = tuple(sorted(mp.items()))
                    if sg in seen_f:
                        continue
                    seen_f.add(sg)
                    placements_f.append(mp)
        out: list[dict] = []
        for hetmap_f in placements_f:
            for edge in list(edges_fn(state) or [])[:max_anchors]:
                try:
                    smi = fused_fn(state, edge, size=request.size,
                                   aromatic=(request.state == "aromatic"),
                                   hetero=hetmap_f)
                except Exception:
                    continue
                rec = _annotate(smi, request, fref, gen)
                if rec:
                    rec["edge"] = tuple(edge)
                    rec["hetero"] = hetmap_f
                    out.append(rec)
                    if len(out) >= max_realizations:
                        return out, None
        return out, (None if out else EXECUTOR_UNSAT)

    if predict_fn is None:
        return [], COMPILER_UNIMPLEMENTED
    placements: list[dict | None]
    if not slots:
        placements = [None]
    else:
        placements, seen = [], set()
        for combo in combinations(range(1, request.size), len(slots)):
            for perm in set(permutations(slots)):
                mapping = {p: ELEMENT_CODE[e] for p, e in zip(combo, perm)}
                sig = tuple(sorted(mapping.items()))
                if sig in seen:
                    continue
                seen.add(sig)
                placements.append(mapping)
    out = []
    for hetmap in placements:
        for anchor in range(min(n_real, max_anchors)):
            try:
                smi = predict_fn(state, anchor, size=request.size,
                                 aromatic=(request.state == "aromatic"),
                                 hetero=hetmap)
            except Exception:
                continue
            rec = _annotate(smi, request, fref, gen)
            if rec:
                rec["anchor"] = anchor
                rec["hetero"] = hetmap
                out.append(rec)
                if len(out) >= max_realizations:
                    return out, None
    return out, (None if out else EXECUTOR_UNSAT)


class SupportMask:
    """Prefix-conditioned hard support for BUILD_RING_SYSTEM parameters.

    Sampling a whole tuple and asking the compiler afterwards wastes most
    draws: measured 15-38% survival on 5HT1B. Worse, the wasted draws teach the
    policy about its own support instead of about reward. So each parameter is
    offered only values for which AT LEAST ONE hard-executable completion
    exists at this state, and the policy then learns preferences strictly
    within support.

    Feasibility is HARD only -- executor legality and molecular-state validity.
    Endpoint criteria (QED, SA, similarity, docking) are deliberately absent:
    intermediates may leave the feasible region and recover.
    """

    def __init__(self, state, seed_smiles: str, domain: dict, predict_fn=None,
                 fused_fn=None, edges_fn=None, max_hetero: int = 3,
                 probe_realizations: int = 1, max_anchors: int = 10):
        self.state = state
        self.seed = seed_smiles
        self.domain = domain
        self.predict_fn = predict_fn
        self.fused_fn = fused_fn
        self.edges_fn = edges_fn
        self.max_hetero = int(max_hetero)
        self.probe = int(probe_realizations)
        self.max_anchors = int(max_anchors)
        self._cache: dict[tuple, tuple[bool, str | None]] = {}
        self.reasons: dict[str, str] = {}

    def _feasible(self, req: RingRequest) -> bool:
        k = (req.topology, req.size, req.stoich, req.state)
        if k not in self._cache:
            reals, why = realize(req, self.state, self.seed,
                                 predict_fn=self.predict_fn,
                                 fused_fn=self.fused_fn, edges_fn=self.edges_fn,
                                 max_anchors=self.max_anchors,
                                 max_realizations=self.probe)
            self._cache[k] = (bool(reals), why)
            if why:
                self.reasons[req.key] = why
        return self._cache[k][0]

    def _completions(self, topology=None, size=None, stoich=None, state=None):
        """Enumerate candidate completions of a prefix, cheapest axis last."""
        topos = [topology] if topology else list(TOPOLOGIES)
        states = [state] if state else list(STATES)
        for t in topos:
            for st in states:
                sizes = [size] if size else list(self.domain.get("size", {}).get(st) or [])
                for k in sizes:
                    stoichs = [stoich] if stoich is not None else self._stoich_candidates(k)
                    for sto in stoichs:
                        yield RingRequest(t, k, sto, st)

    def _stoich_candidates(self, size: int) -> list[tuple]:
        from itertools import combinations_with_replacement
        het_elems = [e for e in RING_ELEMENTS if e != "C"]
        out = [normalize_stoich({}, size)]
        for n in range(1, min(self.max_hetero, size - 1) + 1):
            for combo in combinations_with_replacement(het_elems, n):
                counts: dict[str, int] = {}
                for e in combo:
                    counts[e] = counts.get(e, 0) + 1
                out.append(normalize_stoich(counts, size))
        return out

    def feasible_values(self, axis: str, **prefix) -> list:
        """Values of `axis` retaining at least one hard-executable completion."""
        vals: list = []
        if axis == "topology":
            cand = list(TOPOLOGIES)
        elif axis == "state":
            cand = list(STATES)
        elif axis == "size":
            st = prefix.get("state")
            cand = list(self.domain.get("size", {}).get(st) or [])
        elif axis == "stoich":
            cand = self._stoich_candidates(prefix["size"])
        else:
            raise ValueError(axis)
        for v in cand:
            pf = dict(prefix); pf[{"stoich": "stoich"}.get(axis, axis)] = v
            if any(self._feasible(r) for r in self._completions(**pf)):
                vals.append(v)
        return vals


def sample_masked(policy: "FactorizedRingPolicy", mask: "SupportMask"):
    """Draw a request whose every parameter is conditioned on hard support.

    Order: topology -> state -> size -> stoichiometry. `state` precedes `size`
    because the qualified size set is state-dependent (aromatic is restricted by
    Kekule parity; saturated is not).
    """
    topos = mask.feasible_values("topology")
    if not topos:
        return None
    t = policy._draw("topology", topos)
    states = mask.feasible_values("state", topology=t)
    if not states:
        return None
    st = policy._draw("state", states)
    sizes = mask.feasible_values("size", topology=t, state=st)
    if not sizes:
        return None
    k = int(policy._draw("size", [str(x) for x in sizes]))
    stos = mask.feasible_values("stoich", topology=t, state=st, size=k)
    if not stos:
        return None
    labels = {"".join(f"{e}{n}" for e, n in sto): sto for sto in stos}
    chosen = policy._draw("stoich", list(labels))
    return RingRequest(t, k, labels[chosen], st)


# ---------------------------------------------------------------------------
# QUALIFICATION SET. NOT the controller's vocabulary -- a smoke-test domain
# used to check the compiler still realizes what it used to.
# ---------------------------------------------------------------------------

def qualification_set() -> list[RingRequest]:
    out = []
    for topo in TOPOLOGIES:
        for size in (5, 6):
            for het in ({}, {"N": 1}, {"O": 1}):
                for st in STATES:
                    if st == "aromatic" and size == 5 and not het:
                        continue
                    if st == "aromatic" and size == 6 and het.get("O"):
                        continue
                    out.append(RingRequest(topo, size, normalize_stoich(het, size), st))
    return sorted(out, key=lambda r: r.key)


# ---------------------------------------------------------------------------
# CANONICAL ENDPOINT AGGREGATION + R_theta x H SELECTION
# ---------------------------------------------------------------------------

def aggregate_canonical(realizations: Sequence[dict],
                        prior: Sequence[float] | None = None,
                        per_program: bool = False) -> list[dict]:
    """Collapse realizations onto DISTINCT canonical endpoint molecules.

        P(y | a_sem, x)  proportional to  [ sum_{xi : y_xi ~= y} R_theta(xi|x) ] H(y,z,b)

    Different (anchor, heteroatom-position) programs frequently produce the
    same molecule by symmetry: measured on 5HT1B seed 7,
    ring:linked/6/C5N1/saturated yields 45 realizations covering only 21
    distinct endpoints, with one molecule reachable 4 different ways. Selecting
    over raw realizations would hand that molecule 4x the mass for a purely
    representational reason.

    This is the same canonical-fiber aggregation COMPOSE already performs over
    primitive marks, applied at the macro level: mass is summed over the fiber,
    never counted per syntactic realization.
    """
    from rdkit import Chem
    groups: dict[str, dict] = {}
    for i, r in enumerate(realizations):
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        y = Chem.MolToSmiles(m)
        w = 1.0 if prior is None else float(prior[i])
        g = groups.setdefault(y, {"canonical": y, "mass": 0.0, "n_programs": 0,
                                  "members": [], "record": r})
        # WHETHER TO SUM depends on what `prior` MEANS, and getting this wrong
        # rebuilds the multiplicity bug one level up.
        #
        #   per_program=True : prior[i] is R_theta over PROGRAMS xi. Distinct
        #       programs reaching one endpoint genuinely contribute separate
        #       path mass, so the fiber sum is the correct pushforward.
        #   per_program=False (DEFAULT): prior[i] is R_theta(y_i | x), already a
        #       distribution over CANONICAL SUCCESSORS -- i.e. already
        #       aggregated. Every member of a fiber then carries the SAME value,
        #       and summing would multiply it by the fiber size, which is pure
        #       representational multiplicity. Take it once.
        #
        # COMPOSE's R_theta is defined over canonical successors (the
        # pushforward of the mark law), so False is the correct default here.
        if per_program:
            g["mass"] += w
        else:
            g["mass"] = max(g["mass"], w)
        g["n_programs"] += 1
        g["members"].append(i)
    return sorted(groups.values(), key=lambda g: -g["mass"])


def select_endpoint(realizations: Sequence[dict], prior=None, value_fn=None,
                    rng=None, temperature: float = 1.0,
                    per_program: bool = False):
    """Pick a canonical endpoint under R_theta x H, after fiber aggregation.

    `prior` is R_theta over realizations (goal-independent plausibility).
    `value_fn` is the contextual H(y, z, b) hook, applied ONCE per canonical
    endpoint rather than per realization -- applying it per realization would
    reintroduce the multiplicity it is aggregating away. With value_fn=None
    this is R_theta alone, the honest default while no trustworthy
    realization-level objective value exists.

    Returns (group, probabilities, groups).
    """
    from compose_v4.control.constrained_search import select_realization
    groups = aggregate_canonical(realizations, prior, per_program=per_program)
    if not groups:
        return None, [], []
    cand = [g["record"] for g in groups]
    idx, p = select_realization(cand, prior=[g["mass"] for g in groups],
                                value_fn=value_fn, rng=rng,
                                temperature=temperature)
    return (groups[idx] if idx is not None else None), p, groups


def sample_at_state(policy: "FactorizedRingPolicy", state, seed_smiles: str,
                    predict_fn=None, fused_fn=None, edges_fn=None,
                    cache: dict | None = None, state_key: str | None = None,
                    tries: int = 12, max_anchors: int = 6):
    """Draw a semantic tuple conditioned on the support of THIS state x_t.

    Sampling at program-construction time conditioned the draw on the SEED's
    support, and the tuple then had to execute several edits later at a
    different state. Measured cost of that mismatch: only ~50-70% of sampled
    ring requests executed at their eventual state, and end to end just ~7% of
    draws ever reached credit -- far too sparse to fit a ~47-class space.

    Rejection sampling against the executor gives the support-conditioned
    distribution without materialising a full SupportMask, which cost ~11 s to
    build and is what made the per-particle version unusable. Validity is
    memoised on (canonical state, tuple key), so a state revisited within an
    episode pays nothing and a tuple retried against the same state pays
    nothing.

    Returns (request, realizations, n_tries) or (None, [], tries).
    """
    cache = cache if cache is not None else {}
    for i in range(1, int(tries) + 1):
        try:
            req = policy.sample_request()
        except ValueError:
            continue
        ck = (state_key, req.key)
        hit = cache.get(ck)
        if hit is False:
            continue                      # known-infeasible here; redraw
        reals, _why = realize(req, state, seed_smiles, predict_fn=predict_fn,
                              fused_fn=fused_fn, edges_fn=edges_fn,
                              max_anchors=max_anchors, max_realizations=8)
        cache[ck] = bool(reals)
        if reals:
            return req, reals, i
    return None, [], int(tries)


def parse_request_key(key: str) -> "RingRequest | None":
    """`ring:linked/6/C5N1/saturated` -> RingRequest. Inverse of `.key`.

    Lets the credit path reconstruct requests from what the EXECUTOR reported
    it actually ran, rather than trusting a separately-tracked sampled list.
    """
    if not str(key).startswith("ring:"):
        return None
    try:
        topo, size, stoi, state = str(key)[5:].split("/")
        return RingRequest(topo, int(size), parse_stoich_label(stoi, int(size)),
                           state)
    except Exception:
        return None


def rtheta_semantic_prior(specs, fams, acts, probs, anchors=None) -> dict:
    """rho(s | x) -- GOAL-INDEPENDENT plausibility mass for each semantic spec.

        rho(s | x)  proportional to  sum over admissible first insertions of
                                     the frozen R_theta probability

    Why a prior at all: broadening the space from 3 to 164 classes under a FLAT
    initialization diluted the one compounding move below what ~10 docking
    labels per round can learn. Measured on 5HT1B seed 7 --
    ring:linked/6/C6/aromatic was drawn 1.0% of the time under uniform broad
    semantics versus 33.3% under the narrow menu, while the winning endpoint
    needed that move to fire 2-3 times in one program.

    Why THIS prior: it is not hand-authored and carries no comparator
    information. It is the frozen reference law's own opinion about which
    transformation is plausible to BEGIN at this molecule. Ordinary chemistry
    starts with appreciable mass because R_theta says so; exotic classes stay
    reachable and can be promoted by reward. That is the intended ordering --
    executor says possible, R_theta says plausible, docking says purposeful --
    which a flat prior short-circuited by jumping from possible to purposeful.

    The first committed insertion is used as the tractable proxy for the whole
    program's mass: scoring every multi-step realization of all 164 classes at
    every state is not affordable, and the first step is the point at which
    R_theta actually expresses an opinion about starting the transformation.
    """
    from compose_v4.control.macro_engine import match_growth_descriptors
    # Memoise on the ELEMENT SET. 800 specs collapse to a couple of dozen
    # distinct element sets, and the mass depends only on which elements may
    # start the transformation -- so this is one law scan per set, not per spec.
    by_set: dict[frozenset, float] = {}
    out: dict[str, float] = {}
    for s in specs:
        allowed = frozenset(ELEMENT_CODE[e] for e, n in s.stoich if n > 0)
        if allowed not in by_set:
            try:
                idx = match_growth_descriptors(fams, acts, probs, None,
                                               set(allowed), anchors=anchors)
            except Exception:
                idx = []
            by_set[allowed] = float(sum(float(probs[j]) for j in idx))
        out[s.key] = by_set[allowed]
    tot = sum(out.values())
    if tot > 0:
        out = {k: v / tot for k, v in out.items()}
    return out


def sample_prior_tilted(policy: "FactorizedRingPolicy", prior: dict,
                        state, seed_smiles: str, predict_fn=None, fused_fn=None,
                        edges_fn=None, cache: dict | None = None,
                        state_key: str | None = None, tries: int = 12,
                        max_anchors: int = 6, specs=None):
    """Draw s with  pi(s | x, z)  proportional to  rho(s | x) * W_dock(s, z).

    `prior` is rho from R_theta; the policy's factorized logits supply the
    docking-learned multiplicative tilt W. Docking therefore learns only how to
    REWEIGHT plausibility, instead of rediscovering it from ten labels a round.
    Validity is still confirmed against the CURRENT state before returning.
    """
    import math
    import random as _rnd
    cache = cache if cache is not None else {}
    specs = list(specs if specs is not None else prior.keys())
    if not specs:
        return None, [], 0
    keys = [s if isinstance(s, str) else s.key for s in specs]
    reqs = {k: parse_request_key(k) for k in keys}
    w = []
    for k in keys:
        r = reqs.get(k)
        tilt = 0.0 if r is None else policy._score_tilt(r)
        w.append(max(prior.get(k, 0.0), 1e-12) * math.exp(tilt))
    tot = sum(w) or 1.0
    w = [x / tot for x in w]
    for i in range(1, int(tries) + 1):
        j = policy.rng.choice(len(keys), p=w)
        k = keys[int(j)]
        req = reqs.get(k)
        if req is None:
            continue
        ck = (state_key, k)
        if cache.get(ck) is False:
            continue
        reals, _why = realize(req, state, seed_smiles, predict_fn=predict_fn,
                              fused_fn=fused_fn, edges_fn=edges_fn,
                              max_anchors=max_anchors, max_realizations=8)
        cache[ck] = bool(reals)
        if reals:
            return req, reals, i
    return None, [], int(tries)


def enumerate_specs(domain: dict, max_hetero: int = 3) -> list["RingRequest"]:
    """The candidate semantic space implied by the executor-qualified domain."""
    from itertools import combinations_with_replacement
    het_elems = [e for e in RING_ELEMENTS if e != "C"]
    out: list[RingRequest] = []
    for topo in TOPOLOGIES:
        for state in STATES:
            for size in (domain.get("size", {}).get(state) or []):
                stoichs = [normalize_stoich({}, size)]
                for n in range(1, min(max_hetero, size - 1) + 1):
                    for combo in combinations_with_replacement(het_elems, n):
                        c: dict[str, int] = {}
                        for e in combo:
                            c[e] = c.get(e, 0) + 1
                        stoichs.append(normalize_stoich(c, size))
                for sto in stoichs:
                    out.append(RingRequest(topo, size, sto, state))
    return out

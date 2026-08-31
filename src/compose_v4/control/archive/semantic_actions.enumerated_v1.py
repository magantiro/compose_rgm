"""Stable, parameterized semantic action schema for the T4 controller.

WHY THIS EXISTS
---------------
The frozen T4 ACTION_SPACE hard-codes three ring actions:

    ring:linked/6/C/aromatic   ring:fused/6/C/aromatic   ring:fused/6/C/saturated

all size 6, all all-carbon, and saturated available ONLY when fused. Measured
on 2026-08-26, that is a support ceiling, not a search failure: on 5HT1B seed 7
the controller plateaus at -11.3 kcal/mol from ~90 oracle calls onward and never
moves, while every InVirtuoGen winner on that cell caps its aryl arms with a
SATURATED N-HETEROCYCLE (piperidine, pyrrolidine, azetidine, homopiperazine).
Our best molecule matches the winners on heavy atoms (30 vs 31), similarity
(0.41 vs 0.42), aromatic ring count (3) and linked-not-fused topology -- and is
short exactly one ring, the amine-bearing one. `linked + saturated + N` is not
in the support, so no oracle budget can reach it.

WHAT IS **NOT** A GAP
---------------------
Topology stays pinned to {linked, fused}. Across all 25 IVG winners RDKit gives
CalcNumSpiroAtoms = 0 and CalcNumBridgeheadAtoms = 0. That restriction is
evidence-based and is preserved deliberately -- this module does not widen it.

THE FOUR INVARIANTS THIS MODULE ENFORCES
----------------------------------------
1. SEMANTIC IDENTITY, NOT POSITION. An action is identified by its semantic
   tuple, never by an index into a list. This repo has already produced
   fictitious "elite programs" by decoding a stale checkpoint's integer indices
   against a changed ACTION_SPACE. Keys make that class of bug impossible:
   an unknown key is ignored, a missing key falls back to prior.

2. DYNAMIC EXECUTABLE ENUMERATION. The schema is fixed and total; which of its
   specifications are OFFERED is recomputed per molecular state from what the
   executor can actually realize and what passes the qualification gate.

3. FIXED PARENT MASS. BUILD_RING_SYSTEM holds a constant share of the prior no
   matter how many specifications happen to be executable. A state offering 12
   ring specs must not thereby sample rings 4x more often than a state offering
   3. This project already mistook exactly that for a result: six ring actions
   at 0.04 each vs one at 0.05 read as a -9.6 -> -11.4 "win" that was pure
   sampling rate.

4. REALIZATIONS STAY UNDERNEATH. Attachment site and heteroatom POSITION are
   realizations of a semantic choice, not semantic choices themselves. The
   controller picks "linked 6-ring, exactly one N, saturated"; frozen R_theta
   ranks which anchor and which ring position carries the N. A mass-matched
   ablation already refuted exposing per-site tokens to the controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# THE SCHEMA. Fixed, total, and versioned. Widening it is a deliberate act that
# changes SCHEMA_SHA and therefore invalidates checkpoints keyed to the old one.
# ---------------------------------------------------------------------------

RING_SCHEMA: dict[str, tuple] = {
    # spiro/bridged deliberately absent: 0/25 IVG winners realize them.
    "topology": ("linked", "fused"),
    # 78 six-rings and 18 five-rings across the winners; 3/7/8/9 are marginal.
    "size": (5, 6),
    # STOICHIOMETRY, not an allowed-element set. The old `composition` was a set
    # like {C,N,O}, which can only say "N is permitted somewhere" -- it cannot
    # request "6-ring, exactly one N", which is what piperidine IS.
    "composition": ((), (("N", 1),), (("O", 1),)),
    "state": ("aromatic", "saturated"),
}

SCHEMA_SHA = hashlib.sha256(
    json.dumps({k: [str(x) for x in v] for k, v in RING_SCHEMA.items()},
               sort_keys=True).encode()).hexdigest()[:16]


def composition_label(comp: tuple) -> str:
    return "C" if not comp else "C" + "".join(f"{e}{n}" for e, n in comp)


@dataclass(frozen=True)
class RingSpec:
    """One semantic specification. Hashable; its key is its identity."""
    topology: str
    size: int
    composition: tuple
    state: str

    @property
    def key(self) -> str:
        return (f"ring:{self.topology}/{self.size}/"
                f"{composition_label(self.composition)}/{self.state}")

    @property
    def hetero_count(self) -> int:
        return sum(n for _, n in self.composition)

    def axes(self) -> dict[str, str]:
        """Parameter axes, for factorized credit / partial pooling."""
        return {"topology": self.topology, "size": str(self.size),
                "composition": composition_label(self.composition),
                "state": self.state}


def all_ring_specs() -> list[RingSpec]:
    """Every schema point, canonically ordered. Chemically impossible
    combinations are dropped HERE rather than failing later: a 5-membered
    all-carbon ring is not aromatic, and a 6-aromatic ring with one O would be
    a pyrylium cation, which the executor correctly refuses to build neutral."""
    out: list[RingSpec] = []
    for topo in RING_SCHEMA["topology"]:
        for size in RING_SCHEMA["size"]:
            for comp in RING_SCHEMA["composition"]:
                for st in RING_SCHEMA["state"]:
                    if st == "aromatic" and size == 5 and not comp:
                        continue          # cyclopentadiene is not aromatic
                    if st == "aromatic" and size == 6 and comp == (("O", 1),):
                        continue          # pyrylium needs a formal charge
                    out.append(RingSpec(topo, size, comp, st))
    return sorted(out, key=lambda s: s.key)


RING_FAMILY = "BUILD_RING_SYSTEM"


# ---------------------------------------------------------------------------
# DYNAMIC ENUMERATION. The schema is total; this decides what is OFFERED here.
# ---------------------------------------------------------------------------

ELEMENT_CODE = {"C": 2, "N": 3, "O": 4, "S": 7}


def _hetero_positions(size: int, comp: tuple) -> list[dict]:
    """Candidate heteroatom PLACEMENTS -- these are realizations, not semantics.

    The controller never sees these. It asks for "exactly one N"; which ring
    position carries it is chosen underneath and ranked by frozen R_theta.
    Position 0 is the attachment atom and is excluded: substituting there
    changes the attachment chemistry rather than the ring's composition.
    """
    if not comp:
        return [None]
    out = []
    for elem, n in comp:
        if n != 1:                      # schema currently only carries n == 1
            continue
        for pos in range(1, size):
            out.append({pos: ELEMENT_CODE[elem]})
    return out or [None]


def executable_ring_specs(state, seed_smiles: str, delta: float,
                          predict_fn, gate_fn=None, qed_min: float = 0.6,
                          sa_max: float = 4.0,
                          specs: Sequence[RingSpec] | None = None,
                          max_anchors: int = 24) -> dict[str, dict]:
    """Which schema points this molecular state can actually realize.

    A spec is offered only if at least one (anchor, heteroatom placement)
    realization produces a molecule the executor builds and the qualification
    gate accepts. Returns {key: {"spec":…, "realizations":[…], "n":…}}.

    Cost note: this is descriptor-level prediction, not enumeration of the whole
    closure fiber, and it runs once per state -- the production rule is never to
    enumerate and apply a whole fiber.
    """
    from rdkit import Chem
    from rdkit.Chem import QED, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as _rfg

    gen = _rfg.GetMorganGenerator(radius=2, fpSize=2048)
    ref = Chem.MolFromSmiles(seed_smiles)
    if ref is None:
        return {}
    fref = gen.GetFingerprint(ref)
    n_real = int(getattr(state, "n_real_atoms", 0) or 0)
    offered: dict[str, dict] = {}

    for spec in (specs if specs is not None else all_ring_specs()):
        reals = []
        for het in _hetero_positions(spec.size, spec.composition):
            for anchor in range(min(n_real, max_anchors)):
                try:
                    smi = predict_fn(state, anchor, size=spec.size,
                                     aromatic=(spec.state == "aromatic"),
                                     hetero=het)
                except Exception:
                    continue
                if not smi:
                    continue
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fref, gen.GetFingerprint(m))
                if sim < delta:
                    continue
                q = QED.qed(m)
                if q < qed_min:
                    continue
                if gate_fn is not None and not gate_fn(smi):
                    continue
                reals.append({"anchor": anchor, "hetero": het,
                              "smiles": smi, "sim": round(sim, 4),
                              "qed": round(q, 4)})
        if reals:
            offered[spec.key] = {"spec": spec, "realizations": reals,
                                 "n": len(reals)}
    return offered


# ---------------------------------------------------------------------------
# KEY-ADDRESSED CEM. No positional indices anywhere.
# ---------------------------------------------------------------------------

class KeyedCEM:
    """CEM over semantic keys, with a FIXED parent mass per family.

    Two properties this exists to guarantee:

    * Renormalization is over the CURRENTLY EXECUTABLE set, so a spec that
      cannot be realized in this state simply receives no probability rather
      than being sampled and wasted.

    * BUILD_RING_SYSTEM keeps `ring_mass` of the prior REGARDLESS of how many
      ring specs are executable. Without this, a state offering 12 ring specs
      would sample rings far more often than one offering 3, and the resulting
      docking difference would be a sampling-rate artifact -- which this project
      has already once mistaken for a macro result.

    Credit is FACTORIZED as well as per-key: an elite `linked/6/CN1/saturated`
    raises that key AND the marginal credit of `saturated`, `CN1`, `6`,
    `linked`. Learning therefore transfers across a combinatorial schema that
    individual keys would be far too sparse to fit.
    """

    def __init__(self, rng, ring_mass: float = 0.24, elite_frac: float = 0.25,
                 pool: float = 0.5, smoothing: float = 0.3):
        self.rng = rng
        self.ring_mass = float(ring_mass)
        self.elite_frac = float(elite_frac)
        self.pool = float(pool)
        self.smoothing = float(smoothing)
        self.key_logit: dict[str, float] = {}
        self.axis_logit: dict[str, dict[str, float]] = {}

    # -- scoring -----------------------------------------------------------
    def _score(self, key: str, spec: RingSpec | None) -> float:
        s = self.key_logit.get(key, 0.0)
        if spec is not None:
            for axis, val in spec.axes().items():
                s += self.pool * self.axis_logit.get(axis, {}).get(val, 0.0)
        return s

    def _softmax(self, keys: Sequence[str], specs: dict[str, RingSpec | None],
                 mass: float) -> dict[str, float]:
        if not keys or mass <= 0:
            return {}
        sc = [self._score(k, specs.get(k)) for k in keys]
        mx = max(sc)
        ex = [math.exp(v - mx) for v in sc]
        tot = sum(ex) or 1.0
        return {k: mass * e / tot for k, e in zip(keys, ex)}

    def probs(self, ring_keys: Sequence[str], other_keys: Sequence[str],
              specs: dict[str, RingSpec]) -> dict[str, float]:
        """Distribution over the executable set, family masses preserved."""
        rk, ok = list(ring_keys), list(other_keys)
        # If no ring spec is executable here, the ring family's mass cannot be
        # spent: it goes to the other actions rather than vanishing.
        rmass = self.ring_mass if rk else 0.0
        omass = 1.0 - rmass
        out = self._softmax(rk, specs, rmass)
        out.update(self._softmax(ok, {}, omass))
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()}

    def sample(self, ring_keys, other_keys, specs, n: int) -> list[str]:
        p = self.probs(ring_keys, other_keys, specs)
        if not p:
            return []
        keys = list(p)
        w = [p[k] for k in keys]
        idx = self.rng.choice(len(keys), size=n, p=w)
        return [keys[int(i)] for i in idx]

    # -- learning ----------------------------------------------------------
    def update_by_rank(self, keys: Sequence[str], scores: Sequence[float],
                       specs: dict[str, RingSpec]) -> None:
        """Elites are the LOWEST scores (docking: more negative is better)."""
        if not keys:
            return
        order = sorted(range(len(keys)), key=lambda i: scores[i])
        k = max(1, int(round(self.elite_frac * len(keys))))
        elite = [keys[i] for i in order[:k]]
        bump = {}
        for key in elite:
            bump[key] = bump.get(key, 0.0) + 1.0 / len(elite)
        for key, b in bump.items():
            self.key_logit[key] = ((1 - self.smoothing) * self.key_logit.get(key, 0.0)
                                   + self.smoothing * b * 4.0)
            sp = specs.get(key)
            if sp is None:
                continue
            for axis, val in sp.axes().items():
                d = self.axis_logit.setdefault(axis, {})
                d[val] = (1 - self.smoothing) * d.get(val, 0.0) + self.smoothing * b * 4.0

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict:
        return {"schema_sha": SCHEMA_SHA, "ring_mass": self.ring_mass,
                "key_logit": dict(self.key_logit),
                "axis_logit": {a: dict(v) for a, v in self.axis_logit.items()}}

    def load_state_dict(self, d: dict) -> dict:
        """Keys are self-describing, so a schema change degrades rather than
        corrupts: unknown keys are dropped, absent keys fall back to prior.
        Compare that with positional indices, which silently REMAP."""
        known = {s.key for s in all_ring_specs()}
        kl = d.get("key_logit") or {}
        kept = {k: float(v) for k, v in kl.items()
                if (not k.startswith("ring:")) or k in known}
        dropped = sorted(set(kl) - set(kept))
        self.key_logit.update(kept)
        for a, v in (d.get("axis_logit") or {}).items():
            self.axis_logit.setdefault(a, {}).update({kk: float(vv) for kk, vv in v.items()})
        return {"loaded": len(kept), "dropped": dropped,
                "schema_match": d.get("schema_sha") == SCHEMA_SHA}

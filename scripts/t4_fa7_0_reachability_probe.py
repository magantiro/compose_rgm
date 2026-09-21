"""Can ANY production proposal configuration reach an eligible endpoint for fa7_0 at delta=0.6?

WHAT THIS ANSWERS
-----------------
`fa7_0` is the one unsolved cell in the T4 panel.  Both the live distilled
campaign and the region-repair rescue hold only `round_000_lock.json` for it:
pool 0, terminated `candidate_exhaustion` at round one.  The support-stage audit
established that the binding constraint is the INTERSECTION of the three
thresholds, not any single gate.  What was never run is the obvious thing: drive
the production proposal path itself, hard, across every knob the production
configuration already exposes, and count eligible endpoints.

This driver does exactly that and nothing else.  ZERO ORACLE CALLS -- `Fiber.check`
is similarity, QED, SA and the structural gate, all free.  Nothing docks and no
receptor is loaded.

HOW THE FUNNEL IS MEASURED, AND WHY IT IS WRAPPED RATHER THAN TRANSCRIBED
------------------------------------------------------------------------
`expand` returns ONLY endpoints that already passed `Fiber.check`, so an empty
result cannot say whether the path built nothing or built thousands of things
that were all refused.  To see the funnel interior without changing the decision,
this probe WRAPS the bound `check` method of the real `Fiber` instance: every
call delegates to the ORIGINAL method, whose return value is what the campaign
acts on, and a decomposition is recomputed beside it from the SAME symbols the
production module imports.

The decomposition is then required to PREDICT the original verdict on every
single call.  `gate_disagreements` must be 0; if it is not, the decomposition is
wrong and the per-stage attribution in this report is void.  A transcribed gate
could not fail this way, which is the entire reason for the arrangement.

STAGES, in the order the production path passes through them
------------------------------------------------------------
    draws ......... programs the synthesizer was asked for
    checked ....... endpoints the path actually constructed and offered to the gate
    parsed ........ survived RDKit parse, no disconnection, heavy <= 40
    sim / qed / sa  per-threshold pass counts over `parsed`
    sim AND qed ... the pairwise intersection the support-stage audit named as
                    the binding constraint -- reported because per-gate pass
                    rates alone conceal it
    eligible ...... passed the UNMODIFIED production `Fiber.check`

`eligible_per_draw` is the headline rate; a binary reachable/not is not enough to
size a rescue.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED

# The production symbols, taken from the production module so that a drift in
# either moves this probe with it.
from compose_v4.experiments.t4_fiber_campaign import (
    QED_MIN,
    REPRESENTABLE_HEAVY_ATOMS,
    SA_MAX,
    Fiber,
    expand,
    sascorer,
    structurally_valid,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.control.region_law_contract import (
    CONTRACT_FIELD,
    CONTRACT_LANE,
    FREE_GATE_MARGIN_V1,
    resolve_region_law,
)

SCHEMA_VERSION = "t4_fa7_0_reachability_probe_v1"

#: The T4 proposal executor requires 48 slots. This is the SLOT CAPACITY of the
#: proposal source array and is a different quantity from
#: `REPRESENTABLE_HEAVY_ATOMS`, the 40-heavy-atom ceiling the fiber applies to
#: ENDPOINTS. Both hold at once.
PROPOSAL_SLOTS = 48


class _FunnelFiber:
    """Delegates every decision to the real `Fiber`, and records the funnel.

    The wrapped object is the production instance. `check` returns whatever the
    ORIGINAL method returns -- the campaign's behaviour is unchanged -- and the
    decomposition is recorded beside it and cross-checked against that verdict.
    """

    def __init__(self, fiber: Fiber) -> None:
        self._fiber = fiber
        self.rows: dict[str, dict] = {}
        self.calls = 0
        self.disagreements: list[dict] = []

    # `expand` reads these off the fiber it is handed.
    @property
    def delta(self) -> float:
        return self._fiber.delta

    @property
    def generator(self):
        return self._fiber.generator

    @property
    def support(self) -> str:
        return self._fiber.support

    def check(self, smiles: str) -> dict | None:
        verdict = self._fiber.check(smiles)  # the REAL decision, untouched
        self.calls += 1
        row = self._decompose(smiles)
        row["eligible"] = verdict is not None
        predicted = (
            row["parsed"]
            and row["pass_similarity"]
            and row["pass_qed"]
            and row["pass_sa"]
            and row["pass_structural"]
        )
        if predicted != row["eligible"]:
            self.disagreements.append({"smiles": smiles, **row})
        key = row["canonical"] or smiles
        if key not in self.rows:
            self.rows[key] = row
        return verdict

    def _decompose(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or "." in smiles:
            return {"canonical": None, "parsed": False}
        heavy = mol.GetNumHeavyAtoms()
        if heavy > REPRESENTABLE_HEAVY_ATOMS:
            return {"canonical": None, "parsed": False, "heavy": heavy, "over_ceiling": True}
        similarity = DataStructs.TanimotoSimilarity(
            self._fiber.seed, self._fiber.generator.GetFingerprint(mol)
        )
        quality = QED.qed(mol)
        access = sascorer.calculateScore(mol)
        return {
            "canonical": Chem.MolToSmiles(mol),
            "parsed": True,
            "heavy": heavy,
            "similarity": similarity,
            "qed": quality,
            "sa": access,
            "pass_similarity": similarity >= self._fiber.delta,
            "pass_qed": quality >= QED_MIN,
            "pass_sa": access <= SA_MAX,
            "pass_structural": bool(structurally_valid(smiles)),
        }

    def funnel(self, source_heavy: int) -> dict:
        rows = list(self.rows.values())
        parsed = [r for r in rows if r.get("parsed")]

        def _count(pred) -> int:
            return sum(1 for r in parsed if pred(r))

        sim_pass = [r for r in parsed if r["pass_similarity"]]
        qed_pass = [r for r in parsed if r["pass_qed"]]
        both = [r for r in parsed if r["pass_similarity"] and r["pass_qed"]]
        return {
            "gate_calls": self.calls,
            "distinct_endpoints_checked": len(rows),
            "parsed": len(parsed),
            "pass_similarity": len(sim_pass),
            "pass_qed": len(qed_pass),
            "pass_sa": _count(lambda r: r["pass_sa"]),
            "pass_structural": _count(lambda r: r["pass_structural"]),
            # THE BINDING CONSTRAINT. Per-gate rates alone conceal it.
            "pass_similarity_and_qed": len(both),
            "pass_similarity_and_qed_and_sa": _count(
                lambda r: r["pass_similarity"] and r["pass_qed"] and r["pass_sa"]
            ),
            "eligible": _count(lambda r: r["eligible"]),
            # How far the intersection missed, on each axis, measured over the
            # other axis's passing set.
            "best_qed_among_similarity_passing": max(
                (r["qed"] for r in sim_pass), default=None
            ),
            "best_similarity_among_qed_passing": max(
                (r["similarity"] for r in qed_pass), default=None
            ),
            "best_sa_among_similarity_and_qed_passing": min(
                (r["sa"] for r in both), default=None
            ),
            "max_heavy_excision": max(
                (source_heavy - r["heavy"] for r in parsed), default=None
            ),
            "gate_disagreements": len(self.disagreements),
        }

    def near_misses(self, limit: int = 12) -> list[dict]:
        """Similarity-passing endpoints ranked by how little QED they lack."""
        rows = [r for r in self.rows.values() if r.get("parsed") and r["pass_similarity"]]
        rows.sort(key=lambda r: -r["qed"])
        return [
            {
                "smiles": r["canonical"],
                "similarity": round(r["similarity"], 6),
                "qed": round(r["qed"], 6),
                "sa": round(r["sa"], 4),
                "heavy": r["heavy"],
                "qed_shortfall": round(QED_MIN - r["qed"], 6),
                "pass_sa": r["pass_sa"],
                "pass_structural": r["pass_structural"],
            }
            for r in rows[:limit]
        ]


def _with_region_law(payload: dict, request) -> dict:
    mutated = json.loads(json.dumps(payload))
    mutated["proposal"][CONTRACT_LANE][CONTRACT_FIELD] = request
    return mutated


def _without_region_law(payload: dict) -> dict:
    """ABSENT is the only byte-identical OFF; there is deliberately no uniform law."""
    mutated = json.loads(json.dumps(payload))
    mutated["proposal"][CONTRACT_LANE].pop(CONTRACT_FIELD, None)
    return mutated


def run_arm(
    contract: Path,
    *,
    cell_name: str,
    lane: str,
    law_on: bool,
    horizon: int,
    draws: int,
    seed_offset: int,
    multi_region: bool,
) -> dict:
    payload = unseal(contract)
    cell = next(c for c in payload["cells"] if c["cell"] == cell_name)
    shaped = _with_region_law(payload, FREE_GATE_MARGIN_V1) if law_on else _without_region_law(payload)
    law = resolve_region_law(shaped, delta=shaped["delta"], reference_smiles=cell["smiles"])
    if law_on and law is None:
        raise ValueError("law requested but the contract resolved to none")
    if not law_on and law is not None:
        raise ValueError("an absent field must resolve to no law")
    # A law is consumable only on the shallow lane; `expand` fails closed otherwise.
    if lane != CONTRACT_LANE:
        law = None

    fiber = Fiber(cell["smiles"], payload["delta"], support=payload["support"])
    probe = _FunnelFiber(fiber)
    source_heavy = Chem.MolFromSmiles(cell["smiles"]).GetNumHeavyAtoms()
    seed = int(cell["controller_seed"]) + seed_offset
    started = time.time()
    records = expand(
        cell["smiles"],
        0.0,
        probe,
        np.random.default_rng(seed),
        draws=draws,
        multi_region=multi_region,
        horizon=horizon,
        proposal_lane=lane,
        region_law=law,
    )
    funnel = probe.funnel(source_heavy)
    return {
        "cell": cell_name,
        "delta": payload["delta"],
        "support": payload["support"],
        "source_smiles": cell["smiles"],
        "source_heavy_atoms": source_heavy,
        "lane": lane,
        "region_law": FREE_GATE_MARGIN_V1 if law is not None else None,
        "horizon": horizon,
        "draws": draws,
        "multi_region": multi_region,
        "seed": seed,
        "eligible": len(records),
        "eligible_per_draw": len(records) / draws if draws else 0.0,
        "funnel": funnel,
        "eligible_endpoints": [
            {
                "smiles": r["smiles"],
                "similarity": round(r["similarity"], 6),
                "qed": round(r["qed"], 6),
                "sa": round(r["sa"], 4),
                "heavy": r["heavy"],
                "heavy_delta": r["heavy"] - source_heavy,
                "families": r.get("families"),
                "program_families": r.get("program_families"),
            }
            for r in records
        ],
        "near_misses": probe.near_misses(),
        # Wall clock shares the machine with other jobs, so it is NOT a
        # load-independent cost metric. Read the counts.
        "seconds": round(time.time() - started, 1),
        "oracle_calls": 0,
        "docking_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--cell", default="fa7_0")
    parser.add_argument("--lane", default=CONTRACT_LANE)
    parser.add_argument("--law", choices=("on", "off"), default="on")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--draws", type=int, default=480)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--single-region", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    row = run_arm(
        args.contract,
        cell_name=args.cell,
        lane=args.lane,
        law_on=args.law == "on",
        horizon=args.horizon,
        draws=args.draws,
        seed_offset=args.seed_offset,
        multi_region=not args.single_region,
    )
    row["schema_version"] = SCHEMA_VERSION
    row["contract"] = str(args.contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=1, sort_keys=True))
    f = row["funnel"]
    print(
        f"{row['cell']} lane={row['lane']} law={row['region_law']} h={row['horizon']} "
        f"draws={row['draws']} seed={row['seed']} :: eligible={row['eligible']} "
        f"checked={f['distinct_endpoints_checked']} parsed={f['parsed']} "
        f"sim={f['pass_similarity']} qed={f['pass_qed']} sim&qed={f['pass_similarity_and_qed']} "
        f"disagree={f['gate_disagreements']} {row['seconds']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

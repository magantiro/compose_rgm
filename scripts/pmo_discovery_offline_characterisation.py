"""Offline characterisation of the PMO discovery allocator: arm B against arm C.

ZERO ORACLE CALLS. The scorer is a free synthetic landscape defined in this file.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
Arm C changes WHERE THE BUDGET GOES and nothing else: the proposal law, the realizer,
the fiber and the executor are arm B's. So the instrument drives the live allocation
path -- ``_augment``, ``_fit_value``, ``_credit_allocate``, the real ``PopulationCredit``
or ``DiscoveryCredit``, and the real ``observe_batch`` for every credit write -- over a
candidate pool that both arms see identically at matched seed and matched budget.

It is a SIMULATION of the closed loop, not a chemistry result. Endpoints are real
RDKit-valid molecules and basins are real Bemis-Murcko scaffolds, so the basin algebra
under test is the production one. But which molecule a parent can reach is drawn from a
synthetic basin graph rather than proposed by the fiber, and the score is a synthetic
landscape. No claim about oracle performance is made or implied; the claim is about
basin coverage under a fixed, matched proposal distribution.

THE LOOP IS CLOSED, which is the point. A round's candidate pool is built from molecules
ALREADY SCORED in this run, so a basin the allocator never funds is a basin whose
neighbours are never proposed. That is the compounding the 1,000-call A/B evidence
implies: arm B found its best molecule at 500 charged calls and never improved it, while
its top ten kept filling in.

TWO LANDSCAPES
--------------
``plateaued``   the first basin found is a local plateau and better basins exist. This
                is where discovery should pay.
``adversarial`` the first basin found is overwhelmingly the best and every rival is
                poor. Exploiting is CORRECT here, and the test is that the hard floor
                still holds -- that one early exploit cannot monopolise the run -- at a
                bounded, stated cost. A floor that only survives when it is free is not
                a floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_credit import basin_label
from compose_v4.control.pmo_discovery import DiscoveryConfig
from compose_v4.control.pmo_population_controller import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    PmoPopulationController,
)

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = json.loads(
    (ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json").read_text()
)["payload"]["checkpoints"]["shared_all_routes"]

# Eight distinct Bemis-Murcko cores. Decorated below into analogue series, so a basin
# holds several molecules and "visiting a basin again" is a real option for the
# allocator rather than an artifact of having only one molecule per scaffold.
CORES = (
    "O=C(N{R})c1ccccc1",
    "O=C(N{R})c1ccc2ccccc2c1",
    "c1ccc(-c2ccncc2){R}c1",
    "O=C(N{R})C1CCOCC1",
    "O=C(N{R})c1cccs1",
    "c1ccc(Oc2ccccc2){R}c1",
    "O=C(N{R})C1CCN(C)CC1",
    "O=C(N{R})c1ccno1",
    "O=C(N{R})c1ccccc1F",
    "O=C(N{R})c1ccc(Cl)cc1",
    "O=C(N{R})C1CCCCC1",
    "O=C(N{R})c1ccc2[nH]ccc2c1",
    "O=C(N{R})c1cnc2ccccc2c1",
    "O=C(N{R})C1CCS(=O)(=O)CC1",
    "O=C(N{R})c1ccc(-c2ccccc2)cc1",
    "O=C(N{R})C1CC1",
)
def _substituents() -> tuple[str, ...]:
    """Short C/N/O chains, kept only if the decorated molecule parses.

    The universe must be substantially LARGER THAN THE BUDGET or the comparison is
    vacuous: under `archive_seen` exclusion a run that can reach every molecule scores
    every molecule, both arms finish with the identical set, and basin coverage cannot
    differ however the budget was allocated. Measured at 84 molecules against a 250
    budget, both arms terminated early on 12-40 molecules and agreed exactly. An
    allocator only matters where the budget forces a choice.
    """
    from itertools import product

    chains = []
    for length in (1, 2, 3, 4):
        for combo in product("CNO", repeat=length):
            chain = "".join(combo)
            if "OO" in chain or "NN" in chain or "NO" in chain or "ON" in chain:
                continue  # peroxides / hydrazines: valid but not worth modelling
            chains.append(chain)
    return tuple(dict.fromkeys(chains))


SUBSTITUENTS = _substituents()

FAMILIES = (
    ("atom_insert",),
    ("atom_delete",),
    ("atom_insert", "cycle_close"),
    ("atom_restate_semantic",),
    ("bond_reroute", "cycle_open"),
)
CHANNELS = (SHALLOW_CHANNEL, STRUCTURED_CHANNEL)


def build_basins() -> dict[str, list[str]]:
    """Real, RDKit-valid molecules grouped by their real Murcko scaffold."""
    basins: dict[str, list[str]] = {}
    for core in CORES:
        for sub in SUBSTITUENTS:
            smiles = core.format(R=sub)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            basins.setdefault(basin_label(canonical), []).append(canonical)
    return {key: sorted(set(value)) for key, value in sorted(basins.items()) if value}


def landscape(basins: list[str], *, mode: str) -> dict[str, float]:
    """Hidden per-basin value. Free to evaluate; never an oracle."""
    if mode == "plateaued":
        # The seed basin is a decent plateau; three rivals are strictly better.
        # Value RISES along the chain, so the best basins are the most distant ones
        # and only sustained funding of intermediate basins reaches them.
        return {
            basin: round(0.45 + 0.03 * index, 4) for index, basin in enumerate(basins)
        }
    elif mode == "adversarial":
        # One early exploit is overwhelmingly attractive. Exploiting is CORRECT.
        # The seed basin is overwhelmingly the best and every rival is poor, so
        # exploiting is CORRECT and the floor is paying a real price.
        return {
            basin: (0.95 if index == 0 else round(0.10 + 0.005 * index, 4))
            for index, basin in enumerate(basins)
        }
    raise ValueError(f"unknown landscape {mode!r}")


def score_of(smiles: str, values: dict[str, float]) -> float:
    """Basin plateau plus a deterministic per-molecule jitter. No oracle, no lookup."""
    basin = basin_label(smiles)
    digest = hashlib.blake2b(smiles.encode(), digest_size=8).digest()
    jitter = int.from_bytes(digest, "big") / 2**64  # [0, 1)
    return round(min(1.0, max(0.0, values[basin] + 0.05 * (jitter - 0.5))), 6)


#: The four arms. C is DECOMPOSED, because its two mechanisms can pull against each
#: other: a basin-stratified floor spreads budget, while frontier credit concentrates it
#: by construction -- `frontier_gain` is nonzero only for a molecule that reaches the top
#: ten, and early in a run those are all in the incumbent basin. Running only the full
#: arm would report the NET of the two and attribute it to neither.
ARMS = {
    "B_memory_only": None,
    "C_full": DiscoveryConfig(),
    "C_floor_only": DiscoveryConfig(frontier_weight=0.0),
    "C_frontier_only": DiscoveryConfig(discovery_fraction=0.0),
}


def _controller(*, discovery: bool, seed: int):
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        attempts_per_batch=32,
        candidates_per_batch=16,
        wall_seconds=5,
        parent_allocation="niche_score",
    )
    return PmoPopulationController(
        config,
        source_group="offline-characterisation",
        oracle_protocol="synthetic-free-scorer",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
        enable_online_memory=True,
        enable_discovery=discovery,
    )


def run_arm(
    *, arm: str, seed: int, budget: int, per_round: int, mode: str, stay: float
) -> dict:
    """One arm. Returns coverage counted over molecules that were actually SCORED."""
    basins = build_basins()
    names = list(basins)
    values = landscape(names, mode=mode)
    # Basin adjacency is a CHAIN. With an all-to-all graph every basin is one hop from
    # the seed, coverage saturates in a few rounds and the metric cannot discriminate
    # between the arms -- measured at 6/6 basins for both before this was fixed.
    neighbours: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        adjacent = []
        if index > 0:
            adjacent.append(names[index - 1])
        if index < len(names) - 1:
            adjacent.append(names[index + 1])
        neighbours[name] = adjacent
    config = ARMS[arm]
    controller = _controller(discovery=config is not None, seed=seed)
    if config is not None:
        # The arm variants differ ONLY in these settings; every other input, the seed
        # and the candidate pool are identical across arms.
        controller.credit.config = config
    rng = np.random.default_rng(seed)

    seed_basin = names[0]
    archive: list[tuple[str, float]] = [
        (smiles, score_of(smiles, values)) for smiles in basins[seed_basin][:2]
    ]
    scored: dict[str, float] = dict(archive)
    families_used: set[str] = set()
    parents_used: set[str] = set()
    charged = len(scored)
    rounds = 0

    while charged < budget:
        rounds += 1
        room = min(per_round, budget - charged)
        rows = []
        for index in range(48):
            parent_smiles, parent_score = archive[int(rng.integers(len(archive)))]
            home = basin_label(parent_smiles)
            # A proposal stays in its parent's basin with probability `stay`, else it
            # lands in a neighbouring basin. Both arms draw from the same law.
            target = (
                home
                if rng.random() < stay
                else neighbours[home][int(rng.integers(len(neighbours[home])))]
            )
            endpoint = basins[target][int(rng.integers(len(basins[target])))]
            rules = FAMILIES[index % len(FAMILIES)]
            rows.append(
                controller._augment(
                    {
                        "endpoint": endpoint,
                        "program": {"blocks": []},
                        "trace": {"actions": [{"executor_rule": r} for r in rules]},
                        "provenance": {
                            "planner_channel": CHANNELS[index % len(CHANNELS)],
                            "entry_id": parent_smiles,
                            "parent_measured_score": parent_score,
                        },
                    }
                )
            )
        # Deduplicate on endpoint AND exclude molecules already scored, exactly as
        # `propose_batch` does: it builds `archive_seen` from the observations and the
        # channel pools never re-propose them. Without this the budget is spent
        # re-scoring the archive, which deflates every distinct-count for both arms.
        unique, seen = [], set()
        for row in rows:
            if row["endpoint"] in seen or row["endpoint"] in scored:
                continue
            seen.add(row["endpoint"])
            unique.append(row)

        value, _ = controller._fit_value()
        selected, _ = controller._credit_allocate(unique, value, room)
        if not selected:
            break

        # Score, then hand the outcomes to the REAL `observe_batch`, which is what
        # writes credit, frontier credit and the memory. It raises its archive lock
        # AFTER every write, so the production write path runs in full.
        outcomes = []
        for row in selected:
            value_scored = score_of(row["endpoint"], values)
            outcomes.append(
                {
                    "candidate_id": row["candidate_id"],
                    "score": value_scored,
                    "endpoint": row["endpoint"],
                }
            )
            families_used.add("+".join(sorted({a["executor_rule"] for a in row["trace"]["actions"]})))
            parents_used.add(row["provenance"]["entry_id"])
        controller.pending = {"batch_id": f"r{rounds}", "candidates": selected}
        try:
            controller.observe_batch(f"r{rounds}", outcomes)
        except ValueError as error:
            if "pending candidate lock" not in str(error):
                raise
        controller.pending = None

        for row, outcome in zip(selected, outcomes, strict=True):
            charged += 1
            if row["endpoint"] not in scored:
                scored[row["endpoint"]] = outcome["score"]
                archive.append((row["endpoint"], outcome["score"]))

    ranked = sorted(scored.values(), reverse=True)
    top10 = ranked[:10]
    dominant = seed_basin
    in_dominant = sum(1 for s in scored if basin_label(s) == dominant)
    return {
        "arm": arm,
        "seed": seed,
        "charged_calls": charged,
        "rounds": rounds,
        "distinct_basins_scored": len({basin_label(s) for s in scored}),
        "distinct_molecules_scored": len(scored),
        "distinct_source_molecules_used": len(parents_used),
        "distinct_program_families_used": len(families_used),
        "dominant_basin_call_share": round(in_dominant / max(1, len(scored)), 4),
        "best_score": round(max(ranked), 6) if ranked else None,
        "u10": round(sum(top10) / 10, 6) if top10 else None,
        "basins_available": len(basins),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--per-round", type=int, default=14)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--stay", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "schema_version": "pmo_discovery_offline_characterisation_v1",
        "oracle_calls": 0,
        "scorer": "synthetic free landscape defined in this script",
        "budget": args.budget,
        "per_round": args.per_round,
        "stay_in_basin_probability": args.stay,
        "seeds": args.seeds,
        "landscapes": {},
    }
    for mode in ("plateaued", "adversarial"):
        rows = []
        for seed in args.seeds:
            for arm in ARMS:
                row = run_arm(
                    arm=arm,
                    seed=seed,
                    budget=args.budget,
                    per_round=args.per_round,
                    mode=mode,
                    stay=args.stay,
                )
                rows.append(row)
                print(
                    f"[{mode}] seed={seed} {row['arm']:<14} "
                    f"basins={row['distinct_basins_scored']} "
                    f"mols={row['distinct_molecules_scored']} "
                    f"parents={row['distinct_source_molecules_used']} "
                    f"fams={row['distinct_program_families_used']} "
                    f"dom={row['dominant_basin_call_share']:.3f} "
                    f"best={row['best_score']} u10={row['u10']}",
                    flush=True,
                )

        def mean(arm, field, rows=rows):
            vals = [r[field] for r in rows if r["arm"] == arm and r[field] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary = {
            field: {arm: mean(arm, field) for arm in ARMS}
            for field in (
                "distinct_basins_scored",
                "distinct_molecules_scored",
                "distinct_source_molecules_used",
                "distinct_program_families_used",
                "dominant_basin_call_share",
                "best_score",
                "u10",
            )
        }
        report["landscapes"][mode] = {"rows": rows, "mean": summary}
        print(f"\n[{mode}] MEAN over {len(args.seeds)} seeds:")
        print(f"    {'metric':<36}" + "".join(f"{a:>17}" for a in ARMS))
        for field, row in summary.items():
            print(f"    {field:<36}" + "".join(f"{row[a]!s:>17}" for a in ARMS))
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

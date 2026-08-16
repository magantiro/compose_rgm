#!/usr/bin/env python
"""Offline smoke for a PHASE A exploration policy. ZERO oracle, zero Modal.

Phase A asks a different question from everything before it: can COMPOSE find
the rare JNK3-active basin efficiently from an official-style random-120 start,
where actives are ~2% of draws and a surrogate fit on the initial population is
blind? The objective here is DISCOVERY, not hypervolume.

This checks only that the policy BEHAVES as an explorer, before any evaluation
is charged:

  1 DIFFERENT REGIONS      selections spread out rather than clustering
  2 NO REPEATED EXPANSION  it does not keep expanding the same state
  3 DIVERSE CANDIDATES     selections span many distinct Murcko scaffolds
  4 DIFFERS FROM CONTROL   it and fixed scalarization pick different molecules

⚠️ THE MOLECULAR UNIVERSE HERE IS THE CACHED SEEDED-FIXTURE FIBERS, not fibers
grown from a random-120 start, because those do not exist yet and would cost
money to make. That is fine for what this measures -- these four properties are
about HOW the policy chooses, not about which chemistry it happens to meet --
and it is not fine for anything about discovery rates, which only the charged
test can answer.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from compose_v4.policy.task3.archive import ParetoArchive  # noqa: E402
from compose_v4.policy.task3.steering import (  # noqa: E402
    FixedScalarization,
    NoveltyExploration,
)
from compose_v4.policy.task3.surrogate import TanimotoKNN  # noqa: E402

FIXTURE = Path("artifacts/benchmarks/task3_mechanism_archives_v1/archives.json")
TOP_K = 4


def scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold: the standard unit for 'a distinct chemical series'."""

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:  # noqa: BLE001
        return None


def mean_pairwise_distance(smiles: list[str]) -> float:
    from compose_v4.benchmark.oracles.forest import morgan_bits

    rows = [morgan_bits(s) for s in smiles]
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return 0.0
    fp = np.vstack(rows)
    inter = fp @ fp.T
    norms = fp.sum(axis=1)
    union = norms[:, None] + norms[None, :] - inter
    sim = inter / np.maximum(union, 1e-9)
    upper = sim[np.triu_indices(len(fp), k=1)]
    return float(1.0 - upper.mean())


def simulate(arm, seeded, fibers, *, decisions: int, seed: int) -> dict:
    """Run the policy's selection loop with LABELS FROM THE FIXTURE, not the oracle.

    Newly selected molecules have no labels here, so the archive grows in
    structure only -- enough to exercise start selection, novelty and repetition,
    which is all these four checks are about.
    """

    archive = ParetoArchive()
    archive.add_many(seeded)
    surrogate = TanimotoKNN()
    surrogate.update(list(seeded), list(seeded.values()))
    rng = np.random.default_rng(seed)
    available = set(fibers) & set(archive.values)

    starts, selected = [], []
    for _ in range(decisions):
        target = arm.target(archive, rng)
        if target is None:
            break
        with archive.restricted_to(available):
            start = arm.start(archive, target, rng)
        if start is None or start not in fibers:
            continue
        starts.append(start)
        candidates = [s for s in fibers[start] if s not in archive.values]
        if len(candidates) < TOP_K:
            continue
        predicted, spread = surrogate.predict_with_spread(candidates)
        try:
            ranks = arm.rank(predicted, target, spread, candidates)
        except TypeError:
            try:
                ranks = arm.rank(predicted, target, spread)
            except TypeError:
                ranks = arm.rank(predicted, target)
        picks = [candidates[int(i)] for i in np.argsort(-ranks)[:TOP_K]]
        selected.extend(picks)
        # Structure only: the smoke never learns a label it did not already have.
        for smiles in picks:
            archive.add(smiles, (0.0,) * 5)
    scaffolds = {scaffold(s) for s in selected}
    scaffolds.discard(None)
    return {
        "decisions": len(starts),
        "distinct_starts": len(set(starts)),
        "repeat_rate": 1.0 - len(set(starts)) / max(len(starts), 1),
        "selected": len(selected),
        "distinct_scaffolds": len(scaffolds),
        "scaffolds_per_selection": len(scaffolds) / max(len(selected), 1),
        "mean_pairwise_distance": mean_pairwise_distance(selected[:200]),
        "picks": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fibers", type=Path, default=Path("/tmp/fiber_cache/seed100"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_exploration_smoke.json"))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    seeded = {s: tuple(v) for s, v in
              fixture["archives"][str(args.seed)]["molecules"].items()}
    fibers: dict[str, list[str]] = {}
    for path in glob.glob(str(args.fibers / "*.json")):
        for start, rows in json.loads(Path(path).read_text())["fibers"].items():
            fibers[start] = [r[0] for r in rows]
    if not fibers:
        raise SystemExit(f"no cached fibers under {args.fibers}")
    print(f"{len(fibers)} cached fibers | universe = seeded-fixture states, "
          f"NOT a random-120 start")

    results = {}
    for arm in (NoveltyExploration(), FixedScalarization()):
        results[arm.name] = simulate(arm, seeded, fibers,
                                     decisions=args.decisions, seed=args.seed)
        r = results[arm.name]
        print(f"\n{arm.name}")
        print(f"  distinct starts      {r['distinct_starts']}/{r['decisions']} "
              f"(repeat rate {r['repeat_rate']:.0%})")
        print(f"  distinct scaffolds   {r['distinct_scaffolds']} over "
              f"{r['selected']} selections ({r['scaffolds_per_selection']:.2f} each)")
        print(f"  mean pairwise dist   {r['mean_pairwise_distance']:.3f}")

    explorer = results["novelty-exploration"]
    control = results["fixed-scalarization"]
    n = min(len(explorer["picks"]), len(control["picks"]))
    divergence = (1.0 - len(set(explorer["picks"][:n]) & set(control["picks"][:n]))
                  / max(n, 1)) if n else 0.0

    checks = [
        ("1 different regions", explorer["mean_pairwise_distance"]
         > control["mean_pairwise_distance"]),
        ("2 no repeated expansion", explorer["repeat_rate"] < 0.25),
        ("3 diverse candidates", explorer["scaffolds_per_selection"]
         > control["scaffolds_per_selection"]),
        ("4 differs from control", divergence > 0.5),
    ]
    print(f"\ndivergence from control: {divergence:.0%}\n")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    passed = all(ok for _, ok in checks)
    print(f"\n=> {'PROCEED to the charged Phase A test' if passed else 'DO NOT SPEND: revise offline'}")

    report = {k: {kk: vv for kk, vv in v.items() if kk != "picks"}
              for k, v in results.items()}
    report["divergence_from_control"] = divergence
    report["checks"] = {label: bool(ok) for label, ok in checks}
    report["all_passed"] = passed
    report["universe_caveat"] = ("cached seeded-fixture fibers, not fibers from a "
                                 "random-120 start; behaviour only, never "
                                 "discovery rates")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

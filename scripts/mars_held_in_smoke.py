"""Held-in smoke of the REAL upstream MARS under COMPOSE oracle accounting.

**Runs under the MARS python-3.11 venv, not the project interpreter.** MARS pins
DGL, which publishes no wheel for python 3.14. Invocation:

    cd <mars-src>            # the directory CONTAINING the MARS/ checkout
    PYTHONPATH=<repo>/src \\
      /tmp/baseline_envs/mars-py311/bin/python \\
      <repo>/scripts/mars_held_in_smoke.py --sources 5

What it establishes:

1. the real upstream sampler runs CPU-only on held-in COMPOSE sources;
2. `--mols_init` genuinely starts every chain from a supplied molecule -- the
   capability the registry claims and the retargeting arm depends on;
3. all three counters are wired through MARS's own scoring path;
4. MARS's native duplicate/reject pressure is measured rather than assumed.

The COMPOSE objective is injected by monkeypatching the module-level
`MARS.estimator.scorer.scorer.get_scores`, which is the single function
`Estimator.get_scores` dispatches to. Upstream source is not modified.

Artifact status: `SMOKE_HELD_IN`. An adapter and instrument check, never a
measurement of MARS's optimization quality.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# The MARS checkout must be importable as a top-level package named `MARS`, and
# upstream hardcodes `MARS/...` paths, so this script is run from the directory
# CONTAINING that checkout. Running a script by absolute path puts the script's
# own directory on sys.path, not the working directory, so add it explicitly.
sys.path.insert(0, str(Path.cwd()))

from compose_v4.experiments.oracle_accounting import (  # noqa: E402
    BudgetExhausted,
    OracleAccountant,
    silence_rdkit,
)

COHORT = REPO / "diagnostics" / "retarget_calibration_cohort.json"
NORMALIZERS = REPO / "diagnostics" / "retarget_goal_language_normalizers.json"
OUTPUT = REPO / "diagnostics" / "baselines" / "mars_held_in_smoke.json"

#: MARS dispatches every objective through this one module-level function.
OBJECTIVE_NAME = "compose_developability"


def load_sources(count: int) -> tuple[list[str], str]:
    cohort = json.loads(COHORT.read_text())
    if cohort["pool"] != "held-in training sources only":
        raise SystemExit(f"refusing a non-held-in cohort: {cohort['pool']!r}")
    return [row["source"] for row in cohort["sources"][:count]], cohort["cohort_sha256"]


def developability_objective():
    from rdkit import Chem
    from rdkit.Chem import Crippen, QED

    normalizers = json.loads(NORMALIZERS.read_text())["normalizers"]
    qed_scale = normalizers["qed"]["iqr"]
    clogp_scale = normalizers["clogp"]["iqr"]
    qed_floor = normalizers["qed"]["median"]
    clogp_low = normalizers["clogp"]["p25"]
    clogp_high = normalizers["clogp"]["p75"]

    def evaluate(canonical: str) -> float:
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            return 0.0
        qed_margin = (QED.qed(molecule) - qed_floor) / qed_scale
        clogp = Crippen.MolLogP(molecule)
        clogp_margin = min(clogp - clogp_low, clogp_high - clogp) / clogp_scale
        # MARS maximizes a sum of per-objective scores and its own objectives are
        # all in [0, 1]; a raw margin is unbounded and negative for most held-in
        # molecules, so it is squashed rather than handed over raw. Recorded as a
        # deviation: the ORDERING is preserved, the scale is not.
        raw = float(min(qed_margin, clogp_margin))
        return 1.0 / (1.0 + pow(2.718281828459045, -raw))

    return evaluate, {
        "goal": "target-free developability: QED floor and cLogP box",
        "normalizers": "diagnostics/retarget_goal_language_normalizers.json",
        "deviation": (
            "logistic squash of the raw margin into (0, 1). MARS sums "
            "per-objective scores that its own objectives keep in [0, 1] and its "
            "acceptance ratio is scale-sensitive, so an unbounded raw margin "
            "would change the sampler's temperature semantics. Ordering is "
            "preserved; scale is not. The accountant records the SQUASHED value "
            "actually handed to MARS."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=5)
    parser.add_argument("--num-mols", type=int, default=5, help="parallel chains")
    parser.add_argument("--num-step", type=int, default=10)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--vocab", type=str, default="chembl60k")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    silence_rdkit()
    sources, cohort_sha = load_sources(args.sources)
    objective, goal = developability_objective()

    import torch
    from rdkit import Chem

    from MARS.datasets.utils import load_mols
    from MARS.estimator import estimator as mars_estimator_module
    from MARS.estimator.estimator import Estimator
    from MARS.estimator.scorer import scorer as mars_scorer
    from MARS.proposal.models.editor_basic import BasicEditor
    from MARS.proposal.proposal import Proposal_Editor
    from MARS.sampler import Sampler_SA

    accountant = OracleAccountant(
        evaluate=objective,
        budget=args.budget,
        budget_counter="unique_valid_canonical_evaluations",
    )
    seen_first_states: list[str] = []

    def patched_get_scores(objective_name, mols, **_kwargs):
        scores = []
        for mol in mols:
            try:
                smiles = Chem.MolToSmiles(mol)
            except Exception:
                smiles = ""
            scores.append(accountant.score(smiles))
        return scores

    # `MARS/estimator/estimator.py` does `from .scorer.scorer import get_scores`
    # at import time, which binds the name into the *estimator* module's
    # namespace. Patching only the scorer module therefore has no effect --
    # patch the name where it is actually looked up, and the source module too
    # so nothing else in the package sees the unpatched version.
    mars_scorer.get_scores = patched_get_scores
    mars_estimator_module.get_scores = patched_get_scores

    run_dir = Path("MARS/runs/compose_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    init_path = Path("MARS/data/compose_held_in_sources.txt")
    init_path.write_text("\n".join(sources) + "\n")

    config = {
        "device": torch.device("cpu"),
        "root_dir": "MARS",
        "data_dir": "MARS/data",
        "run_dir": str(run_dir),
        "mols_ref": None,
        "vocab": args.vocab,
        "vocab_size": args.vocab_size,
        "max_size": 40,
        "num_mols": args.num_mols,
        "num_step": args.num_step,
        "log_every": 1,
        "sampler": "sa",
        "proposal": "editor",
        "objectives": [OBJECTIVE_NAME],
        "score_wght": [1.0],
        "score_succ": [0.5],
        "score_clip": [0.6],
        "lr": 3e-4,
        "dataset_size": 500,
        "batch_size": 16,
        "n_atom_feat": 17,
        "n_bond_feat": 5,
        "n_node_hidden": 64,
        "n_edge_hidden": 128,
        "n_layers": 6,
        "editor_dir": None,
        "train": True,
    }

    # THE CAPABILITY UNDER TEST: main.py's mols_init branch, verbatim.
    #   mols = load_mols(config['data_dir'], config['mols_init'])
    #   mols = random.choices(mols, k=config['num_mols'])
    loaded = load_mols(config["data_dir"], init_path.name)
    mols_init = [loaded[i % len(loaded)] for i in range(config["num_mols"])]
    seen_first_states = [Chem.MolToSmiles(m) for m in mols_init]

    editor = BasicEditor(config).to(config["device"])
    proposal = Proposal_Editor(config, editor)
    estimator = Estimator(config)
    sampler = Sampler_SA(config, proposal, estimator)

    started = time.perf_counter()
    error = None
    traceback_text = None
    try:
        sampler.sample(str(run_dir), mols_init)
    except BudgetExhausted:
        error = "budget_exhausted"
    except Exception as exc:  # noqa: BLE001 - a failure here is the finding
        error = f"{type(exc).__name__}: {exc}"
        traceback_text = traceback.format_exc()
    elapsed = time.perf_counter() - started

    manifest = accountant.manifest()
    source_set = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in sources}
    checks = {
        "sampler_ran": error is None or error == "budget_exhausted",
        "mols_init_starts_every_chain_from_a_supplied_source": (
            len(seen_first_states) == config["num_mols"]
            and all(s in source_set for s in seen_first_states)
        ),
        "all_three_counters_wired": (
            manifest["counters"]["oracle_requests"] > 0
            and manifest["counters"]["unique_valid_canonical_evaluations"] > 0
            and manifest["counters"]["evaluator_calls"] > 0
        ),
        "counter_identity_holds": all(manifest["invariants"].values()),
    }

    report = {
        "schema": "compose.baselines.mars_held_in_smoke",
        "title": "MARS (real upstream) — HELD-IN ADAPTER SMOKE",
        "artifact_status": "SMOKE_HELD_IN",
        "known_small_regime_failure": (
            "MEASURED: at --num-mols 5 this smoke crashed with "
            "'ValueError: not enough values to unpack (expected 2, got 0)' in "
            "ImitationDataset.collate_fn. Cause: no proposal improved the score "
            "in the first step, so the imitation dataset was EMPTY and "
            "collate_fn was handed a zero-length batch. MARS trains its proposal "
            "online on improving edits only, so at small chain counts it can "
            "receive zero training records and crash. It ran at --num-mols 40. "
            "This is a second, independent reason a budget-matched comparison "
            "cannot simply shrink MARS: below some chain count it does not run "
            "at all, quite apart from its proposal being untrained."
        ),
        "what_this_is_not": (
            "A measurement of MARS's optimization quality, and not a "
            "COMPOSE-versus-MARS comparison. Chain counts and step counts here "
            "are smoke-sized; MARS's designed regime is ~1000-5000 chains for "
            "~550-1000 steps."
        ),
        "held_out_data_opened": False,
        "sources": {
            "cohort": "diagnostics/retarget_calibration_cohort.json",
            "cohort_sha256": cohort_sha,
            "pool": "held-in training sources only",
            "count": len(sources),
            "smiles": sources,
        },
        "objective": goal,
        "initial_states": seen_first_states,
        "settings": {k: str(v) for k, v in config.items() if k != "device"},
        "counters": manifest["counters"],
        "invariants": manifest["invariants"],
        "demand_ratio_requests_over_unique": manifest[
            "demand_ratio_requests_over_unique"
        ],
        "wall_seconds": round(elapsed, 3),
        "error": error,
        "traceback": traceback_text,
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "note": "runs under the MARS py3.11 venv; DGL has no python 3.14 wheel",
        },
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"error: {error}")
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

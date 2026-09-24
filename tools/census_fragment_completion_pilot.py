"""Hash-bound retrospective census of completed rows, without generation or model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from fetch_official_fragment_evaluator import default_cache_dir, verify_only
from inspect_fragment_program_pilot import describe
from rdkit import Chem, rdBase
from rdkit.Chem import QED
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
    sascorer,
)
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = list(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "min": float(min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(max(values)),
    }


@lru_cache(maxsize=8192)
def molecule_metrics(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"invalid saved endpoint: {smiles}")
    qed, sa = QED.qed(mol), sascorer.calculateScore(mol)
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rdkit_rings": mol.GetRingInfo().NumRings(),
        "qed": qed,
        "sa": sa,
        "joint_pass": qed >= 0.6 and sa <= 4,
    }


def summarize_molecules(rows):
    return {
        "count": len(rows),
        "raw_joint_pass": sum(row["joint_pass"] for row in rows),
        **{
            key: distribution(row[key] for row in rows)
            for key in ("heavy_atoms", "rdkit_rings", "qed", "sa")
        },
    }


def selection_panel(attempt):
    """Reconcile saved selection and measure its conditional size/ring tilt."""
    supported = [r for r in attempt["offered"] if r["status"] == "model_supported"]
    unique = {}
    for row in supported:
        unique.setdefault(row["endpoint"], row)
    if not unique:
        if attempt["committed_smiles"] is not None:
            raise ValueError("output exists without model-supported panel")
        return {"panel_size": 0}
    selected = supported[attempt["selection"]["selected_index"]]
    if selected["endpoint"] != attempt["committed_smiles"]:
        raise ValueError("saved selected index and committed endpoint disagree")
    rows = list(unique.values())
    scores = np.asarray([r["mean_log_mark"] for r in rows], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("supported candidate has nonfinite score")
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    index = next(i for i, r in enumerate(rows) if r["endpoint"] == selected["endpoint"])
    receipt = attempt["selection"]
    if receipt["unique_model_supported_endpoints"] != len(rows):
        raise ValueError("panel uniqueness receipt disagrees")
    if abs(receipt["selected_probability"] - weights[index]) > 1e-12:
        raise ValueError("saved selection probability disagrees with frozen softmax")
    result = {"panel_size": len(rows), "selected_draw": selected["draw"]}
    for key in ("heavy_atoms", "rdkit_rings"):
        values = np.asarray([molecule_metrics(r["endpoint"])[key] for r in rows])
        result[key] = {
            "uniform_mean": float(values.mean()),
            "model_softmax_mean": float(weights @ values),
            "model_minus_uniform": float(weights @ values - values.mean()),
            "selected": int(values[index]),
            "selected_minus_uniform": float(values[index] - values.mean()),
            "highest_score": int(values[np.argmax(scores)]),
            "within_panel_score_covariance": float(
                np.mean((scores - scores.mean()) * (values - values.mean()))
            ),
        }
    return result


def summarize_panels(rows):
    nonempty = [r for r in rows if r["panel_size"]]
    return {
        "attempts": len(rows),
        "nonempty_panels": len(nonempty),
        "unique_model_supported_panel_size": distribution(r["panel_size"] for r in rows),
        **{
            key: {
                name: distribution(r[key][name] for r in nonempty)
                for name in (
                    "uniform_mean",
                    "model_softmax_mean",
                    "model_minus_uniform",
                    "selected",
                    "selected_minus_uniform",
                    "highest_score",
                    "within_panel_score_covariance",
                )
            }
            for key in ("heavy_atoms", "rdkit_rings")
        },
    }


def refinement_summary(rows):
    return {
        "count": len(rows),
        "pass_to_fail": sum(
            r["seed"]["joint_pass"] and not r["endpoint"]["joint_pass"] for r in rows
        ),
        "fail_to_pass": sum(
            not r["seed"]["joint_pass"] and r["endpoint"]["joint_pass"] for r in rows
        ),
        "seed": summarize_molecules([r["seed"] for r in rows]),
        "endpoint": summarize_molecules([r["endpoint"] for r in rows]),
        "deltas": {
            key: distribution(r["endpoint"][key] - r["seed"][key] for r in rows)
            for key in ("heavy_atoms", "rdkit_rings", "qed", "sa")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("use pilot RDKit 2024.03.5 for score parity")
    timestamp = datetime.now(timezone.utc).isoformat()
    prompt_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    prompts = [
        p
        for p in load_genmol_prompts(prompt_path)
        if p.task in (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION)
    ]
    # Freeze this complete-row census once. Never wait for or inspect unfinished attempts.
    included, missing = [], []
    for prompt in prompts:
        path = args.pilot_dir / "rows" / f"{prompt.task.value}_{prompt.drug_name}.json"
        (included if path.is_file() else missing).append((prompt, path))
    inputs = {}

    def bind(path):
        value = Path(path).read_bytes()
        inputs[str(Path(path).resolve())] = hashlib.sha256(value).hexdigest()
        return value

    manifest = json.loads(bind(args.pilot_dir / "manifest.json"))
    bind(args.baseline_dir / "manifest.json")
    bind(prompt_path)
    preflight = json.loads(bind(args.pilot_dir / "evaluator_preflight_final.json"))
    if not preflight["passed"]:
        raise ValueError("evaluator preflight did not pass")
    official = verify_only()
    for name in official:
        bind(default_cache_dir() / "pkg" / name)
    for path in (
        Path(__file__),
        Path("tools/inspect_fragment_program_pilot.py"),
        Path("tools/run_fragment_attachment_library_pilot.py"),
        Path("tools/fetch_official_fragment_evaluator.py"),
        Path("src/compose_v4/benchmark/fragment_official_metrics.py"),
        Path("src/compose_v4/benchmark/fragment_constrained.py"),
        Path(QED.__file__),
        Path(sascorer.__file__),
        Path(sascorer.__file__).with_name("fpscores.pkl.gz"),
    ):
        bind(path)
    results = []
    task_pools = defaultdict(lambda: {"panels": [], "candidates": [], "refinements": []})
    for prompt, path in included:
        row = json.loads(bind(path))
        baseline_path = (
            args.baseline_dir / "shards" / f"{prompt.task.value}__{prompt.drug_name}__baseline.json"
        )
        baseline = json.loads(bind(baseline_path))
        if inputs[str(baseline_path.resolve())] != manifest["inputs"][str(baseline_path.resolve())]:
            raise ValueError("baseline hash differs from frozen pilot input")
        attempts = [
            json.loads(
                bind(
                    args.pilot_dir
                    / "attempts"
                    / f"{prompt.task.value}_{prompt.drug_name}_{i:03d}.json"
                )
            )
            for i in range(20)
        ]
        if row["attempts"] != 20 or not all(
            a["complete"] and len(a["offered"]) == 8 and a["attempt_index"] == i
            for i, a in enumerate(attempts)
        ):
            raise ValueError("completed prompt lacks complete exact attempt census")
        fragment = Chem.MolFromSmiles(prompt.fragments[0])
        core = {
            "heavy_atoms": fragment.GetNumHeavyAtoms(),
            "rdkit_rings": fragment.GetRingInfo().NumRings(),
            "attachments": sum(a.GetAtomicNum() == 0 for a in fragment.GetAtoms()),
        }
        result = {"task": prompt.task.value, "drug": prompt.drug_name, "core": core}
        for arm, records in (("baseline", baseline["attempt_records"]), ("new", attempts)):
            official_now = official_prompt_metrics(
                [r["committed_smiles"] or "" for r in records], expected_samples=20
            )
            if any(abs(official_now[k] - row[arm][k]) > 1e-10 for k in official_now):
                raise ValueError(f"official evaluator parity failed: {path}/{arm}")
            result[arm] = describe(records, row[arm])
            metrics = [
                molecule_metrics(r["committed_smiles"]) for r in records if r["committed_smiles"]
            ]
            result[arm]["distribution"] = summarize_molecules(metrics)
            result[arm]["added_to_core"] = {
                key: distribution(m[key] - core[key] for m in metrics)
                for key in ("heavy_atoms", "rdkit_rings")
            }
            result[arm]["prompt_fidelity"] = sum(
                check_fragment_constraint(prompt, r["committed_smiles"]).satisfied
                for r in records
                if r["committed_smiles"]
            )
        panels, candidates, refinements = [], [], []
        for attempt in attempts:
            panels.append({"attempt": attempt["attempt_index"], **selection_panel(attempt)})
            for offered in attempt["offered"]:
                candidate = {
                    "attempt": attempt["attempt_index"],
                    "draw": offered["draw"],
                    "lane": ("region", "shallow", "structured", "anchored")[offered["draw"] % 4],
                    "status": offered["status"],
                    "reason": offered.get("reason"),
                    "endpoint": offered.get("endpoint"),
                    "mean_log_mark": offered.get("mean_log_mark"),
                    "selected": offered["draw"] == panels[-1].get("selected_draw"),
                }
                if offered.get("endpoint"):
                    candidate["metrics"] = molecule_metrics(offered["endpoint"])
                    if candidate["metrics"]["heavy_atoms"] > 40:
                        raise ValueError("saved endpoint exceeds declared 40-heavy-atom support")
                    candidate["capabilities"] = offered["capabilities"]
                    seed = offered["provenance"].get("seed_endpoint")
                    if seed:
                        refinements.append(
                            {
                                "attempt": attempt["attempt_index"],
                                "draw": offered["draw"],
                                "lane": candidate["lane"],
                                "status": offered["status"],
                                "selected": candidate["selected"],
                                "seed_smiles": seed,
                                "endpoint_smiles": offered["endpoint"],
                                "seed": molecule_metrics(seed),
                                "endpoint": candidate["metrics"],
                            }
                        )
                candidates.append(candidate)
        result.update(panels=panels, candidates=candidates, refinements=refinements)
        results.append(result)
        for key, values in (
            ("panels", panels),
            ("candidates", candidates),
            ("refinements", refinements),
        ):
            task_pools[prompt.task.value][key].extend(values)
    summaries = {}
    for task, pool in task_pools.items():
        group = [r for r in results if r["task"] == task]
        summaries[task] = {
            "completed_prompts": len(group),
            "attempts_per_arm": 20 * len(group),
            "arms": {
                arm: {
                    "groups": dict(sum((Counter(r[arm]["groups"]) for r in group), Counter())),
                    "unique_joint_pass_within_prompt": sum(
                        r[arm]["unique_joint_pass"] for r in group
                    ),
                    "mean_official": {
                        k: float(np.mean([r[arm]["official"][k] for r in group]))
                        for k in ("quality", "uniqueness", "validity", "diversity")
                    },
                    "distribution": summarize_molecules(
                        [
                            molecule_metrics(m["smiles"])
                            for r in group
                            for m in r[arm]["molecules"]
                            if m["smiles"]
                        ]
                    ),
                }
                for arm in ("baseline", "new")
            },
            "panel_selection": summarize_panels(pool["panels"]),
            "status_counts": dict(Counter(c["status"] for c in pool["candidates"])),
            "refusal_reasons": dict(
                Counter(c["reason"] for c in pool["candidates"] if c["reason"])
            ),
            "lanes": {},
            "refinement": {},
        }
        for lane in ("region", "shallow", "structured", "anchored"):
            selected = [c for c in pool["candidates"] if c["lane"] == lane]
            summaries[task]["lanes"][lane] = {
                "offered": len(selected),
                "statuses": dict(Counter(c["status"] for c in selected)),
                **{
                    stage: summarize_molecules(
                        [
                            c["metrics"]
                            for c in selected
                            if c.get("metrics")
                            and (
                                stage == "compiled"
                                or (stage == "model_supported" and c["status"] == stage)
                                or (stage == "selected" and c["selected"])
                            )
                        ]
                    )
                    for stage in ("compiled", "model_supported", "selected")
                },
            }
        for stage in ("compiled", "model_supported", "selected"):
            summaries[task]["refinement"][stage] = refinement_summary(
                [
                    r
                    for r in pool["refinements"]
                    if stage == "compiled"
                    or (stage == "model_supported" and r["status"] == stage)
                    or (stage == "selected" and r["selected"])
                ]
            )
    if any(digest(path) != value for path, value in inputs.items()):
        raise RuntimeError("a bound input changed during the census")
    output = {
        "schema": "fragment_completion_pilot_census_v1",
        "snapshot_utc": timestamp,
        "role": "post-generation descriptive development audit; no causal size claim; no generation or selection change",
        "partial": len(included) < 20,
        "completed_prompts": len(included),
        "expected_prompts": 20,
        "attempts_per_arm": len(included) * 20,
        "excluded_unfinished_prompts": [
            {"task": p.task.value, "drug": p.drug_name, "missing_row": str(path)}
            for p, path in missing
        ],
        "configuration": {
            "only_completed_prompt_rows": True,
            "seed": "none; deterministic saved-endpoint analysis",
            "quality": "QED >= 0.6 and SA <= 4; unique passing per prompt / all attempts",
            "ring_definition": "RDKit perceived ring count; not graph cycle rank",
            "selection_diagnostic": "unit-temperature softmax on saved mean_log_mark; compare conditional within-panel expectations with uniform",
            "splits": "no fitting; fixed public development prompts",
            "workers": 1,
            "device": "cpu",
            "precision": "float64 descriptors and diagnostic reduction; saved float32 model scores",
        },
        "inputs_sha256": inputs,
        "official_blob_hashes": official,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "checks": {
            "all_official_metrics_recomputed": True,
            "input_hashes_unchanged_after_analysis": True,
            "saved_selection_probability_reconciled": True,
            "checkpoint_loads": 0,
            "generated_candidates": 0,
            "new_oracle_or_docking_calls": 0,
        },
        "summaries": summaries,
        "prompts": results,
    }
    args.output_dir.mkdir(parents=True)
    _atomic_json(args.output_dir / "census.json", output)
    print(
        json.dumps(
            {
                "completed_prompts": len(included),
                "partial": output["partial"],
                "summaries": summaries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

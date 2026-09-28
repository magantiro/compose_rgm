"""Pure, count-checked reductions of saved prompt/seed metrics.

No molecular samples are generated or re-scored here. Means average the ten
prompts within each generation seed, then the three seeds. Reported benchmark
SD uses n-1 across those three seed means, never across individual molecules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from statistics import fmean, stdev
from typing import Any

from .evidence import Evidence, EvidenceError

METRICS = ("quality", "uniqueness", "diversity", "validity")
EXTRAS = ("prompt_fidelity", "prompt_faithful_quality_yield")
PANEL_ROLES = {
    "motif_extension": "motif_selection",
    "scaffold_decoration": "decoration_selection",
    "linker_design": "linker_selection",
}


@dataclass(frozen=True)
class Row:
    prompt: str
    seed: int
    budget: int
    arm: str
    metrics: dict[str, float]


def _require(condition: bool, location: str, message: str) -> None:
    if not condition:
        raise EvidenceError(f"{location}: {message}")


def _number(value: Any, location: str, upper: float = 100.0) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= upper,
        location,
        f"expected a finite value in [0, {upper}], found {value!r}",
    )
    return float(value)


def _close(actual: float, expected: Any, location: str) -> None:
    _require(
        type(expected) in (int, float)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10),
        location,
        f"row reduction {actual} disagrees with stored value {expected!r}",
    )


def _population(evidence: Evidence) -> tuple[list[str], dict[str, list[int]]]:
    manifest = evidence.manifest
    prompts = manifest.get("prompts")
    _require(
        isinstance(prompts, list)
        and all(isinstance(p, str) and p for p in prompts)
        and len(set(prompts)) == len(prompts) == 10,
        "manifest/prompts",
        "expected ten distinct prompt names",
    )
    seeds = manifest.get("generation_seeds")
    tasks = {*PANEL_ROLES, "superstructure_generation"}
    _require(isinstance(seeds, dict) and set(seeds) == tasks, "manifest", "wrong task set")
    for task, values in seeds.items():
        _require(
            isinstance(values, list)
            and all(type(seed) is int for seed in values)
            and len(set(values)) == len(values) == 3,
            f"manifest/{task}",
            "expected three distinct integer generation seeds",
        )
    _require(manifest.get("attempts_per_prompt_seed") == 100, "manifest", "expected 100 slots")
    _require(manifest.get("prefix_budgets") == [1, 2, 4, 8], "manifest", "wrong prefix budgets")
    _require(
        manifest.get("aliases") == {"scaffold_morphing": "linker_design"},
        "manifest/aliases",
        "morphing must reuse linker outputs, not become an independent arm",
    )
    return prompts, seeds


def _panel_rows(artifact: dict, task: str, prompts: list[str], seeds: list[int]) -> list[Row]:
    _require(artifact.get("task") == task, task, "artifact task mismatch")
    population = artifact.get("population", {})
    for key, expected in (
        ("generation_seeds", seeds),
        ("prompts", 10),
        ("attempts_per_prompt_seed", 100),
        ("total_attempts", 3000),
    ):
        _require(population.get(key) == expected, f"{task}/population/{key}", "wrong population")
    section = artifact if task == "motif_extension" else artifact["prefix"]
    _require(section.get("budgets") == [1, 2, 4, 8], task, "prefix budgets changed")
    raw = section.get("rows")
    _require(
        isinstance(raw, list) and len(raw) == 120, task, "expected 120 prompt/seed/prefix rows"
    )
    native = "learned" if task == "motif_extension" else "deployed"
    rows = []
    seen = set()
    for item in raw:
        key = (item.get("drug"), item.get("seed"), item.get("budget"))
        _require(
            key[0] in prompts
            and type(key[1]) is int
            and key[1] in seeds
            and type(key[2]) is int
            and key[2] in (1, 2, 4, 8),
            task,
            f"unrecognized row identity {key}",
        )
        _require(key not in seen, task, f"duplicate prompt/seed/prefix {key}")
        seen.add(key)
        _require(set(item.get("arms", {})) == {native, "uniform"}, str(key), "wrong arms")
        for original, arm in ((native, "deployed"), ("uniform", "uniform")):
            record = item["arms"][original]
            metrics = {
                metric: _number(
                    record.get("metrics", {}).get(metric),
                    f"{task}/{key}/{arm}/{metric}",
                    1.0 if metric == "diversity" else 100.0,
                )
                for metric in METRICS
            }
            metrics.update(
                {metric: _number(record.get(metric), f"{key}/{metric}") for metric in EXTRAS}
            )
            outputs = record.get("outputs")
            _require(type(outputs) is int and 0 <= outputs <= 100, str(key), "invalid output count")
            _close(metrics["validity"], outputs, f"{task}/{key}/validity versus output census")
            _require(
                metrics["prompt_faithful_quality_yield"] <= metrics["prompt_fidelity"] <= outputs,
                str(key),
                "faithful yield/fidelity exceeds its denominator",
            )
            rows.append(Row(*key, arm, metrics))
    _require(seen == set(product(prompts, seeds, (1, 2, 4, 8))), task, "incomplete Cartesian panel")
    for budget in (1, 2, 4, 8):
        for original, arm in ((native, "deployed"), ("uniform", "uniform")):
            cohort = [r for r in rows if r.budget == budget and r.arm == arm]
            for metric in (*METRICS, *EXTRAS):
                actual = fmean(r.metrics[metric] for r in cohort)
                _close(
                    actual,
                    section["means"][str(budget)][original][metric],
                    f"{task}/{budget}/{metric}",
                )
                if budget == 8 and task != "motif_extension":
                    _close(
                        actual, artifact["main"]["means"][original][metric], f"{task}/main/{metric}"
                    )
    if task != "motif_extension":
        full = [dict(item, budget=8) for item in artifact["main"]["rows"]]
        by_identity = lambda item: (item["drug"], item["seed"])
        _require(
            sorted(full, key=by_identity)
            == sorted((item for item in raw if item["budget"] == 8), key=by_identity),
            task,
            "full-prefix rows differ from the main comparison",
        )
    one = {(r.prompt, r.seed, r.arm): r.metrics for r in rows if r.budget == 1}
    for prompt, seed in product(prompts, seeds):
        _require(
            one[prompt, seed, "deployed"] == one[prompt, seed, "uniform"],
            task,
            "one-offer arms differ",
        )
    return rows


def _superstructure_rows(
    artifact: dict, arm: str, prompts: list[str], seeds: list[int]
) -> list[Row]:
    location = f"superstructure/{arm}"
    for field in ("attempts", "committed", "fragment_preserving", "prompt_compliant"):
        _require(artifact.get(field) == 3000, location, f"{field} is not 3000")
    raw = artifact.get("rows")
    _require(isinstance(raw, list) and len(raw) == 30, location, "expected 30 rows")
    rows = []
    seen = set()
    for item in raw:
        key = (item.get("drug"), item.get("seed"))
        _require(
            key[0] in prompts and type(key[1]) is int and key[1] in seeds,
            location,
            f"bad row {key}",
        )
        _require(key not in seen, location, f"duplicate prompt/seed {key}")
        seen.add(key)
        for field in ("attempts", "committed", "fragment_preserving", "prompt_compliant"):
            _require(item.get(field) == 100, str(key), f"{field} is not 100")
        metrics = {
            m: _number(
                item.get("official", {}).get(m), f"{key}/{m}", 1.0 if m == "diversity" else 100.0
            )
            for m in METRICS
        }
        rows.append(Row(*key, 0, arm, metrics))
    _require(seen == set(product(prompts, seeds)), location, "incomplete prompt/seed panel")
    for metric in METRICS:
        _close(
            fmean(r.metrics[metric] for r in rows),
            artifact["official_mean"][metric],
            f"{location}/{metric}",
        )
    saved_seeds = {item["seed"]: item for item in artifact["per_seed"]}
    _require(
        len(artifact["per_seed"]) == 3 and set(saved_seeds) == set(seeds),
        location,
        "bad seed summaries",
    )
    for seed, saved in saved_seeds.items():
        for metric in METRICS:
            _close(
                fmean(r.metrics[metric] for r in rows if r.seed == seed),
                saved[metric],
                f"{location}/seed{seed}/{metric}",
            )
    return rows


def summarize(rows: list[Row], arm: str) -> dict:
    cohort = [r for r in rows if r.arm == arm]
    seeds = sorted({r.seed for r in cohort})
    _require(
        len(seeds) == 3 and len(cohort) == 30, arm, "summary requires the complete three-seed panel"
    )
    names = tuple(cohort[0].metrics)
    by_seed = {
        str(seed): {m: fmean(r.metrics[m] for r in cohort if r.seed == seed) for m in names}
        for seed in seeds
    }
    return {
        "seeds": seeds,
        "attempts": 3000,
        "per_seed": by_seed,
        "metrics": {
            m: {
                "mean": fmean(v[m] for v in by_seed.values()),
                "sample_sd": stdev(v[m] for v in by_seed.values()),
            }
            for m in names
        },
    }


def _paired(rows: list[Row], prompts: list[str]) -> list[dict]:
    result = []
    for prompt in prompts:
        names = tuple(rows[0].metrics)
        arms = {
            arm: {
                m: fmean(r.metrics[m] for r in rows if r.prompt == prompt and r.arm == arm)
                for m in names
            }
            for arm in ("deployed", "uniform")
        }
        result.append(
            {
                "prompt": prompt,
                "arms": arms,
                "deployed_minus_uniform": {
                    m: arms["deployed"][m] - arms["uniform"][m] for m in names
                },
            }
        )
    return result


def _intervals(evidence: Evidence, task: str, per_prompt: list[dict]) -> dict:
    if task == "motif_extension":
        source = evidence.artifacts["motif_intervals"]["tasks"][task]["estimates"]
        interval_key, mean_key = "bootstrap_percentile_95", "paired_prompt_mean"
    elif task == "superstructure_generation":
        source = evidence.artifacts["superstructure_comparison"]["estimates"]
        interval_key, mean_key = "prompt_bootstrap_percentile_95", "paired_prompt_mean_difference"
    else:
        source = evidence.artifacts[PANEL_ROLES[task]]["main"]["prompt_bootstrap"]
        interval_key, mean_key = "percentile_95", "paired_prompt_mean"
    result = {}
    for metric in METRICS:
        delta = fmean(p["deployed_minus_uniform"][metric] for p in per_prompt)
        _close(delta, source[metric][mean_key], f"{task}/paired/{metric}")
        bounds = source[metric][interval_key]
        _require(
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
            and bounds[0] <= bounds[1],
            task,
            f"invalid saved interval for {metric}",
        )
        result[metric] = {
            "mean_difference": delta,
            "percentile_95": bounds,
            "status": "saved_interval_not_recomputed",
        }
    return result


def reduce_evidence(evidence: Evidence) -> dict:
    prompts, seeds = _population(evidence)
    benchmark, comparisons, prefixes = {}, {}, {}
    for task, role in PANEL_ROLES.items():
        rows = _panel_rows(evidence.artifacts[role], task, prompts, seeds[task])
        full = [r for r in rows if r.budget == 8]
        benchmark[task] = summarize(full, "deployed")
        per_prompt = _paired(full, prompts)
        comparisons[task] = {
            "arms": {arm: summarize(full, arm) for arm in ("deployed", "uniform")},
            "per_prompt": per_prompt,
            "intervals": _intervals(evidence, task, per_prompt),
        }
        prefixes[task] = {
            str(budget): {
                arm: summarize([r for r in rows if r.budget == budget], arm)
                for arm in ("deployed", "uniform")
            }
            for budget in evidence.manifest["prefix_budgets"]
        }
    task = "superstructure_generation"
    super_rows = []
    for role, arm in (
        ("superstructure_learned", "deployed"),
        ("superstructure_uniform", "uniform"),
    ):
        super_rows.extend(_superstructure_rows(evidence.artifacts[role], arm, prompts, seeds[task]))
    benchmark[task] = summarize(super_rows, "deployed")
    per_prompt = _paired(super_rows, prompts)
    comparisons[task] = {
        "arms": {arm: summarize(super_rows, arm) for arm in ("deployed", "uniform")},
        "per_prompt": per_prompt,
        "intervals": _intervals(evidence, task, per_prompt),
    }
    comparison = evidence.artifacts["superstructure_comparison"]
    for original, role in (
        ("learned", "superstructure_learned"),
        ("uniform", "superstructure_uniform"),
    ):
        _require(
            comparison["arm_inputs"][original]["sha256"]
            == evidence.manifest["artifacts"][role]["sha256"],
            task,
            "comparison binds a different result",
        )
        _require(
            comparison["input_contract_payload_sha256"][original]
            == evidence.artifacts[role]["contract_payload_sha256"],
            task,
            "comparison binds a different contract",
        )
    motif_hash = evidence.artifacts["motif_intervals"]["input_sha256"][
        "diagnostics/fragment_common_panel_motif_v1/result.json"
    ]
    _require(
        motif_hash
        == evidence.artifacts["motif_selection"]["provenance"]["existing_selection_sha256"],
        "motif_extension",
        "intervals bind a different motif selection",
    )
    return {
        "schema": "compose_fragment_tables_v1",
        "scope": evidence.manifest["scope"],
        "inputs": evidence.manifest["artifacts"],
        "aliases": evidence.manifest["aliases"],
        "aggregation": "ten prompt means per generation seed; mean and sample SD (n-1) across three seeds",
        "bootstrap": evidence.manifest["bootstrap"],
        "benchmark": benchmark,
        "comparisons": comparisons,
        "prefixes": prefixes,
        "limitations": evidence.manifest["limitations"]
        + [
            "Submitted Table 1 prints motif validity as 100%; saved rows give 99.9667%. No manuscript is edited.",
            "Saved intervals are carried with provenance unless --recompute-intervals is requested.",
            "Published external comparator columns are not independently verified or regenerated here.",
        ],
    }


def recompute_intervals(report: dict) -> None:
    """Repeat the saved bootstrap algorithm in the declared reduction environment."""
    import numpy as np

    protocol = report["bootstrap"]
    expected = protocol["reduction_numpy"]
    if np.__version__ != expected:
        raise EvidenceError(f"bootstrap requires numpy=={expected}, found {np.__version__}")
    for task, comparison in report["comparisons"].items():
        prompts = comparison["per_prompt"]
        rng = np.random.default_rng(protocol["seeds"][task])
        indices = rng.integers(0, len(prompts), size=(protocol["draws"], len(prompts)))
        for metric in METRICS:
            values = np.asarray(
                [p["deployed_minus_uniform"][metric] for p in prompts], dtype=np.float64
            )
            estimates = values[indices].mean(axis=1)
            bounds = [float(np.quantile(estimates, q)) for q in (0.025, 0.975)]
            saved = comparison["intervals"][metric]
            for actual, expected in zip(bounds, saved["percentile_95"], strict=True):
                _close(actual, expected, f"{task}/{metric}/bootstrap")
            saved["percentile_95"] = bounds
            saved["status"] = "recomputed_and_matched_saved_interval"

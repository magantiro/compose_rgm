"""Replay selection on the locked decoration and linker fragment panels.

No molecular proposals are generated. The full-budget learned arm is the
recorded deployed selection; every other selection is a deterministic replay
on the same recorded offer panels. Smaller budgets truncate the original
attempted-offer order and are retrospective, not fresh generator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import RDLogger, rdBase

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "scaffold_decoration": (
        ROOT / ".worktrees/fragment-decoration-source-coupled-v1",
        "fragment_decoration_official_v2",
        "frozen",
    ),
    "linker_design": (
        ROOT / ".worktrees/fragment-linker-novelty-v1",
        "fragment_linker_novelty_official_v2",
        None,
    ),
}
METRICS = ("quality", "uniqueness", "diversity", "validity")
BUDGETS = (1, 2, 4, 8)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260926


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seeded_rng(*parts: object) -> np.random.Generator:
    material = "|".join(str(part) for part in parts)
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class Offer:
    draw: int
    endpoint: str
    score: float
    provenance: dict


def admitted_offers(raw: list[dict], budget: int) -> tuple[Offer, ...]:
    if len(raw) != 8 or [item.get("draw") for item in raw] != list(range(8)):
        raise ValueError("expected all eight original attempted offers in draw order")
    if budget not in BUDGETS:
        raise ValueError(f"unsupported attempted-offer budget: {budget}")
    seen: set[str] = set()
    result: list[Offer] = []
    for item in raw[:budget]:
        if item.get("status") != "model_supported":
            continue
        endpoint = item.get("endpoint")
        score = item.get("mean_log_mark")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("supported offer lacks endpoint")
        if endpoint in seen:
            raise ValueError("duplicate endpoint survived native panel deduplication")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("supported offer lacks a finite native score")
        seen.add(endpoint)
        result.append(
            Offer(item["draw"], endpoint, float(score), item.get("provenance", {}))
        )
    return tuple(result)


def probabilities(
    offers: tuple[Offer, ...], *, selector: str, emitted: frozenset[str]
) -> np.ndarray:
    if not offers:
        return np.empty(0, dtype=np.float64)
    if selector == "uniform":
        return np.full(len(offers), 1.0 / len(offers), dtype=np.float64)
    if selector not in ("learned", "learned_novelty4"):
        raise ValueError(f"unknown selector {selector}")
    scores = np.asarray([offer.score for offer in offers], dtype=np.float64)
    if selector == "learned_novelty4":
        scores += np.asarray(
            [math.log(4.0) if offer.endpoint not in emitted else 0.0 for offer in offers]
        )
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()


def draw_endpoint(
    offers: tuple[Offer, ...], *, selector: str, emitted: frozenset[str], seed: tuple
) -> str:
    if not offers:
        return ""
    p = probabilities(offers, selector=selector, emitted=emitted)
    return offers[int(seeded_rng(*seed).choice(len(offers), p=p))].endpoint


def verify_deployed_selection(
    panel: dict, offers: tuple[Offer, ...], *, selector: str, emitted: frozenset[str]
) -> None:
    selected = panel.get("selected_smiles")
    receipt = panel.get("selection")
    if not offers:
        if selected is not None or receipt is not None:
            raise ValueError("empty eligible panel has a selected molecule")
        return
    if not isinstance(receipt, dict) or selected not in {o.endpoint for o in offers}:
        raise ValueError("deployed selection is absent or outside native support")
    index = receipt.get("selected_index")
    if not isinstance(index, int) or not 0 <= index < len(offers):
        raise ValueError("deployed selection index is outside native support")
    chosen = offers[index]
    if chosen.endpoint != selected or receipt.get("selected_draw") != chosen.draw:
        raise ValueError("deployed selection endpoint/draw mismatch")
    p = probabilities(offers, selector=selector, emitted=emitted)
    if not math.isclose(float(receipt["selected_probability"]), float(p[index]), abs_tol=1e-12):
        raise ValueError("deployed selected probability differs from scorer normalization")
    if not math.isclose(
        float(receipt["mean_native_log_mark_probability"]), chosen.score, abs_tol=1e-10
    ):
        raise ValueError("deployed native score differs from offer")
    if receipt.get("unique_model_supported_endpoints") != len(offers):
        raise ValueError("deployed eligible count differs from panel")
    if selector == "learned_novelty4" and (
        receipt.get("prior_unique_emissions") != len(emitted)
        or receipt.get("selected_already_emitted") != (selected in emitted)
        or receipt.get("novelty_multiplier") != 4.0
    ):
        raise ValueError("deployed novelty archive/weight is inconsistent")


def install_evaluator(source_root: Path) -> None:
    for path in (
        source_root / "src",
        source_root / "tools",
        source_root / ".official_eval_cache/pkg",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from fetch_official_fragment_evaluator import verify_only

    verify_only()


def load_locked(task: str) -> tuple[dict, dict, list[dict], dict]:
    source_root, name, arm = SPECS[task]
    config_path = source_root / "configs" / f"{name}.json"
    envelope = json.loads(config_path.read_text())
    contract = envelope["payload"]
    if json_identity(contract) != envelope["payload_sha256"]:
        raise ValueError(f"contract self-hash mismatch: {config_path}")
    if contract["schema"] != name or contract["task"] != task:
        raise ValueError(f"contract task/schema mismatch: {config_path}")
    if contract["attempts_per_prompt_seed"] != 100 or len(contract["drugs"]) != 10:
        raise ValueError(f"contract population mismatch: {config_path}")
    output = source_root / contract["output_dir"]
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest_hash = sha256(manifest_path)
    if manifest["contract_payload_sha256"] != envelope["payload_sha256"]:
        raise ValueError(f"manifest does not bind contract: {manifest_path}")
    if summary["manifest_sha256"] != manifest_hash or summary["attempts"] != 3000:
        raise ValueError(f"summary does not bind full completed run: {summary_path}")
    if summary["outputs"] != 3000:
        raise ValueError(f"locked run contains output failures: {summary_path}")

    prompt_path = source_root / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    expected_prompt_hash = manifest["development_contract"]["material_sha256"][
        "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    ]
    if sha256(prompt_path) != expected_prompt_hash:
        raise ValueError(f"prompt population hash mismatch: {prompt_path}")
    install_evaluator(source_root)
    from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts

    task_enum = (
        FragmentTask.SCAFFOLD_DECORATION
        if task == "scaffold_decoration"
        else FragmentTask.LINKER_DESIGN
    )
    prompts = {
        prompt.drug_name: prompt
        for prompt in load_genmol_prompts(prompt_path)
        if prompt.task == task_enum
    }
    if set(prompts) != set(contract["drugs"]):
        raise ValueError(f"prompt names mismatch: {prompt_path}")

    records: list[dict] = []
    attempt_hashes: dict[str, str] = {}
    selector = "learned_novelty4" if task == "linker_design" else "learned"
    for seed in contract["seeds"]:
        for drug in contract["drugs"]:
            sub = output / f"seed{seed}"
            if arm:
                attempt_dir = sub / "attempts" / arm
                lock_path = sub / "locks" / arm / f"{drug}.json"
                row_path = sub / "rows" / arm / f"{drug}.json"
            else:
                attempt_dir = sub / "attempts"
                lock_path = sub / "locks" / f"{drug}.json"
                row_path = sub / "rows" / f"{drug}.json"
            lock = json.loads(lock_path.read_text())
            source_row = json.loads(row_path.read_text())
            if lock["manifest_sha256"] != manifest_hash:
                raise ValueError(f"lock manifest mismatch: {lock_path}")
            if source_row["lock_sha256"] != sha256(lock_path):
                raise ValueError(f"row lock hash mismatch: {row_path}")
            if source_row["manifest_sha256"] != manifest_hash:
                raise ValueError(f"row manifest mismatch: {row_path}")
            if len(lock["samples"]) != 100 or len(lock["attempt_hashes"]) != 100:
                raise ValueError(f"incomplete locked prompt population: {lock_path}")
            if summary["row_sha256"][str(row_path.relative_to(output))] != sha256(row_path):
                raise ValueError(f"summary row hash mismatch: {row_path}")
            attempts = []
            emitted: set[str] = set()
            for index in range(100):
                path = attempt_dir / f"{drug}_{index:03d}.json"
                relative = (
                    f"attempts/{arm}/{drug}_{index:03d}.json"
                    if arm else f"attempts/{drug}_{index:03d}.json"
                )
                observed_hash = sha256(path)
                if lock["attempt_hashes"][relative] != observed_hash:
                    raise ValueError(f"locked attempt hash mismatch: {path}")
                receipt = json.loads(path.read_text())
                if (receipt["drug"], receipt["attempt_index"]) != (drug, index):
                    raise ValueError(f"attempt identity mismatch: {path}")
                if task == "linker_design" and receipt["seed"] != seed:
                    raise ValueError(f"attempt seed mismatch: {path}")
                panel = receipt["panel"]
                offers = admitted_offers(panel["offered"], 8)
                if panel["model_supported_count"] != len(offers):
                    raise ValueError(f"native support count mismatch: {path}")
                verify_deployed_selection(
                    panel, offers, selector=selector, emitted=frozenset(emitted)
                )
                selected = panel["selected_smiles"] or ""
                if selected != lock["samples"][index]:
                    raise ValueError(f"selected molecule differs from locked output: {path}")
                if selected:
                    emitted.add(selected)
                attempt_hashes[str(path)] = observed_hash
                attempts.append({
                    "index": index,
                    "offers": offers,
                    "raw_offer_statuses": [item["status"] for item in panel["offered"]],
                    "selected": selected,
                })
            records.append(
                {
                    "seed": seed,
                    "drug": drug,
                    "attempts": attempts,
                    "source_row": source_row,
                    "source_row_sha256": sha256(row_path),
                    "source_lock_sha256": sha256(lock_path),
                }
            )
    if len(records) != 30 or len(attempt_hashes) != 3000:
        raise ValueError("incomplete 30-row or 3,000-attempt record")
    provenance = {
        "source_root": str(source_root),
        "contract_path": str(config_path),
        "contract_sha256": sha256(config_path),
        "contract_payload_sha256": envelope["payload_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "prompt_path": str(prompt_path),
        "prompt_sha256": expected_prompt_hash,
        "source_code_revision": manifest["code_revision"],
        "source_checkpoint_sha256": manifest["development_contract"]["material_sha256"][
            manifest["development_contract"]["checkpoint_path"]
        ],
        "official_evaluator_sha256": manifest["development_contract"][
            "official_evaluator_sha256"
        ],
        "attempt_sha256": attempt_hashes,
    }
    return contract, summary, records, {"prompts": prompts, "provenance": provenance}


def summarize_rows(rows: list[dict]) -> dict:
    return {
        name: float(np.mean([row["metrics"][name] for row in rows])) for name in METRICS
    } | {
        "prompt_fidelity": float(np.mean([row["prompt_fidelity"] for row in rows])),
        "prompt_faithful_quality_yield": float(
            np.mean([row["prompt_faithful_quality_yield"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=tuple(SPECS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    RDLogger.DisableLog("rdApp.warning")
    contract, source_summary, records, context = load_locked(args.task)
    from compose_v4.benchmark.fragment_constrained import check_fragment_constraint
    from compose_v4.benchmark.fragment_linker_assembly import linker_fidelity
    from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
    from in_virtuo_gen.utils.mol import compute_properties, is_drug_like_and_synthesizable

    fidelity_cache: dict[tuple[str, str], bool] = {}
    quality_cache: dict[str, bool] = {}

    def faithful(drug: str, smiles: str) -> bool:
        if not smiles:
            return False
        key = (drug, smiles)
        if key not in fidelity_cache:
            prompt = context["prompts"][drug]
            fidelity_cache[key] = bool(
                linker_fidelity(prompt, smiles)["satisfied"]
                if args.task == "linker_design"
                else check_fragment_constraint(prompt, smiles).satisfied
            )
        return fidelity_cache[key]

    def quality_flags(samples: list[str]) -> None:
        missing = list(dict.fromkeys(sample for sample in samples if sample and sample not in quality_cache))
        if not missing:
            return
        props = compute_properties(missing)
        if len(props) != len(missing):
            raise ValueError("official quality function returned incorrect row count")
        quality_cache.update(
            (sample, bool(prop is not None and is_drug_like_and_synthesizable(prop)))
            for sample, prop in zip(missing, props, strict=True)
        )

    def score_row(drug: str, samples: list[str]) -> dict:
        if len(samples) != 100:
            raise ValueError("attempt denominator changed")
        metrics = official_prompt_metrics(samples, expected_samples=100)
        quality_flags(samples)
        faithful_count = sum(faithful(drug, sample) for sample in samples)
        faithful_quality_count = sum(
            faithful(drug, sample) and quality_cache.get(sample, False) for sample in samples
        )
        return {
            "metrics": metrics,
            "outputs": sum(bool(sample) for sample in samples),
            "prompt_fidelity": faithful_count,
            "prompt_faithful_quality_yield": faithful_quality_count,
        }

    deployed = "learned_novelty4" if args.task == "linker_design" else "learned"
    main_rows = []
    for record in records:
        seed, drug = record["seed"], record["drug"]
        learned = [item["selected"] for item in record["attempts"]]
        uniform = [
            draw_endpoint(
                item["offers"], selector="uniform", emitted=frozenset(),
                seed=("locked-fragment-selection-v1", contract["schema"], seed, drug, item["index"], "uniform"),
            )
            for item in record["attempts"]
        ]
        arms = {"deployed": score_row(drug, learned), "uniform": score_row(drug, uniform)}
        for metric in METRICS:
            if not math.isclose(
                arms["deployed"]["metrics"][metric],
                record["source_row"]["metrics"][metric],
                abs_tol=1e-8,
            ):
                raise ValueError(f"deployed {metric} failed official row reproduction: {seed}/{drug}")
        if arms["deployed"]["outputs"] != record["source_row"]["outputs"]:
            raise ValueError(f"deployed output count failed row reproduction: {seed}/{drug}")
        main_rows.append({"seed": seed, "drug": drug, "arms": arms})
    main_means = {
        arm: summarize_rows([row["arms"][arm] for row in main_rows])
        for arm in ("deployed", "uniform")
    }
    for metric in METRICS:
        if not math.isclose(main_means["deployed"][metric], source_summary["official_mean"][metric], abs_tol=1e-8):
            raise ValueError(f"deployed {metric} failed locked summary reproduction")
    main_by_seed = {
        str(seed): {
            arm: summarize_rows([
                row["arms"][arm] for row in main_rows if row["seed"] == seed
            ])
            for arm in ("deployed", "uniform")
        }
        for seed in contract["seeds"]
    }

    prompts = []
    for drug in contract["drugs"]:
        matched = [row for row in main_rows if row["drug"] == drug]
        arms = {
            arm: summarize_rows([row["arms"][arm] for row in matched])
            for arm in ("deployed", "uniform")
        }
        prompts.append({
            "drug": drug,
            "arms": arms,
            "deployed_minus_uniform": {
                name: arms["deployed"][name] - arms["uniform"][name]
                for name in (*METRICS, "prompt_fidelity", "prompt_faithful_quality_yield")
            },
        })
    bootstrap = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if args.task == "linker_design" else 1))
    indices = rng.integers(0, len(prompts), size=(BOOTSTRAP_DRAWS, len(prompts)))
    for name in (*METRICS, "prompt_fidelity", "prompt_faithful_quality_yield"):
        differences = np.asarray([p["deployed_minus_uniform"][name] for p in prompts])
        estimates = differences[indices].mean(axis=1)
        bootstrap[name] = {
            "paired_prompt_mean": float(differences.mean()),
            "percentile_95": [float(np.quantile(estimates, q)) for q in (0.025, 0.975)],
            "deployed_higher": int(np.count_nonzero(differences > 0)),
            "tied": int(np.count_nonzero(differences == 0)),
            "uniform_higher": int(np.count_nonzero(differences < 0)),
        }
    print(json.dumps({"stage": "main_complete", "task": args.task, "means": main_means}), flush=True)

    # All eight original attempts were recorded, including rejected offers, so
    # these are honest prefix truncations of the recorded proposal-attempt order.
    budget_rows = []
    for budget in BUDGETS:
        if budget == 8:
            budget_rows.extend({"budget": budget, **row} for row in main_rows)
            continue
        for record in records:
            seed, drug = record["seed"], record["drug"]
            archives = {"deployed": set(), "uniform": set()}
            samples = {"deployed": [], "uniform": []}
            for item in record["attempts"]:
                eligible_by_draw = {offer.draw: offer for offer in item["offers"]}
                prefix = admitted_offers(
                    [
                        {
                            "draw": i, "status": "model_supported",
                            "endpoint": offer.endpoint, "mean_log_mark": offer.score,
                            "provenance": offer.provenance,
                        }
                        if (offer := eligible_by_draw.get(i)) is not None
                        else {"draw": i, "status": item["raw_offer_statuses"][i]}
                        for i in range(8)
                    ],
                    budget,
                )
                for arm, selector in (("deployed", deployed), ("uniform", "uniform")):
                    chosen = draw_endpoint(
                        prefix, selector=selector, emitted=frozenset(archives[arm]),
                        seed=("locked-fragment-prefix-v1", contract["schema"], seed, drug, item["index"], budget, arm),
                    )
                    samples[arm].append(chosen)
                    if chosen:
                        archives[arm].add(chosen)
                if budget == 1 and samples["deployed"][-1] != samples["uniform"][-1]:
                    raise ValueError("one attempted offer changed endpoint across selectors")
            budget_rows.append({
                "budget": budget, "seed": seed, "drug": drug,
                "arms": {arm: score_row(drug, values) for arm, values in samples.items()},
            })
        print(json.dumps({"stage": "budget_complete", "task": args.task, "budget": budget}), flush=True)
    budget_means = {
        str(budget): {
            arm: summarize_rows([
                row["arms"][arm] for row in budget_rows if row["budget"] == budget
            ])
            for arm in ("deployed", "uniform")
        }
        for budget in BUDGETS
    }

    # Additive expectation, conditional on the original saved panels. This is
    # not the official nonlinear cohort quality or uniqueness metric.
    all_offers = [
        offer.endpoint for record in records for item in record["attempts"] for offer in item["offers"]
    ]
    quality_flags(all_offers)
    expected = {"deployed": 0.0, "uniform": 0.0}
    for record in records:
        emitted: set[str] = set()
        for item in record["attempts"]:
            offers = item["offers"]
            for arm, selector in (("deployed", deployed), ("uniform", "uniform")):
                weights = probabilities(offers, selector=selector, emitted=frozenset(emitted))
                expected[arm] += sum(
                    float(weight) * faithful(record["drug"], offer.endpoint)
                    * quality_cache[offer.endpoint]
                    for offer, weight in zip(offers, weights, strict=True)
                )
            if item["selected"]:
                emitted.add(item["selected"])
    expected = {arm: value / 3000.0 * 100.0 for arm, value in expected.items()}

    report = {
        "schema": "fragment_locked_panel_selection_replay_v1",
        "task": args.task,
        "interpretation": (
            "deployed reference-plus-novelty4 versus uniform selection on identical saved offers"
            if args.task == "linker_design" else
            "deployed reference-score softmax versus uniform selection on identical saved offers"
        ),
        "limitations": [
            "Selection replay is conditional on proposals and native-support admission from the locked run.",
            "Prefix budgets retrospectively truncate the original eight attempted offers, including failures; they are not fresh smaller-panel generator runs.",
            "Uniform selector draws are not independent generation replicates.",
        ] + ([
            "The linker deployed selector includes a fourfold preference for as-yet-unemitted endpoints, so its contrast with uniform is not an isolated learned-reference effect.",
            "The full-budget learned archive is the recorded deployed archive. Counterfactual selector histories can alter future proposal RNG consumption in an autonomous run.",
        ] if args.task == "linker_design" else []),
        "provenance": context["provenance"] | {
            "analysis_path": str(Path(__file__).resolve()),
            "analysis_sha256": sha256(Path(__file__)),
            "analysis_git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion, "platform": platform.platform(),
            },
            "seed_derivation": "SHA256 domain-separated task|seed|prompt|attempt|arm|budget, first 64 bits",
            "bootstrap": {"unit": "prompt after averaging three generation seeds", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        },
        "population": {"prompts": 10, "generation_seeds": contract["seeds"], "attempts_per_prompt_seed": 100, "total_attempts": 3000, "full_panel_attempted_offers": 8},
        "main": {"means": main_means, "per_seed": main_by_seed, "rows": main_rows, "per_prompt": prompts, "prompt_bootstrap": bootstrap, "conditional_expected_prompt_faithful_quality_yield": expected},
        "prefix": {"budgets": list(BUDGETS), "means": budget_means, "rows": budget_rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, args.output)
    print(json.dumps({"stage": "written", "task": args.task, "output": str(args.output), "main": main_means, "prefix": budget_means}), flush=True)


if __name__ == "__main__":
    main()

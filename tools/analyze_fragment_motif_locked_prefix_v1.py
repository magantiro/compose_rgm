"""Retrospective attempted-offer prefixes on the locked motif panels.

The existing exact-panel motif learned-versus-uniform result is reused at the
full eight-offer budget. This program only derives smaller prefixes from the
same 3,000 recorded eight-offer attempts; it never generates a molecule.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from analyze_fragment_locked_panel_selection_v1 import (
    METRICS,
    admitted_offers,
    draw_endpoint,
    install_evaluator,
    json_identity,
    probabilities,
    sha256,
    summarize_rows,
    verify_deployed_selection,
)
from rdkit import RDLogger, rdBase

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".worktrees/fragment-attachment-library-20260924"
RESULT = ROOT / ".worktrees/fragment-qed-ablations-20260925/diagnostics/fragment_common_panel_motif_v1/result.json"
EXPECTED_RESULT_SHA256 = "3d05f15e29dcf5f72f89480841ee9a67e380a8d1d2f176580f7a4e3c23a9e559"
OUTPUT = ROOT / "diagnostics/fragment_locked_panel_selection_v1/motif_prefix_result.json"


def analyze() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(RESULT) != EXPECTED_RESULT_SHA256:
        raise ValueError(f"existing motif selection result changed: {RESULT}")
    prior = json.loads(RESULT.read_text())
    config_path = SOURCE / "configs/fragment_motif_official_v1.json"
    envelope = json.loads(config_path.read_text())
    contract = envelope["payload"]
    if json_identity(contract) != envelope["payload_sha256"]:
        raise ValueError("motif contract self-hash mismatch")
    run = SOURCE / contract["output_dir"]
    summary_path = run / "summary.json"
    manifest_path = run / "manifest.json"
    manifest_hash = sha256(manifest_path)
    if (
        sha256(summary_path) != prior["source_summary_sha256"]
        or manifest_hash != prior["source_manifest_sha256"]
        or envelope["payload_sha256"] != prior["source_contract_payload_sha256"]
    ):
        raise ValueError("existing motif replay is not bound to the locked result")
    if len(prior["selections"]) != 3000 or len(prior["rows"]) != 30:
        raise ValueError("existing motif replay has incomplete population")
    source_summary = json.loads(summary_path.read_text())
    if source_summary["attempts"] != 3000 or source_summary["manifest_sha256"] != manifest_hash:
        raise ValueError("motif official summary is incomplete or unbound")
    prompt_path = SOURCE / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    expected_prompt = contract["material_sha256"][
        "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    ]
    if sha256(prompt_path) != expected_prompt:
        raise ValueError("motif prompt data changed")
    install_evaluator(SOURCE)
    from compose_v4.benchmark.fragment_constrained import (
        FragmentTask,
        check_fragment_constraint,
        load_genmol_prompts,
    )
    from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
    from in_virtuo_gen.utils.mol import compute_properties, is_drug_like_and_synthesizable

    prompts = {
        item.drug_name: item
        for item in load_genmol_prompts(prompt_path)
        if item.task == FragmentTask.MOTIF_EXTENSION
    }
    if set(prompts) != set(contract["drugs"]):
        raise ValueError("motif prompt names differ from contract")
    indexed_prior = {
        (item["seed"], item["drug"], item["attempt_index"]): item
        for item in prior["selections"]
    }
    if len(indexed_prior) != 3000:
        raise ValueError("existing motif replay has duplicate attempt identities")
    quality_cache: dict[str, bool] = {}
    fidelity_cache: dict[tuple[str, str], bool] = {}

    def quality_flags(samples: list[str]) -> None:
        missing = list(dict.fromkeys(s for s in samples if s and s not in quality_cache))
        if missing:
            props = compute_properties(missing)
            if len(props) != len(missing):
                raise ValueError("pinned quality evaluator returned wrong count")
            quality_cache.update(
                (s, bool(p is not None and is_drug_like_and_synthesizable(p)))
                for s, p in zip(missing, props, strict=True)
            )

    def faithful(drug: str, smiles: str) -> bool:
        if not smiles:
            return False
        key = (drug, smiles)
        if key not in fidelity_cache:
            fidelity_cache[key] = bool(check_fragment_constraint(prompts[drug], smiles).satisfied)
        return fidelity_cache[key]

    def score_row(drug: str, samples: list[str]) -> dict:
        metrics = official_prompt_metrics(samples, expected_samples=100)
        quality_flags(samples)
        return {
            "metrics": metrics,
            "outputs": sum(bool(s) for s in samples),
            "prompt_fidelity": sum(faithful(drug, s) for s in samples),
            "prompt_faithful_quality_yield": sum(
                faithful(drug, s) and quality_cache.get(s, False) for s in samples
            ),
        }

    rows: list[dict] = []
    attempt_hashes: dict[str, str] = {}
    expected = {"learned": 0.0, "uniform": 0.0}
    for seed in contract["seeds"]:
        for drug in contract["drugs"]:
            lock_path = run / "locks" / f"seed{seed}" / f"{drug}.json"
            lock = json.loads(lock_path.read_text())
            if lock["manifest_sha256"] != manifest_hash or len(lock["samples"]) != 100:
                raise ValueError(f"motif lock incomplete: {lock_path}")
            samples = {budget: {"learned": [], "uniform": []} for budget in (1, 2, 4)}
            all_offers = []
            for index in range(100):
                relative = f"attempts/seed{seed}/{drug}_{index:03d}.json"
                path = run / relative
                observed_hash = sha256(path)
                if lock["attempt_hashes"][relative] != observed_hash:
                    raise ValueError(f"motif attempt hash mismatch: {path}")
                item = json.loads(path.read_text())
                panel = item["panel"]
                offers = admitted_offers(panel["offered"], 8)
                verify_deployed_selection(panel, offers, selector="learned", emitted=frozenset())
                original = panel["selected_smiles"] or ""
                replay = indexed_prior[(seed, drug, index)]
                uniform_full = draw_endpoint(
                    offers, selector="uniform", emitted=frozenset(),
                    seed=(envelope["payload_sha256"], seed, drug, index, "uniform-v1"),
                )
                if (
                    original != lock["samples"][index]
                    or replay["learned"] != original
                    or replay["uniform"] != uniform_full
                    or replay["source_sha256"] != observed_hash
                ):
                    raise ValueError(f"existing motif replay did not reproduce: {path}")
                attempt_hashes[str(path)] = observed_hash
                all_offers.append(offers)
                for budget in (1, 2, 4):
                    eligible = admitted_offers(panel["offered"], budget)
                    for selector in ("learned", "uniform"):
                        picked = draw_endpoint(
                            eligible, selector=selector, emitted=frozenset(),
                            seed=("locked-motif-prefix-v1", envelope["payload_sha256"], seed, drug, index, budget, selector),
                        )
                        samples[budget][selector].append(picked)
                    if budget == 1 and samples[budget]["learned"][-1] != samples[budget]["uniform"][-1]:
                        raise ValueError("motif one-offer prefix differs across selectors")
            for budget in (1, 2, 4):
                rows.append({
                    "budget": budget, "seed": seed, "drug": drug,
                    "arms": {
                        selector: score_row(drug, samples[budget][selector])
                        for selector in ("learned", "uniform")
                    },
                })
            quality_flags([offer.endpoint for offers in all_offers for offer in offers])
            for offers in all_offers:
                for selector in ("learned", "uniform"):
                    weights = probabilities(offers, selector=selector, emitted=frozenset())
                    expected[selector] += sum(
                        float(weight) * faithful(drug, offer.endpoint)
                        * quality_cache[offer.endpoint]
                        for offer, weight in zip(offers, weights, strict=True)
                    )
        print(json.dumps({"stage": "seed_complete", "seed": seed}), flush=True)

    if len(rows) != 90 or len(attempt_hashes) != 3000:
        raise ValueError("motif prefix population incomplete")
    old_rows = {
        (item["seed"], item["drug"]): item for item in prior["rows"]
    }
    for seed in contract["seeds"]:
        for drug in contract["drugs"]:
            old = old_rows[(seed, drug)]
            arms = {
                selector: {
                    "metrics": old["arms"][selector]["metrics"],
                    "outputs": old["arms"][selector]["outputs"],
                    "prompt_fidelity": old["arms"][selector]["prompt_faithful_outputs"],
                    "prompt_faithful_quality_yield": 100.0 * old["arms"][selector][
                        "prompt_faithful_quality_yield"
                    ],
                }
                for selector in ("learned", "uniform")
            }
            rows.append({"budget": 8, "seed": seed, "drug": drug, "arms": arms})
    means = {
        str(budget): {
            selector: summarize_rows([
                row["arms"][selector] for row in rows if row["budget"] == budget
            ])
            for selector in ("learned", "uniform")
        }
        for budget in (1, 2, 4, 8)
    }
    for metric in METRICS:
        if not math.isclose(
            means["8"]["learned"][metric], source_summary["official_mean"][metric],
            abs_tol=1e-8,
        ):
            raise ValueError(f"motif full-budget {metric} did not reproduce locked summary")
    return {
        "schema": "fragment_motif_locked_prefix_v1",
        "task": "motif_extension",
        "interpretation": "Retrospective attempted-offer prefix on the locked three-seed motif run; full budget reuses the completed selection replay.",
        "limitations": [
            "The original eight-offer generator was not rerun at smaller configured panel sizes.",
            "Counterfactual selector draws are conditional on the original recorded offers and are not additional generation seeds.",
            "Failed attempts and model-support abstentions remain inside the attempted-offer prefix and output denominator.",
        ],
        "provenance": {
            "contract_path": str(config_path), "contract_sha256": sha256(config_path),
            "contract_payload_sha256": envelope["payload_sha256"],
            "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
            "summary_path": str(summary_path), "summary_sha256": sha256(summary_path),
            "existing_selection_path": str(RESULT), "existing_selection_sha256": EXPECTED_RESULT_SHA256,
            "prompt_path": str(prompt_path), "prompt_sha256": expected_prompt,
            "attempt_sha256": attempt_hashes,
            "analysis_path": str(Path(__file__).resolve()),
            "analysis_sha256": sha256(Path(__file__)),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
            "seed_derivation": "SHA256 task contract|seed|prompt|attempt|budget|selector, first 64 bits",
        },
        "population": {"prompts": 10, "generation_seeds": contract["seeds"], "attempts_per_prompt_seed": 100, "total_attempts": 3000},
        "budgets": [1, 2, 4, 8], "means": means, "rows": rows,
        "conditional_expected_prompt_faithful_quality_yield": {
            key: value / 3000.0 * 100.0 for key, value in expected.items()
        },
    }


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    result = analyze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT.parent, suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, OUTPUT)
    print(json.dumps({"stage": "written", "output": str(OUTPUT), "means": result["means"]}), flush=True)


if __name__ == "__main__":
    main()

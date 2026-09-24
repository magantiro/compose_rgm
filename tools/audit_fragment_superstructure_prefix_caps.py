"""Replay saved superstructure trajectories at earlier event-count endpoints.

This is a paired, post-hoc development diagnostic. It is not equivalent to a
fresh rollout with a smaller max-events value: the released runner uses one RNG
stream per prompt, so stopping an attempt early changes later attempts' draws.
No endpoint is chosen by its score. Every available trajectory is truncated at
each predeclared cap and scored by the pinned official evaluator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import rdkit
from fetch_official_fragment_evaluator import OFFICIAL_BLOBS, verify_only

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics
from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "diagnostics/fragment_initial_family_dev_v1"
SOURCE = FOLDER / "conditioned_strict_seed10_n20_all10.json"
PROMPTS = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
OUTPUT = FOLDER / "prefix_caps_seed10_n20_all10.json"
CAPS = (4, 8, 12, 16, 32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    verified = verify_only()
    evaluator_hash = OFFICIAL_BLOBS["in_virtuo_gen/train_utils/metrics.py"][1]
    if evaluator_hash not in verified.values():
        raise RuntimeError("official evaluator hash not verified")
    payload = json.loads(SOURCE.read_text())
    if payload["attachment_control"]["config"]["condition_initial_locked_family"] is not True:
        raise RuntimeError("source did not use initial-family conditioning")
    if payload["attachment_control"]["config"]["hard_lock_effective_chemistry"] is not True:
        raise RuntimeError("source did not use the effective-chemistry lock")
    if payload["sampler"]["config"]["max_events"] != 32:
        raise RuntimeError("source does not have the expected 32-event cap")
    prompt_map = {
        prompt.drug_name: prompt
        for prompt in load_genmol_prompts(PROMPTS)
        if prompt.task == FragmentTask.SUPERSTRUCTURE_GENERATION
    }
    if len(prompt_map) != 10:
        raise RuntimeError("expected ten superstructure prompts")
    per_drug = payload["results"]["superstructure_generation"]["per_drug"]
    if set(per_drug) != set(prompt_map):
        raise RuntimeError("source prompt set does not match the frozen manifest")

    system = de_novo_rewrite_system()
    rows = []
    for drug, prompt in sorted(prompt_map.items()):
        [entry] = per_drug[drug]
        attempts = entry["attempt_records"]
        if len(attempts) != 20:
            raise RuntimeError(f"incomplete attempt census: {drug}")
        context = build_prompt_context(prompt)
        per_cap = {cap: [] for cap in CAPS}
        for attempt in attempts:
            state = pad_molecular_graph(context.start_state, 48)
            actions = attempt["accepted_actions"]
            if len(actions) != attempt["events"]:
                raise RuntimeError(
                    f"event/action count mismatch: {drug}/{attempt['attempt_index']}"
                )
            endpoints = {}
            for index, record in enumerate(actions, start=1):
                rule = record["rule"]
                action_type = system.rules[rule].action_type
                if record["action_type"] != action_type.__name__:
                    raise RuntimeError(f"action type mismatch: {drug}/{attempt['attempt_index']}")
                action = action_type(**record["payload"])
                state = system.apply(state, rule, action)
                if index in CAPS:
                    endpoints[index] = molecular_graph_to_smiles(state)
            final = molecular_graph_to_smiles(state) if actions else None
            if final != attempt["committed_smiles"]:
                raise RuntimeError(f"full exact replay mismatch: {drug}/{attempt['attempt_index']}")
            for cap in CAPS:
                # If a trajectory ended before the cap, its recorded final
                # endpoint is the endpoint of that counterfactual as well.
                smiles = endpoints.get(cap, final if len(actions) < cap else None)
                if smiles and not check_fragment_constraint(prompt, smiles).satisfied:
                    raise RuntimeError(
                        f"prefix loses fragment: {drug}/{attempt['attempt_index']}/{cap}"
                    )
                per_cap[cap].append(smiles or "")
        if per_cap[32] != entry["emitted_samples"]:
            raise RuntimeError(f"full-cap emitted samples changed under replay: {drug}")
        for cap in CAPS:
            samples = per_cap[cap]
            metrics = official_prompt_metrics(samples, expected_samples=20)
            if cap == 32:
                for name in ("validity", "uniqueness", "quality", "diversity"):
                    if abs(metrics[name] - entry["official"][name]) > 1e-10:
                        raise RuntimeError(f"official replay metric mismatch: {drug}/{name}")
            rows.append(
                {
                    "drug": drug,
                    "cap": cap,
                    "attempts": 20,
                    "nonempty": sum(bool(sample) for sample in samples),
                    "metrics": metrics,
                    "samples": samples,
                }
            )

    summary = []
    for cap in CAPS:
        selected = [row for row in rows if row["cap"] == cap]
        summary.append(
            {
                "cap": cap,
                "attempts": 200,
                "nonempty": sum(row["nonempty"] for row in selected),
                **{
                    name: sum(row["metrics"][name] for row in selected) / 10
                    for name in ("validity", "uniqueness", "quality", "diversity")
                },
            }
        )
    result = {
        "schema": "fragment_superstructure_prefix_caps_audit_v1",
        "evidence_role": "Paired exact-prefix development diagnostic, not a fresh reduced-cap sampler run",
        "caps": list(CAPS),
        "input_sha256": {
            str(SOURCE.relative_to(ROOT)): sha256(SOURCE),
            str(PROMPTS.relative_to(ROOT)): sha256(PROMPTS),
        },
        "official_evaluator_sha256": evaluator_hash,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
                ROOT / "src/compose_v4/benchmark/fragment_constrained.py",
                ROOT / "src/compose_v4/rewrite/kernel.py",
            )
        },
        "auditor_sha256": sha256(Path(__file__)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "software": {"python": sys.version.split()[0], "rdkit": rdkit.__version__},
        "rows": rows,
        "summary": summary,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=FOLDER, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

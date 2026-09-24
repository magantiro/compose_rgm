"""Measure frozen-prior first-event mass on the legal superstructure fiber.

This is a zero-oracle, source-state diagnostic. It does not select an action or
change the sampler; every sampled mark and its executor/lock outcome is saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import rdkit
from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

from compose_v4.benchmark.fragment_attachment_control import AttachmentControlConfig
from compose_v4.benchmark.fragment_conditioned_sampler import (
    RegionLock,
    build_prompt_context,
)
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.model.time_convention import frozen_time
from compose_v4.rewrite.kernel import de_novo_rewrite_system

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
MANIFEST = ROOT / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
OUTPUT = ROOT / "diagnostics/fragment_superstructure_initial_marks_v1/result.json"
DRUGS = ("SPIRAPRIL", "LOVASTATIN", "BARICITINIB")
DRAWS = 256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if digest(CHECKPOINT) != "24117dfeaee91729bb4ebccb5eb5b993b4a6218605045e51917823d086b4c1e4":
        raise RuntimeError("fragment checkpoint hash mismatch")
    model, metadata = load_factorized_rollout_checkpoint(CHECKPOINT)
    system = de_novo_rewrite_system()
    prompts = {
        prompt.drug_name: prompt
        for prompt in load_genmol_prompts(MANIFEST)
        if prompt.task is FragmentTask.SUPERSTRUCTURE_GENERATION
    }
    observations = []
    for drug in DRUGS:
        context = build_prompt_context(prompts[drug], control=AttachmentControlConfig(enabled=True))
        lock = RegionLock(context.start_state, context.locked_slots)
        seed = int.from_bytes(
            hashlib.blake2b(f"superstructure_initial_mark|{drug}".encode(), digest_size=8).digest(),
            "big",
        ) % (2**32)
        rng = np.random.default_rng(seed)
        records = []
        for index in range(DRAWS):
            try:
                mark = model.sample_rewrite_mark(context.start_state, frozen_time(0.0), rng)
            except Exception as error:  # noqa: BLE001 - record every failed sampled mark
                records.append(
                    {
                        "draw": index,
                        "outcome": "sample_exception",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            record = {
                "draw": index,
                "rule": mark.rule_name,
                "action_type": type(mark.action).__name__,
                "action": asdict(mark.action) if is_dataclass(mark.action) else repr(mark.action),
            }
            try:
                successor = system.apply(context.start_state, mark.rule_name, mark.action)
            except Exception as error:  # noqa: BLE001 - record executor refusal exactly
                record["outcome"] = "executor_refusal"
                record["error"] = f"{type(error).__name__}: {error}"
            else:
                record["outcome"] = (
                    "legal_first_event" if lock.permits(successor) else "lock_refusal"
                )
            records.append(record)
        outcomes = Counter(r["outcome"] for r in records)
        families = Counter(r.get("rule", "sample_exception") for r in records)
        legal_families = Counter(r["rule"] for r in records if r["outcome"] == "legal_first_event")
        observations.append(
            {
                "drug": drug,
                "seed": seed,
                "start_smiles": context.start_smiles,
                "locked_slots": list(context.locked_slots),
                "draws": DRAWS,
                "outcomes": dict(sorted(outcomes.items())),
                "families": dict(sorted(families.items())),
                "legal_families": dict(sorted(legal_families.items())),
                "records": records,
            }
        )
    result = {
        "schema": "fragment_superstructure_initial_marks_v1",
        "evidence_role": "Exploratory zero-oracle first-event support audit, not a controller selection gate",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "auditor_sha256": digest(Path(__file__)),
        "checkpoint_sha256": digest(CHECKPOINT),
        "checkpoint_completed_steps": metadata.get("completed_steps"),
        "manifest_sha256": digest(MANIFEST),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "draws_per_prompt": DRAWS,
        "observations": observations,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    print(
        json.dumps(
            [
                {key: row[key] for key in ("drug", "draws", "outcomes", "legal_families")}
                for row in observations
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Zero-oracle motif-extension pilot over the frozen GenMol fragment manifest.

Motif extension is the easiest of the five released fragment-constrained tasks
for COMPOSE: each prompt supplies exactly ONE retained fragment carrying exactly
ONE declared attachment site, so a proposal needs no bridge between separate
components and never has to invent a site.  This driver sweeps the runner's
``variant_index`` over that task and reports the official denominators together
with the constraint diagnostics COMPOSE additionally needs.

Invariants checked here and recorded in the artifact:

* the manifest is the frozen GenMol asset (SHA-256 pinned below);
* every committed program state is padded, never a TIGHT SMILES rebuild, so the
  ``atom_insert`` family stays in the legal support;
* an abstention is scored as an empty sample, so it lowers validity rather than
  silently shrinking the denominator.

No oracle, scoring, Modal, or network call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rdkit import rdBase

from compose_v4.benchmark.fragment_constrained import (
    SCHEMA_VERSION,
    FragmentTask,
    evaluate_prompt,
    load_genmol_prompts,
)
from compose_v4.benchmark.fragment_constrained_runner import (
    ProposalLimits,
    prompt_id,
    propose_prompt,
)
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "compose_fragment_motif_extension_pilot_v1"
DEFAULT_ASSET = Path(
    "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
)
DEFAULT_OUTPUT = Path("diagnostics/fragment_motif_extension_pilot_v1/result.json")
EXPECTED_ASSET_SHA256 = (
    "a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9"
)
MATERIAL_CODE = (
    Path("src/compose_v4/benchmark/fragment_constrained.py"),
    Path("src/compose_v4/benchmark/fragment_constrained_runner.py"),
    Path("scripts/fragment_motif_extension_pilot_v1.py"),
)


# ---- Provenance helpers ----


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    temporary.replace(path)


# ---- Pilot ----


def _slot_evidence(receipt: dict) -> dict:
    """Prove the committed states carry free slots (not a TIGHT SMILES rebuild)."""

    states = tuple(decode_state(state) for state in receipt["states"])
    slots = sorted({int(state.atom_types.shape[0]) for state in states})
    real = sorted({int(state.n_real_atoms) for state in states})
    return {
        "committed_states": len(states),
        "distinct_slot_counts": slots,
        "distinct_real_atom_counts": real,
        "every_state_has_free_slot": all(
            int(state.atom_types.shape[0]) > int(state.n_real_atoms)
            for state in states
        ),
    }


def run_pilot(asset: Path, variants: int, limits: ProposalLimits) -> dict:
    manifest_sha = _sha256(asset)
    if manifest_sha != EXPECTED_ASSET_SHA256:
        raise SystemExit(
            "fragment manifest drift: expected "
            f"{EXPECTED_ASSET_SHA256}, found {manifest_sha}"
        )

    prompts = load_genmol_prompts(asset)
    selected = tuple(p for p in prompts if p.task is FragmentTask.MOTIF_EXTENSION)
    if len(selected) != 10:
        raise SystemExit(f"expected 10 motif-extension prompts, found {len(selected)}")
    for prompt in selected:
        if len(prompt.fragments) != 1:
            raise SystemExit(
                f"{prompt.drug_name}: motif extension must carry one retained fragment"
            )

    per_prompt: list[dict] = []
    proposals: list[dict] = []
    reason_codes: Counter[str] = Counter()
    slot_checks: list[dict] = []

    for prompt in selected:
        samples: list[str] = []
        completed = 0
        for variant_index in range(variants):
            result = propose_prompt(
                prompt, variant_index=variant_index, limits=limits
            )
            if result["status"] == "complete":
                completed += 1
                samples.append(result["endpoint"])
                if variant_index == 0:
                    slot_checks.append(
                        {
                            "drug_name": prompt.drug_name,
                            **_slot_evidence(result["receipt"]),
                        }
                    )
            else:
                # An abstention is an attempted sample that produced nothing.
                samples.append("")
                reason_codes[result.get("reason_code", "unknown")] += 1
            proposals.append(
                {
                    "prompt_id": result["prompt_id"],
                    "drug_name": result["drug_name"],
                    "variant_index": variant_index,
                    "status": result["status"],
                    "endpoint": result.get("endpoint"),
                    "endpoint_active_atoms": result.get("endpoint_active_atoms"),
                    "reason_code": result.get("reason_code"),
                    "construction": result.get("construction"),
                    "oracle_calls": result["oracle_calls"],
                    "scoring_calls": result["scoring_calls"],
                }
            )

        metrics = evaluate_prompt(prompt, samples, expected_samples=variants)
        distinct_endpoints = sorted({s for s in samples if s})
        per_prompt.append(
            {
                "prompt_id": prompt_id(prompt),
                "drug_name": prompt.drug_name,
                "retained_fragment": prompt.fragments[0],
                "attempts": variants,
                "completed": completed,
                "abstained": variants - completed,
                "distinct_raw_endpoints": len(distinct_endpoints),
                "metrics": asdict(metrics),
            }
        )

    attempts = len(selected) * variants
    completed_total = sum(row["completed"] for row in per_prompt)
    valid_total = sum(row["metrics"]["valid_count"] for row in per_prompt)
    compliant_total = sum(
        row["metrics"]["constraint_valid_count"] for row in per_prompt
    )
    unique_total = sum(
        row["metrics"]["unique_constraint_valid_count"] for row in per_prompt
    )
    quality_total = sum(row["metrics"]["quality_count"] for row in per_prompt)

    def _mean(key: str) -> float:
        return float(sum(row["metrics"][key] for row in per_prompt) / len(per_prompt))

    overall = {
        "prompts": len(selected),
        "attempts_per_prompt": variants,
        "attempts_total": attempts,
        "completed_total": completed_total,
        "abstained_total": attempts - completed_total,
        # Pooled counts (single ratio over all attempts).
        "pooled_chemical_validity": valid_total / attempts,
        "pooled_benchmark_validity": compliant_total / attempts,
        "pooled_constraint_precision_given_valid": compliant_total
        / max(valid_total, 1),
        "pooled_uniqueness": unique_total / max(compliant_total, 1),
        "pooled_quality": quality_total / attempts,
        # Per-drug means, which is the shape the published tables report.
        "mean_chemical_validity": _mean("chemical_validity"),
        "mean_benchmark_validity": _mean("benchmark_validity"),
        "mean_constraint_validity": _mean("constraint_validity"),
        "mean_uniqueness": _mean("uniqueness"),
        "mean_diversity": _mean("diversity"),
        "mean_quality": _mean("quality"),
        "mean_central_distance": _mean("central_distance"),
        "valid_count": valid_total,
        "constraint_valid_count": compliant_total,
        "unique_constraint_valid_count": unique_total,
        "quality_count": quality_total,
        "abstention_reason_codes": dict(reason_codes),
    }

    return {
        "schema_version": SCHEMA,
        "adapter_schema_version": SCHEMA_VERSION,
        "task": FragmentTask.MOTIF_EXTENSION.value,
        "task_selection_rationale": (
            "motif extension supplies exactly one retained fragment with exactly one "
            "declared attachment site, so it needs neither an inter-component bridge "
            "(linker design / scaffold morphing) nor an invented site "
            "(superstructure generation)"
        ),
        "evidence_class": "zero_oracle_local_support_and_constraint_pilot",
        "claim_boundary": (
            "This is NOT the released benchmark result. The released protocol is 100 "
            "samples per drug and task over three seeds from a learned generator. "
            "This pilot sweeps a deterministic, non-learned variant law, so its "
            "uniqueness and diversity are properties of that placeholder proposal "
            "law and are NOT comparable to SAFE-GPT, GenMol, or InVirtuoGen."
        ),
        "configuration": {
            "variants_per_prompt": variants,
            "limits": asdict(limits),
            "abstention_scored_as_empty_sample": True,
        },
        "inputs": {
            "manifest_path": str(asset),
            "manifest_sha256": manifest_sha,
            "manifest_sha256_matches_frozen": True,
        },
        "state_semantics": {
            "assert_production_state_semantics_applicable": False,
            "reason": (
                "that preflight compares a trained model's realized legal-family "
                "census against a corpus-recorded census; this path has neither a "
                "model nor a compiler-recorded census, so the padding invariant is "
                "checked directly instead"
            ),
            "tight_graph_gotcha_avoided": all(
                row["every_state_has_free_slot"] for row in slot_checks
            ),
            "per_prompt_slot_evidence": slot_checks,
        },
        "overall": overall,
        "by_prompt": per_prompt,
        "proposals": proposals,
        "provenance": {
            "git_revision": _revision(),
            "rdkit_version": rdBase.rdkitVersion,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "material_code_sha256": {
                str(path): _sha256(path) for path in MATERIAL_CODE if path.exists()
            },
            "oracle_calls": 0,
            "scoring_calls": 0,
            "network_calls": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=12)
    parser.add_argument("--max-active-atoms", type=int, default=40)
    parser.add_argument("--max-primitives", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=8)
    args = parser.parse_args()

    if args.variants < 1:
        raise SystemExit("--variants must be positive")

    limits = ProposalLimits(
        max_active_atoms=args.max_active_atoms,
        max_primitives=args.max_primitives,
        max_blocks=args.max_blocks,
    )
    payload = run_pilot(args.asset, args.variants, limits)
    _atomic_json(args.output, payload)

    overall = payload["overall"]
    print(f"wrote {args.output}")
    print(
        f"attempts {overall['attempts_total']} "
        f"completed {overall['completed_total']} "
        f"abstained {overall['abstained_total']}"
    )
    print(
        f"mean chemical validity   {overall['mean_chemical_validity']:.4f}\n"
        f"mean benchmark validity  {overall['mean_benchmark_validity']:.4f}\n"
        f"mean constraint validity {overall['mean_constraint_validity']:.4f}\n"
        f"mean uniqueness          {overall['mean_uniqueness']:.4f}\n"
        f"mean diversity           {overall['mean_diversity']:.4f}\n"
        f"mean quality             {overall['mean_quality']:.4f}"
    )


if __name__ == "__main__":
    main()

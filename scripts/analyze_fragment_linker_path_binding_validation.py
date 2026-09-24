"""Adjudicate the frozen fresh-seed linker binding signal without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(artifact: dict, *, seed: int) -> dict[str, dict]:
    result = artifact["results"]["linker_design"]
    if result["build_failures"]:
        raise ValueError(f"prompt construction failed: {result['build_failures']}")
    if result["prompts_declared"] != result["prompts_scored"]:
        raise ValueError("not every declared prompt was scored")
    output = {}
    for drug, entries in result["per_drug"].items():
        if len(entries) != 1 or entries[0]["seed"] != seed:
            raise ValueError(f"expected exactly seed {seed} for {drug}")
        output[drug] = entries[0]
    return output


def mean(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)


def summarize(prompt_rows: dict[str, dict]) -> dict:
    attempts = sum(row["attempts"] for row in prompt_rows.values())
    commits = sum(row["committed_endpoints"] for row in prompt_rows.values())
    valid_commits = sum(row["committed_chemically_valid"] for row in prompt_rows.values())
    preserving = sum(row["committed_fragment_preserving"] for row in prompt_rows.values())
    emitted = sum(row["emitted_nonempty"] for row in prompt_rows.values())
    lengths: Counter[int] = Counter()
    genuine = 0
    per_prompt = {}
    census: Counter[str] = Counter()
    for drug, row in sorted(prompt_rows.items()):
        if row["attempts"] != 20:
            raise ValueError(f"wrong attempt count for {drug}: {row['attempts']}")
        if len(row["emitted_samples"]) != 20:
            raise ValueError(f"missing attempt-level emitted molecules for {drug}")
        if len(row["committed_endpoint_smiles"]) != row["committed_endpoints"]:
            raise ValueError(f"missing committed molecules for {drug}")
        local_lengths = row["realized_linker_lengths"]
        if len(local_lengths) != row["committed_endpoints"]:
            raise ValueError(f"linker length/commit mismatch for {drug}")
        seeded = row["seeded_linker_length"]
        local_genuine = sum(length > seeded for length in local_lengths)
        genuine += local_genuine
        lengths.update(local_lengths)
        census.update(row.get("path_census", {}))
        per_prompt[drug] = {
            "attempts": row["attempts"],
            "commits": row["committed_endpoints"],
            "valid_commits": row["committed_chemically_valid"],
            "fragment_containing_commits": row["committed_fragment_preserving"],
            "emitted_nonempty": row["emitted_nonempty"],
            "full_task_successes": (
                local_genuine
                if row["committed_fragment_preserving"] == row["committed_endpoints"]
                else None
            ),
            "seeded_linker_length": seeded,
            "realized_linker_length_histogram": dict(sorted(Counter(local_lengths).items())),
            "genuine_linkers_longer_than_seed": local_genuine,
            "official": row["official"],
        }
    if commits != sum(lengths.values()):
        raise ValueError("linker length histogram does not cover all commits")
    return {
        "prompt_count": len(prompt_rows),
        "attempts": attempts,
        "commits": commits,
        "valid_commits": valid_commits,
        "committed_exact_chemical_validity": valid_commits / commits if commits else None,
        "fragment_containing_commits": preserving,
        "emitted_nonempty": emitted,
        "full_task_successes": (genuine if preserving == commits else None),
        "genuine_linkers_longer_than_seed": genuine,
        "genuine_linker_rate_per_attempt": genuine / attempts,
        "realized_linker_length_histogram": dict(sorted(lengths.items())),
        "official_quality_fraction": mean(
            [row["official"]["quality"] / 100 for row in prompt_rows.values()]
        ),
        "official_uniqueness_fraction": mean(
            [row["official"]["uniqueness"] / 100 for row in prompt_rows.values()]
        ),
        "official_diversity": mean([row["official"]["diversity"] for row in prompt_rows.values()]),
        "per_prompt": per_prompt,
        "path_refusal_census": dict(sorted(census.items())),
    }


def check_identity(contract: dict, old: dict, new: dict, old_dir: Path, new_dir: Path) -> None:
    expected = contract["payload"]
    payload_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload_hash != contract["payload_sha256"]:
        raise ValueError("frozen validation contract payload hash changed")
    protocol = expected["protocol"]
    for name, artifact, source_dir in (
        ("old_site_matching", old, old_dir),
        ("constraint_bound", new, new_dir),
    ):
        arm = expected["arms"][name]
        for key, relative in (
            ("sampler_sha256", "src/compose_v4/benchmark/fragment_conditioned_sampler.py"),
            ("runner_sha256", "tools/run_fragment_constrained_suite.py"),
        ):
            observed = digest(source_dir / relative)
            if observed != arm[key]:
                raise ValueError(f"{name} {key} changed: {observed} != {arm[key]}")
        if artifact["protocol"]["linker_bridge_atoms"] != protocol["seeded_bridge_atoms"]:
            raise ValueError(f"{name} seeded bridge changed")
        if artifact["protocol"]["samples_per_prompt"] != protocol["attempts_per_prompt"]:
            raise ValueError(f"{name} attempt count changed")
        if not artifact["attachment_control"]["config"]["enabled"]:
            raise ValueError(f"{name} attachment control is off")
        if not artifact["attachment_control"]["config"]["path_program"]:
            raise ValueError(f"{name} path program is off")
        if (
            artifact["sampler"]["config"]["mark_attempts_per_event"]
            != protocol["mark_attempts_per_event"]
        ):
            raise ValueError(f"{name} event attempt budget changed")
        prompt_manifest = (
            source_dir / "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
        )
        if digest(prompt_manifest) != expected["inputs"]["prompt_manifest_sha256"]:
            raise ValueError(f"{name} prompt manifest changed")
        official_metrics = (
            source_dir / ".official_eval_cache/pkg/in_virtuo_gen/train_utils/metrics.py"
        )
        if digest(official_metrics) != expected["inputs"]["official_ivg_metrics_sha256"]:
            raise ValueError(f"{name} official evaluator changed")
    for key in ("checkpoint", "protocol", "sampler", "attachment_control"):
        if old[key] != new[key]:
            raise ValueError(f"arms differ in {key}")
    if digest(Path(old["checkpoint"]["path"])) != expected["inputs"]["checkpoint_sha256"]:
        raise ValueError("checkpoint physical hash changed")


def publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    check_identity(contract, old, new, args.old_source, args.new_source)
    protocol = contract["payload"]["protocol"]
    seed_ids = protocol["seed_ids"]
    if len(seed_ids) != 1:
        raise ValueError("this validator requires exactly one frozen seed")
    baseline_rows = rows(old, seed=seed_ids[0])
    treatment_rows = rows(new, seed=seed_ids[0])
    if (
        baseline_rows.keys() != treatment_rows.keys()
        or len(baseline_rows) != protocol["prompt_count"]
    ):
        raise ValueError("arms do not cover the same frozen prompt set")
    baseline = summarize(baseline_rows)
    treatment = summarize(treatment_rows)
    gate = contract["payload"]["signal_gate"]
    deltas = {
        key: treatment[key] - baseline[key]
        for key in (
            "genuine_linker_rate_per_attempt",
            "official_quality_fraction",
            "official_uniqueness_fraction",
            "official_diversity",
        )
    }
    numeric_gates = {
        "committed_exact_chemical_validity": (
            treatment["committed_exact_chemical_validity"]
            == gate["committed_exact_chemical_validity"]
        ),
        "genuine_linker_rate": (
            deltas["genuine_linker_rate_per_attempt"]
            >= gate["newly_bound_genuine_linker_rate_gain_min_absolute"]
        ),
        "official_quality": (
            deltas["official_quality_fraction"] >= -gate["official_quality_loss_max_absolute"]
        ),
        "official_uniqueness": (
            deltas["official_uniqueness_fraction"]
            >= -gate["mean_emitted_uniqueness_loss_max_absolute"]
        ),
        "official_diversity": (
            deltas["official_diversity"] >= -gate["mean_emitted_diversity_loss_max_absolute"]
        ),
    }
    payload = {
        "schema": "compose.fragment_linker_path_binding_validation_result.v1",
        "status": "development signal only, not an official benchmark row",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "software": {"python": platform.python_version()},
        "inputs": {
            key: {"path": str(path), "sha256": digest(path)}
            for key, path in {
                "contract": args.contract,
                "old_artifact": args.old,
                "new_artifact": args.new,
                "analyzer": Path(__file__),
            }.items()
        },
        "units": "quality and uniqueness converted from official percent to fractions; diversity is already a fraction",
        "baseline": baseline,
        "treatment": treatment,
        "deltas": deltas,
        "numeric_gates": numeric_gates,
        "numeric_signal_pass": all(numeric_gates.values()),
        "locked_graph_gate": (
            "not directly observable from canonical SMILES; production RegionLock "
            "invariant and focused tests are separate evidence"
        ),
        "interpretation_limit": (
            "A numeric pass qualifies only a larger frozen test, not an official "
            "paper row; the one-atom bridge was supplied by the harness."
        ),
    }
    publish(args.output, payload)
    print(json.dumps({"numeric_gates": numeric_gates, "deltas": deltas}, sort_keys=True))


if __name__ == "__main__":
    main()

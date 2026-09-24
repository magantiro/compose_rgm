"""Audit the predeclared three-prompt linker path-binding development signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

PROMPTS = ("eliglustat", "futibatinib", "lovastatin")
SCHEMA = "compose_fragment_linker_path_binding_dev_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(payload: dict, name: str) -> dict:
    return payload["results"]["linker_design"]["per_drug"][name.upper()][0]


def _load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing linker artifact: {path}")
    return json.loads(path.read_text())


def _qualified_count(row: dict) -> int:
    count = float(row["official"]["quality"]) * int(row["attempts"]) / 100.0
    rounded = round(count)
    if abs(count - rounded) > 1e-8:
        raise ValueError(f"nonintegral qualified count from official quality: {count}")
    return rounded


def _require_frozen_prefix(old: dict, original: dict, name: str) -> None:
    n_attempts = len(old["emitted_samples"])
    if old["emitted_samples"] != original["emitted_samples"][:n_attempts]:
        raise ValueError(f"{name}: census control does not replay frozen first ten samples")


def analyze(before_dir: Path, after_dir: Path, frozen_dir: Path, root: Path) -> dict:
    inputs: dict[str, str] = {}
    rows: list[dict] = []
    for name in PROMPTS:
        before_path = before_dir / f"{name}_seed0_10.json"
        after_path = after_dir / f"after_{name}_seed0_10.json"
        frozen_path = frozen_dir / f"linker_design__{name.upper()}__seed0.json"
        before, after, frozen = map(_load, (before_path, after_path, frozen_path))
        inputs[str(before_path)] = sha256(before_path)
        inputs[str(after_path)] = sha256(after_path)
        inputs[str(frozen_path)] = sha256(frozen_path)
        old = _row(before, name)
        new = _row(after, name)
        original = _row(frozen, name)
        _require_frozen_prefix(old, original, name)
        for field in ("rng_seed", "attempts", "seeded_linker_length"):
            if old[field] != new[field]:
                raise ValueError(f"{name}: matched {field} differs across arms")
        for field in ("sampler", "attachment_control", "protocol"):
            if before[field] != after[field]:
                raise ValueError(f"{name}: matched {field} differs across arms")
        if old["attempts"] != 10 or new["attempts"] != 10:
            raise ValueError(f"{name}: expected ten attempts per arm")
        if any(length <= 1 for length in new["realized_linker_lengths"]):
            raise ValueError(f"{name}: candidate committed a seed-length linker")
        rows.append(
            {
                "prompt": name,
                "before": {
                    "committed": old["committed_endpoints"],
                    "chemically_valid_commits": old["committed_chemically_valid"],
                    "qualified_emitted": _qualified_count(old),
                    "unique_valid": old["unique_valid_count"],
                    "path_payload_absent": old["path_census"]["path_payload_absent"],
                    "path_transactions": old["path_census"]["path_transactions"],
                    "genuine_linkers": sum(x > 1 for x in old["realized_linker_lengths"]),
                },
                "after": {
                    "committed": new["committed_endpoints"],
                    "chemically_valid_commits": new["committed_chemically_valid"],
                    "qualified_emitted": _qualified_count(new),
                    "unique_valid": new["unique_valid_count"],
                    "path_payload_absent": new["path_census"]["path_payload_absent"],
                    "path_transactions": new["path_census"]["path_transactions"],
                    "genuine_linkers": sum(x > 1 for x in new["realized_linker_lengths"]),
                },
            }
        )
    totals = {
        arm: {key: sum(row[arm][key] for row in rows) for key in rows[0][arm]}
        for arm in ("before", "after")
    }
    signal_pass = (
        totals["after"]["path_payload_absent"] < totals["before"]["path_payload_absent"]
        and totals["after"]["path_transactions"] > totals["before"]["path_transactions"]
        and totals["after"]["genuine_linkers"] > totals["before"]["genuine_linkers"]
        and all(
            row["after"]["committed"] == row["after"]["chemically_valid_commits"] for row in rows
        )
        and all(
            row["after"]["qualified_emitted"] >= row["before"]["qualified_emitted"] - 1
            for row in rows
        )
    )
    implementation_paths = (
        "src/compose_v4/benchmark/fragment_conditioned_sampler.py",
        "src/compose_v4/benchmark/fragment_attachment_control.py",
        "tools/run_fragment_constrained_suite.py",
        "src/compose_v4/benchmark/fragment_official_metrics.py",
        "docs/FRAGMENT_LINKER_PATH_BINDING_DEV_2026-09-24.md",
        "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv",
        ".official_eval_cache/pkg/in_virtuo_gen/train_utils/metrics.py",
    )
    checkpoint_path = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
    return {
        "schema": SCHEMA,
        "evidence_class": "selected_prompt_same_seed_development_signal",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "input_sha256": dict(sorted(inputs.items())),
        "implementation_and_asset_sha256": {
            path: sha256(root / path) for path in implementation_paths
        },
        "checkpoint_sha256": sha256(checkpoint_path),
        "python_version": platform.python_version(),
        "attempts_per_arm": 30,
        "rows": rows,
        "totals": totals,
        "predeclared_signal_pass": signal_pass,
        "limitations": [
            "Three prompts were selected after seeing the original path failure.",
            "Ten attempts per prompt are insufficient for an official benchmark claim.",
            "The one-atom bridge was constructed before sampling and is not a generated linker.",
            "The old census-control implementation lacks a physical source hash; its emitted-prefix identity is verified against frozen shards.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite linker analysis: {args.output}")
    root = Path(__file__).resolve().parents[1]
    payload = analyze(args.before_dir, args.after_dir, args.frozen_dir, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary linker analysis: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {"signal_pass": payload["predeclared_signal_pass"], "totals": payload["totals"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

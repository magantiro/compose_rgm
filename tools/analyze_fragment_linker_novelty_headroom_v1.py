"""Measure score-only novelty headroom on the locked seed-zero linker pilot.

This is a one-step diagnostic conditional on the *observed* selection history.
It does not replay the generator, evaluate molecular quality, or predict a
counterfactual full-run uniqueness metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_supported_offers(panel: dict) -> list[tuple[str, float]]:
    kept: dict[str, float] = {}
    for offer in panel["offered"]:
        endpoint = offer.get("endpoint")
        score = offer.get("mean_log_mark")
        if (
            offer["status"] == "model_supported"
            and isinstance(endpoint, str)
            and endpoint
            and isinstance(score, (int, float))
            and math.isfinite(score)
        ):
            kept.setdefault(endpoint, float(score))
    return list(kept.items())


def unseen_probability(
    offers: list[tuple[str, float]], previous: set[str], novelty_multiplier: float
) -> float:
    if not offers:
        return 0.0
    maximum = max(score for _, score in offers)
    weights = [
        math.exp(score - maximum) * (novelty_multiplier if endpoint not in previous else 1.0)
        for endpoint, score in offers
    ]
    return sum(
        weight for (endpoint, _), weight in zip(offers, weights, strict=True)
        if endpoint not in previous
    ) / sum(weights)


def summarize_prompt(attempts: list[dict]) -> dict:
    previous: set[str] = set()
    rows = []
    for index, attempt in enumerate(attempts):
        if attempt["attempt_index"] != index:
            raise ValueError("pilot attempt order or index changed")
        panel = attempt["panel"]
        offers = unique_supported_offers(panel)
        selected = panel["selected_smiles"]
        if selected is not None and selected not in dict(offers):
            raise ValueError("selected endpoint absent from finite supported panel")
        repeated = selected is not None and selected in previous
        available_unseen = any(endpoint not in previous for endpoint, _ in offers)
        rows.append(
            {
                "attempt_index": index,
                "supported_distinct_offers": len(offers),
                "unseen_distinct_offers": sum(
                    endpoint not in previous for endpoint, _ in offers
                ),
                "selected_repeated": repeated,
                "selected_repeated_despite_unseen_offer": repeated and available_unseen,
                "conditional_p_unseen_frozen": unseen_probability(offers, previous, 1.0),
                "conditional_p_unseen_novelty4": unseen_probability(offers, previous, 4.0),
            }
        )
        if selected is not None:
            previous.add(selected)
    return {
        "attempts": len(rows),
        "selected_distinct": len(previous),
        "selected_repeated": sum(row["selected_repeated"] for row in rows),
        "selected_repeated_despite_unseen_offer": sum(
            row["selected_repeated_despite_unseen_offer"] for row in rows
        ),
        "mean_supported_distinct_offers": sum(
            row["supported_distinct_offers"] for row in rows
        ) / len(rows),
        "mean_unseen_distinct_offers": sum(
            row["unseen_distinct_offers"] for row in rows
        ) / len(rows),
        "conditional_expected_unseen_frozen": sum(
            row["conditional_p_unseen_frozen"] for row in rows
        ),
        "conditional_expected_unseen_novelty4": sum(
            row["conditional_p_unseen_novelty4"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("attempts/*.json"))
    if len(paths) != 200:
        raise ValueError(f"expected exactly 200 seed-zero pilot attempts, found {len(paths)}")
    grouped: dict[str, list[dict]] = {}
    hashes = {}
    for path in paths:
        attempt = json.loads(path.read_text())
        drug = attempt["drug"]
        if path.stem != f"{drug}_{attempt['attempt_index']:03d}":
            raise ValueError(f"attempt identity mismatch: {path}")
        grouped.setdefault(drug, []).append(attempt)
        hashes[path.name] = sha256(path)
    if len(grouped) != 10 or any(len(attempts) != 20 for attempts in grouped.values()):
        raise ValueError("expected ten complete, 20-attempt development prompts")
    by_prompt = {
        drug: summarize_prompt(sorted(attempts, key=lambda row: row["attempt_index"]))
        for drug, attempts in sorted(grouped.items())
    }
    totals = {
        key: sum(row[key] for row in by_prompt.values())
        for key in (
            "attempts",
            "selected_distinct",
            "selected_repeated",
            "selected_repeated_despite_unseen_offer",
            "conditional_expected_unseen_frozen",
            "conditional_expected_unseen_novelty4",
        )
    }
    result = {
        "schema": "fragment_linker_novelty_headroom_v1",
        "role": "locked seed-zero development diagnostic; not a counterfactual benchmark result",
        "input_dir": str(args.input_dir.resolve()),
        "input_sha256": dict(sorted(hashes.items())),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "config": {"novelty_multiplier": 4.0, "expected_prompts": 10, "attempts_per_prompt": 20},
        "software": {"python": __import__("platform").python_version()},
        "totals": totals,
        "per_prompt": by_prompt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"novelty diagnostic output exists: {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()

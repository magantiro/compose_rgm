"""Reconcile the frozen superstructure validity row without regenerating samples.

The historical shards retain counts but not attempt-aligned traces or all SMILES.
Consequently this audit can identify no-commit attempts exactly, but cannot
retroactively assign each one to zero-event versus rejection exhaustion.
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

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "diagnostics/fragment_official_suite_v2/shards"
OUTPUT = ROOT / "diagnostics/fragment_superstructure_gap_audit_v1/result.json"
TASK = "superstructure_generation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    verified = verify_only()
    evaluator_hash = OFFICIAL_BLOBS["in_virtuo_gen/train_utils/metrics.py"][1]
    if evaluator_hash not in verified.values():
        raise RuntimeError("pinned official fragment evaluator not verified")

    rows: list[dict] = []
    shard_hashes: dict[str, str] = {}
    sampler_hashes: set[str] = set()
    for path in sorted(SHARDS.glob("superstructure_generation__*.json")):
        shard_hashes[str(path.relative_to(ROOT))] = sha256(path)
        payload = json.loads(path.read_text())
        sampler_hashes.add(payload["sampler"]["config_sha256"])
        result = payload["results"][TASK]
        if result["build_failures"]:
            raise RuntimeError(f"build failures in {path}")
        for drug, drug_rows in result["per_drug"].items():
            for source in drug_rows:
                attempts = int(source["attempts"])
                committed = int(source["committed_endpoints"])
                valid = int(source["committed_chemically_valid"])
                preserving = int(source["committed_fragment_preserving"])
                emitted = int(source["emitted_nonempty"])
                official_validity = float(source["official"]["validity"])
                if attempts != 100 or int(source["validity_denominator"]) != attempts:
                    raise RuntimeError(f"unexpected official denominator: {path}")
                if not (0 <= emitted <= preserving <= valid <= committed <= attempts):
                    raise RuntimeError(f"inconsistent counts in {path}")
                if abs(official_validity - 100.0 * emitted / attempts) > 1e-10:
                    raise RuntimeError(f"official validity/count mismatch in {path}")
                if int(source["constraint_failures"]) != committed - emitted:
                    raise RuntimeError(f"constraint-failure count mismatch in {path}")
                rows.append(
                    {
                        "drug": drug,
                        "seed": int(source["seed"]),
                        "attempts": attempts,
                        "no_committed_endpoint": attempts - committed,
                        "committed": committed,
                        "chemically_invalid_committed": committed - valid,
                        "chemically_valid_committed": valid,
                        "fragment_preservation_failures": committed - preserving,
                        "fragment_preserving_committed": preserving,
                        "official_emitted": emitted,
                        "official_validity_pct": official_validity,
                        "trajectories_hitting_rejection_budget": int(source["budget_exhausted"]),
                        "mean_events": float(source["mean_events"]),
                    }
                )
    if len(rows) != 30 or len({(r["drug"], r["seed"]) for r in rows}) != 30:
        raise RuntimeError("expected exactly 10 drugs x 3 seeds")
    if len(sampler_hashes) != 1:
        raise RuntimeError("shards disagree on sampler configuration")
    if sorted({r["seed"] for r in rows}) != [0, 1, 2]:
        raise RuntimeError("expected three frozen seeds")

    count_keys = (
        "attempts",
        "no_committed_endpoint",
        "committed",
        "chemically_invalid_committed",
        "chemically_valid_committed",
        "fragment_preservation_failures",
        "fragment_preserving_committed",
        "official_emitted",
        "trajectories_hitting_rejection_budget",
    )
    totals = {key: sum(row[key] for row in rows) for key in count_keys}
    if totals["attempts"] != 3000:
        raise RuntimeError("historical superstructure denominator changed")
    if totals["no_committed_endpoint"] + totals["committed"] != totals["attempts"]:
        raise RuntimeError("attempt accounting does not close")
    if totals["chemically_invalid_committed"] != 0:
        raise RuntimeError("unexpected invalid committed molecule")
    if totals["committed"] - totals["fragment_preservation_failures"] != totals["official_emitted"]:
        raise RuntimeError("official emission/constraint accounting does not close")

    # The official code has validity = valid_count / len(all_smiles). It counts
    # our empty failure placeholder as invalid. Verify that branch directly
    # with the pinned code, independently of the historical aggregate values.
    package_root = ROOT / ".official_eval_cache/pkg"
    sys.path.insert(0, str(package_root))
    from in_virtuo_gen.train_utils.metrics import ids_to_smiles

    class IdentityTokenizer:
        @staticmethod
        def decode(value: str) -> str:
            return value

    for emitted in (0, 47, 95, 100):
        samples = ["CC"] * emitted + [""] * (100 - emitted)
        all_smiles, valid_smiles, *_ = ids_to_smiles(
            samples, print_flag=False, tokenizer=IdentityTokenizer(), already_smiles=True
        )
        if len(all_smiles) != 100 or len(valid_smiles) != emitted:
            raise RuntimeError("official evaluator denominator probe failed")

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows.sort(key=lambda row: (row["drug"], row["seed"]))
    return {
        "schema": "fragment_superstructure_gap_audit_v1",
        "evidence_role": "Retrospective audit of frozen official-suite aggregate shards; no new sampling",
        "input_sha256": shard_hashes,
        "official_evaluator": {
            "commit": "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb",
            "path": "in_virtuo_gen/train_utils/metrics.py",
            "sha256": evaluator_hash,
            "validity_definition": "valid parsed SMILES / all attempted samples, including failure placeholders",
            "denominator_probe_valid_counts": [0, 47, 95, 100],
        },
        "code": {
            "git_head": git_head,
            "auditor_sha256": sha256(Path(__file__)),
            "sampler_config_sha256": sampler_hashes.pop(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
            "hardware": "offline integer aggregation; hardware-independent",
        },
        "totals": totals,
        "rates_pct": {
            "official_validity_per_attempt": 100.0
            * totals["official_emitted"]
            / totals["attempts"],
            "chemical_validity_per_attempt": 100.0
            * totals["chemically_valid_committed"]
            / totals["attempts"],
            "chemical_validity_per_committed": 100.0
            * totals["chemically_valid_committed"]
            / totals["committed"],
            "preservation_per_committed": 100.0
            * totals["fragment_preserving_committed"]
            / totals["committed"],
        },
        "rows": rows,
        "limit": (
            "Historical shards store aggregate rejection-budget counts and mean events, "
            "not attempt-aligned events/refusal reasons or all committed SMILES. "
            "The exact split of no-commit attempts into zero-event, budget-exhausted, "
            "or serialization failure cannot be recovered. Budget exhaustion is not "
            "disjoint from successful trajectories."
        ),
    }


def main() -> None:
    payload = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"totals": payload["totals"], "rates_pct": payload["rates_pct"]}, indent=2))


if __name__ == "__main__":
    main()

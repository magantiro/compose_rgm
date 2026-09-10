"""Audit the complete locked docking batch and describe its molecular graphs."""

import inspect
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer
from scipy.stats import spearmanr

from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.ivg_winner_paths import ROOT as SOURCE_ROOT
from tools.ivg_winner_paths import implementation_closure
from tools.t4_compare_winners import describe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run():
    folder = HERE / "attempt_1/run"
    result = json.loads((folder / "result.json").read_text())
    receipt = json.loads((HERE / "attempt_1/launch.json").read_text())
    contract_path = ROOT / "configs/t4_proposal_docking.json"
    contract = json.loads(contract_path.read_text())
    verify_file(contract_path, receipt["task"]["contract_sha256"])
    assert result["configuration"] == contract
    assert result["code_revision"] == receipt["task"]["image_revision"]["commit"]
    assert rdBase.rdkitVersion == contract["required_rdkit"]
    assert result["status"] == "complete"
    assert not (folder / "failure.json").exists()
    assert result["runtime_gate"]["input_sha256"] == contract["expected_input_sha256"]
    lock_path = folder / "candidate_lock.json"
    verify_file(lock_path, result["candidate_lock_sha256"])
    lock = unseal(lock_path)
    rows = result["docked"]
    assert len(rows) == result["new_oracle_attempts"] == 16
    assert {r["smiles"] for r in rows} == set(contract["authorized_smiles"])
    assert result["new_generator_calls"] == 0
    assert result["cumulative_source_and_diagnostic_calls"] == 67
    assert not result["optimizer_round_completed"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed = lock["task"]["smiles"]
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(seed))
    prior = result["best_prior"]
    annotation, families, fingerprints = [], Counter(), []
    for index, (row, candidate) in enumerate(zip(rows, lock["take"], strict=True)):
        saved = unseal(folder / f"rows/{index:02d}/result.json")
        started = json.loads((folder / f"rows/{index:02d}/started.json").read_text())
        assert all(row[k] == v for k, v in candidate.items())
        assert all(row[k] == v for k, v in saved.items())
        assert all(saved[k] == v for k, v in started.items())
        assert row["index"] == index
        assert row["candidate_lock_sha256"] == result["candidate_lock_sha256"]
        assert lock["locked_at_utc"] < row["started_at_utc"] <= row["completed_at_utc"]
        assert acceptable_endpoint(row)
        assert canonical_state_key(decode_state(row["state"])) == row["smiles"]
        structure = describe(row["smiles"], seed_fp, 0.4)
        assert structure["feasible"]
        for key, target in (("qed", "qed"), ("sa", "sa"), ("sim", "similarity_to_seed")):
            assert abs(row[key] - structure[target]) < 1e-10
        origins = []
        for origin in row["origins"]:
            parent = describe(origin["source_smiles"], seed_fp, 0.4)
            changes = [w["model_family"] for w in origin["witnesses"]]
            families.update(changes)
            origins.append(
                {
                    "parent_smiles": origin["source_smiles"],
                    "witness_families": changes,
                    "deltas": {
                        k: structure[k] - parent[k]
                        for k in ("cycle_rank", "ring_system_count", "heavy_atoms")
                    },
                    "parent_structure": parent,
                }
            )
        annotation.append(
            {
                "index": index,
                "smiles": row["smiles"],
                "observed_docking": row["ds"],
                "predicted_docking": row["predicted_docking"],
                "prediction_rank_min": 1
                + sum(r["predicted_docking"] < row["predicted_docking"] for r in rows),
                "observed_rank_min": None
                if row["ds"] is None
                else 1 + sum(r["ds"] is not None and r["ds"] < row["ds"] for r in rows),
                "docking_seconds": row["docking_seconds"],
                "structure": structure,
                "origins": origins,
            }
        )
        fingerprints.append(generator.GetFingerprint(Chem.MolFromSmiles(row["smiles"])))
    values = [r for r in annotation if r["observed_docking"] is not None]
    assert len(rows) - len(values) == result["oracle_failures"]
    distances = [
        1 - DataStructs.TanimotoSimilarity(a, b)
        for i, a in enumerate(fingerprints)
        for b in fingerprints[i + 1 :]
    ]
    route_path = ROOT / "diagnostics/t4_route_diagnosis/result.json"
    verify_file(route_path, "6dd20679073978791ec6bd97a9583a203cf92810b7695124a8976cd2f85480fa")
    winner = json.loads(route_path.read_text())["route"][-1]["smiles"]
    inputs = [contract_path, route_path, *sorted((HERE / "attempt_1").rglob("*.json"))]
    return {
        "schema_version": "t4_proposal_docking_audit_v1",
        "input_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs},
        "analysis_sha256": sha256_file(Path(__file__)),
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_root": str(SOURCE_ROOT),
        "source_closure": {
            path: digest
            for helper in (describe, canonical_state_key, unseal)
            for path, digest in implementation_closure(Path(inspect.getfile(helper))).items()
        },
        "sa_assets_sha256": {
            p.name: sha256_file(p)
            for p in (Path(sascorer.__file__), Path(sascorer.__file__).with_name("fpscores.pkl.gz"))
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "accelerator": None,
            "analysis_workers": 1,
            "precision": "float64",
        },
        "configuration": {
            "sample": "complete saved repair census",
            "randomness": "none in audit; inherited unseeded docking",
            "fingerprint": {"radius": 2, "bits": 2048},
            "exclusions": [],
        },
        "evidence_role": "inspected development diagnostic, no fitting or winner-informed selection",
        "new_oracle_attempts": 16,
        "failures": result["oracle_failures"],
        "best_prior": {k: prior[k] for k in ("smiles", "ds")},
        "best_batch": min(values, key=lambda r: r["observed_docking"], default=None),
        "count_better_than_prior": sum(r["observed_docking"] < prior["ds"] for r in values),
        "spearman_predicted_observed": float(
            spearmanr(
                [r["predicted_docking"] for r in values], [r["observed_docking"] for r in values]
            ).statistic
        ),
        "mean_pairwise_fingerprint_distance": float(np.mean(distances)),
        "canonical_unique": len({r["smiles"] for r in rows}),
        "witness_family_counts": dict(sorted(families.items())),
        "rows": annotation,
        "original_seed": describe(seed, seed_fp, 0.4),
        "incumbent": describe(prior["smiles"], seed_fp, 0.4),
        "known_winner_diagnostic_only": describe(winner, seed_fp, 0.4),
        "remote_seconds": result["elapsed_seconds"],
        "docking_phase_seconds": result["docking_phase_seconds_this_invocation"],
        "peak_rss_native_units": result["peak_rss_native_units"],
        "limitations": [
            "single unseeded score per molecule; no significance claim",
            "local repair census is not a completed adaptive search round",
            "SMILES graph differences are not a shortest-path proof or a causal docking prescription",
            "the historical winner was not docked in this batch",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    destination = HERE / "summary.json"
    if destination.exists():
        raise ValueError("refusing to overwrite a banked audit")
    report = run()
    publish_json(destination, report)
    print(
        {
            k: report[k]
            for k in (
                "new_oracle_attempts",
                "failures",
                "count_better_than_prior",
                "spearman_predicted_observed",
                "remote_seconds",
            )
        }
    )
    for row in sorted(
        report["rows"], key=lambda r: (r["observed_docking"] is None, r["observed_docking"] or 0)
    ):
        print(row["index"], row["observed_docking"], row["predicted_docking"], row["smiles"])

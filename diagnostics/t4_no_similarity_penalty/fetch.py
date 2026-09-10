"""Fetch one ablation and reuse the existing locked-control audit, without launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import canonical_bytes, publish_json, sha256_file
from compose_v4.experiments.t4_macro_beam import no_similarity_desirability
from compose_v4.experiments.t4_matched_pilot import unseal
from diagnostics.t4_constraint_recovery.summarize import distribution, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--capture-lifecycle", action="store_true")
    args = parser.parse_args()
    destination = HERE / "attempt_1"
    if args.capture_lifecycle:
        command = ["modal", "app", "logs", "genmol-t4-opt", "--timestamps"]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            log = process.communicate(timeout=8)[0]
        except subprocess.TimeoutExpired:
            process.terminate()
            log = process.communicate(timeout=5)[0]
        publish_json(
            destination / "lifecycle.json",
            {
                "command": command,
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "app-wide corroboration, not exact per-call attribution",
                "preemption_lines": [
                    line for line in log.splitlines() if "preemption" in line.lower()
                ],
                "collector_sha256": sha256_file(Path(__file__)),
                "timing_limit": "heartbeat elapsed counter reset during this run; final counters cover only the resumed invocation; do not claim a whole-job speedup",
            },
        )
    launch_path = destination / "launch.json"
    if args.receipt:
        incoming = json.loads(args.receipt.read_text())
        if launch_path.exists() and incoming != json.loads(launch_path.read_text()):
            raise ValueError("refusing to mix similarity ablation launches")
        publish_json(launch_path, incoming)
    launch = json.loads(launch_path.read_text())
    contract_path = ROOT / "configs/t4_no_similarity_penalty.json"
    contract = json.loads(contract_path.read_text())
    assert rdBase.rdkitVersion == contract["required_rdkit"]
    assert launch["oracle_calls"] == contract["oracle_calls"] == 0
    assert len(launch["cases"]) == 1 and launch["cases"][0]["case_index"] == 0
    assert sha256_file(contract_path) == launch["task"]["contract_sha256"]
    volume = modal.Volume.from_name(launch["volume"])
    remote_root = launch["cases"][0]["volume_path"].lstrip("/")
    remote_hashes = {}

    def get(name):
        remote = f"{remote_root}/{name}.json"
        try:
            data = b"".join(volume.read_file(remote))
        except FileNotFoundError:
            return None
        remote_hashes[remote] = hashlib.sha256(data).hexdigest()
        value = json.loads(data)
        publish_json(destination / f"{name}.json", value)
        if isinstance(value, dict) and "payload_sha256" in value:
            assert (
                hashlib.sha256(canonical_bytes(value["payload"])).hexdigest()
                == value["payload_sha256"]
            )
            return value["payload"]
        return value

    result, failure, heartbeat = get("result"), get("failure"), get("heartbeat")
    if failure:
        raise RuntimeError(f"similarity ablation failed: {failure}")
    if result is None:
        print(json.dumps({"status": "pending", "heartbeat": heartbeat}, sort_keys=True))
        return
    assert result["configuration"] == contract and result["case_index"] == 0
    assert result["code_revision"] == launch["task"]["image_revision"]["commit"]
    assert result["runtime_gate"]["input_sha256"] == contract["expected_input_sha256"]
    for name in (
        "generation_lock",
        "scored_lock",
        "runtime_gate",
        "law_cache_inventory",
    ):
        assert get(name) is not None, name
    snapshot_bytes = b"".join(volume.read_file(contract["value_snapshot"]["path"]))
    assert hashlib.sha256(snapshot_bytes).hexdigest() == contract["value_snapshot"]["sha256"]
    remote_hashes[contract["value_snapshot"]["path"]] = contract["value_snapshot"]["sha256"]
    publish_json(destination / "task_snapshot.json", json.loads(snapshot_bytes))
    snapshot = unseal(destination / "task_snapshot.json")
    for candidate in result["candidates"]:
        expected = no_similarity_desirability(
            candidate, candidate["predicted_docking"], snapshot, contract["no_similarity_guidance"]
        )
        assert np.isclose(candidate["no_similarity_desirability"], expected, rtol=1e-12, atol=0)
    proof_path = HERE / "control_reuse.json"
    proof = json.loads(proof_path.read_text())
    assert proof["new_revision"] == result["code_revision"]
    assert proof["contract_sha256"] == sha256_file(contract_path)
    assert proof["old_revision"] == contract["controls"]["implementation_revision"]
    inputs = [contract_path, proof_path, ROOT / contract["controls"]["contract"]]
    inputs.extend(sorted(destination.glob("*.json")))
    arms, overlaps = [], []
    new_lock = unseal(destination / "generation_lock.json")
    new_smiles = {r["smiles"] for r in result["candidates"]}
    for case, label in enumerate(("post_hoc", "terminal", "recovery")):
        prior_dir = ROOT / contract["controls"]["results"] / f"case_{case}"
        inputs.extend(
            prior_dir / f"{n}.json"
            for n in (
                "result",
                "generation_lock",
                "scored_lock",
                "runtime_gate",
                "law_cache_inventory",
            )
        )
        prior = json.loads((prior_dir / "result.json").read_text())
        old_lock = unseal(prior_dir / "generation_lock.json")
        assert prior["code_revision"] == proof["old_revision"]
        assert prior["runtime_gate"]["input_sha256"] == result["runtime_gate"]["input_sha256"]
        assert prior["value_snapshot_sha256"] == result["value_snapshot_sha256"]
        assert old_lock["root"] == new_lock["root"]
        assert old_lock["levels"][0]["attempts"] == new_lock["levels"][0]["attempts"]
        arms.append(summarize(prior_dir, label))
        old_smiles = {r["smiles"] for r in prior["candidates"]}
        overlaps.append(
            {
                "control": label,
                "intersection": len(old_smiles & new_smiles),
                "only_new_arm": len(new_smiles - old_smiles),
            }
        )
    arms.append(summarize(destination, "no_similarity"))
    arms[-1]["timing_scope"] = (
        "final resumed invocation only; interrupted work excluded; not a whole-job timing or call count"
    )
    candidates = result["candidates"]
    all_candidate_diagnostics = {
        "qed_below_threshold": sum(r["qed"] < 0.6 for r in candidates),
        "sa_above_threshold": sum(r["sa"] > 4 for r in candidates),
        "similarity_below_threshold": sum(r["sim"] < 0.4 for r in candidates),
        "final_gate_failures_overlap": True,
        "original_seed_similarity": distribution([r["sim"] for r in candidates]),
        "predicted_docking": distribution([r["predicted_docking"] for r in candidates]),
        "ring_system_deltas_from_root": dict(
            sorted(Counter(r["cumulative_change"]["d_ring_systems"] for r in candidates).items())
        ),
        "cycle_rank_deltas_from_root": dict(
            sorted(Counter(r["cumulative_change"]["d_cycle_rank"] for r in candidates).items())
        ),
        "cycle_rank_deltas_per_option": dict(
            sorted(Counter(r["structural_change"]["d_cycle_rank"] for r in candidates).items())
        ),
        "realized_change_from_root": distribution(
            [r["cumulative_change"]["largest_changed_fraction"] for r in candidates]
        ),
        "constructive_programs": [
            {
                "option": r["bundle"]["option"],
                "smiles": r["smiles"],
                "change": r["structural_change"],
                "sim": r["sim"],
                "qed": r["qed"],
                "sa": r["sa"],
                "oracle_eligible": r["oracle_eligible"],
            }
            for r in candidates
            if r["bundle"]["option"].startswith("construct:")
        ],
    }
    report = {
        "schema_version": "t4_no_similarity_summary_v1",
        "arms": arms,
        "candidate_overlap": overlaps,
        "all_candidate_diagnostics": all_candidate_diagnostics,
        "oracle_calls": 0,
        "winner_used": False,
        "input_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(inputs)},
        "remote_file_sha256": remote_hashes,
        "analysis": {
            "script_sha256": sha256_file(Path(__file__)),
            "reused_reducer_sha256": sha256_file(
                ROOT / "diagnostics/t4_constraint_recovery/summarize.py"
            ),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "claim_limit": "one inspected development parent and RNG seed; old controls reused after compatibility verification; no new docking observations",
    }
    print(publish_json(HERE / "summary.json", report))
    for arm in arms:
        print(
            arm["arm"],
            "new eligible",
            arm["unique_new_eligible"],
            "recovered",
            arm["unique_new_recovered_molecules"],
            "proposal seconds",
            round(arm["proposal_seconds"], 1),
        )


if __name__ == "__main__":
    main()

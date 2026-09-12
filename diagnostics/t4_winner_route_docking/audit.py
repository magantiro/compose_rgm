"""Verify the six saved docking results and prepared molecular identities."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase

from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[2]
RUN = "fece76dedf244574e46892c7c8eb4bf3d5a9ae114e3abcd2be9e47bb779f03c9"
DIRECTORY = Path(__file__).resolve().parent / RUN


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("requires production-matched RDKit")
    inputs = {}

    def read(path, sealed=False):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return unseal(path) if sealed else json.loads(path.read_text())

    result = read(DIRECTORY / "result.json")
    lock = read(DIRECTORY / "candidate_lock.json", True)
    barrier = read(DIRECTORY / "docking_started.json")
    gate = read(DIRECTORY / "runtime_gate.json")
    expected = result["configuration"]["expected_input_sha256"]
    assert gate["input_sha256"] == expected
    digest = sha(DIRECTORY / "candidate_lock.json")
    assert result["candidate_lock_sha256"] == barrier["candidate_lock_sha256"] == digest
    assert result["status"] == "complete" and result["new_oracle_attempts"] == 6
    assert len(result["docked"]) == len(lock["take"]) == 6
    diagnosis = read(
        ROOT
        / "diagnostics/t4_macro_lookahead/8776743bbb813429663f339f60cbce587e59d427248dfaee80020014b11a6b28/diagnosis.json"
    )["summary"]
    predictions = {r["name"]: r["final_predicted_docking"] for r in diagnosis["route_stages"]}
    predictions["incumbent_redock"] = diagnosis["final_prediction_of_best"]
    rows = []
    for i, candidate in enumerate(lock["take"]):
        started = read(DIRECTORY / f"rows/{i:02}/started.json")
        actual = read(DIRECTORY / f"rows/{i:02}/result.json", True)
        assert started["candidate_lock_sha256"] == actual["candidate_lock_sha256"] == digest
        assert actual["index"] == i and actual["smiles"] == candidate["smiles"]
        assert result["docked"][i] == {**candidate, **actual}
        assert started["started_at_utc"] >= barrier["started_at_utc"]
        if i:
            assert started["started_at_utc"] >= rows[0]["completed_at_utc"]
        poses = DIRECTORY / f"poses/{i:02}"
        manifest = read(poses / "manifest.json")
        for name, checksum in manifest["sha256"].items():
            path = poses / name
            assert sha(path) == checksum
            inputs[str(path.relative_to(ROOT))] = checksum
        molecule = Chem.MolFromMolFile(str(poses / "l.mol"), removeHs=True)
        assert molecule is not None
        # The production representation is achiral; the 3D conformer may carry
        # stereotags assigned during preparation, outside that representation.
        prepared_identity = Chem.MolToSmiles(molecule, isomericSmiles=False)
        locked_identity = Chem.MolToSmiles(
            Chem.MolFromSmiles(candidate["smiles"]), isomericSmiles=False
        )
        assert prepared_identity == locked_identity, candidate["role"]
        pose_scores = [
            float(line.split()[3])
            for line in (poses / "o.pdbqt").read_text().splitlines()
            if line.startswith("REMARK VINA RESULT")
        ]
        assert pose_scores and pose_scores[0] == actual["ds"]
        rows.append(
            {
                "role": candidate["role"],
                "smiles": candidate["smiles"],
                "ds": actual["ds"],
                "prior_surrogate_prediction": predictions[candidate["role"]],
                "qed": candidate["qed"],
                "sa": candidate["sa"],
                "sim": candidate["sim"],
                "docking_seconds": actual["docking_seconds"],
                "completed_at_utc": actual["completed_at_utc"],
                "prepared_achiral_graph_matches": True,
            }
        )
    report = {
        "schema_version": "t4_winner_route_audit_v1",
        "rows": rows,
        "run_revision": result["code_revision"],
        "input_sha256": inputs,
        "analysis_sha256": sha(Path(__file__)),
        "software": {"rdkit": rdBase.rdkitVersion, "python": platform.python_version()},
        "configuration": result["configuration"],
        "hardware": {"machine": platform.machine(), "analysis": "CPU, integer molecular graphs"},
        "split_identity": "answer-known PARP1 seed0 development diagnostic, excluded from optimizer fitting",
        "randomness": result["randomness"],
        "new_oracle_calls": 0,
        "run_oracle_calls": 6,
        "run_elapsed_seconds": result["elapsed_seconds"],
        "run_docking_seconds": sum(r["docking_seconds"] for r in rows),
        "checks": [
            "sealed payloads",
            "candidate and barrier identities",
            "winner-first ordering",
            "all six rows included",
            "input and pose hashes",
            "prepared achiral molecular identity",
            "pose scores equal reported scores",
        ],
        "limitations": [
            "one docking per molecule, no uncertainty estimate",
            "answer-known route, not autonomous discovery",
            "score is not experimental affinity",
            "unset production docking seeds",
            "prior predictions are diagnostic only, not newly fit",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = DIRECTORY / "audit.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(target)
    print(
        json.dumps(
            {
                "rows": rows,
                "audit_sha256": sha(target),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

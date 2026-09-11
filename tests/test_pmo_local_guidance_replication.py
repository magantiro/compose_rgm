import json
import shutil
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments import pmo_local_guidance_replication as experiment
from compose_v4.experiments.continuation_profile import publish_json, sha256_file


def test_replication_preserves_original_inputs_and_rejects_rehashed_exclusion_drift(tmp_path):
    root = Path(__file__).resolve().parents[1]
    # Versioned prepared artifacts are the bounded integration fixture; no remote data.
    c = experiment.load_contract(root)
    paths = set(c["replication"]["input_files"]) | {
        experiment.CONTRACT,
        experiment.PREPARED,
        experiment.PROTOCOL,
    }
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, target)
    experiment.load_contract(tmp_path)
    prepared = tmp_path / experiment.PREPARED
    data = json.loads(prepared.read_text())
    # This would remove a paid-but-still-proposable molecule in a naive cache merge.
    smiles, score = next(iter(data["requested_only_scores"].items()))
    data["observed"][smiles] = score
    publish_json(prepared, data)
    c["prepared"]["sha256"] = sha256_file(prepared)
    c["contract_sha256"] = identity({k: v for k, v in c.items() if k != "contract_sha256"})
    publish_json(tmp_path / experiment.CONTRACT, c)
    with pytest.raises(ValueError, match="original starts, model, donors or exclusions"):
        experiment.load_contract(tmp_path)

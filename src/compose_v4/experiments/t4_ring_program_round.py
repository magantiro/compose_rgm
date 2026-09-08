"""One bounded new-recipe round from the existing 46-call exact archive."""

from __future__ import annotations

import json

from rdkit import rdBase

from compose_v4.control.ring_program import default_ring_options
from compose_v4.experiments.continuation_profile import verify_file
from compose_v4.experiments.t4_matched_pilot import run_remote as common_remote
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import run_rounds

CONTRACT_PATH = "configs/t4_ring_program_round.json"
KIND = "t4_ring_program_round"


def load_archive(root, contract):
    source = contract["source"]
    path = root / source["path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("ring-round archive escapes artifact root")
    verify_file(path, source["sha256"])
    warm = unseal(path)
    if (
        warm["schema_version"] != "t4_exact_archive_v1"
        or warm["round"] != 3
        or warm["oracle_attempts"] != 46
        or len(warm["archive"]) != 47
    ):
        raise ValueError("ring round requires the complete 46-call archive")
    return warm


def run_remote(
    task, repo_root, artifact_root, volume, runtime_factory, validate_revision, prepare, dock
):
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())

    def runner(actual_task, prepare, dock, output, **kwargs):
        if (
            contract["additional_rounds"] != 1
            or contract["compute"]["oracle_call_limit"] != 20
            or actual_task["budget"] != 20
            or actual_task["per_round"] != 20
            or actual_task["lineages"] != 8
            or actual_task["workers"] != 1
            or actual_task["executor_calls_per_parent"] != 2500
            or actual_task["max_executor_applications"] != 20000
            or tuple(actual_task["ring_program_options"]) != default_ring_options()
        ):
            raise ValueError("ring round exceeds its declared one-round recipe/budget")
        if rdBase.rdkitVersion != contract["required_rdkit"]:
            raise ValueError("ring round requires pinned RDKit 2024.03.5")
        warm = load_archive(artifact_root, contract)
        result = run_rounds(
            actual_task,
            warm,
            prepare,
            dock,
            output,
            additional_rounds=1,
            endpoint_policy=contract["endpoint_selection_policy"],
            source_identity=contract["source"],
            **kwargs,
        )
        result.update(
            schema_version="t4_ring_program_round_v1",
            automatic_continuation=False,
            interpretation_scope="new parameterized-option recipe; inspected development continuation, not a bit-identical resume",
        )
        return result

    volume.reload()
    return common_remote(
        task,
        repo_root,
        artifact_root,
        volume,
        runtime_factory,
        validate_revision,
        prepare,
        dock,
        contract_path=CONTRACT_PATH,
        run_kind=KIND,
        runner=runner,
    )

"""Focused gates for the thin Process-V2 Gate 0 Modal surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_apps.run_process_v2_gate_zero_app as launcher
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    authority_false_block,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def _revision() -> dict[str, object]:
    return {"image_revision_sha256": "1" * 64}


def _completion() -> dict[str, object]:
    return {"completion_sha256": "2" * 64}


def _contracts() -> SimpleNamespace:
    return SimpleNamespace(
        contract_sha256="3" * 64,
        binding_sha256="4" * 64,
        process_identity_sha256="5" * 64,
    )


def _request() -> dict[str, object]:
    return launcher.build_run_request(
        image_revision=_revision(),
        active8_run_root="/artifacts/editing_v2/process_v2_active8/" + "6" * 64,
        active8_completion=_completion(),
        active8_completion_file_sha256="7" * 64,
        contracts=_contracts(),
        output_artifact_prefix=launcher.OUTPUT_ARTIFACT_PREFIX,
    )


def test_request_is_content_addressed_and_grants_no_authority() -> None:
    request = launcher.validate_run_request(_request())
    run_body = {
        key: request[key]
        for key in request
        if key not in {"run_identity_sha256", "run_artifact_root", "request_sha256"}
    }

    assert request["run_identity_sha256"] == canonical_sha256(run_body)
    assert request["run_artifact_root"] == (
        f"{launcher.OUTPUT_ARTIFACT_PREFIX}/{request['run_identity_sha256']}"
    )
    assert {field: request[field] for field in AUTHORITY_FIELDS} == authority_false_block()


@pytest.mark.parametrize(
    "mutation",
    ["active8_completion", "completion_file", "contract", "revision"],
)
def test_every_material_input_moves_the_request_identity(mutation: str) -> None:
    reference = _request()
    kwargs = {
        "image_revision": _revision(),
        "active8_run_root": "/artifacts/editing_v2/process_v2_active8/" + "6" * 64,
        "active8_completion": _completion(),
        "active8_completion_file_sha256": "7" * 64,
        "contracts": _contracts(),
        "output_artifact_prefix": launcher.OUTPUT_ARTIFACT_PREFIX,
    }
    if mutation == "active8_completion":
        kwargs["active8_completion"] = {"completion_sha256": "8" * 64}
    elif mutation == "completion_file":
        kwargs["active8_completion_file_sha256"] = "8" * 64
    elif mutation == "contract":
        kwargs["contracts"] = SimpleNamespace(
            contract_sha256="8" * 64,
            binding_sha256="4" * 64,
            process_identity_sha256="5" * 64,
        )
    else:
        kwargs["image_revision"] = {"image_revision_sha256": "8" * 64}

    observed = launcher.build_run_request(**kwargs)
    assert observed["run_identity_sha256"] != reference["run_identity_sha256"]


def test_request_refuses_granted_authority_after_a_valid_reseal() -> None:
    request = _request()
    request["t1_authorized"] = True
    body = dict(request)
    body.pop("request_sha256")
    request["request_sha256"] = canonical_sha256(body)

    with pytest.raises(ValueError, match="grants authority"):
        launcher.validate_run_request(request)


def test_request_refuses_a_malformed_digest_after_a_valid_reseal() -> None:
    request = _request()
    request["active8_completion_file_sha256"] = "not-a-digest"
    run_body = {
        key: request[key]
        for key in request
        if key not in {"run_identity_sha256", "run_artifact_root", "request_sha256"}
    }
    request["run_identity_sha256"] = canonical_sha256(run_body)
    request["run_artifact_root"] = (
        f"{request['output_artifact_prefix']}/{request['run_identity_sha256']}"
    )
    body = dict(request)
    body.pop("request_sha256")
    request["request_sha256"] = canonical_sha256(body)

    with pytest.raises(RuntimeError, match="malformed active8_completion_file_sha256"):
        launcher.validate_run_request(request)


def test_request_refuses_a_path_outside_the_artifact_mount() -> None:
    with pytest.raises(RuntimeError, match="below /artifacts"):
        launcher.build_run_request(
            image_revision=_revision(),
            active8_run_root="/tmp/not-an-artifact",
            active8_completion=_completion(),
            active8_completion_file_sha256="7" * 64,
            contracts=_contracts(),
            output_artifact_prefix=launcher.OUTPUT_ARTIFACT_PREFIX,
        )


def test_local_revision_refuses_a_dirty_or_wrong_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = {
        ("rev-parse", "HEAD"): "1" * 40,
        ("rev-parse", "HEAD^{tree}"): "2" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): " M tracked.py",
    }
    monkeypatch.setattr(launcher, "_git", lambda _root, *arguments: answers[arguments])

    with pytest.raises(RuntimeError, match="exact clean committed worktree"):
        launcher.local_image_revision(expected_commit="1" * 40, repo_root=ROOT)

    answers[("status", "--porcelain=v1", "--untracked-files=all")] = ""
    with pytest.raises(RuntimeError, match="exact clean committed worktree"):
        launcher.local_image_revision(expected_commit="9" * 40, repo_root=ROOT)


def test_immutable_request_publication_reuses_only_identical_bytes(tmp_path: Path) -> None:
    target = tmp_path / launcher.REQUEST_FILENAME
    request = _request()

    assert launcher._publish_immutable(target, request) is False
    assert launcher._publish_immutable(target, request) is True
    tampered = dict(request)
    tampered["active8_completion_sha256"] = "8" * 64
    with pytest.raises(RuntimeError, match="immutable Gate 0 collision"):
        launcher._publish_immutable(target, tampered)


def test_remote_surface_authenticates_before_publish_and_commits_once() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_gate_zero"
    )
    body = ast.get_source_segment(source, function)
    decorator = ast.get_source_segment(source, function.decorator_list[0])
    assert body is not None and decorator is not None

    assert body.index("artifact_volume.reload()") < body.index(
        "load_process_v2_active8_completion"
    )
    assert body.index("load_process_v2_active8_completion") < body.index(
        "_publish_immutable(staging_request_path, request)"
    )
    assert body.index("run_gate_zero(") < body.index("os.replace(staging_root, output_root)")
    assert body.index("os.replace(staging_root, output_root)") < body.index(
        "artifact_volume.commit()"
    )
    assert body.count("artifact_volume.commit()") == 1
    assert "cpu=GATE_ZERO_CPU" in decorator
    assert "memory=GATE_ZERO_MEMORY_MB" in decorator
    assert "max_containers=GATE_ZERO_MAX_CONTAINERS" in decorator
    assert launcher.GATE_ZERO_CPU == 1.0
    assert launcher.GATE_ZERO_MEMORY_MB == 8 * 1024
    assert launcher.GATE_ZERO_TIMEOUT_SECONDS == 60 * 60
    assert launcher.GATE_ZERO_MAX_CONTAINERS == 1


def test_surface_uses_only_the_process_v2_gate_zero_core() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    assert "editing_v2_process_v2_gate_zero" in source
    assert "editing_v2_semantic_gate_zero" not in source
    assert "run_gate_zero(" in source
    assert "reduce_gate_zero(" not in source


def test_reused_decision_is_validated_before_return() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_gate_zero"
    )
    body = ast.get_source_segment(source, function)
    assert body is not None
    reuse_start = body.index("if request_path.exists() or decision_path.exists():")
    reuse_end = body.index("output_root.parent.mkdir(parents=True, exist_ok=True)")
    reuse_body = body[reuse_start:reuse_end]
    assert "validate_run_request" in reuse_body
    assert "_validate_decision" in reuse_body


def test_request_bytes_are_canonical_and_round_trip(tmp_path: Path) -> None:
    target = tmp_path / launcher.REQUEST_FILENAME
    request = _request()
    launcher._publish_immutable(target, request)
    assert launcher.validate_run_request(json.loads(target.read_bytes())) == request

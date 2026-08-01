"""Focused safety tests for the candidate-provenance Modal launcher."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_apps.materialize_editing_v2_candidate_provenance_app as launcher

SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps/materialize_editing_v2_candidate_provenance_app.py"


def _source_revision(source_hashes: dict[str, str] | None = None) -> dict[str, object]:
    hashes = source_hashes or {"source.py": SHA}
    body: dict[str, object] = {
        "schema": launcher.SOURCE_REVISION_SCHEMA,
        "schema_version": launcher.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": launcher._canonical_sha256(hashes),
    }
    return {**body, "source_revision_sha256": launcher._canonical_sha256(body)}


def _candidate_identity() -> dict[str, str]:
    return {
        "manifest_file_sha256": SHA,
        "manifest_sha256": "b" * 64,
        "rows_file_sha256": "c" * 64,
        "rows_semantic_sha256": "d" * 64,
        "address_stream_sha256": "e" * 64,
    }


def test_run_request_is_deterministic_content_addressed_and_nonauthorizing() -> None:
    arguments = {
        "source_revision": _source_revision(),
        "candidate_root": "/artifacts/editing_v2/candidate",
        "candidate_identity": _candidate_identity(),
        "decisions_path": "/root/compose/configs/decisions.json",
        "decisions_file_sha256": SHA,
        "decisions_sha256": "f" * 64,
        "decisions_registry_inputs_sha256": "1" * 64,
        "editing_corpus_contract_path": "/root/compose/configs/contract.json",
        "editing_corpus_contract_file_sha256": "2" * 64,
        "registry_sha256": "3" * 64,
        "max_row_bytes": 1024,
    }
    first = launcher.build_run_request(**arguments)
    second = launcher.build_run_request(**arguments)

    assert first == second
    assert first["training_authorized"] is False
    assert first["bounded_p50_authorized"] is False
    assert len(first["run_identity_sha256"]) == 64
    changed = launcher.build_run_request(**{**arguments, "max_row_bytes": 2048})
    assert changed["run_identity_sha256"] != first["run_identity_sha256"]


def test_local_source_revision_rejects_dirty_or_wrong_full_commit(monkeypatch) -> None:
    source_hashes = {"source.py": SHA}
    monkeypatch.setattr(launcher, "_serialized_source_hashes", lambda root: source_hashes)

    def clean_git(root: Path, *arguments: str) -> str:
        del root
        if arguments == ("rev-parse", "HEAD"):
            return "1" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "2" * 40
        return ""

    monkeypatch.setattr(launcher, "_git", clean_git)
    revision = launcher.local_source_revision(expected_commit="1" * 40)
    assert revision["serialized_source_hashes"] == source_hashes

    with pytest.raises(RuntimeError, match="exact clean committed"):
        launcher.local_source_revision(expected_commit="3" * 40)

    def dirty_git(root: Path, *arguments: str) -> str:
        if arguments[:2] == ("status", "--porcelain=v1"):
            return "?? untracked.py"
        return clean_git(root, *arguments)

    monkeypatch.setattr(launcher, "_git", dirty_git)
    with pytest.raises(RuntimeError, match="exact clean committed"):
        launcher.local_source_revision(expected_commit="1" * 40)


@pytest.mark.parametrize(
    ("candidate_path", "manifest_sha256"),
    [
        ("/artifacts/editing_v2/other/PACKED_CANDIDATE_MATERIALIZATION.json", "b" * 64),
        (
            "/artifacts/editing_v2/candidate/PACKED_CANDIDATE_MATERIALIZATION.json",
            "0" * 64,
        ),
    ],
)
def test_decisions_binding_rejects_another_candidate_path_or_identity(
    candidate_path: str,
    manifest_sha256: str,
) -> None:
    identity = _candidate_identity()
    decisions = SimpleNamespace(
        training_authorized=False,
        candidate_artifact=SimpleNamespace(
            candidate_materialization=SimpleNamespace(
                path=candidate_path,
                **{
                    field: manifest_sha256 if field == "manifest_sha256" else value
                    for field, value in identity.items()
                },
            ),
            editing_corpus_contract=SimpleNamespace(file_sha256="9" * 64),
        ),
    )
    with pytest.raises(RuntimeError, match="exact candidate or contract"):
        launcher._validate_decisions_candidate_binding(
            decisions,
            candidate_root_address="/artifacts/editing_v2/candidate",
            candidate_identity=identity,
            materialization_filename="PACKED_CANDIDATE_MATERIALIZATION.json",
            contract_file_sha256="9" * 64,
        )


def test_driver_validates_candidate_once_and_reuses_exact_value_on_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    remote_root = tmp_path / "remote"
    candidate_address = "/artifacts/editing_v2/candidate"
    candidate_root = artifact_root / "editing_v2/candidate"
    candidate_root.mkdir(parents=True)
    manifest_path = candidate_root / "PACKED_CANDIDATE_MATERIALIZATION.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    decisions_path = remote_root / "configs/decisions.json"
    contract_path = remote_root / "configs/contract.json"
    decisions_path.parent.mkdir(parents=True)
    decisions_path.write_text("{}\n", encoding="utf-8")
    contract_path.write_text("{}\n", encoding="utf-8")

    validated_values: list[dict[str, object]] = []
    registry_reuse: list[object] = []
    bridge_reuse: list[object] = []

    class FakeMaterializer:
        MATERIALIZATION_FILENAME = "PACKED_CANDIDATE_MATERIALIZATION.json"

        @staticmethod
        def validate_packed_candidate_materialization(path, *, max_row_bytes):
            assert Path(path) == candidate_root
            assert max_row_bytes == 4096
            value = {
                "manifest_sha256": "b" * 64,
                "rows": {
                    "file_sha256": "c" * 64,
                    "semantic_sha256": "d" * 64,
                    "address_stream_sha256": "e" * 64,
                },
            }
            validated_values.append(value)
            return value

    registry_inputs = {
        "compiler_identity": {},
        "policy_bindings": {},
        "evidence_identity": {},
        "identity_definitions": {},
        "source_assets": [],
    }

    class FakeDecisions:
        training_authorized = False
        decision_sha256 = "f" * 64

        def __init__(self) -> None:
            declared = SimpleNamespace(
                path=f"{candidate_address}/PACKED_CANDIDATE_MATERIALIZATION.json",
                manifest_file_sha256=launcher._file_sha256(manifest_path),
                manifest_sha256="b" * 64,
                rows_file_sha256="c" * 64,
                rows_semantic_sha256="d" * 64,
                address_stream_sha256="e" * 64,
            )
            self.candidate_artifact = SimpleNamespace(
                candidate_materialization=declared,
                editing_corpus_contract=SimpleNamespace(
                    file_sha256=launcher._file_sha256(contract_path)
                ),
            )

        def registry_inputs(self):
            return registry_inputs

    class FakeBridge:
        REGISTRY_FILENAME = "CANDIDATE_PROVENANCE_REGISTRY.json"
        BRIDGE_MANIFEST_FILENAME = "CANDIDATE_PROVENANCE_BRIDGE.json"

        @staticmethod
        def build_candidate_provenance_registry(**kwargs):
            registry_reuse.append(kwargs["_validated_candidate_materialization"])
            assert all(name in kwargs for name in registry_inputs)
            return {"training_authorized": False, "registry_sha256": "3" * 64}

        @staticmethod
        def write_candidate_provenance_registry(registry, path):
            path.write_bytes(launcher._canonical_bytes(registry, pretty=True))

        @staticmethod
        def materialize_candidate_provenance_bridge(*, output_dir, **kwargs):
            bridge_reuse.append(kwargs["_validated_candidate_materialization"])
            manifest = {
                "training_authorized": False,
                "manifest_sha256": "4" * 64,
                "counts": {"attempted": 1, "routed": 1, "rejected": 0, "split_rows": 1},
                "blockers": ["training.not_authorized"],
            }
            output_dir.mkdir()
            (output_dir / FakeBridge.BRIDGE_MANIFEST_FILENAME).write_bytes(
                launcher._canonical_bytes(manifest, pretty=True)
            )
            return manifest

        @staticmethod
        def validate_candidate_provenance_bridge(output_dir, **kwargs):
            bridge_reuse.append(kwargs["_validated_candidate_materialization"])
            return json.loads((output_dir / FakeBridge.BRIDGE_MANIFEST_FILENAME).read_bytes())

    source_hashes = {"source.py": SHA}
    monkeypatch.setattr(launcher, "_serialized_source_hashes", lambda root: source_hashes)
    monkeypatch.setattr(
        launcher,
        "_imports",
        lambda remote: {
            "bridge": FakeBridge,
            "materializer": FakeMaterializer,
            "load_candidate_provenance_decisions": lambda path: FakeDecisions(),
        },
    )
    arguments = {
        "source_revision": _source_revision(source_hashes),
        "candidate_root": candidate_address,
        "decisions_path": str(decisions_path),
        "editing_corpus_contract_path": str(contract_path),
        "max_row_bytes": 4096,
        "output_prefix": launcher.OUTPUT_PREFIX,
        "artifact_root": artifact_root,
        "remote_root": remote_root,
    }
    first = launcher._materialize_impl(**arguments)
    second = launcher._materialize_impl(**arguments)

    assert first["reused"] is False
    assert second["reused"] is True
    assert len(validated_values) == 2
    assert registry_reuse == validated_values
    assert bridge_reuse == validated_values
    assert first["completion"] == second["completion"]
    assert first["completion"]["training_authorized"] is False
    assert first["completion"]["bounded_p50_authorized"] is False


def test_static_surface_uses_public_decisions_api_and_waits_for_remote_completion() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    implementation = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_materialize_impl"
    )
    implementation_source = ast.get_source_segment(source, implementation)
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main)
    assert implementation_source is not None
    assert main_source is not None
    assert "load_candidate_provenance_decisions" in implementation_source
    assert "registry_inputs = decisions.registry_inputs()" in implementation_source
    assert implementation_source.count("validate_packed_candidate_materialization(") == 1
    assert (
        implementation_source.count("_validated_candidate_materialization=validated_candidate") == 3
    )
    assert "materialize_candidate_provenance.remote(" in main_source
    assert "materialize_candidate_provenance.spawn(" not in main_source
    assert '"training_launched": False' in main_source

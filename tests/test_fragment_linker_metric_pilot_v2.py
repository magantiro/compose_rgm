"""The linker repair preserves the failed v1 measurement and fails closed on drift."""

import json

import pytest
from run_fragment_training_linker_metric_pilot_v2 import V1, _copy_verified, prepare_manifest

from compose_v4.benchmark.training_attachment_fragments import physical_sha256


def test_copy_verified_preserves_bytes_and_refuses_changed_sources(tmp_path):
    source = tmp_path / "source.json"
    target = tmp_path / "imported" / "source.json"
    source.write_bytes(b'{"old":true}\n')
    digest = physical_sha256(source)
    _copy_verified(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    _copy_verified(source, target, digest)
    source.write_bytes(b'{"old":false}\n')
    with pytest.raises(ValueError, match="source v1 artifact changed"):
        _copy_verified(source, target, digest)
    assert target.read_bytes() == b'{"old":true}\n'


@pytest.mark.skipif(not (V1 / "manifest.json").exists(), reason="sealed v1 pilot asset absent")
def test_repair_manifest_binds_exact_failed_prefix_and_single_replay():
    manifest, prompts, original = prepare_manifest()
    assert manifest["source_v1_manifest_sha256"] == physical_sha256(V1 / "manifest.json")
    assert len(prompts) == 10
    assert len(manifest["v1_imported_artifact_sha256"]) == 127 + 6 + 6
    assert manifest["v1_failure"] == {
        "kind": "nonadditive_rdkit_perceived_ring_count",
        "offered_draw": 4,
        "connector": "[1*]N1C2CNCC1C2[2*]",
        "planned_cell": [21, 4],
        "actual_cell": [21, 3],
    }
    assert (
        manifest["source_v1_sampler_sha256"]
        == original["inputs"][
            next(path for path in original["inputs"] if path.endswith("fragment_linker_sampler.py"))
        ]
    )
    start = json.loads((V1 / "starts/LIOTHYRONINE_007.json").read_text())
    assert start["attempt_index"] == 7
    assert start["manifest_sha256"] == manifest["source_v1_manifest_sha256"]

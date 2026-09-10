import json
from types import SimpleNamespace

import pytest
from test_carbonyl_option import engineering_law, initial

from compose_v4.experiments import saved_marked_law as saved
from compose_v4.experiments.continuation_profile import encode_action, sha256_file
from compose_v4.experiments.t4_matched_pilot import seal
from compose_v4.rewrite.trace_shard import encode_state


@pytest.mark.parametrize("cache_path", [".", "caches/00"])
def test_saved_law_reuse_checks_inputs_and_rejects_corrupt_mass(tmp_path, monkeypatch, cache_path):
    graph = initial("CCC").graph
    families, actions, probabilities = engineering_law(graph)
    payload = {
        "source": encode_state(graph),
        "marks": [encode_action(f, a) for f, a in zip(families, actions, strict=True)],
        "probabilities": list(probabilities),
    }
    module = tmp_path / "src/compose_v4/model/test_dependency.py"
    module.parent.mkdir(parents=True)
    module.write_text("fixture = 1\n")
    old = tmp_path / "old"
    old.mkdir()
    (old / "launch.json").write_text(
        json.dumps(
            {
                "image_revision": {
                    "serialized_sources": {
                        "src/compose_v4/model/test_dependency.py": sha256_file(module)
                    }
                }
            }
        )
    )
    expected = {"checkpoint": "a" * 64}
    (old / "runtime_gate.json").write_text(json.dumps({"input_sha256": expected}))
    key = saved.identity(payload["source"])
    seal(old / cache_path / f"laws/{key}.json", payload)
    contract = {
        "law_caches": [
            {
                "path": "old",
                "launch_sha256": sha256_file(old / "launch.json"),
                "cache_paths": [cache_path],
            }
        ],
        "expected_input_sha256": expected,
    }
    records = {}

    def save(name, value):
        records[name] = json.loads(json.dumps(value))

    def forbidden(*args):
        raise AssertionError("compatible saved law was recomputed")

    monkeypatch.setattr(saved, "enumerate_factorized_marked_law", forbidden)
    cache = saved.SavedMarkedLaw(
        None,
        tmp_path,
        save,
        records.get,
        repo_root=tmp_path,
        artifact_root=tmp_path,
        contract=contract,
        progress={},
    )
    row = cache(graph)
    assert (
        cache(graph) == row and cache.counts["old_hits"] == 1 and cache.counts["memory_hits"] == 1
    )
    assert records["law_cache_inventory"][0]["available"]
    records[f"laws/{key}"]["probabilities"][0] = -1
    cache = saved.SavedMarkedLaw(
        None,
        tmp_path,
        save,
        records.get,
        repo_root=tmp_path,
        artifact_root=tmp_path,
        contract=contract,
        progress={},
    )
    with pytest.raises(ValueError, match="invalid probability"):
        cache(graph)
    module.write_text("fixture = 2\n")
    fresh = {}
    monkeypatch.setattr(
        saved,
        "enumerate_factorized_marked_law",
        lambda *args: SimpleNamespace(
            marks=[
                SimpleNamespace(executor_rule_name=f, action=a, probability=p)
                for f, a, p in zip(families, actions, probabilities, strict=True)
            ]
        ),
    )
    cache = saved.SavedMarkedLaw(
        None,
        tmp_path,
        lambda n, p: fresh.update({n: p}),
        fresh.get,
        repo_root=tmp_path,
        artifact_root=tmp_path,
        contract=contract,
        progress={},
    )
    assert not fresh["law_cache_inventory"][0]["available"]
    cache(graph)
    assert cache.counts["fresh_laws"] == 1 and cache.counts["old_hits"] == 0

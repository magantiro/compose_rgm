"""Exact-state model-law cache with conservative prior-artifact containment."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import encode_action, sha256_file, verify_file
from compose_v4.experiments.production_successor_kernel import enumerate_factorized_marked_law
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.trace_shard import encode_state


class SavedMarkedLaw:
    def __init__(self, model, output, save, read, *, repo_root, artifact_root, contract, progress):
        self.model, self.output, self.save, self.read = model, output, save, read
        self.progress, self.memory, self.priors = progress, {}, []
        self.counts = {
            "fresh_laws": 0,
            "memory_hits": 0,
            "saved_hits": 0,
            "old_hits": 0,
            "law_seconds": 0.0,
        }
        inventory = []
        for source in contract.get("law_caches", []):
            directory = artifact_root / source["path"]
            if not (directory / "launch.json").exists():
                inventory.append({"path": source["path"], "available": False})
                continue
            verify_file(directory / "launch.json", source["launch_sha256"])
            old = json.loads((directory / "launch.json").read_text())["image_revision"][
                "serialized_sources"
            ]
            dependencies = {
                p: h
                for p, h in old.items()
                if p.startswith(
                    tuple(f"src/compose_v4/{d}/" for d in ("chem", "model", "rewrite", "data"))
                )
                or p
                in (
                    "src/compose_v4/experiments/production_successor_kernel.py",
                    "src/compose_v4/experiments/factorized_mark_conditional.py",
                    "src/compose_v4/experiments/tracelet_conditional.py",
                    "src/compose_v4/experiments/successor_kernel.py",
                )
            }
            mismatch = [
                p
                for p, h in dependencies.items()
                if not (repo_root / p).exists() or sha256_file(repo_root / p) != h
            ]
            gate = json.loads((directory / "runtime_gate.json").read_text())
            compatible = (
                bool(dependencies)
                and not mismatch
                and gate["input_sha256"] == contract["expected_input_sha256"]
            )
            inventory.append(
                {
                    "path": source["path"],
                    "available": compatible,
                    "mismatched_files": mismatch,
                    "matched_files": len(dependencies) - len(mismatch),
                    "launch_sha256": source["launch_sha256"],
                }
            )
            if compatible:
                for relative in source.get("cache_paths", ["."]):
                    if (
                        PurePosixPath(relative).is_absolute()
                        or ".." in PurePosixPath(relative).parts
                    ):
                        raise ValueError("law cache must remain inside its validated source run")
                    self.priors.append(directory / relative)
        save("law_cache_inventory", inventory)

    def __call__(self, graph):
        key = identity(encode_state(graph))
        if key in self.memory:
            self.counts["memory_hits"] += 1
            return self.memory[key]
        name = f"laws/{key}"
        payload = self.read(name)
        if payload is not None:
            self.counts["saved_hits"] += 1
        else:
            for directory in self.priors:
                path = directory / f"{name}.json"
                if path.exists():
                    payload = unseal(path)
                    self.counts["old_hits"] += 1
                    self.save(f"law_reuse/{key}", {"path": str(path), "sha256": sha256_file(path)})
                    self.save(name, payload)
                    break
        if payload is None:
            self.progress.update(phase="law_enumeration", latest_law=key, **self.counts)
            started = perf_counter()
            row = enumerate_factorized_marked_law(self.model, graph, 0.5)
            self.counts["fresh_laws"] += 1
            self.counts["law_seconds"] += perf_counter() - started
            payload = {
                "source": encode_state(graph),
                "marks": [encode_action(m.executor_rule_name, m.action) for m in row.marks],
                "probabilities": [m.probability for m in row.marks],
            }
            self.save(name, payload)
        if payload["source"] != encode_state(graph):
            raise ValueError("cached marked law disagrees with exact persistent state")
        pairs = [
            (action_codec_v4 if m["schema_version"] == 4 else action_codec).decode_action(m)
            for m in payload["marks"]
        ]
        weights = tuple(payload["probabilities"])
        if (
            len(weights) != len(pairs)
            or not np.isfinite(weights).all()
            or any(p < 0 for p in weights)
            or (weights and not np.isclose(sum(weights), 1, atol=1e-12))
        ):
            raise ValueError("cached marked law has invalid probability mass")
        result = tuple(f for f, _ in pairs), tuple(a for _, a in pairs), weights
        self.memory[key] = result
        return result

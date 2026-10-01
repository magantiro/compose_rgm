"""Replay learned and uniform selection on the same saved fragment offer panels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.experiments.core_runtime import core_runtime_identity

from .assets import child_path, load_registry, sha256
from .evaluator import load_evaluator

PANEL_TASKS = frozenset({"motif_extension", "scaffold_decoration", "linker_design"})


def _offers(panel: dict) -> list[tuple[str, float]]:
    raw = panel.get("offered")
    if not isinstance(raw, list) or len(raw) != 8:
        raise ValueError("a locked fragment panel must retain all eight attempted offers")
    if [row.get("draw") for row in raw] != list(range(8)):
        raise ValueError("fragment offers are not in their original draw order")
    offers: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in raw:
        if row.get("status") != "model_supported":
            continue
        endpoint, score = row.get("endpoint"), row.get("mean_log_mark")
        if not isinstance(endpoint, str) or not endpoint or endpoint in seen:
            raise ValueError("supported fragment endpoint is empty or duplicated")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(f"supported fragment endpoint lacks a finite native score: {endpoint}")
        seen.add(endpoint)
        offers.append((endpoint, float(score)))
    if panel.get("model_supported_count") != len(offers):
        raise ValueError("fragment panel native-support count changed")
    if panel.get("selected_smiles") is not None and panel["selected_smiles"] not in seen:
        raise ValueError("saved fragment selection is outside the supported panel")
    return offers


def _quantile(*parts: object) -> float:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (integer + 0.5) / 2**64


def _pick(
    offers: list[tuple[str, float]], *, law: str, emitted: set[str], task: str, u: float
) -> tuple[str, float | None]:
    if not offers:
        return "", None
    if law == "uniform":
        weights = np.ones(len(offers), dtype=np.float64)
    elif law == "reference":
        scores = np.asarray([score for _, score in offers], dtype=np.float64)
        if task == "linker_design":
            scores += np.asarray(
                [math.log(4.0) if endpoint not in emitted else 0.0 for endpoint, _ in offers]
            )
        weights = np.exp(scores - max(scores))
    else:
        raise ValueError(f"unknown fragment selection law: {law}")
    probabilities = weights / weights.sum()
    index = min(int(np.searchsorted(np.cumsum(probabilities), u, side="right")), len(offers) - 1)
    return offers[index][0], float(probabilities[index])


def replay(result: dict, *, seed: int) -> dict:
    """Keep each attempted offer fixed, including failures and empty panels."""
    task = result.get("task")
    if task not in PANEL_TASKS:
        raise ValueError(f"locked-panel selection is unavailable for task {task!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("replay seed must be a nonnegative integer")
    cells = result.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("fragment generation result has no cells")
    output = []
    seen_cells = set()
    for cell in cells:
        prompt, generation_seed = cell.get("prompt"), cell.get("seed")
        key = (prompt, generation_seed)
        if key in seen_cells:
            raise ValueError(f"duplicate fragment prompt and seed: {key}")
        seen_cells.add(key)
        attempts = cell.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"fragment cell has no attempts: {key}")
        emitted = {"reference": set(), "uniform": set()}
        selections = {"reference": [], "uniform": []}
        for index, attempt in enumerate(attempts):
            if attempt.get("attempt_index") != index:
                raise ValueError(f"fragment attempt order changed: {key}, {index}")
            offers = _offers(attempt["panel"])
            u = _quantile(seed, task, prompt, generation_seed, index)
            for law in ("reference", "uniform"):
                endpoint, probability = _pick(offers, law=law, emitted=emitted[law], task=task, u=u)
                selections[law].append(
                    {"attempt_index": index, "endpoint": endpoint, "probability": probability}
                )
                if endpoint:
                    emitted[law].add(endpoint)
        output.append(
            {
                "prompt": prompt,
                "seed": generation_seed,
                "attempts": len(attempts),
                "selections": selections,
            }
        )
    return {
        "schema_version": "compose.fragment.locked_selection.v1",
        "task": task,
        "replay_seed": seed,
        "shared_quantiles": True,
        "proposal_panels_fixed": True,
        "cells": output,
    }


def _metric_evaluator(root: Path, assets: Path):
    registry = load_registry(root)
    paths = {}
    hashes = {}
    for name in ("evaluator_mol", "evaluator_metrics"):
        record = registry["assets"][name]
        path = child_path(assets, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"fragment evaluator asset identity mismatch: {path}")
        paths[name] = path
        hashes[name] = record["sha256"]
    return load_evaluator(paths, hashes), hashes


def run(*, root: Path, input_path: Path, output_path: Path, seed: int, assets: Path | None) -> Path:
    """Write one immutable replay with input, code and evaluator identities."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite fragment ablation: {output_path}")
    result = json.loads(input_path.read_text())
    provenance_path = input_path.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text())
    registry = load_registry(root)
    expected_checkpoint = registry["assets"]["checkpoint"]["sha256"]
    if (
        result.get("schema") != "compose_fragment_generation_v1"
        or provenance.get("mode") != "local_generation"
        or provenance.get("result_sha256") != sha256(input_path)
        or provenance.get("configuration", {}).get("task") != result.get("task")
        or result.get("checkpoint_sha256") != expected_checkpoint
    ):
        raise ValueError("fragment selection input is not a bound shared-reference generation")
    report = replay(result, seed=seed)
    report["input"] = {"path": str(input_path.resolve()), "sha256": sha256(input_path)}
    report["provenance"] = {
        "path": str(provenance_path.resolve()),
        "sha256": sha256(provenance_path),
    }
    report["reference_checkpoint_sha256"] = expected_checkpoint
    report["runtime"] = core_runtime_identity()
    report["interpretation"] = (
        "retrospective selector comparison on fixed attempted offer panels; "
        "not independent generation campaigns"
    )
    if assets is not None:
        from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics

        evaluator, hashes = _metric_evaluator(root, assets)
        report["evaluator_sha256"] = hashes
        for cell in report["cells"]:
            cell["metrics"] = {
                law: official_prompt_metrics(
                    [row["endpoint"] for row in cell["selections"][law]],
                    evaluator=evaluator,
                    expected_samples=cell["attempts"],
                )
                for law in ("reference", "uniform")
            }
    content = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    write_bytes_if_absent(output_path, content + b"\n")
    return output_path

"""Conditional quality-aware selection replay on locked linker panels."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from analyze_fragment_linker_quality_headroom_v1 import (
    EVALUATOR_SHA256,
    PILOT,
    PROPERTY_MODULE_SHA256,
    ROOT,
    SOURCE,
    immutable_json,
    physical_sha256,
    quality_flags,
)
from rdkit import Chem, rdBase

HEADROOM = ROOT / "diagnostics/fragment_linker_quality_headroom_v1/result.json"
HEADROOM_SHA256 = "65e52a2835675c897dc38ed7baf9df89d21ee72d094041f6af2a40b170be3b5f"
OUTPUT = ROOT / "diagnostics/fragment_linker_quality_guided_replay_v1/result.json"
STREAMS = 16
METRICS = ("quality", "uniqueness", "diversity", "validity")


def stream_seed(drug: str, stream: int) -> int:
    literal = f"fragment-linker-quality-guided-replay-v1|{drug}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(literal).digest()[:8], "little")


def select_quality_guided(
    offers: tuple[tuple[str, float], ...],
    emitted: set[str],
    is_quality: dict[str, bool],
    rng: np.random.Generator,
) -> tuple[str, str]:
    if not offers or len({smiles for smiles, _ in offers}) != len(offers):
        raise ValueError("empty or duplicate model-supported panel")
    if any(not math.isfinite(score) or smiles not in is_quality for smiles, score in offers):
        raise ValueError("panel has unsupported score or missing property value")
    unseen_good = tuple(row for row in offers if row[0] not in emitted and is_quality[row[0]])
    unseen = tuple(row for row in offers if row[0] not in emitted)
    if unseen_good:
        pool, tier = unseen_good, "unseen_quality"
    elif unseen:
        pool, tier = unseen, "unseen_other"
    else:
        pool, tier = offers, "forced_repeat"
    values = np.asarray([score for _, score in pool], dtype=float)
    weights = np.exp(values - values.max())
    weights /= weights.sum()
    selected = pool[int(rng.choice(len(pool), p=weights))][0]
    return selected, tier


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"quality-guided replay already exists: {OUTPUT}")
    if physical_sha256(HEADROOM) != HEADROOM_SHA256:
        raise ValueError("locked quality-headroom diagnostic changed")
    headroom = json.loads(HEADROOM.read_text())
    package = SOURCE / ".official_eval_cache/pkg"
    if physical_sha256(package / "in_virtuo_gen/train_utils/metrics.py") != EVALUATOR_SHA256:
        raise ValueError("official evaluator changed")
    if physical_sha256(package / "in_virtuo_gen/utils/mol.py") != PROPERTY_MODULE_SHA256:
        raise ValueError("official property computation changed")
    sys.path.insert(0, str(package))
    sys.path.insert(0, str(SOURCE / "src"))
    from in_virtuo_gen.utils.mol import compute_single_property

    from compose_v4.benchmark.fragment_official_metrics import official_prompt_metrics

    property_values: dict[str, tuple[float, float]] = {}
    is_quality: dict[str, bool] = {}

    def quality(smiles: str) -> bool:
        if smiles not in is_quality:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or Chem.MolToSmiles(mol, canonical=True) != smiles:
                raise ValueError(f"invalid or noncanonical supported endpoint: {smiles}")
            values = compute_single_property(smiles)
            if values is None or len(values) != 2:
                raise ValueError(f"upstream property computation failed: {smiles}")
            property_values[smiles] = (float(values[0]), float(values[1]))
            is_quality[smiles] = quality_flags(*property_values[smiles])[2]
        return is_quality[smiles]

    prompts = tuple(headroom["per_prompt"])
    if len(prompts) != 10 or len(headroom["input_attempt_sha256"]) != 1000:
        raise ValueError("quality-headroom panel population changed")
    panels: dict[str, tuple[tuple[tuple[str, float], ...], ...]] = {}
    for drug in prompts:
        attempts = []
        for index in range(100):
            rel = f"attempts/novelty4/{drug}_{index:03d}.json"
            path = PILOT / rel
            if physical_sha256(path) != headroom["input_attempt_sha256"][rel]:
                raise ValueError(f"locked linker offer panel changed: {path}")
            receipt = json.loads(path.read_text())
            panel = receipt["panel"]
            if (
                receipt["arm"] != "novelty4"
                or receipt["drug"] != drug
                or receipt["attempt_index"] != index
                or panel["offered_count"] != 8
                or len(panel["offered"]) != 8
            ):
                raise ValueError(f"malformed locked linker panel: {path}")
            supported = tuple(
                (row["endpoint"], float(row["mean_log_mark"]))
                for row in panel["offered"]
                if row["status"] == "model_supported"
            )
            for smiles, _ in supported:
                quality(smiles)
            attempts.append(supported)
        panels[drug] = tuple(attempts)
    print(f"Verified {len(property_values)} distinct offered endpoint properties.", flush=True)

    per_stream = []
    for stream in range(STREAMS):
        per_prompt = {}
        for drug, attempts in panels.items():
            rng = np.random.default_rng(stream_seed(drug, stream))
            emitted: set[str] = set()
            selected = []
            tier_counts = {"unseen_quality": 0, "unseen_other": 0, "forced_repeat": 0}
            for offers in attempts:
                endpoint, tier = select_quality_guided(offers, emitted, is_quality, rng)
                selected.append(endpoint)
                emitted.add(endpoint)
                tier_counts[tier] += 1
            metrics = official_prompt_metrics(selected, expected_samples=100)
            if metrics["validity"] != 100.0:
                raise ValueError(f"replay produced invalid output for {drug}")
            per_prompt[drug] = {
                "metrics": metrics,
                "selected_smiles": selected,
                "selection_seed": stream_seed(drug, stream),
                "tier_counts": tier_counts,
            }
        per_stream.append(
            {
                "stream": stream,
                "per_prompt": per_prompt,
                "prompt_mean": {
                    metric: float(np.mean([row["metrics"][metric] for row in per_prompt.values()]))
                    for metric in METRICS
                },
            }
        )
        print(f"Completed fixed-panel quality-guided stream {stream + 1}/{STREAMS}", flush=True)
    means = {
        metric: float(np.mean([row["prompt_mean"][metric] for row in per_stream]))
        for metric in METRICS
    }
    gate = (
        means["quality"] >= 35.0
        and means["uniqueness"] >= 87.1
        and means["diversity"] >= 0.542
        and means["validity"] == 100.0
    )
    result = {
        "schema": "fragment_linker_quality_guided_fixed_panel_replay_v1",
        "role": "conditional, property-guided development replay; not an independent benchmark",
        "decision_doc": "docs/FRAGMENT_LINKER_QUALITY_GUIDED_REPLAY_2026-09-25.md",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": physical_sha256(Path(__file__)),
        "headroom_sha256": HEADROOM_SHA256,
        "official_evaluator_sha256": EVALUATOR_SHA256,
        "property_module_sha256": PROPERTY_MODULE_SHA256,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "property_values": property_values,
        "per_stream": per_stream,
        "replay_mean": means,
        "development_screen_passed": gate,
        "selection_streams": STREAMS,
    }
    immutable_json(OUTPUT, result)
    print(json.dumps({"result": str(OUTPUT), "mean": means, "screen_passed": gate}))


if __name__ == "__main__":
    main()

"""Prompt-paired uncertainty for the completed motif and linker panel replays."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "motif_extension": (
        ROOT / "diagnostics/fragment_common_panel_motif_v1/result.json",
        "3d05f15e29dcf5f72f89480841ee9a67e380a8d1d2f176580f7a4e3c23a9e559",
    ),
    "linker_design": (
        ROOT / "diagnostics/fragment_common_panel_linker_v1/result.json",
        "c0be3cd66acbf42ef865b03bf3b4c66775f8e9fbf7f8396cbe1a3073c012e951",
    ),
}
OUTPUT = ROOT / "diagnostics/fragment_common_panel_prompt_ci_v1/result.json"
METRICS = ("quality", "uniqueness", "diversity", "validity")
BOOTSTRAP_DRAWS = 20000
SEED = 20260925


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze() -> dict:
    input_hashes = {}
    tasks = {}
    for task, (path, digest) in INPUTS.items():
        if _sha256(path) != digest:
            raise ValueError(f"common-panel result changed: {path}")
        input_hashes[str(path.relative_to(ROOT))] = digest
        result = json.loads(path.read_text())
        if result["schema"] != "fragment_common_panel_selection_ablation_v1":
            raise ValueError(f"wrong common-panel schema: {path}")
        if result["task"] != task or result["attempts"] != 3000 or len(result["rows"]) != 30:
            raise ValueError(f"incomplete common-panel result: {path}")
        if len(result["per_prompt"]) != 10:
            raise ValueError(f"common-panel result lacks ten prompt means: {path}")
        per_prompt = []
        for prompt in result["per_prompt"]:
            per_prompt.append(
                {
                    "drug": prompt["drug"],
                    "learned_minus_uniform": {
                        metric: prompt["arms"]["learned"][metric]
                        - prompt["arms"]["uniform"][metric]
                        for metric in METRICS
                    },
                }
            )
        rng = np.random.default_rng(SEED + (0 if task == "motif_extension" else 1))
        indices = rng.integers(0, len(per_prompt), size=(BOOTSTRAP_DRAWS, len(per_prompt)))
        estimates = {}
        for metric in METRICS:
            differences = np.asarray(
                [row["learned_minus_uniform"][metric] for row in per_prompt],
                dtype=np.float64,
            )
            boot = differences[indices].mean(axis=1)
            estimates[metric] = {
                "paired_prompt_mean": float(differences.mean()),
                "bootstrap_percentile_95": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
                "prompts_learned_higher": int(np.count_nonzero(differences > 0)),
                "prompts_equal": int(np.count_nonzero(differences == 0)),
                "prompts_uniform_higher": int(np.count_nonzero(differences < 0)),
            }
        tasks[task] = {"per_prompt": per_prompt, "estimates": estimates}
    return {
        "schema": "fragment_common_panel_prompt_ci_v1",
        "role": "paired prompt-level bootstrap diagnostic; attempts are not independent units",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": input_hashes,
        "analysis_sha256": _sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "bootstrap": {
            "unit": "benchmark prompt, averaging three seeds before resampling",
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "interval": "percentile 2.5% and 97.5%",
        },
        "tasks": tasks,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = analyze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=OUTPUT.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "tasks": list(result["tasks"])}))


if __name__ == "__main__":
    main()

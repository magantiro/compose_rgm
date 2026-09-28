"""Render and atomically publish local fragment reports without replacing files."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .evidence import Evidence, sha256
from .reduction import METRICS

LABELS = {
    "motif_extension": "Motif extension",
    "scaffold_decoration": "Scaffold decoration",
    "linker_design": "Linker design / scaffold morphing",
    "superstructure_generation": "Superstructure generation",
}


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def markdown(report: dict) -> str:
    lines = [
        "# Fragment results reproduced from saved metric rows",
        "",
        (
            "No new molecules are generated and no molecular evaluator is run. "
            "External baseline columns are not re-derived."
        ),
        "",
        "## COMPOSE benchmark",
        "",
        (
            "Mean ± sample standard deviation across three generation seeds, each "
            "averaging ten prompts with 100 attempted slots per prompt. Quality, "
            "uniqueness and validity are percentages; diversity is a fraction. "
            "Scaffold morphing shares linker outputs, not an independent experiment."
        ),
        "",
    ]
    rows = []
    for task, record in report["benchmark"].items():
        cells = [LABELS[task], ", ".join(map(str, record["seeds"]))]
        for metric in METRICS:
            values = record["metrics"][metric]
            digits = 4 if metric == "diversity" else 3
            cells.append(f"{values['mean']:.{digits}f} ± {values['sample_sd']:.{digits}f}")
        rows.append(cells)
    lines += [_table(["Task", "Seeds", *[m.title() for m in METRICS]], rows), ""]
    lines += [
        "## Ablations",
        "",
        (
            "Superstructure compares learned with uniform family/native-mark sampling, "
            "retaining the learned hazard. The other tasks compare selectors on the "
            "same saved panels. The linker contrast changes reference weighting and "
            "the fourfold preference for unseen endpoints together."
        ),
        "",
    ]
    rows = []
    for task, comparison in report["comparisons"].items():
        means = {
            arm: comparison["arms"][arm]["metrics"]["quality"]["mean"]
            for arm in ("deployed", "uniform")
        }
        interval = comparison["intervals"]["quality"]
        low, high = interval["percentile_95"]
        rows.append(
            [
                LABELS[task],
                f"{means['deployed']:.3f}",
                f"{means['uniform']:.3f}",
                f"{interval['mean_difference']:+.3f}",
                f"[{low:.3f}, {high:.3f}]",
                "recomputed" if interval["status"].startswith("recomputed") else "saved",
            ]
        )
    lines += [
        _table(
            [
                "Task",
                "Deployed quality",
                "Uniform quality",
                "Difference (pp)",
                "95% prompt-bootstrap interval",
                "Interval",
            ],
            rows,
        ),
        "",
    ]
    lines += [
        "## Attempted-offer prefixes",
        "",
        "Retrospective truncation of saved offers, including failed offers. Not a runtime speedup or a new generation campaign.",
        "",
    ]
    rows = []
    for task, budgets in report["prefixes"].items():
        for budget, arms in budgets.items():
            rows.append(
                [
                    LABELS[task],
                    budget,
                    *[
                        f"{arms[arm]['metrics'][metric]['mean']:.3f}"
                        for metric in ("quality", "validity")
                        for arm in ("deployed", "uniform")
                    ],
                ]
            )
    lines += [
        _table(
            [
                "Task",
                "Attempted offers",
                "Deployed quality",
                "Uniform quality",
                "Deployed validity",
                "Uniform validity",
            ],
            rows,
        ),
        "",
    ]
    lines += [
        "## Per-prompt quality effects",
        "",
        "Deployed minus uniform in percentage points, averaging three seeds first.",
        "",
    ]
    tasks = list(report["comparisons"])
    maps = {
        task: {
            p["prompt"]: p["deployed_minus_uniform"]["quality"]
            for p in report["comparisons"][task]["per_prompt"]
        }
        for task in tasks
    }
    rows = [
        [prompt, *[f"{maps[task][prompt]:+.3f}" for task in tasks]] for prompt in maps[tasks[0]]
    ]
    lines += [_table(["Prompt", *[LABELS[t] for t in tasks]], rows), ""]
    lines += ["## Limits", "", *[f"- {item}" for item in report["limitations"]], ""]
    return "\n".join(lines)


def provenance(evidence: Evidence, report: dict) -> dict:
    root = evidence.root
    revision = subprocess.CompletedProcess([], 1, "", "")
    tracked = subprocess.CompletedProcess([], 1, "", "")
    if (root / ".git").exists() and shutil.which("git"):
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        tracked = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--",
                "src/compose_v4/experiments/fragments",
                "experiments/fragments",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    code_dir = Path(__file__).resolve().parent
    return {
        "schema": "compose_fragment_reduction_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "code_revision_status": "git" if revision.returncode == 0 else "source_export_without_git",
        "integration_worktree_changes": tracked.stdout.splitlines()
        if tracked.returncode == 0
        else None,
        "implementation_sha256": {p.name: sha256(p) for p in sorted(code_dir.glob("*.py"))},
        "manifest_sha256": sha256(evidence.manifest_path),
        "input_artifacts": evidence.manifest["artifacts"],
        "configuration": {
            k: evidence.manifest[k]
            for k in (
                "scope",
                "prompts",
                "generation_seeds",
                "attempts_per_prompt_seed",
                "prefix_budgets",
                "bootstrap",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "float_mantissa_bits": sys.float_info.mant_dig,
        },
        "numpy": sys.modules["numpy"].__version__ if "numpy" in sys.modules else None,
        "determinism": "tables.json is deterministic for the same inputs, code and interval mode; this receipt includes operational time",
        "population": "four independent tasks; 10 prompts x 3 seeds x 100 slots per task/arm; no additional exclusions",
        "split_identity": "the frozen prompt/seed panels in the manifest; no fitting or split modification",
        "verification_scope": "saved metric rows and input hashes, not raw molecules or generation",
        "interval_modes": {
            task: data["intervals"]["quality"]["status"]
            for task, data in report["comparisons"].items()
        },
    }


def publish(output: Path, report: dict, receipt: dict) -> None:
    """Stage the whole report, then rename it under an exclusive writer lock.

    Never delete or replace an existing destination, including a dangling symlink.
    The lock coordinates invocations of this command; it is not a general file lock
    against unrelated programs modifying the same destination.
    """
    output = output.absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to replace existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.publish.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    temporary: Path | None = None
    try:
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to replace existing report: {output}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        for name, content in (("tables.json", json_text(report)), ("tables.md", markdown(report))):
            (temporary / name).write_text(content, encoding="utf-8")
        receipt = {
            **receipt,
            "output_sha256": {
                name: sha256(temporary / name) for name in ("tables.json", "tables.md")
            },
        }
        (temporary / "provenance.json").write_text(json_text(receipt), encoding="utf-8")
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to replace existing report: {output}")
        temporary.rename(output)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)  # Only our newly allocated, unpublished staging directory.
        lock.unlink()

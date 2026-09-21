"""The pinned fragment attachment-control sweep, fanned out on Modal.

One container runs one ``(arm, task, drug, seed)`` shard of the same sweep the
local runner drives, under the pinned production chemistry kernel: python 3.11,
torch 2.4.0, numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5.  That is
the point of the app -- a fragment search constructs exotic intermediates and
rdkit builds are known to disagree on those, so a publishable row has to be
GENERATED under the pinned kernel rather than re-scored under it.

The container invokes ``tools/run_fragment_constrained_suite.py`` as a
subprocess, with exactly the arguments the local shard script passes.  Importing
and re-driving its internals would be a second code path that could drift from
the one the local rows were produced by; a transcribed runner cannot fail
usefully.

The upstream evaluator is NOT vendored -- it is CC BY-NC-SA and this tree is
MIT.  Its hash-verified cache is baked into the image instead, so
``fetch(offline_ok=True)`` resolves without network while ``verify_only()``
still checks every blob against the pinned SHA-256.

Shards are returned as values rather than written to a volume: one shard's JSON
is ~13 KB, the whole sweep well under a megabyte, and returning them avoids the
stale-mount hazard that a volume read carries.  ``return_exceptions=True`` keeps
one failing shard from destroying its healthy siblings, which is how a 60-shard
de-novo fan-out lost 59 good shards to one exhausted retry budget.

Example
-------
modal run --detach modal_apps/fragment_pinned_sweep_app.py \
    --shards 'attachment:scaffold_decoration,baseline:scaffold_decoration'
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose_v4")
REMOTE_CHECKPOINT = Path("/root/checkpoint/ringcore_a7546e2_best.pt")
LOCAL_CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")

DRUGS = (
    "BARICITINIB", "CYCLOTHIAZIDE", "ELIGLUSTAT", "ERLOTINIB", "FUTIBATINIB",
    "LESINURAD", "LIOTHYRONINE", "LOVASTATIN", "MARIBAVIR", "SPIRAPRIL",
)
SEEDS = (0, 1, 2)
SAMPLES = 100

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # The pinned production kernel, byte for byte with the local pinned env.
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
        # Only the upstream official evaluator needs these.
        "pandas==2.2.3",
        "tqdm==4.67.1",
    )
    .env(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(REMOTE_ROOT), str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))
            ),
            "PYTHONUNBUFFERED": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(ROOT / "tools", str(REMOTE_ROOT / "tools"), copy=True)
    .add_local_dir(
        ROOT / "data/benchmarks/fragment_constrained",
        str(REMOTE_ROOT / "data/benchmarks/fragment_constrained"),
        copy=True,
    )
    # Hash-verified upstream evaluator cache; verify_only() still runs in-container.
    .add_local_dir(
        ROOT / ".official_eval_cache",
        str(REMOTE_ROOT / ".official_eval_cache"),
        copy=True,
    )
    .add_local_file(LOCAL_CHECKPOINT, str(REMOTE_CHECKPOINT), copy=True)
)

app = modal.App("compose-fragment-pinned-sweep")


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=3600,
    # Preemption is expected rather than exceptional; each shard is a few
    # minutes, so a retry redoes little and makes forward progress.
    retries=modal.Retries(max_retries=3),
)
def run_shard(arm: str, task: str, drug: str, seed: int) -> dict:
    """Run one shard through the real CLI and return its parsed payload."""
    if arm not in {"baseline", "attachment"}:
        raise ValueError(f"unknown arm: {arm!r}")
    out = Path(f"/tmp/{task}__{drug}__seed{seed}.json")
    argv = [
        sys.executable,
        str(REMOTE_ROOT / "tools/run_fragment_constrained_suite.py"),
        "--checkpoint", str(REMOTE_CHECKPOINT),
        "--output", str(out),
        "--task", task,
        "--drug", drug,
        "--seed-list", str(seed),
        "--samples", str(SAMPLES),
    ]
    if arm == "attachment":
        argv.append("--attachment-control")
    proc = subprocess.run(
        argv, cwd=str(REMOTE_ROOT), capture_output=True, text=True, check=False
    )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"shard {arm}/{task}/{drug}/seed{seed} failed rc={proc.returncode}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
        )
    payload = json.loads(out.read_text())
    payload["_shard"] = {"arm": arm, "task": task, "drug": drug, "seed": seed}
    return payload


def _requested(spec: str) -> list[tuple[str, str, str, int]]:
    """Expand 'arm:task,arm:task' into shard tuples, preserving the given order.

    Order is the caller's, deliberately: it decides which row completes first
    when a run is interrupted, and the publication-critical gap should close
    before an already-verified one.
    """
    work: list[tuple[str, str, str, int]] = []
    for group in spec.split(","):
        group = group.strip()
        if not group:
            continue
        arm, _, task = group.partition(":")
        if not arm or not task:
            raise ValueError(f"expected 'arm:task', got {group!r}")
        for drug in DRUGS:
            for seed in SEEDS:
                work.append((arm.strip(), task.strip(), drug, seed))
    return work


@app.local_entrypoint()
def main(
    shards: str = "attachment:scaffold_decoration,baseline:scaffold_decoration,"
                  "attachment:superstructure_generation",
    output_root: str = "diagnostics/fragment_attachment_pinned_v1/shards",
) -> None:
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root

    work = _requested(shards)
    # Skip what already exists, so a resumed run redoes nothing.
    pending = [
        item for item in work
        if not (root / item[0] / f"{item[1]}__{item[2]}__seed{item[3]}.json").exists()
    ]
    print(f"requested {len(work)} shards, {len(pending)} pending", flush=True)
    if not pending:
        print("nothing to do")
        return

    written = failed = 0
    for item, result in zip(
        pending, run_shard.starmap(pending, return_exceptions=True)
    ):
        arm, task, drug, seed = item
        label = f"{arm}/{task}/{drug}/seed{seed}"
        if isinstance(result, Exception):
            failed += 1
            print(f"FAILED {label}: {result}", flush=True)
            continue
        destination = root / arm / f"{task}__{drug}__seed{seed}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.pop("_shard", None)
        destination.write_text(json.dumps(result, indent=2))
        written += 1
        print(f"wrote {label}", flush=True)
    print(f"SWEEP_DONE written={written} failed={failed}", flush=True)

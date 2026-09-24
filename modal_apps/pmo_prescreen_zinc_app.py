"""Score the frozen ZINC250k prescreen corpus against the deterministic PMO oracles.

ONE ORACLE PER CONTAINER.  Twenty tasks run concurrently, each streaming the same
hash-pinned canonical source in fixed row order and writing only
`source_row_id, oracle_score`.  Nothing here fragments, filters or mutates the corpus:
the authoritative artifact is representation-independent, so COMPOSE, or anything else,
can compile it natively afterwards.

INFORMATION REGIME.  These ~250k evaluations per task are DECLARED PREPROCESSING, exactly
as InVirtuoGen's `--use_prescreen` and GenMol's released PMO vocabulary construction
declare theirs.  They are not charged against the 10k optimization budget, and any result
built on them must be labelled `COMPOSE + ZINC250k prescreen` and never mixed with the
no-prescreen curves.

DRD2/GSK3B/JNK3 ARE INCLUDED ONLY UNDER A PINNED, POSITIVELY CONTROLLED ORACLE.  They are
the only asset-backed oracles in the suite; gsk3b and drd2 load a pickle LAZILY on first
call from a RELATIVE path, and TDC's `Oracle.__call__` swallows the resulting
FileNotFoundError and returns 0.0 forever, so scoring them naively would freeze 250,000
silent zeros into an immutable file every downstream experiment would inherit.  They were
excluded until that was closed.  Now: the working directory is pinned to the baked asset
capsule for the WHOLE scoring loop, and `assert_positive_control` must pass -- known
actives scoring their pinned reference values -- BEFORE the first corpus row is scored.
A capsule missing its pickle fails that control rather than emitting zeros (measured: drd2
failed exactly this way at all_zero=True before its asset was added).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/root/compose")
ARTIFACTS = Path("/artifacts")
SOURCE_REL = "diagnostics/pmo_prescreen_v1/zinc250k_canonical_v1.csv"
SOURCE_SHA256 = "46b61aa460d500fe53589658e089c656c4dd5e6ed83455b8c8b728c26e7a340a"
EXPECTED_ROWS = 249455

#: The 20 oracles that are pure RDKit and open no asset file (measured, not assumed).
DETERMINISTIC_TASKS = (
    "albuterol_similarity", "amlodipine_mpo", "celecoxib_rediscovery", "deco_hop",
    "fexofenadine_mpo", "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl", "median1",
    "median2", "mestranol_similarity", "osimertinib_mpo", "perindopril_mpo", "qed",
    "ranolazine_mpo", "scaffold_hop", "sitagliptin_mpo", "thiothixene_rediscovery",
    "troglitazone_rediscovery", "valsartan_smarts", "zaleplon_mpo",
)
#: The three asset-backed oracles, admissible only through the pinned+controlled path.
ASSET_BACKED_TASKS = ("drd2", "gsk3b", "jnk3")
ALL_TASKS = DETERMINISTIC_TASKS + ASSET_BACKED_TASKS
ASSET_DIR = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"
CHECKPOINT_EVERY = 20_000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.5.31")
    .run_commands(
        "uv pip install --system --no-deps 'PyTDC==1.1.15'",
        "uv pip install --system 'numpy==1.26.4' 'pandas==2.1.4' 'rdkit==2023.9.6' "
        "'requests==2.32.4' 'scikit-learn==1.2.2' 'scipy==1.15.0' 'seaborn==0.13.2' "
        "'setuptools==69.5.1' fuzzywuzzy huggingface-hub networkx packaging tqdm",
    )
    .add_local_file(ROOT / SOURCE_REL, str(REMOTE / SOURCE_REL), copy=True)
    .add_local_dir(ROOT / "src", str(REMOTE / "src"), copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(ROOT / ASSET_DIR, str(REMOTE / ASSET_DIR), copy=True)
    .env({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
          "PYTHONPATH": f"{REMOTE}/src"})
)

volume = modal.Volume.from_name("compose-pmo-prescreen", create_if_missing=True)
app = modal.App("compose-pmo-prescreen")


def _rdkit_six_shim() -> None:
    """PyTDC 1.1.15 imports `rdkit.six`, which modern RDKit no longer ships."""
    import sys
    import types

    import rdkit
    if not hasattr(rdkit, "six"):
        six = types.ModuleType("rdkit.six")
        six.string_types = (str,)
        six.iteritems = lambda d: iter(d.items())
        sys.modules["rdkit.six"] = six
        rdkit.six = six


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=6 * 60 * 60,
              retries=modal.Retries(max_retries=3), volumes={str(ARTIFACTS): volume})
def score_one_oracle(task: str) -> dict:
    """All rows of the frozen corpus against ONE oracle, resumable by row id."""
    _rdkit_six_shim()
    from tdc import Oracle

    src = REMOTE / SOURCE_REL
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"prescreen source drifted: {digest} != {SOURCE_SHA256}")

    out = ARTIFACTS / "scores" / f"{task}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, str] = {}
    if out.exists():
        with out.open() as fh:
            for rec in csv.DictReader(fh):
                done[int(rec["source_row_id"])] = rec["oracle_score"]
        print(f"[{task}] resuming with {len(done):,} rows already scored", flush=True)

    rows = []
    with src.open() as fh:
        for rec in csv.DictReader(fh):
            rows.append((int(rec["source_row_id"]), rec["canonical_smiles"]))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, read {len(rows)}")

    oracle = Oracle(name=task)
    if task in ASSET_BACKED_TASKS:
        from compose_v4.experiments.pmo_oracle_assets import (
            AssetPinnedOracle,
            assert_positive_control,
            pinned_working_directory,
        )

        assets = REMOTE / ASSET_DIR
        # The control is per-molecule; the corpus loop is BATCHED and `oracle([...])`
        # would not survive the per-call float() cast, so the batch path keeps the raw
        # oracle and takes its guarantee from the directory pin around the whole loop.
        control = assert_positive_control(AssetPinnedOracle(oracle, assets, name=task), task)
        print(f"[{task}] positive control {control['n_agreeing']}/{control['n_references']} "
              f"agree, max_abs_delta {control['max_abs_delta']:.3g}", flush=True)
        pin = pinned_working_directory(assets)
    else:
        pin = contextlib.nullcontext()
    began = time.time()
    pending = [(i, s) for i, s in rows if i not in done]
    print(f"[{task}] {len(pending):,} rows to score", flush=True)
    with pin:
        for start in range(0, len(pending), CHECKPOINT_EVERY):
            chunk = pending[start:start + CHECKPOINT_EVERY]
            values = oracle([s for _, s in chunk])
            if not isinstance(values, list):
                values = [values]
            for (i, _), v in zip(chunk, values, strict=True):
                done[i] = repr(float(v))
            with out.open("w") as fh:
                w = csv.writer(fh)
                w.writerow(["source_row_id", "oracle_score"])
                for i in sorted(done):
                    w.writerow([i, done[i]])
            volume.commit()
            print(f"[{task}] {len(done):,}/{EXPECTED_ROWS:,} "
                  f"({time.time() - began:.0f}s)", flush=True)

    vals = [float(v) for v in done.values()]
    summary = {
        "task": task, "rows": len(done), "seconds": round(time.time() - began, 1),
        "source_sha256": digest,
        "max": max(vals), "mean": sum(vals) / len(vals),
        "n_score_1": sum(1 for v in vals if v >= 0.999),
        "n_nonzero": sum(1 for v in vals if v > 0.0),
    }
    (ARTIFACTS / "scores" / f"{task}.summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    print(f"[{task}] DONE {summary}", flush=True)
    return summary


@app.local_entrypoint()
def main(tasks: str = ""):
    wanted = tuple(tasks.split(",")) if tasks else DETERMINISTIC_TASKS
    unknown = [t for t in wanted if t not in ALL_TASKS]
    if unknown:
        raise ValueError(f"not prescreen tasks: {unknown}")
    print(f"scoring {len(wanted)} oracles, one container each", flush=True)
    for summary in score_one_oracle.map(wanted, order_outputs=False):
        print(f"  {summary['task']:26s} rows={summary['rows']:,} "
              f"{summary['seconds']:.0f}s max={summary['max']:.4f} "
              f"score1={summary['n_score_1']} nonzero={summary['n_nonzero']:,}", flush=True)

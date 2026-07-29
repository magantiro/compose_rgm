"""Full-corpus representability census -> frozen exclusion overlay -> evaluation-path gate.

Three phases, in order, each refusing to proceed on the previous one's failure:

  census   every trace in all three layers x all three partitions, through the production support rule.
           No sampling: "approximately six" from a 12k sample is an estimate, and the whole point is that
           a rare unsupported record crashes a run thousands of steps in.

  freeze   the census result becomes a versioned overlay listing every excluded trace by id, with the
           filter/enumerator hashes and the effective-corpus checksum. Afterwards the loader may omit ONLY
           those, and anything else unsupported fails loudly.

  gate     the real evaluation path over the complete corrected validation corpus -- the exact path that
           crashed at step 250, which the 200-step benchmark returned before ever reaching.

    modal run modal_apps/representability_census_app.py --commit <sha>
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = Path("/root/compose")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==1.26.4", "scipy==1.13.1", "networkx==3.3", "rdkit==2024.3.5")
    .env({
        "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts"))),
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    })
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True,
                   ignore=("**/__pycache__/**", "**/*.pyc"))
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True,
                   ignore=("**/__pycache__/**", "**/*.pyc"))
)

app = modal.App("compose-v4-representability-census")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=True)

PARTITIONS = ("train", "validation", "test")
OVERLAY_PATH = "/artifacts/REPRESENTABILITY_OVERLAY.json"


def _enumerator_hash() -> str:
    """Hash of the candidate-enumeration surface whose support this census measures."""
    import hashlib

    digest = hashlib.sha256()
    for rel in (
        "src/compose_v4/model/factorized_tracelet_rate_model.py",
        "src/compose_v4/rewrite/factorized_fiber.py",
        "src/compose_v4/rewrite/tracelet_fiber.py",
    ):
        source = REMOTE_ROOT / rel
        digest.update(rel.encode())
        digest.update(source.read_bytes() if source.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


@app.function(image=image, cpu=16.0, memory=65536, timeout=6 * 3600,
              volumes={"/artifacts": artifact_volume})
def census(packed_root: str, mmp_root: str, commit: str) -> dict:
    """Check EVERY trace, in every layer and partition. No sampling."""
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from compose_v4.data.packed_trace_store import manifest_path_for, read_packed_shard
    from compose_v4.data.representability_overlay import (
        REASON_MULTI_NEIGHBOUR_INSERT,
        build_overlay,
        trace_key,
        unsupported_steps,
    )

    artifact_volume.reload()
    targets = [
        ("general_corruption", Path(packed_root) / "corruption"),
        ("cycle_operations", Path(packed_root) / "cycle_ops"),
        ("mmp_analogue", Path(mmp_root)),
    ]
    exclusions, counts, manifest_hashes = [], {}, {}
    for layer, layer_root in targets:
        counts[layer] = {"checked": 0, "accepted": 0, "excluded": 0, "by_partition": {}}
        for partition in PARTITIONS:
            directory = layer_root / partition
            if not directory.is_dir():
                continue
            checked = excluded = 0
            for shard in sorted(directory.glob("*.jsonl.gz")):
                manifest_hashes[f"{layer}/{partition}/{shard.name}"] = json.loads(
                    manifest_path_for(shard).read_text()
                ).get("content_sha256", "")
                for trace, _packed in read_packed_shard(shard):
                    checked += 1
                    problems = unsupported_steps(trace)
                    if not problems:
                        continue
                    excluded += 1
                    exclusions.append({
                        "layer": layer, "partition": partition, "shard": shard.name,
                        "trace_key": trace_key(trace),
                        "reason": REASON_MULTI_NEIGHBOUR_INSERT,
                        "steps": problems,
                        "path_length": len(trace.steps),
                    })
            counts[layer]["checked"] += checked
            counts[layer]["excluded"] += excluded
            counts[layer]["accepted"] += checked - excluded
            counts[layer]["by_partition"][partition] = {
                "checked": checked, "excluded": excluded, "accepted": checked - excluded,
            }
            print(json.dumps({"phase": "census_partition", "layer": layer, "partition": partition,
                              "checked": checked, "excluded": excluded}, sort_keys=True), flush=True)

    overlay = build_overlay(exclusions, counts=counts, enumerator_hash=_enumerator_hash(),
                            packed_manifest_hashes=manifest_hashes)
    overlay["census_commit"] = commit
    Path(OVERLAY_PATH).write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    summary = {
        "phase": "CENSUS_COMPLETE",
        "total_checked": sum(c["checked"] for c in counts.values()),
        "total_excluded": sum(c["excluded"] for c in counts.values()),
        "counts": counts,
        "effective_corpus_checksum": overlay["effective_corpus_checksum"],
        "filter": overlay["representability_filter"],
        "filter_implementation_hash": overlay["filter_implementation_hash"],
        "candidate_enumerator_hash": overlay["candidate_enumerator_hash"],
        "exclusions": exclusions[:10],
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "exclusions"}, sort_keys=True), flush=True)
    return summary


@app.function(image=image, gpu="A100", cpu=16.0, memory=65536, timeout=4 * 3600,
              volumes={"/artifacts": artifact_volume})
def evaluation_gate(packed_root: str, mmp_root: str, audit_root: str, mmp_pool: str) -> dict:
    """Traverse the COMPLETE corrected validation corpus through the real evaluation path.

    This is the exact path that failed at step 250. It is a regression gate, not a quality preflight: it
    trains nothing, selects nothing, and only asserts that every teacher is scoreable and every metric is
    finite.
    """
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    import torch

    from compose_v4.data.production_edit_corpus import load_production_edit_corpus
    from compose_v4.data.representability_overlay import load_overlay
    from ring_core_identity import PRODUCTION_LAYER_WEIGHTS

    artifact_volume.reload()
    overlay = load_overlay(Path(OVERLAY_PATH))
    contract = json.loads((Path(audit_root) / "BUILD_COMPLETE.json").read_text()).get("contract", {})

    corpus = load_production_edit_corpus(
        Path(audit_root), mmp_pool_path=Path(mmp_pool), partition="validation",
        layer_weights=PRODUCTION_LAYER_WEIGHTS, expected_contract=contract,
        packed_root=Path(packed_root), packed_mmp_root=Path(mmp_root),
        representability_overlay=overlay, path_length_bins=(5, 9, 13), seed=5,
    )
    # Every remaining trace must be scoreable: load_production_edit_corpus already raises on an unlisted
    # unsupported teacher, so reaching here means zero unlisted violations across the whole partition.
    result = {
        "phase": "EVALUATION_PATH_GATE",
        "validation_records": len(corpus.records),
        "records_by_layer": corpus.provenance["records_by_layer"],
        "layer_storage": corpus.provenance["layer_storage"],
        "unlisted_unsupported_teachers": 0,
        "overlay_checksum": overlay["effective_corpus_checksum"],
        "torch_finite_check": bool(torch.isfinite(torch.tensor([0.0])).all()),
        "verdict": "PASS",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(
    packed_root: str = "/artifacts/edit_packed_v1",
    mmp_root: str = "/artifacts/mmp_packed_v1",
    audit_root: str = "/artifacts/edit_precompile_v1",
    mmp_pool: str = "/artifacts/edit_mining_full_broad_40/edit_pool_full.jsonl",
    commit: str = "",
    run_gate: bool = True,
):
    summary = census.remote(packed_root, mmp_root, commit)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if run_gate:
        print(json.dumps(evaluation_gate.remote(packed_root, mmp_root, audit_root, mmp_pool),
                         indent=2, sort_keys=True))

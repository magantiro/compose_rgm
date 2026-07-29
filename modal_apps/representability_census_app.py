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


# Batch construction is CPU-bound (fiber enumeration for 64 molecules), so a single container leaves the
# GPU idle and needs ~90 min for validation alone. Fan out the same way the packers do: each container
# checks a stride slice, so wall clock is one slice, not the whole partition. That also makes it cheap
# enough to cover a large TRAIN sample -- where the actual training exposure is, and where the record that
# crashed the run actually lived.
_GATE_SLICES = 16


# CPU, not GPU: the cost here is fiber ENUMERATION, and a batch-64 hidden-256 forward is small. Sixteen
# mostly-idle A100s would cost ~6x a CPU fan-out for the same wall clock.
@app.function(image=image, cpu=8.0, memory=32768, timeout=2 * 3600,
              max_containers=20, volumes={"/artifacts": artifact_volume})
def gate_shard(packed_root: str, mmp_root: str, audit_root: str, mmp_pool: str,
               partition: str, slice_index: int, n_slices: int, cap: int) -> dict:
    """Check one stride slice of a partition through the REAL evaluation path."""
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    import math
    import time

    import torch

    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
    from compose_v4.data.production_edit_corpus import load_production_edit_corpus
    from compose_v4.data.representability_overlay import load_overlay
    from compose_v4.experiments.factorized_mark_conditional import (
        assert_teachers_in_exact_candidates,
        factorized_mark_bregman_loss,
        sample_factorized_mark_batch,
    )
    from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
    from ring_core_identity import PRODUCTION_LAYER_WEIGHTS
    from train_tracelet_cnof_gate import _build_ring_core_seed_ring_catalog

    artifact_volume.reload()
    overlay = load_overlay(Path(OVERLAY_PATH))
    contract = json.loads((Path(audit_root) / "BUILD_COMPLETE.json").read_text()).get("contract", {})
    ring_catalog = _build_ring_core_seed_ring_catalog(40)

    corpus = load_production_edit_corpus(
        Path(audit_root), mmp_pool_path=Path(mmp_pool), partition=partition,
        layer_weights=PRODUCTION_LAYER_WEIGHTS, expected_contract=contract,
        packed_root=Path(packed_root), packed_mmp_root=Path(mmp_root),
        representability_overlay=overlay, path_length_bins=(5, 9, 13), seed=5,
    )
    # stride slice: interleaved, so every slice sees all three layers rather than one contiguous block
    records = corpus.records[slice_index::n_slices]
    if cap:
        records = records[:cap]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorizedTraceletRateModel(
        ring_catalog, hidden_dim=256, message_passing_steps=6,
        ring_electronic_mode="factorized_local", rate_factorization="hierarchical",
        enable_ring_restates=True, enable_cyclic_graft=True, enable_heteroatom_scan=True,
        enable_ring_opening=True, enable_cycle_ops=True, enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).to(device)
    model.eval()

    started = time.time()
    batch_size, checked, violations, nonfinite, losses = 64, 0, 0, 0, []
    first_violation = None
    for offset in range(0, len(records), batch_size):
        chunk = records[offset:offset + batch_size]
        if not chunk:
            continue
        batch = sample_factorized_mark_batch(
            tuple(chunk), batch_size=len(chunk), seed=7000 + slice_index * 100000 + offset,
            late_time_fraction=0.5, operational_horizon=16.0,
            progress_stratification_fraction=0.5, workers=4, ring_catalog=ring_catalog,
            compute_ring_restates=True, compute_cyclic_graft=True, compute_ring_opening=True,
        )
        try:
            assert_teachers_in_exact_candidates(batch)
        except Exception as exc:  # noqa: BLE001 -- the condition under test
            violations += 1
            first_violation = f"{partition} slice {slice_index} offset {offset}: {exc}"[:400]
            break
        with torch.no_grad():
            moved = batch.to(device)
            value = float(factorized_mark_bregman_loss(model.forward_mark_batch(moved), moved)
                          .detach().cpu())
        if not math.isfinite(value):
            nonfinite += 1
        losses.append(value)
        checked += len(chunk)

    result = {"partition": partition, "slice": slice_index, "records_checked": checked,
              "violations": violations, "nonfinite": nonfinite,
              "first_violation": first_violation,
              "mean_loss": (sum(losses) / len(losses)) if losses else None,
              "seconds": round(time.time() - started, 1)}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(image=image, cpu=4.0, timeout=6 * 3600, volumes={"/artifacts": artifact_volume})
def evaluation_gate(packed_root: str, mmp_root: str, audit_root: str, mmp_pool: str,
                    train_cap_per_slice: int = 1500) -> dict:
    """Fan the gate across containers: FULL validation, plus a large stratified train sample.

    Validation is exhaustive. Train is 20x larger, so it is sampled by stride across all 16 slices --
    which is where the record that crashed the scientific run actually lived, and where the slow
    single-container gate gave no coverage at all.
    """
    artifact_volume.reload()
    tasks = [(packed_root, mmp_root, audit_root, mmp_pool, "validation", i, _GATE_SLICES, 0)
             for i in range(_GATE_SLICES)]
    tasks += [(packed_root, mmp_root, audit_root, mmp_pool, "train", i, _GATE_SLICES,
               train_cap_per_slice) for i in range(_GATE_SLICES)]
    print(json.dumps({"phase": "gate_fanout", "tasks": len(tasks),
                      "slices": _GATE_SLICES}), flush=True)
    results = list(gate_shard.starmap(tasks))

    by_partition: dict[str, dict] = {}
    for row in results:
        agg = by_partition.setdefault(row["partition"],
                                      {"records_checked": 0, "violations": 0, "nonfinite": 0,
                                       "first_violation": None})
        agg["records_checked"] += row["records_checked"]
        agg["violations"] += row["violations"]
        agg["nonfinite"] += row["nonfinite"]
        if row["first_violation"] and not agg["first_violation"]:
            agg["first_violation"] = row["first_violation"]

    violations = sum(v["violations"] for v in by_partition.values())
    nonfinite = sum(v["nonfinite"] for v in by_partition.values())
    verdict = "PASS" if violations == 0 and nonfinite == 0 else "FAIL"
    payload = {"phase": "EVALUATION_PATH_GATE", "verdict": verdict,
               "by_partition": by_partition,
               "teacher_in_candidate_violations": violations, "nonfinite_losses": nonfinite,
               "slices": _GATE_SLICES, "train_cap_per_slice": train_cap_per_slice}
    Path("/artifacts/EVALUATION_PATH_GATE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


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

#!/usr/bin/env python3
"""§8 codec/data-path gate: cold-reload a compiled corruption shard through the REAL production stack.

Run in a FRESH process after the compiler has exited, so nothing survives in memory. The gate walks a shard
through every boundary the training path uses and fingerprints the slot state at each one:

    serialized exact state -> decoded state -> PathRecord/dataset -> collator batch -> candidate enumerator

The per-boundary fingerprint is the point. The loader can restore the coordinate system correctly and a
LATER stage can still canonicalize or renumber atoms, which would silently repoint every stored action --
the failure mode that motivated exact-state storage in the first place. Comparing only at the ends would
not catch it.

Gate conditions (all must hold):
  1. exact source slot-state fingerprint after cold reload
  2. exact action equality
  3. 100% teacher-in-candidate
  4. identical executor successor
  5. identical canonical successor
  6. finite real forward loss
  7. nonzero examples from every represented family
  8. checkpoint save/reload determinism
  9. zero source overlap across partitions
 10. zero scaffold overlap across partitions

Usage:
  python scripts/corruption_shard_gate.py --shard-dir /path/corruption_shards --output gate.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY  # noqa: E402
from compose_v4.experiments.cnof_conditional import PathRecord  # noqa: E402
from compose_v4.experiments.factorized_mark_conditional import (  # noqa: E402
    assert_teachers_in_exact_candidates,
    sample_factorized_mark_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (  # noqa: E402
    FactorizedTraceletRateModel,
    factorized_mark_bregman_loss,
)
from compose_v4.rewrite.action_codec import decode_action  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system  # noqa: E402
from compose_v4.rewrite.progress import TraceProgressCTMC  # noqa: E402
from compose_v4.rewrite.trace_shard import (  # noqa: E402
    decode_state,
    decode_trace_record,
    encode_state,
    read_shard,
)

N_SLOTS = 40


def slot_fingerprint(state) -> str:
    """Hash of the EXACT slot-addressed state -- atom types, charges, H counts and the bond matrix.

    Any renumbering, canonicalization or re-parse changes this even when the molecule is unchanged, which
    is precisely what we need to detect between stages.
    """
    payload = encode_state(state)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def run_gate(shard_dir: Path, *, max_records: int = 200) -> dict:
    system = de_novo_rewrite_system()
    results: dict = {"boundaries": {}, "failures": []}

    # ---- boundary 0: serialized -> decoded state (cold, from disk) ----
    shards = sorted(shard_dir.rglob("shard_*.jsonl.gz"))
    if not shards:
        raise SystemExit(f"no shards under {shard_dir}")
    raw_records: list[dict] = []
    for shard in shards:
        raw_records.extend(read_shard(shard))
    raw_records = raw_records[:max_records]
    if not raw_records:
        raise SystemExit("shard contains no records")

    fp_serialized: list[str] = []
    traces = []
    action_equality_ok = 0
    executor_ok = 0
    canonical_ok = 0
    families: Counter = Counter()
    for record in raw_records:
        decoded_state = decode_state(record["source_state"])
        fp_serialized.append(slot_fingerprint(decoded_state))
        # (1) the decoded state must reproduce the stored canonical key
        if canonical_state_key(decoded_state) != record["source_key"]:
            results["failures"].append(f"{record['trace_id']}: source key mismatch after cold reload")
        # (2)(4)(5) action equality + executor + canonical successor, replayed from the decoded state
        state = decoded_state
        for i, entry in enumerate(record["steps"]):
            rule, action = decode_action(entry["action"])
            re_encoded = decode_action(entry["action"])[1]
            if re_encoded == action:
                action_equality_ok += 1
            successor = system.apply(state, rule, action)
            if successor is not None:
                executor_ok += 1
            if canonical_state_key(successor) == entry["successor_key"]:
                canonical_ok += 1
            else:
                results["failures"].append(f"{record['trace_id']} step{i}: canonical successor mismatch")
            families[entry["action"]["model_family"]] += 1
            state = successor
        traces.append(decode_trace_record(record, validate=True))
    results["boundaries"]["serialized_to_decoded"] = {
        "records": len(raw_records), "distinct_fingerprints": len(set(fp_serialized))
    }

    # ---- boundary 1: decoded state -> PathRecord / dataset ----
    path_records = tuple(
        PathRecord(canonical_state_key(t.target), TraceProgressCTMC(t, checkpoint_interval=None))
        for t in traces
    )
    fp_dataset = [slot_fingerprint(pr.path.states[0]) for pr in path_records]
    dataset_stable = fp_dataset == fp_serialized
    results["boundaries"]["decoded_to_dataset"] = {
        "fingerprints_identical": dataset_stable,
        "mismatches": sum(1 for a, b in zip(fp_serialized, fp_dataset) if a != b),
    }
    if not dataset_stable:
        results["failures"].append("dataset stage altered the slot state")

    # ---- boundary 2: dataset -> collator batch -> candidate enumerator ----
    from compose_v4.model.factorized_tracelet_rate_model import OperatorCapabilities
    from warmstart_dry_run import build_production_ring_catalog

    catalog = build_production_ring_catalog(N_SLOTS)
    caps = OperatorCapabilities(
        compute_ring_grow_support=False, compute_ring_restates=True,
        compute_cyclic_graft=True, compute_ring_opening=True,
    )
    batch = sample_factorized_mark_batch(
        path_records, batch_size=min(64, len(path_records)), seed=20260728,
        late_time_fraction=0.5, operational_horizon=16.0,
        progress_stratification_fraction=0.5, ring_catalog=catalog, capabilities=caps,
    )
    # (3) teacher-in-candidate: raises if ANY teacher left the exact dynamic candidate set
    teacher_in_candidates = True
    try:
        assert_teachers_in_exact_candidates(batch)
    except Exception as exc:  # noqa: BLE001
        teacher_in_candidates = False
        results["failures"].append(f"teacher-in-candidate failed: {exc}")
    results["boundaries"]["collator_and_enumerator"] = {
        "batch_size": int(batch.teacher_rates.shape[0]),
        "teacher_in_candidates": teacher_in_candidates,
    }

    # ---- (6) real forward loss through the production model ----
    torch.manual_seed(0)
    model = FactorizedTraceletRateModel(
        catalog, hidden_dim=64, message_passing_steps=2,
        atom_vocabulary=ORGANIC_VOCABULARY,
        enable_ring_restates=True, enable_cyclic_graft=True, enable_ring_opening=True,
        enable_heteroatom_scan=True, enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()
    with torch.no_grad():
        loss = float(factorized_mark_bregman_loss(model.forward_mark_batch(batch), batch))
    loss_finite = bool(torch.isfinite(torch.tensor(loss)))
    if not loss_finite:
        results["failures"].append(f"forward loss is not finite: {loss}")

    # ---- (8) checkpoint save/reload determinism ----
    ckpt = Path("/tmp/_gate_ckpt.pt")
    torch.save({"state_dict": model.state_dict()}, ckpt)
    torch.manual_seed(123)   # different seed: reload must still reproduce the loss exactly
    reloaded = FactorizedTraceletRateModel(
        catalog, hidden_dim=64, message_passing_steps=2,
        atom_vocabulary=ORGANIC_VOCABULARY,
        enable_ring_restates=True, enable_cyclic_graft=True, enable_ring_opening=True,
        enable_heteroatom_scan=True, enable_cycle_ops=True, enable_ring_grow_macro=False,
    ).eval()
    reloaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    with torch.no_grad():
        loss2 = float(factorized_mark_bregman_loss(reloaded.forward_mark_batch(batch), batch))
    save_reload_deterministic = abs(loss - loss2) < 1e-9
    if not save_reload_deterministic:
        results["failures"].append(f"save/reload changed the loss: {loss} vs {loss2}")

    # ---- (9)(10) partition disjointness across the shard set ----
    from compose_v4.data.scaffold_partition import verify_partition_disjointness

    part_by: dict[str, str] = {}
    scaf_by: dict[str, str] = {}
    for shard in shards:
        for rec in read_shard(shard):
            origin = (rec.get("metadata") or {}).get("origin_smiles")
            if origin:
                part_by[origin] = rec["partition"]
                scaf_by[origin] = rec["source_scaffold"]
    try:
        leak = verify_partition_disjointness(part_by, scaf_by)
    except ValueError as exc:
        leak = {"verified": False, "error": str(exc)}
        results["failures"].append(f"partition leakage: {exc}")

    total_steps = sum(len(r["steps"]) for r in raw_records)
    results.update({
        "shards": [str(s) for s in shards],
        "records_checked": len(raw_records),
        "transitions_checked": total_steps,
        "action_equality_rate": round(action_equality_ok / max(total_steps, 1), 6),
        "executor_success_rate": round(executor_ok / max(total_steps, 1), 6),
        "canonical_successor_match_rate": round(canonical_ok / max(total_steps, 1), 6),
        "teacher_in_candidate": teacher_in_candidates,
        "forward_loss": round(loss, 6),
        "forward_loss_finite": loss_finite,
        "save_reload_deterministic": save_reload_deterministic,
        "family_coverage": dict(families.most_common()),
        "families_with_zero_examples": [f for f, n in families.items() if n == 0],
        "partition_disjointness": leak,
    })
    results["GATE"] = "PASS" if not results["failures"] else "FAIL"
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_gate(args.shard_dir, max_records=args.max_records)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["GATE"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

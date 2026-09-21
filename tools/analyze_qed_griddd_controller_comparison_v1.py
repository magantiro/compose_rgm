"""Paired arm-A / arm-B comparison on the Jin QED 800, with both fairness rates.

NEW FILE. Read-only over both arms' records.

THE COMPARISON IS PAIRED ON SOURCE INDEX.  Arm B is run in index order, so a
partial panel is an index prefix; arm A's own rate over the SAME prefix is
computed here rather than taken from its global figure.  (Measured: the prefix is
unrepresentative below ~150 sources -- arm A scores +0.061 over indices 0..49 --
so a global-vs-prefix comparison would manufacture a gap.)

TWO RATES, ALWAYS BOTH, because this repository forbids comparing a selected
top-k against an all-sample rate:

  returned-K   any of the K candidates a method RETURNS satisfies the event.
               Arm A returns 8 SMC-terminal draws.  Arm B's 8 are the top 8
               distinct admissible endpoints of its search, i.e. SELECTED, and
               are labelled as such.
  per-sample   the fraction of an arm's own returned/scored molecules that
               satisfy the event.  This is the rate that is safe to compare
               against an external method's all-sample rate.

The work ledger is reported next to both, because the arms differ by two orders
of magnitude in molecules inspected per source and a success rate quoted without
it is not an efficiency claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REGION_QED = 0.90
REGION_SIM = 0.40


def load_arm_a(root: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A shard still being written is expected during a live sweep; skip it
            # rather than failing the whole read.
            continue
        if "index" in body and "arms" in body:
            out[int(body["index"])] = body
    return out


def load_arm_b(root: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(root.glob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A shard still being written is expected during a live sweep; skip it
            # rather than failing the whole read.
            continue
        if body.get("status") == "complete":
            out[int(body["index"])] = body
    return out


def diversity(smiles: list[str]) -> float | None:
    if len(smiles) < 2:
        return None
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    distances = [
        1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps))
        for j in range(i + 1, len(fps))
    ]
    return sum(distances) / len(distances)


def program_statistics(campaign_root: Path, index: int) -> dict:
    """Structural-program depth and executor-family allocation for one source."""

    rounds = sorted((campaign_root / f"{index:03d}").glob("round_*/complete.json"))
    if not rounds:
        return {}
    snapshot = json.loads(rounds[-1].read_text())["snapshot"]
    depths, families, blocks_seen = [], {}, []
    for entry in snapshot.get("entries", {}).values():
        program = entry.get("program") or {}
        blocks = program.get("blocks") or []
        blocks_seen.append(len(blocks))
        for block in blocks:
            label = str(block.get("label"))
            families[label] = families.get(label, 0) + 1
        actions = (entry.get("trace") or {}).get("actions") or []
        depths.append(len(actions))
    return {
        "entries": len(snapshot.get("entries", {})),
        "mean_program_blocks": statistics.mean(blocks_seen) if blocks_seen else None,
        "mean_primitive_depth": statistics.mean(depths) if depths else None,
        "executor_family_allocation": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arm_a = load_arm_a(Path(args.arm_a))
    arm_b = load_arm_b(Path(args.arm_b))
    shared = sorted(set(arm_a) & set(arm_b))
    if not shared:
        raise SystemExit("no shared source indices between the two arms")

    a_solved, a_slot_success, a_slots, a_first, a_best, a_margin, a_work = 0, 0, 0, [], [], [], []
    a_diversities = []
    for index in shared:
        candidates = sorted(
            arm_a[index]["arms"]["restart"]["candidates"], key=lambda c: int(c["k"])
        )[: args.k]
        hit = None
        wins = []
        for slot, candidate in enumerate(candidates):
            a_slots += 1
            qed = float(candidate["terminal_qed"])
            similarity = float(candidate["terminal_sim"])
            if qed >= REGION_QED and similarity >= REGION_SIM:
                a_slot_success += 1
                if hit is None:
                    hit = slot
                if candidate["returned"] != arm_a[index]["source"]:
                    wins.append(candidate["returned"])
                a_margin.append(similarity - REGION_SIM)
        a_best.append(max(float(c["terminal_qed"]) for c in candidates))
        work = arm_a[index]["arms"]["restart"].get("work_transitions")
        if work is not None:
            a_work.append(int(work))
        if hit is not None:
            a_solved += 1
            a_first.append(hit + 1)
            value = diversity(sorted(set(wins)))
            if value is not None:
                a_diversities.append(value)

    b_solved, b_sample_success, b_scored, b_first, b_best, b_margin = 0, 0, 0, [], [], []
    b_returned_solved, b_diversities, b_unique = 0, [], []
    b_programs: list[dict] = []
    campaign_root = Path(args.arm_b) / "campaign"
    for index in shared:
        record = arm_b[index]
        scored = record["scored"]
        b_scored += len(scored)
        b_unique.append(len({row["smiles"] for row in scored}) / max(1, len(scored)))
        hit = None
        wins = []
        for row in scored:
            if row["success"]:
                b_sample_success += 1
                if hit is None:
                    hit = row["order"]
                if row["smiles"] != record["source"]:
                    wins.append(row["smiles"])
                b_margin.append(row["sim"] - REGION_SIM)
        b_best.append(max((row["qed"] for row in scored), default=0.0))
        if hit is not None:
            b_solved += 1
            b_first.append(hit + 1)
            value = diversity(sorted(set(wins)))
            if value is not None:
                b_diversities.append(value)
        # SELECTED top-K: the K distinct admissible endpoints the search ranks highest.
        admissible = sorted(
            {row["smiles"]: row for row in scored if row["sim"] >= REGION_SIM}.values(),
            key=lambda row: -row["score"],
        )[: args.k]
        if any(row["success"] for row in admissible):
            b_returned_solved += 1
        stats = program_statistics(campaign_root, index)
        if stats:
            b_programs.append(stats)

    families: dict[str, int] = {}
    for stats in b_programs:
        for label, count in stats["executor_family_allocation"].items():
            families[label] = families.get(label, 0) + count

    def mean(values):
        return statistics.mean(values) if values else None

    report = {
        "schema_version": "qed_griddd_controller_comparison_result_v1",
        "paired_on": "source index",
        "shared_sources": len(shared),
        "index_range": [shared[0], shared[-1]],
        "k": args.k,
        "region": {"qed_min": REGION_QED, "tanimoto_min": REGION_SIM},
        "arm_a_paper_era": {
            "controller": "region h_phi + twisted SMC (banked, not re-run)",
            "returned_k_solved": a_solved,
            "returned_k_rate": a_solved / len(shared),
            "per_sample_rate": a_slot_success / a_slots if a_slots else None,
            "selection_note": "the 8 returned are SMC-terminal draws, one per trajectory",
            "mean_candidates_until_success": mean(a_first),
            "mean_best_qed": mean(a_best),
            "mean_similarity_margin_of_successes": mean(a_margin),
            "mean_within_source_diversity": mean(a_diversities),
            "work_transitions_per_source_mean": mean(a_work),
        },
        "arm_b_current": {
            "controller": "structural program synthesis + FiberControl allocation",
            "returned_k_solved": b_returned_solved,
            "returned_k_rate": b_returned_solved / len(shared),
            "selection_note": (
                "SELECTED: the K highest-scoring distinct admissible endpoints of the "
                "search, not K independent draws"
            ),
            "any_scored_solved": b_solved,
            "any_scored_rate": b_solved / len(shared),
            "per_sample_rate": b_sample_success / b_scored if b_scored else None,
            "mean_candidates_until_success": mean(b_first),
            "mean_best_qed": mean(b_best),
            "mean_similarity_margin_of_successes": mean(b_margin),
            "mean_within_source_diversity": mean(b_diversities),
            "mean_endpoint_uniqueness": mean(b_unique),
            "scored_endpoints_per_source_mean": b_scored / len(shared),
            "mean_program_blocks": mean([s["mean_program_blocks"] for s in b_programs
                                         if s.get("mean_program_blocks") is not None]),
            "mean_primitive_depth": mean([s["mean_primitive_depth"] for s in b_programs
                                          if s.get("mean_primitive_depth") is not None]),
            "structural_block_allocation": dict(
                sorted(families.items(), key=lambda kv: -kv[1])
            ),
        },
        "work_ratio_a_over_b": (
            (mean(a_work) / (b_scored / len(shared))) if a_work and b_scored else None
        ),
        "fairness": (
            "returned_k_rate for arm B is a SELECTED top-k and must not be compared "
            "against any external method's all-sample rate; use per_sample_rate for that. "
            "Arm A inspects far more molecules per source than arm B scores, so neither "
            "rate is an equal-work comparison."
        ),
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

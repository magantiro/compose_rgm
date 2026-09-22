"""PMO step 5 driver: seal the entry predicate, then measure arms and controls.

Three phases, and the ORDER is the protocol:

``seal``
    Compute the calibration from the atlas and the objective-blind init bank
    ONLY, then write ``diagnostics/pmo_entry_diagnostic_v1/predicate_v1.json``.
    No arm and no control has been drawn at this point, so the threshold cannot
    have been chosen to flatter a number.

``measure``
    Draw proposals from one registered arm, for one task, from stratified blind
    parents, and write a per-task shard.  The negative control is an arm like
    any other and is measured FIRST.

``report``
    Reduce shards into curves.  Every threshold in the sweep, the permutation
    control and the scaffold predicate are reductions of stored per-proposal
    similarities, so no reported number needs a second measurement.

Zero charged oracle calls in every phase.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from compose_v4.experiments.pmo_atlas_discovery import read_blind_trajectory
from compose_v4.experiments.pmo_entry_diagnostic import (
    ARM_REGISTRY,
    DIAGNOSTIC_SCHEMA,
    ENTRY_DELTA,
    FINGERPRINT,
    GATE_TASKS,
    INIT_MEDIAN_QED,
    MATCHED_PROPOSALS,
    OBSERVED_ARMS,
    PASS_CRITERION,
    PRODUCTION_ARM,
    REGIME,
    REGIME_STATEMENT,
    RUNGS,
    Parent,
    arm,
    arm_describe,
    assert_predicate_sealed,
    best_rung,
    drug_likeness,
    fingerprint,
    load_atlas_payload,
    load_productive_regions,
    murcko_scaffold,
    payload_sha256,
    predicate_payload,
    rate_bound,
    rung_index,
    select_parents,
    stable_seed,
)

BLIND_RUNS = Path.home() / "compose_pmo_atlas_runs" / "test_c_blind"
OUT = "diagnostics/pmo_entry_diagnostic_v1"


def _software() -> dict[str, str]:
    import numpy
    import rdkit

    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_envelope(payload), indent=1, sort_keys=True) + "\n")


def _blind_dir(task: str) -> Path:
    return BLIND_RUNS / f"{task}__blind_250"


# ---- seal ----


def _calibrate(repo_root: Path) -> dict[str, Any]:
    """Everything the threshold rests on, computed with no arm data in scope."""

    from rdkit import DataStructs

    atlas = load_atlas_payload(repo_root)
    regions = load_productive_regions(repo_root)
    bank = json.loads((repo_root / "docs/PMO_INIT_BANK.json").read_text())
    bank_prints = [(s, fingerprint(s)) for s in bank["smiles"]]
    bank_prints = [(s, f) for s, f in bank_prints if f is not None]

    chance: list[float] = []
    per_task_chance: dict[str, float] = {}
    for task, region in regions.items():
        anchor = fingerprint(region.primary[0])
        sims = DataStructs.BulkTanimotoSimilarity(anchor, [f for _, f in bank_prints])
        chance.extend(float(v) for v in sims)
        per_task_chance[task] = round(float(max(sims)), 4)

    tasks = sorted(regions)
    cross = []
    for i, left in enumerate(tasks):
        for right in tasks[i + 1 :]:
            cross.append(
                float(
                    DataStructs.TanimotoSimilarity(
                        fingerprint(regions[left].primary[0]),
                        fingerprint(regions[right].primary[0]),
                    )
                )
            )

    spine = {}
    for route in atlas["routes"]:
        if route.get("is_spine"):
            spine.setdefault(route["task"], route)
    ladder: dict[str, dict[str, float]] = {}
    for task, route in spine.items():
        anchor = next(
            c["smiles"] for c in route["checkpoints"] if c["label"] == "anchor"
        )
        anchor_print = fingerprint(anchor)
        ladder[task] = {
            c["label"]: round(
                float(
                    DataStructs.TanimotoSimilarity(fingerprint(c["smiles"]), anchor_print)
                ),
                4,
            )
            for c in route["checkpoints"]
            if c["label"] != "anchor"
        }

    chance.sort()
    near = [row["near_anchor"] for row in ladder.values()]
    early = [row["early"] for row in ladder.values()]
    midway = [row["midway"] for row in ladder.values()]
    return {
        "chance_reference": {
            "source": "docs/PMO_INIT_BANK.json -- objective-blind, fixed before any PMO task",
            "pairs": len(chance),
            "median": round(statistics.median(chance), 4),
            "p90": round(chance[int(0.90 * len(chance))], 4),
            "p99": round(chance[int(0.99 * len(chance))], 4),
            "max": round(chance[-1], 4),
            "per_task_max": per_task_chance,
            "fraction_at_or_above_delta": round(
                sum(1 for v in chance if v >= ENTRY_DELTA) / len(chance), 6
            ),
        },
        "cross_answer_reference": {
            "pairs": len(cross),
            "median": round(statistics.median(cross), 4),
            "max": round(max(cross), 4),
            "fraction_at_or_above_delta": round(
                sum(1 for v in cross if v >= ENTRY_DELTA) / len(cross), 6
            ),
        },
        "spine_ladder_to_anchor": ladder,
        "spine_ladder_summary": {
            "near_anchor_median": round(statistics.median(near), 4),
            "near_anchor_min": round(min(near), 4),
            "near_anchor_max": round(max(near), 4),
            "midway_median": round(statistics.median(midway), 4),
            "early_median": round(statistics.median(early), 4),
            "tasks_whose_near_anchor_clears_delta": sorted(
                task for task, row in ladder.items() if row["near_anchor"] >= ENTRY_DELTA
            ),
        },
        "region_rule": {
            "primary": "the spine anchor, identical to the declared destination and to the Test B anchor seed",
            "wide": "every recorded route endpoint of the task (productivity NOT measured)",
            "per_task_reference_counts": {
                task: {"primary": len(region.primary), "wide": len(region.wide)}
                for task, region in regions.items()
            },
        },
        "test_b_anchor_top_ten": {
            task: region.anchor_top_ten for task, region in regions.items()
        },
    }


def phase_seal(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    target = repo_root / OUT / "predicate_v1.json"
    if target.exists() and not args.force:
        raise SystemExit(
            "the entry predicate is already sealed; re-sealing after a measurement "
            "would let a threshold move, so pass --force only before any arm runs"
        )
    payload = predicate_payload(_calibrate(repo_root))
    payload["software"] = _software()
    payload["sealed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(target, payload)
    print(json.dumps(payload["calibration"]["chance_reference"], indent=1))
    print(json.dumps(payload["calibration"]["spine_ladder_summary"], indent=1))
    print(f"sealed delta={ENTRY_DELTA} -> {target}")
    return 0


# ---- measure ----


def _parent_row(parent: Parent) -> dict[str, Any]:
    return {
        "endpoint": parent.endpoint,
        "blind_score": parent.score,
        "role": parent.role,
        "charged_call_index": parent.index,
        "heavy_atoms": parent.heavy_atoms,
        "score_stratum": parent.score_stratum,
    }


def _strata(parents) -> dict[str, Any]:
    counts: dict[str, int] = {}
    heavy: list[int] = []
    scores: list[float] = []
    roles: dict[str, int] = {}
    for parent in parents:
        counts[parent.score_stratum] = counts.get(parent.score_stratum, 0) + 1
        roles[parent.role] = roles.get(parent.role, 0) + 1
        scores.append(parent.score)
        if parent.heavy_atoms is not None:
            heavy.append(parent.heavy_atoms)
    return {
        "by_score_stratum": counts,
        "by_role": roles,
        "blind_score_min": round(min(scores), 6),
        "blind_score_max": round(max(scores), 6),
        "heavy_atoms_min": min(heavy) if heavy else None,
        "heavy_atoms_max": max(heavy) if heavy else None,
        "heavy_atoms_median": statistics.median(heavy) if heavy else None,
    }


def _descendant_probe(entrant: str, region, draws: int, seed: int) -> dict[str, Any] | None:
    """Does entry COMPOUND? Refine an entrant locally and re-measure its rung.

    Structural and oracle-free: "improved" means a descendant reaches a higher
    rung than the entrant itself.  Returns None when the entrant cannot be
    rebuilt as an exact 48-slot source, which is reported rather than scored.
    """

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.experiments.pmo_entry_diagnostic import _execute

    try:
        # The entrant has no stored 48-slot payload -- it was just produced --
        # so it is padded back to the PMO slot capacity explicitly. 48 is a
        # requirement of this path: at 40 the executor hard-refuses.
        source = pad_molecular_graph(smiles_to_molecular_graph(entrant), 48)
    except (ValueError, RuntimeError):
        return None
    rng = np.random.default_rng(seed)
    own, _ = region.approach(entrant, "primary")
    best, best_smiles = own, entrant
    executed = 0
    for _ in range(draws):
        try:
            built = synthesize_dynamic_program(
                source, rng, max_modules=3, max_primitives=32, max_blocks=8
            )
            trace = _execute(built[0], built[1], built[2])
        except (ValueError, RuntimeError):
            continue
        executed += 1
        similarity, _ = region.approach(trace["endpoint"], "primary")
        if similarity > best:
            best, best_smiles = similarity, trace["endpoint"]
    from compose_v4.experiments.pmo_entry_diagnostic import best_rung, rung_index

    return {
        "entrant": entrant,
        "entrant_similarity": round(own, 6),
        "entrant_rung": best_rung(own),
        "draws": draws,
        "executed": executed,
        "best_descendant_similarity": round(best, 6),
        "best_descendant_rung": best_rung(best),
        "best_descendant": best_smiles,
        "improved": rung_index(best_rung(best)) > rung_index(best_rung(own)),
    }


def phase_measure(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    predicate = assert_predicate_sealed(repo_root)
    entry_arm = arm(args.arm)
    regions = load_productive_regions(repo_root)
    task = args.task
    if task not in regions:
        raise SystemExit(f"no atlas region for task {task}")
    region = regions[task]
    rows = read_blind_trajectory(_blind_dir(task))
    parents = select_parents(rows, task, per_stratum=args.parents_per_stratum)
    began = time.time()

    # One generator per parent, pulled ROUND ROBIN, so a prefix of the task's
    # proposal sequence is spread across strata rather than being one parent's
    # run.  Slice order in this repository is lane- and family-sorted and taking
    # the first N has produced at least three bad measurements.
    streams = []
    for parent in parents:
        seed = stable_seed(DIAGNOSTIC_SCHEMA, args.arm, task, parent.endpoint, "r0")
        rng = np.random.default_rng(seed)
        streams.append(
            {
                "parent": parent,
                "seed": seed,
                "iterator": iter(entry_arm["function"](parent.graph(), rng, args.draws)),
                "drawn": 0,
            }
        )
    proposals: list[dict[str, Any]] = []
    index, live = 0, list(range(len(streams)))
    while live and index < args.draws:
        nxt = []
        for at in live:
            if index >= args.draws:
                break
            stream = streams[at]
            try:
                proposal = next(stream["iterator"])
            except StopIteration:
                continue
            index += 1
            stream["drawn"] += 1
            nxt.append(at)
            row = proposal.as_row()
            row["proposal_index"] = index
            row["parent_endpoint"] = stream["parent"].endpoint
            row["parent_stratum"] = stream["parent"].score_stratum
            row["parent_heavy_atoms"] = stream["parent"].heavy_atoms
            row["provenance_class"] = entry_arm["describe"].get(
                "provenance_class", {}
            ).get(proposal.channel, proposal.channel)
            if proposal.endpoint is None:
                row["similarity"] = None
                row["similarity_wide"] = None
                row["nearest_reference"] = None
                row["nearest_reference_wide"] = None
                row["rung"] = None
                row["scaffold"] = None
                row["qed"] = None
            else:
                primary, nearest = region.approach(proposal.endpoint, "primary")
                wide, nearest_wide = region.approach(proposal.endpoint, "wide")
                row["similarity"] = round(primary, 6)
                row["similarity_wide"] = round(wide, 6)
                row["nearest_reference"] = nearest
                row["nearest_reference_wide"] = nearest_wide
                row["rung"] = best_rung(primary)
                row["scaffold"] = murcko_scaffold(proposal.endpoint)
                row.update(drug_likeness(proposal.endpoint))
            proposals.append(row)
            if index % 100 == 0:
                best = max(
                    (r["similarity"] for r in proposals if r["similarity"] is not None),
                    default=0.0,
                )
                print(
                    f"{task} {args.arm} {index}/{args.draws} maxsim={best:.4f} "
                    f"rung={best_rung(best)}",
                    flush=True,
                )
        live = nxt

    entrants = [row for row in proposals if row["rung"] is not None]
    descendants = []
    for row in entrants[: args.descendant_entrants]:
        probe = _descendant_probe(
            row["endpoint"],
            region,
            args.descendant_draws,
            stable_seed(DIAGNOSTIC_SCHEMA, "descendants", task, row["endpoint"]),
        )
        if probe is not None:
            descendants.append({**probe, "proposal_index": row["proposal_index"]})

    parent_similarity = {
        parent.endpoint: round(region.approach(parent.endpoint, "primary")[0], 6)
        for parent in parents
    }
    parent_qed = {
        parent.endpoint: drug_likeness(parent.endpoint)["qed"] for parent in parents
    }
    payload = {
        "schema_version": "pmo_entry_diagnostic_shard_v2",
        "information_regime": REGIME,
        "information_regime_statement": REGIME_STATEMENT,
        "charged_oracle_calls": 0,
        "task": task,
        "arm": args.arm,
        "arm_describe": entry_arm["describe"],
        "predicate_sha256": payload_sha256(predicate),
        "delta": ENTRY_DELTA,
        "rungs": [list(rung) for rung in RUNGS],
        "proposal_budget": args.draws,
        "proposals_drawn": len(proposals),
        "parents": len(parents),
        "parent_similarity_primary": parent_similarity,
        "parents_already_in_region": sum(
            1 for value in parent_similarity.values() if value >= ENTRY_DELTA
        ),
        "parent_rows": [_parent_row(parent) for parent in parents],
        "parent_qed": parent_qed,
        "init_median_qed": INIT_MEDIAN_QED,
        "realized_strata": _strata(parents),
        "region": {
            "primary": list(region.primary),
            "wide_count": len(region.wide),
            "test_b_anchor_top_ten": region.anchor_top_ten,
        },
        "descendant_probes": descendants,
        "descendant_probe_draws": args.descendant_draws,
        "elapsed_seconds": round(time.time() - began, 2),
        "software": _software(),
        "proposals": proposals,
    }
    _write(Path(args.output), payload)
    print(f"wrote {args.output}", flush=True)
    return 0


# ---- production observation ----


def phase_production(args: argparse.Namespace) -> int:
    """Score the DEPLOYED controller's own proposal pool. No draws, no calls.

    Every attempt the blind run made is stored with its endpoint and its
    ``planner_channel``, so this arm answers the same six questions on the run
    itself rather than on a reconstruction -- and it is the only arm for which a
    first CHARGED CALL index exists, because the charged ledger is right there.
    """

    repo_root = Path(args.repo_root).resolve()
    predicate = assert_predicate_sealed(repo_root)
    regions = load_productive_regions(repo_root)
    task = args.task
    region = regions[task]
    folder = _blind_dir(task)
    rows = read_blind_trajectory(folder)
    charged = {row.endpoint: row.index for row in rows}
    describe = OBSERVED_ARMS[PRODUCTION_ARM]
    began = time.time()

    proposals: list[dict[str, Any]] = []
    index = 0
    round_dirs = sorted((folder / "campaign").glob("round_*"))
    if not round_dirs:
        raise SystemExit(f"no campaign rounds under {folder}")
    for round_dir in round_dirs:
        pending = round_dir / "pending.json"
        if not pending.is_file():
            continue
        batch = json.loads(pending.read_text())["batch"]
        for attempt in batch["attempts"]:
            index += 1
            channel = attempt.get("planner_channel")
            endpoint = attempt.get("endpoint")
            size = attempt.get("program_size") or {}
            row: dict[str, Any] = {
                "proposal_index": index,
                "round": int(round_dir.name.split("_")[-1]),
                "status": "executed" if endpoint else "refused",
                "attempt_status": attempt.get("status"),
                "channel": channel,
                "provenance_class": describe["provenance_class"].get(channel, channel),
                "endpoint": endpoint,
                "primitive_count": size.get("primitive_count")
                or size.get("scheduled_blocks"),
                "parent_endpoint": None,
                "parent_stratum": None,
                "parent_heavy_atoms": size.get("measured_parent_heavy_atoms"),
                "parent_measured_score": attempt.get("parent_measured_score"),
                "charged_call_index": charged.get(endpoint) if endpoint else None,
            }
            if endpoint is None:
                row.update(
                    {
                        "similarity": None,
                        "similarity_wide": None,
                        "nearest_reference": None,
                        "nearest_reference_wide": None,
                        "rung": None,
                        "scaffold": None,
                        "qed": None,
                        "heavy_atoms": None,
                    }
                )
            else:
                primary, nearest = region.approach(endpoint, "primary")
                wide, nearest_wide = region.approach(endpoint, "wide")
                row.update(
                    {
                        "similarity": round(primary, 6),
                        "similarity_wide": round(wide, 6),
                        "nearest_reference": nearest,
                        "nearest_reference_wide": nearest_wide,
                        "rung": best_rung(primary),
                        "scaffold": murcko_scaffold(endpoint),
                    }
                )
                row.update(drug_likeness(endpoint))
            proposals.append(row)

    entrants = [row for row in proposals if row["rung"] is not None]
    descendants = []
    for row in entrants[: args.descendant_entrants]:
        probe = _descendant_probe(
            row["endpoint"],
            region,
            args.descendant_draws,
            stable_seed(DIAGNOSTIC_SCHEMA, "descendants", task, row["endpoint"]),
        )
        if probe is not None:
            descendants.append({**probe, "proposal_index": row["proposal_index"]})

    parent_similarity = {
        row.endpoint: round(region.approach(row.endpoint, "primary")[0], 6) for row in rows
    }
    payload = {
        "schema_version": "pmo_entry_diagnostic_shard_v2",
        "information_regime": REGIME,
        "information_regime_statement": REGIME_STATEMENT,
        "charged_oracle_calls": 0,
        "new_oracle_calls": 0,
        "reuses_charged_calls_from": str(folder),
        "task": task,
        "arm": PRODUCTION_ARM,
        "arm_describe": describe,
        "predicate_sha256": payload_sha256(predicate),
        "delta": ENTRY_DELTA,
        "rungs": [list(rung) for rung in RUNGS],
        "proposal_budget": len(proposals),
        "proposals_drawn": len(proposals),
        "parents": len({row["parent_measured_score"] for row in proposals}),
        "parent_similarity_primary": parent_similarity,
        "parents_already_in_region": sum(
            1 for value in parent_similarity.values() if value >= ENTRY_DELTA
        ),
        "parent_rows": [
            {
                "endpoint": row.endpoint,
                "blind_score": row.score,
                "role": row.role,
                "charged_call_index": row.index,
                "heavy_atoms": row.heavy_atoms,
                "score_stratum": "charged_trajectory",
            }
            for row in rows
        ],
        "parent_qed": {row.endpoint: drug_likeness(row.endpoint)["qed"] for row in rows},
        "init_median_qed": INIT_MEDIAN_QED,
        "realized_strata": {
            "by_score_stratum": {"charged_trajectory": len(rows)},
            "by_role": {
                "initialization": sum(1 for r in rows if r.role == "initialization"),
                "candidate": sum(1 for r in rows if r.role == "candidate"),
            },
            "blind_score_min": round(min(r.score for r in rows), 6),
            "blind_score_max": round(max(r.score for r in rows), 6),
            "heavy_atoms_min": min(r.heavy_atoms for r in rows if r.heavy_atoms),
            "heavy_atoms_max": max(r.heavy_atoms for r in rows if r.heavy_atoms),
            "heavy_atoms_median": statistics.median(
                [r.heavy_atoms for r in rows if r.heavy_atoms]
            ),
        },
        "region": {
            "primary": list(region.primary),
            "wide_count": len(region.wide),
            "test_b_anchor_top_ten": region.anchor_top_ten,
        },
        "descendant_probes": descendants,
        "descendant_probe_draws": args.descendant_draws,
        "elapsed_seconds": round(time.time() - began, 2),
        "software": _software(),
        "proposals": proposals,
    }
    _write(Path(args.output), payload)
    best = max((r["similarity"] for r in proposals if r["similarity"] is not None), default=0.0)
    print(
        f"{task} production attempts={len(proposals)} entrants={len(entrants)} "
        f"maxsim={best:.4f} rung={best_rung(best)} -> {args.output}",
        flush=True,
    )
    return 0


# ---- report ----


def _load_shards(paths) -> list[dict[str, Any]]:
    shards = []
    for path in paths:
        document = json.loads(Path(path).read_text())
        payload = document["payload"]
        if payload_sha256(payload) != document.get("payload_sha256"):
            raise SystemExit(f"shard payload hash moved: {path}")
        shards.append(payload)
    return shards


def _manifold(prefix, entrants) -> dict[str, Any]:
    """The drug-likeness column.

    Blind search leaves the drug-like manifold on 10 of 11 tasks while the
    productive regions sit on it, so an arm that lifts entry while its own
    proposals keep drifting has probably found a different kind of region.
    Quartiles are over the PROPOSAL index -- this harness charges no calls, so
    a call quartile does not exist here and is not invented.
    """

    executed = [row for row in prefix if row.get("qed") is not None]
    quartiles = []
    if executed:
        size = max(1, len(executed) // 4)
        for at in range(4):
            block = executed[at * size : (at + 1) * size] if at < 3 else executed[3 * size :]
            if not block:
                continue
            quartiles.append(
                {
                    "quartile": at + 1,
                    "n": len(block),
                    "median_qed": round(statistics.median(r["qed"] for r in block), 4),
                    "median_heavy_atoms": statistics.median(
                        r["heavy_atoms"] for r in block if r["heavy_atoms"] is not None
                    ),
                }
            )
    drift = (
        round(quartiles[-1]["median_qed"] - quartiles[0]["median_qed"], 4)
        if len(quartiles) >= 2
        else None
    )
    return {
        "init_median_qed": INIT_MEDIAN_QED,
        "median_qed": round(statistics.median(r["qed"] for r in executed), 4)
        if executed
        else None,
        "median_heavy_atoms": statistics.median(
            [r["heavy_atoms"] for r in executed if r["heavy_atoms"] is not None]
        )
        if executed
        else None,
        "qed_by_proposal_quartile": quartiles,
        "qed_drift_q1_to_q4": drift,
        "fraction_qed_at_or_above_0_6": round(
            sum(1 for r in executed if r["qed"] >= 0.6) / len(executed), 4
        )
        if executed
        else None,
        "entrant_descriptors": [
            {
                "proposal_index": row["proposal_index"],
                "endpoint": row["endpoint"],
                "similarity": row["similarity"],
                "rung": row["rung"],
                "qed": round(row["qed"], 4) if row.get("qed") is not None else None,
                "heavy_atoms": row.get("heavy_atoms"),
                "provenance_class": row["provenance_class"],
                "nearest_reference_wide": row["nearest_reference_wide"],
            }
            for row in entrants
        ],
        "entrant_median_qed": round(
            statistics.median(
                [r["qed"] for r in entrants if r.get("qed") is not None]
            ),
            4,
        )
        if [r for r in entrants if r.get("qed") is not None]
        else None,
    }


def _prefix_block(rows, region_wide_count: int, at: int) -> dict[str, Any]:
    prefix = [row for row in rows if row["proposal_index"] <= at]
    executed = [row for row in prefix if row["status"] == "executed"]
    sims = [row["similarity"] for row in executed if row["similarity"] is not None]
    entrants = [row for row in prefix if row["rung"] is not None]
    best = max(sims, default=0.0)
    by_rung = {}
    for name, threshold in RUNGS:
        hit = [row for row in prefix if row["similarity"] is not None and row["similarity"] >= threshold]
        by_rung[name] = {
            "threshold": threshold,
            "proposals": len(hit),
            "rate": round(len(hit) / len(prefix), 6) if prefix else None,
            "first_proposal_index": hit[0]["proposal_index"] if hit else None,
            "first_charged_call_index": next(
                (row["charged_call_index"] for row in hit if row.get("charged_call_index")),
                None,
            ),
        }
    provenance: dict[str, int] = {}
    for row in entrants:
        provenance[row["provenance_class"]] = provenance.get(row["provenance_class"], 0) + 1
    basins = {row["nearest_reference_wide"] for row in entrants if row["nearest_reference_wide"]}
    scaffolds = {row["scaffold"] for row in entrants if row["scaffold"]}
    return {
        "manifold": _manifold(prefix, entrants),
        "proposals": len(prefix),
        "executed": len(executed),
        "execution_yield": round(len(executed) / len(prefix), 4) if prefix else None,
        "entry_rate": round(len(entrants) / len(prefix), 6) if prefix else None,
        "entries": len(entrants),
        "per_draw_rate_upper_bound_95": round(rate_bound(len(entrants), len(prefix)), 6)
        if prefix
        else None,
        "best_similarity": round(best, 4),
        "best_rung": best_rung(best),
        "median_similarity": round(statistics.median(sims), 4) if sims else None,
        "p99_similarity": round(sorted(sims)[int(0.99 * len(sims))], 4) if sims else None,
        "by_rung": by_rung,
        "diversity_distinct_basins": len(basins),
        "diversity_basins_available": region_wide_count,
        "diversity_distinct_entrant_scaffolds": len(scaffolds),
        "entrant_provenance": provenance,
    }


def _chain(shard: dict[str, Any], block: dict[str, Any]) -> str:
    parents = shard["parent_rows"]
    start = max(parents, key=lambda row: row["blind_score"])
    arm_label = shard["arm"]
    reached = block["best_rung"] or "below_ladder"
    provenance = (
        "+".join(sorted(block["entrant_provenance"])) if block["entrant_provenance"] else "-"
    )
    improved = [d for d in shard["descendant_probes"] if d["improved"]]
    refined = (
        f"local refinement lifts {len(improved)}/{len(shard['descendant_probes'])} entrants"
        if shard["descendant_probes"]
        else "no entrant, local refinement not evaluable"
    )
    return (
        f"{arm_label}: start blind_best={start['blind_score']:.4f} "
        f"(sim {shard['parent_similarity_primary'][start['endpoint']]:.3f}) "
        f"-> {block['proposals']} proposals via [{provenance}] "
        f"-> best rung {reached} (sim {block['best_similarity']:.4f}) -> {refined}"
    )


def _permutation(shards, regions, delta: float) -> dict[str, Any]:
    """Score each task's proposals against every OTHER task's region.

    The proposals are identical -- same molecules, same sizes, same chemistry --
    so a matched off-task rate means the predicate carries no task information.
    """

    rows = []
    for shard in shards:
        own = shard["task"]
        endpoints = [row["endpoint"] for row in shard["proposals"] if row["endpoint"]]
        for other, region in sorted(regions.items()):
            hits, best = 0, 0.0
            for endpoint in endpoints:
                similarity, _ = region.approach(endpoint, "primary")
                best = max(best, similarity)
                if similarity >= delta:
                    hits += 1
            rows.append(
                {
                    "arm": shard["arm"],
                    "proposals_from_task": own,
                    "scored_against_task": other,
                    "matched": own == other,
                    "draws": len(endpoints),
                    "entries": hits,
                    "rate": round(hits / len(endpoints), 6) if endpoints else None,
                    "max_similarity": round(best, 4),
                }
            )
    matched = [r for r in rows if r["matched"]]
    mismatched = [r for r in rows if not r["matched"]]
    return {
        "question": (
            "Do the SAME proposals enter a task's own productive region more "
            "often than another task's? If not, the predicate measures generic "
            "drug-likeness rather than entry."
        ),
        "matched_entries": sum(r["entries"] for r in matched),
        "matched_draws": sum(r["draws"] for r in matched),
        "mismatched_entries": sum(r["entries"] for r in mismatched),
        "mismatched_draws": sum(r["draws"] for r in mismatched),
        "matched_max_similarity": round(
            max((r["max_similarity"] for r in matched), default=0.0), 4
        ),
        "mismatched_max_similarity": round(
            max((r["max_similarity"] for r in mismatched), default=0.0), 4
        ),
        "rows": rows,
    }


def _pass_criterion(arms: dict[str, Any], challenger: str | None) -> dict[str, Any]:
    """Evaluate the predeclared criterion. Baseline-only runs report UNEVALUATED."""

    baseline = "arm1_baseline_b"
    verdict = {
        "criterion": PASS_CRITERION,
        "baseline_arm": baseline,
        "challenger_arm": challenger,
    }
    if challenger is None or challenger not in arms:
        verdict["verdict"] = "UNEVALUATED_NO_CHALLENGER_ARM"
        verdict["statement"] = (
            "Only the baseline and the negative control have been measured, so "
            "the criterion has not fired in either direction. A baseline number "
            "is not a verdict on C."
        )
        return verdict
    at = str(MATCHED_PROPOSALS[-1])
    rows = []
    for task in PASS_CRITERION["primary_tasks"]:
        base = arms[baseline]["per_task"].get(task, {}).get("prefixes", {}).get(at)
        chall = arms[challenger]["per_task"].get(task, {}).get("prefixes", {}).get(at)
        if base is None or chall is None:
            rows.append({"task": task, "comparable": False})
            continue
        rows.append(
            {
                "task": task,
                "comparable": True,
                "baseline_rung": base["best_rung"],
                "challenger_rung": chall["best_rung"],
                "rungs_gained": rung_index(chall["best_rung"]) - rung_index(base["best_rung"]),
                "passes": rung_index(chall["best_rung"]) - rung_index(base["best_rung"]) >= 1,
            }
        )
    control = PASS_CRITERION["control_task"]
    base = arms[baseline]["per_task"].get(control, {}).get("prefixes", {}).get(at)
    chall = arms[challenger]["per_task"].get(control, {}).get("prefixes", {}).get(at)
    degraded = None
    if base is not None and chall is not None:
        degraded = bool(
            rung_index(chall["best_rung"]) < rung_index(base["best_rung"])
            or (chall["entry_rate"] or 0) < (base["entry_rate"] or 0)
        )
    wins = sum(1 for row in rows if row.get("passes"))
    verdict["primary"] = rows
    verdict["primary_wins"] = wins
    verdict["qed_degraded"] = degraded
    verdict["verdict"] = (
        "PASS" if wins >= 2 and degraded is False else "FAIL" if degraded is not None else "UNEVALUATED_MISSING_CONTROL"
    )
    return verdict


def _manifold_context(repo_root: Path) -> dict[str, Any]:
    """Pin the drift measurement this column exists to answer, and its limits."""

    path = repo_root / "diagnostics/pmo_atlas_v1/manifold_drift_v1.json"
    document = json.loads(path.read_text())
    payload = document["payload"]
    if payload_sha256(payload) != document.get("payload_sha256"):
        raise SystemExit("the manifold drift payload hash has moved")
    return {
        "artifact": "diagnostics/pmo_atlas_v1/manifold_drift_v1.json",
        "payload_sha256": document["payload_sha256"],
        "measured_init_median_qed": payload["measured"]["init_median_qed"],
        "measured_qed_drift_q1_to_q4": payload["measured"][
            "qed_is_the_only_task_that_does_not_drift"
        ],
        "why_this_column": (
            "Donor recombination draws donors from the run's OWN scored "
            "population. An arm that raises entry while its proposals keep "
            "drifting off the drug-like manifold has probably found a different "
            "kind of region, and the rung alone would not show it."
        ),
        "not_established": (
            "The cross-task correlation r(%anchor, QED drift) = 0.475 at n=11 "
            "(p~0.14) is INDICATIVE and is not reported here as a finding. The "
            "measured parts are the qed contrast and the per-task drift table."
        ),
        "qed_task_caveat": (
            "On the qed task this column IS the task objective, so it is not an "
            "axis independent of the score there. It is computed after the fact "
            "and never returned to a proposal mechanism on any task."
        ),
    }


def phase_report(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    predicate = assert_predicate_sealed(repo_root)
    regions = load_productive_regions(repo_root)
    shards = _load_shards(args.inputs)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        by_arm.setdefault(shard["arm"], []).append(shard)
    arms: dict[str, Any] = {}
    chains: dict[str, list[str]] = {}
    for name, group in sorted(by_arm.items()):
        per_task = {}
        for shard in sorted(group, key=lambda s: s["task"]):
            prefixes = {}
            for at in MATCHED_PROPOSALS:
                if at > shard["proposals_drawn"]:
                    continue
                prefixes[str(at)] = _prefix_block(
                    shard["proposals"], shard["region"]["wide_count"], at
                )
            full = _prefix_block(
                shard["proposals"], shard["region"]["wide_count"], shard["proposals_drawn"]
            )
            prefixes.setdefault(str(shard["proposals_drawn"]), full)
            improved = [d for d in shard["descendant_probes"] if d["improved"]]
            per_task[shard["task"]] = {
                "proposals_drawn": shard["proposals_drawn"],
                "parents": shard["parents"],
                "realized_strata": shard["realized_strata"],
                "parents_already_in_region": shard["parents_already_in_region"],
                "parent_similarity_max": round(
                    max(shard["parent_similarity_primary"].values()), 4
                ),
                "parent_median_qed": round(
                    statistics.median(
                        [v for v in shard.get("parent_qed", {}).values() if v is not None]
                    ),
                    4,
                )
                if [v for v in shard.get("parent_qed", {}).values() if v is not None]
                else None,
                "test_b_anchor_top_ten": shard["region"]["test_b_anchor_top_ten"],
                "prefixes": prefixes,
                "descendant_probes": len(shard["descendant_probes"]),
                "descendants_improved": len(improved)
                if shard["descendant_probes"]
                else None,
                "descendant_detail": shard["descendant_probes"],
                "primitive_count_histogram": _histogram(
                    [
                        row["primitive_count"]
                        for row in shard["proposals"]
                        if row["primitive_count"] is not None
                    ]
                ),
                "heavy_delta_summary": _heavy_delta(shard),
                "elapsed_seconds": shard["elapsed_seconds"],
            }
            chains.setdefault(shard["task"], []).append(_chain(shard, full))
        total = sum(block["prefixes"][str(block["proposals_drawn"])]["proposals"] for block in per_task.values())
        entries = sum(
            block["prefixes"][str(block["proposals_drawn"])]["entries"]
            for block in per_task.values()
        )
        arms[name] = {
            "role": (arm_describe(name) or {}).get("role", "unknown"),
            "describe": arm_describe(name),
            "tasks": sorted(per_task),
            "total_proposals": total,
            "total_entries": entries,
            "per_draw_entry_rate": round(entries / total, 6) if total else None,
            "per_draw_rate_upper_bound_95": round(rate_bound(entries, total), 6)
            if total
            else None,
            "best_similarity": max(
                block["prefixes"][str(block["proposals_drawn"])]["best_similarity"]
                for block in per_task.values()
            ),
            "per_task": per_task,
        }
    payload = {
        "schema_version": "pmo_entry_diagnostic_report_v2",
        "information_regime": REGIME,
        "information_regime_statement": REGIME_STATEMENT,
        "charged_oracle_calls": 0,
        "question": (
            "Starting from the parents a blind run actually visited, does the "
            "general proposal mechanism generate molecules that enter a known "
            "productive region, at matched proposal counts?"
        ),
        "predicate": predicate,
        "predicate_sha256": payload_sha256(predicate),
        "fingerprint": FINGERPRINT,
        "delta": ENTRY_DELTA,
        "rungs": [list(rung) for rung in RUNGS],
        "matched_proposals": list(MATCHED_PROPOSALS),
        "negative_control": {
            "arm": "nc1_uniform_legal_edits",
            "reported_before_any_arm_number": True,
            "result": arms.get("nc1_uniform_legal_edits"),
        },
        "arms": arms,
        "causal_chains": chains,
        "manifold_context": _manifold_context(repo_root),
        "permutation_control": _permutation(shards, regions, ENTRY_DELTA),
        "pass_criterion": _pass_criterion(arms, args.challenger),
        "software": _software(),
    }
    _write(Path(args.output), payload)
    for name, block in arms.items():
        print(
            f"{name:28s} role={block['role']:24s} proposals={block['total_proposals']:6d} "
            f"entries={block['total_entries']:4d} "
            f"rate<={block['per_draw_rate_upper_bound_95']:.5f} "
            f"maxsim={block['best_similarity']}"
        )
    print(f"wrote {args.output}")
    return 0


def _histogram(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _heavy_delta(shard: dict[str, Any]) -> dict[str, Any]:
    deltas = []
    for row in shard["proposals"]:
        base = row.get("parent_heavy_atoms")
        if base is not None and row.get("heavy_atoms") is not None:
            deltas.append(row["heavy_atoms"] - base)
    if not deltas:
        return {"n": 0}
    return {
        "n": len(deltas),
        "mean": round(sum(deltas) / len(deltas), 3),
        "median": statistics.median(deltas),
        "min": min(deltas),
        "max": max(deltas),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="phase", required=True)

    seal = sub.add_parser("seal")
    seal.add_argument("--force", action="store_true")
    seal.set_defaults(handler=phase_seal)

    measure = sub.add_parser("measure")
    measure.add_argument("--arm", required=True, choices=sorted(ARM_REGISTRY))
    measure.add_argument("--task", required=True, choices=sorted(GATE_TASKS))
    measure.add_argument("--draws", type=int, default=1000)
    measure.add_argument("--parents-per-stratum", type=int, default=2)
    measure.add_argument("--descendant-entrants", type=int, default=8)
    measure.add_argument("--descendant-draws", type=int, default=32)
    measure.add_argument("--output", required=True)
    measure.set_defaults(handler=phase_measure)

    production = sub.add_parser("production")
    production.add_argument("--task", required=True, choices=sorted(GATE_TASKS))
    production.add_argument("--descendant-entrants", type=int, default=8)
    production.add_argument("--descendant-draws", type=int, default=32)
    production.add_argument("--output", required=True)
    production.set_defaults(handler=phase_production)

    report = sub.add_parser("report")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--challenger", default=None)
    report.add_argument("--output", required=True)
    report.set_defaults(handler=phase_report)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

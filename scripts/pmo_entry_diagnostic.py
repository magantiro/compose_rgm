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
    PARENT_SOURCES,
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
    parent_source,
    payload_sha256,
    predicate_payload,
    rate_bound,
    rung_index,
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

#: The round-0 population every shard used before the population became an
#: explicit axis. Runs over it keep their original seed stream.
DEFAULT_PARENT_SOURCE = "blind_visited_stratified"


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
        if parent.score is not None:
            scores.append(parent.score)
        if parent.heavy_atoms is not None:
            heavy.append(parent.heavy_atoms)
    return {
        "by_score_stratum": counts,
        "by_role": roles,
        "parents_carrying_a_score": len(scores),
        "blind_score_min": round(min(scores), 6) if scores else None,
        "blind_score_max": round(max(scores), 6) if scores else None,
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
    source = parent_source(args.parent_source)
    parents = source["function"](
        repo_root, task, per_stratum=args.parents_per_stratum
    )
    began = time.time()

    # One generator per parent, pulled ROUND ROBIN, so a prefix of the task's
    # proposal sequence is spread across strata rather than being one parent's
    # run.  Slice order in this repository is lane- and family-sorted and taking
    # the first N has produced at least three bad measurements.
    streams = []
    for parent in parents:
        # The population joins the seed only when it is NOT the default, so a
        # run over the deployed population reproduces the seeds every shard
        # written before the population became an explicit axis used. A new
        # population gets its own stream; an old shard stays re-derivable.
        label = (
            ()
            if args.parent_source == DEFAULT_PARENT_SOURCE
            else (args.parent_source,)
        )
        seed = stable_seed(
            DIAGNOSTIC_SCHEMA, args.arm, *label, task, parent.endpoint, "r0"
        )
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
        "parent_source": args.parent_source,
        "parent_source_describe": source["describe"],
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
                # The run records no primitive count -- ``program_size`` carries
                # ``scheduled_blocks``, a list of block INDICES -- so the column
                # stays None here rather than silently holding a different
                # quantity from the one the regenerated arms put in it. The
                # block count is published beside it under its own name.
                "primitive_count": size.get("primitive_count"),
                "scheduled_block_count": (
                    len(size["scheduled_blocks"])
                    if isinstance(size.get("scheduled_blocks"), list)
                    else None
                ),
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
    # With no entrant there is no entrant provenance, and reporting an empty map
    # says nothing about which channel searches best. The nearest approaches
    # carry the same synthesis-time tag and are the informative substitute: they
    # answer "which channel gets closest" when the answer to "which channel
    # enters" is nobody.
    approaching = sorted(
        (row for row in prefix if row["similarity"] is not None),
        key=lambda row: -row["similarity"],
    )[:25]
    near_provenance: dict[str, int] = {}
    for row in approaching:
        near_provenance[row["provenance_class"]] = (
            near_provenance.get(row["provenance_class"], 0) + 1
        )
    all_provenance: dict[str, int] = {}
    for row in prefix:
        all_provenance[row["provenance_class"]] = (
            all_provenance.get(row["provenance_class"], 0) + 1
        )
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
        "nearest_25_provenance": near_provenance,
        "all_proposal_provenance": all_provenance,
        "nearest_approaches": [
            {
                "proposal_index": row["proposal_index"],
                "similarity": row["similarity"],
                "provenance_class": row["provenance_class"],
                "endpoint": row["endpoint"],
                "qed": round(row["qed"], 4) if row.get("qed") is not None else None,
                "heavy_atoms": row.get("heavy_atoms"),
            }
            for row in approaching[:5]
        ],
    }


def _arm_key(shard: dict[str, Any]) -> str:
    """One report row per (mechanism, round-0 population).

    Two runs of ONE mechanism over different initializations are different arms
    of the ladder and must not be pooled. Shards written before the population
    became an explicit axis carry no field and keep their original key, so the
    numbers already committed are still addressed by the name they were filed
    under.
    """

    source = shard.get("parent_source")
    if source in (None, DEFAULT_PARENT_SOURCE):
        return shard["arm"]
    return f"{shard['arm']}@{source}"


def _chain(shard: dict[str, Any], block: dict[str, Any]) -> str:
    parents = shard["parent_rows"]
    scored = [row for row in parents if row.get("blind_score") is not None]
    # An objective-blind round-0 population carries no score by construction --
    # this harness charges nothing -- so the chain opens on the parent nearest
    # the region instead of inventing a best-scoring one.
    if scored:
        start = max(scored, key=lambda row: row["blind_score"])
        opening = f"start blind_best={start['blind_score']:.4f}"
    else:
        start = max(
            parents,
            key=lambda row: shard["parent_similarity_primary"].get(row["endpoint"], 0.0),
        )
        opening = f"start objective_blind (n={len(parents)})"
    arm_label = _arm_key(shard)
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
        f"{arm_label}: {opening} "
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


def _charged_approach_ladder(shards) -> dict[str, Any]:
    """How the best structural approach moves as a real run spends its budget.

    Only the observed production arm has a charged-call ordering -- it is the
    run itself -- so this is computed for that arm alone and is absent, never
    zero, for a regenerated one. It separates two readings a single best-of
    number cannot: a search still climbing when its budget ran out, and a search
    that reached a ceiling and stopped.
    """

    out: dict[str, Any] = {}
    for shard in shards:
        if shard["arm"] != PRODUCTION_ARM:
            continue
        rows = sorted(
            (
                row
                for row in shard["parent_rows"]
                if row.get("charged_call_index") is not None
            ),
            key=lambda row: row["charged_call_index"],
        )
        similarity = shard["parent_similarity_primary"]
        running, ladder, last_gain = 0.0, [], None
        for row in rows:
            value = similarity.get(row["endpoint"])
            if value is None:
                continue
            if value > running:
                running = value
                last_gain = row["charged_call_index"]
            ladder.append({"charged_call": row["charged_call_index"], "best": round(running, 4)})
        if not ladder:
            continue
        marks = {}
        for at in (16, 50, 100, 150, 200, 250):
            reached = [row["best"] for row in ladder if row["charged_call"] <= at]
            if reached:
                marks[str(at)] = reached[-1]
        final = ladder[-1]["best"]
        proposals = _prefix_block(
            shard["proposals"], shard["region"]["wide_count"], shard["proposals_drawn"]
        )
        out[shard["task"]] = {
            "charged_calls": len(ladder),
            "best_by_charged_call": marks,
            "final_best_over_the_run": final,
            "last_charged_call_that_improved_it": last_gain,
            "further_proposals_from_those_parents": proposals["proposals"],
            "further_proposals_best": proposals["best_similarity"],
            "further_proposals_lift": round(proposals["best_similarity"] - final, 4),
            "entry_threshold": ENTRY_DELTA,
            "short_of_entry_by": round(ENTRY_DELTA - max(final, proposals["best_similarity"]), 4),
            "reading": (
                "A run still climbing at its last call is budget-limited. A run "
                "whose best stopped moving long before its last call, and whose "
                "parents then yield almost nothing over hundreds more proposals, "
                "has reached what this proposal law can reach from this "
                "population -- more calls would not close the gap."
            ),
        }
    return out


def _novelty_versus_b(shards) -> dict[str, Any]:
    """Did an arm generate chemistry B NEVER generated -- not merely more proposals?

    The owner's headline question, as a reported quantity. B's output is
    everything the deployed controller was observed to produce on this task:
    every endpoint of every proposal it made, charged or not. An arm's novel set
    is what it produced that is absent from that.

    Novelty ALONE is cheap -- a random walk invents new molecules constantly --
    so it is reported beside two qualifications: how much of it is nearer a
    productive region than ANYTHING B produced, and how much is nearer than B's
    own best. Both use the structural proxy, whose measured fidelity to V_local
    is partial and teacher-referenced, so a zero here does not prove an arm
    found nothing valuable -- it proves it found nothing valuable THAT THIS
    PROXY CAN SEE.
    """

    by_task: dict[str, Any] = {}
    for task in sorted({shard["task"] for shard in shards}):
        rows = [shard for shard in shards if shard["task"] == task]
        production = [shard for shard in rows if shard["arm"] == PRODUCTION_ARM]
        if not production:
            continue
        b_endpoints, b_best = set(), 0.0
        for shard in production:
            for row in shard["proposals"]:
                if row.get("endpoint"):
                    b_endpoints.add(row["endpoint"])
                if row.get("similarity") is not None:
                    b_best = max(b_best, row["similarity"])
        arms = {}
        for shard in rows:
            key = _arm_key(shard)
            if shard["arm"] == PRODUCTION_ARM:
                continue
            produced = [row for row in shard["proposals"] if row.get("endpoint")]
            distinct = {row["endpoint"] for row in produced}
            novel = [row for row in produced if row["endpoint"] not in b_endpoints]
            novel_distinct = {row["endpoint"] for row in novel}
            beats = [
                row
                for row in novel
                if row.get("similarity") is not None and row["similarity"] > b_best
            ]
            arms[key] = {
                "proposals": len(shard["proposals"]),
                "executed": len(produced),
                "distinct_molecules": len(distinct),
                "novel_versus_b": len(novel_distinct),
                "novel_fraction_of_distinct": round(len(novel_distinct) / len(distinct), 4)
                if distinct
                else None,
                "novel_and_nearer_a_region_than_b_ever_got": len(
                    {row["endpoint"] for row in beats}
                ),
                "best_novel_similarity": round(
                    max((row["similarity"] for row in novel if row["similarity"] is not None), default=0.0),
                    4,
                ),
                "novel_entrants_at_delta": len(
                    {
                        row["endpoint"]
                        for row in novel
                        if row.get("similarity") is not None
                        and row["similarity"] >= ENTRY_DELTA
                    }
                ),
                "examples_of_novel_and_nearer": [
                    {
                        "endpoint": row["endpoint"],
                        "similarity": row["similarity"],
                        "qed": round(row["qed"], 4) if row.get("qed") is not None else None,
                        "provenance_class": row["provenance_class"],
                    }
                    for row in sorted(
                        beats, key=lambda r: -r["similarity"]
                    )[:5]
                ],
            }
        by_task[task] = {
            "b_distinct_molecules": len(b_endpoints),
            "b_best_similarity": round(b_best, 4),
            "arms": arms,
        }
    return {
        "question": (
            "Did the arm generate useful chemistry that B NEVER generated, as "
            "opposed to more proposals?"
        ),
        "b_is": (
            "every endpoint the deployed controller was observed to produce on "
            "this task, charged or not, from the production arm's shard"
        ),
        "caveat": (
            "novel_and_nearer uses the STRUCTURAL proxy. Its agreement with "
            "V_local is partial and teacher-referenced (see "
            "vlocal_proxy_validation_v1.json), so a zero means the arm found "
            "nothing this proxy can see, NOT that it found nothing valuable. A "
            "novel high-V_local molecule in an unrelated basin is invisible here "
            "by construction and needs charged calls."
        ),
        "by_task": by_task,
    }


def _pass_criterion(
    arms: dict[str, Any], challenger: str | None, baseline: str | None = None
) -> dict[str, Any]:
    """Evaluate the predeclared criterion. An unevaluable run reports UNEVALUATED.

    The criterion is sealed over three primary tasks plus a control, so a run
    covering fewer tasks CANNOT fire it in either direction and must say so
    rather than return a verdict computed on a subset.
    """

    baseline = baseline or "arm1_baseline_b"
    verdict = {
        "criterion": PASS_CRITERION,
        "baseline_arm": baseline,
        "challenger_arm": challenger,
    }
    if baseline not in arms:
        verdict["verdict"] = "UNEVALUATED_NO_BASELINE_ARM"
        verdict["statement"] = (
            f"The sealed criterion compares against {baseline}, which is not "
            f"among the measured arms {sorted(arms)}. Naming a different arm as "
            "the baseline here would be re-tuning a sealed criterion after "
            "seeing numbers, so nothing is substituted."
        )
        return verdict
    if challenger is None or challenger not in arms:
        verdict["verdict"] = "UNEVALUATED_NO_CHALLENGER_ARM"
        verdict["statement"] = (
            "Only the baseline and the negative control have been measured, so "
            "the criterion has not fired in either direction. A baseline number "
            "is not a verdict on C."
        )
        return verdict
    missing = [
        task
        for task in (*PASS_CRITERION["primary_tasks"], PASS_CRITERION["control_task"])
        if task not in arms[challenger]["per_task"] or task not in arms[baseline]["per_task"]
    ]
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
    verdict["tasks_missing_from_this_run"] = missing
    if missing:
        verdict["verdict"] = "UNEVALUATED_CRITERION_SPANS_TASKS_NOT_MEASURED"
        verdict["statement"] = (
            "The sealed criterion spans "
            f"{list(PASS_CRITERION['primary_tasks'])} plus the "
            f"{PASS_CRITERION['control_task']} control, and this run is missing "
            f"{missing}. A subset verdict would be a different criterion from "
            "the one that was sealed, so none is emitted."
        )
        return verdict
    verdict["verdict"] = (
        "PASS"
        if wins >= 2 and degraded is False
        else "FAIL"
        if degraded is not None
        else "UNEVALUATED_MISSING_CONTROL"
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


#: Ladder row order. A row is added when its arm is measured; an arm that has
#: not run is ABSENT rather than reported as a zero.
LADDER_ORDER = (
    "nc1_uniform_legal_edits",
    "arm0_production_blind_run",
    "arm1_baseline_b",
    "arm1_baseline_b@init_bank_16",
    "arm1_baseline_b@init_bank_100",
)


def _ladder_table(arms: dict[str, Any], predicate: dict[str, Any]) -> dict[str, Any]:
    """The ladder as one table per task, at each matched proposal count.

    Every cell is a lookup into a block already computed above; nothing here is
    a second measurement. ``first_entry_proposal_index`` is the committed delta
    rung, so a task with no entrant reports ``None`` and never a zero.
    """

    # A SEALED calibration value, looked up and reported beside the ladder. The
    # rungs themselves are untouched: ``above_chance`` is deliberately the
    # GLOBAL maximum over every task's objective-blind pairs, which is a
    # conservative task-agnostic bar, and a task whose own chance ceiling is far
    # below it can sit above its own ceiling while below the named rung. Both
    # readings are published so neither can be quoted alone.
    per_task_chance = (
        predicate.get("calibration", {}).get("chance_reference", {}).get("per_task_max", {})
    )
    tasks: dict[str, Any] = {}
    for name in [n for n in LADDER_ORDER if n in arms] + [
        n for n in sorted(arms) if n not in LADDER_ORDER
    ]:
        for task, block in arms[name]["per_task"].items():
            for at, prefix in block["prefixes"].items():
                manifold = prefix["manifold"]
                # Arms start from different populations, so an absolute maximum
                # is not comparable across rows. The LIFT over the best parent
                # already in the pool is: it asks what the proposals ADDED.
                parent_best = block["parent_similarity_max"]
                tasks.setdefault(task, {}).setdefault(at, []).append(
                    {
                        "arm": name,
                        "role": arms[name]["role"],
                        "parent_source": arms[name]["parent_source"],
                        "parents": block["parents"],
                        "proposals": prefix["proposals"],
                        "qed_drift_q1_to_q4": manifold["qed_drift_q1_to_q4"],
                        "median_qed": manifold["median_qed"],
                        "parent_median_qed": block["parent_median_qed"],
                        # The comparable manifold quantity across rows. A
                        # regenerated arm draws every proposal ONE program from a
                        # FIXED parent set, so nothing compounds and its drift
                        # across proposal quartiles cannot show what a real
                        # multi-round trajectory shows. The one-step drop from
                        # the parent pool to the proposals is defined the same
                        # way for every row.
                        "one_step_qed_drop": (
                            round(manifold["median_qed"] - block["parent_median_qed"], 4)
                            if manifold["median_qed"] is not None
                            and block["parent_median_qed"] is not None
                            else None
                        ),
                        "drift_is_a_trajectory": arms[name]["role"]
                        == "observed_production",
                        "best_basin_rung": prefix["best_rung"],
                        "best_similarity": prefix["best_similarity"],
                        "best_parent_similarity": parent_best,
                        "best_lift_over_best_parent": round(
                            prefix["best_similarity"] - parent_best, 4
                        ),
                        "task_own_chance_ceiling": per_task_chance.get(task),
                        "best_over_task_own_chance_ceiling": (
                            round(prefix["best_similarity"] / per_task_chance[task], 3)
                            if per_task_chance.get(task)
                            else None
                        ),
                        "best_over_entry_threshold": round(
                            prefix["best_similarity"] / ENTRY_DELTA, 3
                        ),
                        "first_entry_proposal_index": prefix["by_rung"]["entry"][
                            "first_proposal_index"
                        ],
                        "distinct_basins": prefix["diversity_distinct_basins"],
                        "distinct_entrant_scaffolds": prefix[
                            "diversity_distinct_entrant_scaffolds"
                        ],
                        "entrant_median_qed": manifold["entrant_median_qed"],
                        "entries": prefix["entries"],
                        "entry_rate": prefix["entry_rate"],
                        "entry_rate_upper_bound_95": prefix[
                            "per_draw_rate_upper_bound_95"
                        ],
                        "entrant_provenance": prefix["entrant_provenance"],
                        "nearest_25_provenance": prefix["nearest_25_provenance"],
                        "all_proposal_provenance": prefix["all_proposal_provenance"],
                        "nearest_approaches": prefix["nearest_approaches"],
                    }
                )
    # Is a zero RARE or OUT OF REACH? A zero whose best approach is still
    # climbing with budget is under-powered and a larger budget may settle it; a
    # zero whose best approach stopped climbing is a reach statement about the
    # mechanism. This repository has been wrong in both directions -- a fragment
    # probe read 0.0000 at 2,048 draws and lifted to 0.80 at 8,192 -- so the
    # growth curve is published rather than the verdict being asserted.
    growth: dict[str, Any] = {}
    for task, byat in tasks.items():
        per_arm: dict[str, list] = {}
        for at, rows in byat.items():
            for row in rows:
                per_arm.setdefault(row["arm"], []).append((int(at), row["best_similarity"]))
        for name, points in per_arm.items():
            points.sort()
            if len(points) < 2:
                continue
            first, last = points[0], points[-1]
            gained = last[1] - points[-2][1]
            drawn = last[0] - points[-2][0]
            growth.setdefault(task, {})[name] = {
                # An ordered LIST of pairs, not a dict: this payload is written
                # with sort_keys, which orders "1000" before "300" and makes a
                # budget curve read as though it went backwards.
                "best_by_prefix": [[at, value] for at, value in points],
                "budget_multiple": round(last[0] / first[0], 2),
                "best_gained_over_that_multiple": round(last[1] - first[1], 4),
                "gained_in_the_final_step": round(gained, 4),
                "still_climbing": gained > 0,
                # still_climbing alone invites over-reading a rounding-level
                # gain as progress, so the rate and what it implies are beside
                # it. Maxima grow sub-linearly, so this is an OPTIMISTIC bound.
                "gain_per_thousand_proposals_in_the_final_step": round(
                    gained / drawn * 1000, 5
                )
                if drawn
                else None,
                "proposals_to_reach_entry_at_that_rate": (
                    int((ENTRY_DELTA - last[1]) / (gained / drawn))
                    if gained > 0 and drawn
                    else None
                ),
            }
    return {
        "columns": (
            "arm | qed_drift_q1_to_q4 | best_basin_rung | first_entry_proposal_index "
            "| distinct_basins | entrant_median_qed | entry_rate | entrant_provenance"
        ),
        "best_similarity_growth_with_budget": growth,
        "growth_reading": (
            "still_climbing distinguishes an under-powered zero from a reach "
            "statement. An arm whose best approach stopped moving while its "
            "budget grew several fold is not short of draws. Read it WITH the "
            "rate: a gain of a few thousandths over hundreds of proposals is "
            "technically still climbing and implies a budget nobody will spend. "
            "proposals_to_reach_entry_at_that_rate extrapolates the final step "
            "linearly and is therefore an OPTIMISTIC bound, because the maximum "
            "of a sample grows sub-linearly in the sample size."
        ),
        "entry_threshold": ENTRY_DELTA,
        "reading": (
            "best_basin_rung is None when no proposal reached the lowest named "
            "rung (0.25). first_entry_proposal_index is None when nothing reached "
            "the committed 0.30 entry threshold. Neither is a zero and neither "
            "may be reported as one."
        ),
        "drift_reading": (
            "qed_drift_q1_to_q4 is a TRAJECTORY quantity and is only that for "
            "the observed production arm, where later proposals descend from "
            "earlier ones. A regenerated arm draws every proposal one program "
            "from a fixed parent set, so nothing compounds and its quartile "
            "drift is close to noise by construction; drift_is_a_trajectory "
            "marks which rows it means. one_step_qed_drop is defined identically "
            "for every row and is the column to compare across arms."
        ),
        "lift_reading": (
            "best_lift_over_best_parent is the headline comparison across rows. "
            "Each arm starts from its own population, so best_similarity alone "
            "compares starting points as much as mechanisms; the lift asks what "
            "the proposals themselves added to the best approach already in the "
            "pool."
        ),
        "chance_ceiling_reading": (
            "task_own_chance_ceiling is the maximum similarity any of 100 "
            "objective-blind init-bank molecules reaches to THIS task's region, "
            "from the sealed calibration. The above_chance RUNG is the global "
            "maximum across all tasks and is deliberately more conservative, so "
            "an arm can exceed its task's own ceiling while sitting below the "
            "named rung. Report both; the rungs are sealed and unchanged."
        ),
        "by_task": tasks,
    }


def phase_report(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    predicate = assert_predicate_sealed(repo_root)
    regions = load_productive_regions(repo_root)
    shards = _load_shards(args.inputs)
    # Arms can end at different draw counts -- a control measured at 300 against
    # an arm measured at 1,000 is not a comparison. Every arm therefore also
    # reports a prefix at every OTHER arm's full count that it can reach, so
    # every pair has a matched point. These are reductions of stored
    # per-proposal similarities, never a second measurement, and the sealed
    # MATCHED_PROPOSALS points are unchanged.
    matched_points = sorted(
        {int(shard["proposals_drawn"]) for shard in shards} | set(MATCHED_PROPOSALS)
    )
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        by_arm.setdefault(_arm_key(shard), []).append(shard)
    arms: dict[str, Any] = {}
    chains: dict[str, list[str]] = {}
    for name, group in sorted(by_arm.items()):
        per_task = {}
        for shard in sorted(group, key=lambda s: s["task"]):
            prefixes = {}
            for at in matched_points:
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
                        if row.get("primitive_count") is not None
                    ]
                ),
                "scheduled_block_count_histogram": _histogram(
                    [
                        row["scheduled_block_count"]
                        for row in shard["proposals"]
                        if row.get("scheduled_block_count") is not None
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
        mechanism = name.split("@", 1)[0]
        population = group[0].get("parent_source", DEFAULT_PARENT_SOURCE)
        arms[name] = {
            "role": (arm_describe(mechanism) or {}).get("role", "unknown"),
            "describe": arm_describe(mechanism),
            "mechanism": mechanism,
            "parent_source": population,
            "parent_source_describe": group[0].get("parent_source_describe"),
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
        "matched_points_reported": matched_points,
        "matched_points_note": (
            "The sealed points are "
            f"{list(MATCHED_PROPOSALS)}. Every arm's own full draw count is also "
            "reported for every arm that can reach it, so a control measured at "
            "one budget and an arm measured at another still have a matched "
            "comparison. All of it is a reduction of stored per-proposal "
            "similarities."
        ),
        "negative_control": {
            "arm": "nc1_uniform_legal_edits",
            "reported_before_any_arm_number": True,
            "result": arms.get("nc1_uniform_legal_edits"),
        },
        "arms": arms,
        "causal_chains": chains,
        "manifold_context": _manifold_context(repo_root),
        "permutation_control": _permutation(shards, regions, ENTRY_DELTA),
        "charged_approach_ladder": _charged_approach_ladder(shards),
        "novelty_versus_b": _novelty_versus_b(shards),
        "pass_criterion": _pass_criterion(arms, args.challenger, args.baseline),
        "ladder_table": _ladder_table(arms, predicate),
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
    """Count by value, sorted numerically when every key is a number.

    A non-numeric key is not forced through ``int`` -- that raised on the
    production arm, whose column held a list -- it falls back to a lexical sort
    so a malformed column is visible in the output instead of crashing the whole
    report.
    """

    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    try:
        return dict(sorted(out.items(), key=lambda kv: int(kv[0])))
    except ValueError:
        return dict(sorted(out.items()))


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
    measure.add_argument(
        "--parent-source",
        default=DEFAULT_PARENT_SOURCE,
        choices=sorted(PARENT_SOURCES),
        help=(
            "round-0 parent population. The default is the deployed run's own "
            "visited molecules, which is what every shard written before this "
            "flag existed used."
        ),
    )
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
    report.add_argument(
        "--baseline",
        default=None,
        help=(
            "arm the sealed criterion compares against. Defaults to "
            "arm1_baseline_b; naming another does not change the sealed rule."
        ),
    )
    report.add_argument("--output", required=True)
    report.set_defaults(handler=phase_report)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

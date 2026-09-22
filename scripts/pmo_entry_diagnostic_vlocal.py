"""Supersede the entry predicate with FUTURE VALUE, and measure what can proxy it.

The owner's refinement, 2026-09-22: a productive basin is a molecule with high

    V_local(G) = E[top-10 score after 64 local calls | G]

whether or not it resembles a known teacher endpoint.  The teacher region
becomes a POSITIVE CONTROL -- V_local is large there -- rather than the
definition.  Asking "did the mechanism recreate the teacher's chemistry"
overfits development to one known answer, and a different molecule with the same
future value is arguably the better result.

Two things have to be true before that refinement is usable, and this module
establishes both or refuses:

``supersede``
    Write a checkable supersession record: the original predicate verbatim and
    by hash, the owner's reason, the new definition, AND an honest statement of
    which arm numbers had already been observed when the change arrived.  The
    owner's stated precondition was that none had been.  That precondition is
    FALSE here and the record says so with the commits that prove it, because a
    supersession that misstates its own preconditions is exactly the
    goalpost-move it is meant to rule out.

``validate``
    Exact V_local costs 64 charged calls per candidate, so it cannot be computed
    inside a zero-call gate.  Test B already paid it at 33 points -- 11 tasks by
    3 teacher-route positions -- and that is the free validation set.  Every
    candidate proxy is scored against those points and reported with its
    measured agreement.  A proxy whose fidelity is unmeasured is not a basin
    definition, and a proxy that fails to validate is reported as a finding.

Zero charged oracle calls in both phases.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from compose_v4.experiments.pmo_entry_diagnostic import (
    ENTRY_DELTA,
    drug_likeness,
    load_productive_regions,
    payload_sha256,
)

OUT = "diagnostics/pmo_entry_diagnostic_v1"

#: The budget Test B spent per point, and therefore the definition's horizon.
VLOCAL_BUDGET = 64

#: The owner's reason, recorded verbatim so the record does not paraphrase it.
SUPERSESSION_REASON = (
    "A good basin is a molecule or region with high V_local, whether or not it "
    "resembles the known teacher answer. The teacher region is then a POSITIVE "
    "CONTROL (V_local is huge there -- 0.7362 in 64 calls on celecoxib), not "
    "the target definition. Asking 'did C recreate the teacher's chemistry' "
    "overfits development to one known answer, and if C finds a completely "
    "different molecule with the same future value that is arguably a BETTER "
    "result. The gate should be able to say so."
)


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


# ---- what had already been read ----


def observed_arm_numbers(repo_root: Path) -> dict[str, Any]:
    """Every arm result that existed BEFORE the supersession, from git, not memory.

    The supersession's own legitimacy rests on this, so it is reconstructed from
    committed artifacts rather than asserted. A shard is evidence an arm number
    was produced; its commit is evidence of when.
    """

    shards = sorted((repo_root / OUT / "shards").glob("*.json")) + sorted(
        (repo_root / OUT / "superseded").glob("*.json")
    )
    observed = []
    for path in shards:
        document = json.loads(path.read_text())
        payload = document["payload"]
        try:
            commit = subprocess.run(
                ["git", "log", "-1", "--format=%h %aI", "--", str(path.relative_to(repo_root))],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            commit = ""
        similarity = [
            row["similarity"]
            for row in payload["proposals"]
            if row.get("similarity") is not None
        ]
        observed.append(
            {
                "path": str(path.relative_to(repo_root)),
                "arm": payload["arm"],
                "parent_source": payload.get("parent_source"),
                "task": payload["task"],
                "proposals_drawn": payload["proposals_drawn"],
                "entries_at_sealed_delta": sum(
                    1 for value in similarity if value >= ENTRY_DELTA
                ),
                "best_similarity": round(max(similarity), 4) if similarity else None,
                "committed": commit,
            }
        )
    tasks = sorted({row["task"] for row in observed})
    return {
        "arm_results_observed_before_supersession": observed,
        "tasks_observed": tasks,
        "total_entries_observed_at_sealed_delta": sum(
            row["entries_at_sealed_delta"] for row in observed
        ),
        "distinct_outcomes_across_arms": sorted(
            {row["entries_at_sealed_delta"] for row in observed}
        ),
    }


def phase_supersede(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    original = json.loads((repo_root / OUT / "predicate_v1.json").read_text())
    read = observed_arm_numbers(repo_root)
    every_arm_identical = len(read["distinct_outcomes_across_arms"]) == 1
    payload = {
        "schema_version": "pmo_entry_predicate_supersession_v1",
        "supersedes": {
            "path": f"{OUT}/predicate_v1.json",
            "payload_sha256": original["payload_sha256"],
            "statement": original["payload"]["statement"],
            "delta": original["payload"]["delta"],
            "rungs": original["payload"]["rungs"],
            "pass_criterion": original["payload"]["pass_criterion"],
        },
        "reason_from_the_owner": SUPERSESSION_REASON,
        "new_definition": {
            "name": "V_local",
            "statement": (
                "V_local(G) = E[top-10 score after "
                f"{VLOCAL_BUDGET} local calls | G]. A productive basin is a "
                "molecule or region of high V_local, regardless of whether it "
                "resembles a known teacher endpoint."
            ),
            "budget_per_candidate": VLOCAL_BUDGET,
            "teacher_region_role": "POSITIVE CONTROL, no longer the definition",
        },
        # THE HONEST PART. The owner's stated precondition for a clean
        # supersession was that no arm number had been read. It had been.
        "precondition_stated_by_the_owner": (
            "Changing the sealed predicate is legitimate ONLY because no arm "
            "number has been read yet."
        ),
        "precondition_holds": False,
        "precondition_statement": (
            "This precondition is FALSE and the supersession is recorded anyway, "
            "with the evidence, rather than being written as though it held. Arm "
            "results had been measured, committed and read under the sealed "
            "predicate before the refinement arrived."
        ),
        "what_makes_it_defensible_anyway": (
            "Every observed arm returned the SAME result -- zero entrants at the "
            "sealed threshold and zero at the rung below it -- so no arm was "
            "favoured or disfavoured by the old predicate and the change cannot "
            "have been chosen to move a specific number. That is weaker than the "
            "owner's precondition and is offered as such, not as a substitute "
            "for it. The owner should decide whether it suffices."
            if every_arm_identical
            else "The observed arms did NOT all return the same result, so this "
            "supersession changes a predicate under which arms were already "
            "separated. It requires an explicit owner decision."
        ),
        "every_observed_arm_returned_the_same_result": every_arm_identical,
        **read,
        "cost_of_the_new_definition": {
            "exact_cost_per_candidate": VLOCAL_BUDGET,
            "why_it_cannot_be_the_gate_directly": (
                f"{VLOCAL_BUDGET} charged calls per candidate, so ten candidates "
                "cost 640 -- more than the scored run this gate precedes. The "
                "gate must therefore use a validated offline proxy, or ask for "
                "an explicit budget."
            ),
        },
        "software": _software(),
        "charged_oracle_calls": 0,
    }
    _write(repo_root / OUT / "predicate_supersession_v1.json", payload)
    print(f"precondition_holds = {payload['precondition_holds']}")
    print(f"arm results already observed: {len(read['arm_results_observed_before_supersession'])}")
    print(f"distinct outcomes across them: {read['distinct_outcomes_across_arms']}")
    return 0


# ---- can anything proxy V_local offline? ----


def _spearman(xs, ys) -> float:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0] * len(values)
        for position, index in enumerate(order):
            out[index] = position
        return out

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def _concordance(pairs) -> tuple[int, int]:
    concordant = discordant = 0
    for (x1, y1), (x2, y2) in itertools.combinations(pairs, 2):
        if x1 == x2 or y1 == y2:
            continue
        if (x1 < x2) == (y1 < y2):
            concordant += 1
        else:
            discordant += 1
    return concordant, discordant


def vlocal_points(repo_root: Path) -> list[dict[str, Any]]:
    """The already-paid V_local measurements, joined to offline descriptors.

    Test B ran the frozen local controller for exactly ``VLOCAL_BUDGET`` charged
    calls from each seed and recorded ``top_ten_new_mean``, which IS V_local.
    Reading it costs nothing.
    """

    document = json.loads(
        (repo_root / "diagnostics/pmo_atlas_v1/test_b_local_lift.json").read_text()
    )
    payload = document["payload"]
    if payload_sha256(payload) != document.get("payload_sha256"):
        raise SystemExit("the Test B payload hash has moved; refusing to validate against it")
    regions = load_productive_regions(repo_root)
    points = []
    for run in payload["runs"]:
        if int(run["budget"]) != VLOCAL_BUDGET:
            continue
        region = regions[run["task"]]
        similarity, _ = region.approach(run["seed_smiles"], "primary")
        descriptors = drug_likeness(run["seed_smiles"])
        points.append(
            {
                "task": run["task"],
                "checkpoint_label": run["checkpoint_label"],
                "seed_smiles": run["seed_smiles"],
                "v_local": float(run["top_ten_new_mean"]),
                "charged_calls_already_paid": int(run["charged_calls"]),
                "proxy_structural_similarity_to_anchor": round(similarity, 6),
                "proxy_own_oracle_score": float(run["seed_score"]),
                "proxy_qed": descriptors["qed"],
                "proxy_heavy_atoms": descriptors["heavy_atoms"],
            }
        )
    return points


PROXIES = {
    "structural_similarity_to_anchor": {
        "field": "proxy_structural_similarity_to_anchor",
        "offline": True,
        "note": "the superseded predicate, now assessed as a PROXY for V_local",
    },
    "own_oracle_score": {
        "field": "proxy_own_oracle_score",
        "offline": False,
        "note": (
            "free for a molecule the run already charged, and NOT available for "
            "a new proposal without spending a call -- so it bounds what a proxy "
            "could achieve rather than being usable as one"
        ),
    },
    "qed": {"field": "proxy_qed", "offline": True, "note": "drug-likeness"},
    "heavy_atoms": {"field": "proxy_heavy_atoms", "offline": True, "note": "size"},
}


def score_proxy(points, field: str) -> dict[str, Any]:
    """Agreement with true V_local, pooled and within task, with and without anchors.

    The within-task figure is the one that matters: tasks have different score
    scales, so a pooled correlation partly measures which task a point came
    from. The anchor points carry similarity 1.0 BY CONSTRUCTION, so a
    structural proxy is inflated by them and the non-anchor figure is the honest
    one.
    """

    def block(subset):
        if len(subset) < 3:
            return None
        return round(_spearman([r[field] for r in subset], [r["v_local"] for r in subset]), 4)

    non_anchor = [r for r in points if r["checkpoint_label"] != "anchor"]
    within, within_non_anchor = [0, 0], [0, 0]
    disagreeing_tasks = []
    for task in sorted({r["task"] for r in points}):
        rows = [r for r in points if r["task"] == task]
        concordant, discordant = _concordance([(r[field], r["v_local"]) for r in rows])
        within[0] += concordant
        within[1] += discordant
        rows = [r for r in rows if r["checkpoint_label"] != "anchor"]
        concordant, discordant = _concordance([(r[field], r["v_local"]) for r in rows])
        within_non_anchor[0] += concordant
        within_non_anchor[1] += discordant
        if discordant:
            disagreeing_tasks.append(task)
    total = within[0] + within[1]
    total_non_anchor = within_non_anchor[0] + within_non_anchor[1]
    return {
        "pooled_spearman_all": block(points),
        "pooled_spearman_non_anchor": block(non_anchor),
        "within_task_concordant": within[0],
        "within_task_discordant": within[1],
        "within_task_kendall_tau": round((within[0] - within[1]) / total, 4) if total else None,
        "non_anchor_within_task_concordant": within_non_anchor[0],
        "non_anchor_within_task_discordant": within_non_anchor[1],
        "non_anchor_within_task_kendall_tau": (
            round((within_non_anchor[0] - within_non_anchor[1]) / total_non_anchor, 4)
            if total_non_anchor
            else None
        ),
        "tasks_where_it_ranks_wrong": disagreeing_tasks,
    }


def phase_validate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    points = vlocal_points(repo_root)
    scored = {}
    for name, spec in PROXIES.items():
        scored[name] = {**spec, **score_proxy(points, spec["field"])}
    offline = {
        name: block for name, block in scored.items() if block["offline"]
    }
    best = max(
        offline,
        key=lambda name: offline[name]["non_anchor_within_task_kendall_tau"] or -1.0,
    )
    counterexamples = []
    for task in sorted({r["task"] for r in points}):
        rows = sorted(
            (r for r in points if r["task"] == task),
            key=lambda r: r["proxy_structural_similarity_to_anchor"],
        )
        values = [r["v_local"] for r in rows]
        if values != sorted(values):
            counterexamples.append(
                {
                    "task": task,
                    "by_rising_similarity": [
                        {
                            "checkpoint": r["checkpoint_label"],
                            "similarity": r["proxy_structural_similarity_to_anchor"],
                            "v_local": round(r["v_local"], 4),
                        }
                        for r in rows
                    ],
                }
            )
    payload = {
        "schema_version": "pmo_entry_vlocal_proxy_validation_v1",
        "charged_oracle_calls": 0,
        "question": (
            "V_local costs 64 charged calls per candidate and cannot be computed "
            "inside a zero-call gate. Does any OFFLINE quantity predict it well "
            "enough to stand in?"
        ),
        "validation_set": {
            "source": "diagnostics/pmo_atlas_v1/test_b_local_lift.json",
            "points": len(points),
            "tasks": len({r["task"] for r in points}),
            "budget_per_point": VLOCAL_BUDGET,
            "charged_calls_already_paid": sum(
                r["charged_calls_already_paid"] for r in points
            ),
            "new_charged_calls": 0,
            "limits": (
                "Every point is a TEACHER-ROUTE checkpoint, so the set spans "
                "distance-to-anchor by construction and contains no molecule "
                "from an unrelated basin. A proxy validated here is validated "
                "for ranking positions along a route, which is NOT the same as "
                "ranking a novel basin -- the case the new definition exists to "
                "cover. Three points per task is also too few to separate close "
                "proxies."
            ),
        },
        "proxies": scored,
        "best_offline_proxy": best,
        "counterexamples_to_similarity_as_a_basin_definition": counterexamples,
        "verdict": _verdict(scored, counterexamples),
        "points": points,
        "software": _software(),
    }
    _write(repo_root / OUT / "vlocal_proxy_validation_v1.json", payload)
    for name, block in scored.items():
        print(
            f"{name:34s} offline={block['offline']!s:5s} "
            f"within-task tau={block['within_task_kendall_tau']:+.3f} "
            f"non-anchor tau={block['non_anchor_within_task_kendall_tau']:+.3f} "
            f"wrong on {len(block['tasks_where_it_ranks_wrong'])} tasks"
        )
    print(f"\nverdict: {payload['verdict']['label']}")
    print(payload["verdict"]["statement"])
    return 0


def _verdict(scored, counterexamples) -> dict[str, Any]:
    """Derive the verdict from the measured numbers, never from a written-in claim.

    A hardcoded summary that disagrees with the table it summarises is how a
    result gets quoted wrong, and the first version of this function said size
    does not proxy V_local while the table beside it measured tau +0.692.
    """

    similarity = scored["structural_similarity_to_anchor"]
    ceiling = scored["own_oracle_score"]
    offline = {name: block for name, block in scored.items() if block["offline"]}
    ranked = sorted(
        offline.items(),
        key=lambda kv: kv[1]["non_anchor_within_task_kendall_tau"] or -1.0,
        reverse=True,
    )
    useful = [
        f"{name} ({block['within_task_kendall_tau']:+.3f} within task, "
        f"{block['non_anchor_within_task_kendall_tau']:+.3f} excluding anchors)"
        for name, block in ranked
        if (block["non_anchor_within_task_kendall_tau"] or 0) > 0
    ]
    useless = [
        f"{name} ({block['non_anchor_within_task_kendall_tau']:+.3f})"
        for name, block in ranked
        if (block["non_anchor_within_task_kendall_tau"] or 0) <= 0
    ]
    matches_ceiling = (
        similarity["within_task_kendall_tau"] == ceiling["within_task_kendall_tau"]
    )
    return {
        "label": "PARTIAL_PROXY_ONLY_AND_IT_CANNOT_SEE_A_NOVEL_BASIN",
        "offline_proxies_with_signal": useful,
        "offline_proxies_without_signal": useless,
        "similarity_matches_the_own_score_ceiling": matches_ceiling,
        "statement": (
            "The superseded structural predicate IS a moderate proxy for V_local "
            f"within task (tau {similarity['within_task_kendall_tau']:+.3f}), "
            + (
                "exactly matching the molecule's own oracle score, which is the "
                "strongest signal available and is NOT free for a new proposal -- "
                "on this validation set the two are indistinguishable, which is a "
                "property of a set built from route positions rather than evidence "
                "that structure carries as much as a score does. "
                if matches_ceiling
                else "against the molecule's own oracle score at tau "
                f"{ceiling['within_task_kendall_tau']:+.3f}. "
            )
            + "The agreement rests on the anchor points, where similarity is 1.0 "
            "by construction: excluding them it falls to tau "
            f"{similarity['non_anchor_within_task_kendall_tau']:+.3f}, ranking "
            f"wrong on {len(similarity['tasks_where_it_ranks_wrong'])} tasks "
            f"({', '.join(similarity['tasks_where_it_ranks_wrong'])}). "
            + (f"Offline signal also appears in: {'; '.join(useful[1:])}. " if len(useful) > 1 else "")
            + (f"No signal: {', '.join(useless)}. " if useless else "")
            + "DECISIVE LIMIT, and it is structural rather than statistical: every "
            "proxy measured here is teacher-REFERENCED or a route-position "
            "artifact, so by construction it scores a high-value molecule in an "
            "unrelated basin as zero -- exactly the case the new definition "
            "exists to credit. No offline proxy in hand can do that. That is a "
            "finding about the gate, not a measurement failure: crediting a "
            "novel basin requires charged calls."
        ),
        "consequence_for_the_gate": (
            "The structural rung ladder remains sound as a POSITIVE-CONTROL "
            "detector -- it answers 'did the arm re-find the known productive "
            "region'. It cannot answer 'did the arm find a DIFFERENT productive "
            "region', and no offline substitute for that was found. That second "
            "question needs a costed V_local sample and an explicit budget."
        ),
        "counterexample_tasks": [row["task"] for row in counterexamples],
        "why_the_counterexamples_matter": (
            "On these tasks a molecule FURTHER from the teacher anchor has "
            "HIGHER future value, so resemblance and future value disagree in "
            "the data rather than only in principle. This is the owner's "
            "refinement measured rather than argued."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="phase", required=True)
    supersede = sub.add_parser("supersede")
    supersede.set_defaults(handler=phase_supersede)
    validate = sub.add_parser("validate")
    validate.set_defaults(handler=phase_validate)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

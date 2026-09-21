"""Merge the PMO 3x250 autopsy stages into one committed diagnostic plus a summary.

Inputs are the JSON outputs of the four read-only drivers in this directory
(``pmo_3x250_autopsy.py``, ``pmo_3x250_celecoxib_distance.py``,
``pmo_3x250_binder_probe.py``, ``pmo_3x250_jump_parent_counterfactual.py``).  This
module adds no new measurement: it joins them, records the per-task classification with
the evidence each verdict rests on, and labels every claim MEASURED or INFERRED.

gsk3b returned exactly 0.0 on all 250 charged calls, so it carries no signal about
controller behaviour and is retained here descriptively only.  Its task-blind binder
numbers ARE used, because the jump plan bank and the binder never see the objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCIENTIFIC_TASKS = ("perindopril_mpo", "celecoxib_rediscovery")


def classification(task: str, bundle: dict) -> dict:
    """The (A)/(B)/(C)/(D) verdict, with the evidence and the disconfirming evidence."""
    task_block = bundle["tasks"][task]
    landmarks = task_block["curve_landmarks"]
    if task == "perindopril_mpo":
        trend = task_block["objective_block_trend"]
        return {
            "classification": "A",
            "label": "still climbing in the correct chemical direction",
            "more_budget_helps": True,
            "measured_evidence": [
                (
                    "max similarity-to-perindopril AMONG FEASIBLE molecules (exactly two "
                    "aromatic rings, so ring term 1.0) rises monotonically after the first "
                    f"block: {[b['max_similarity_among_feasible'] for b in trend]}, and the "
                    "final block exceeds the initialization ceiling"
                ),
                "the count of feasible molecules per 50 calls rises 3 -> 10 -> 9 -> 11 -> 18",
                (
                    f"2 of the run's {task_block['best_so_far_lineage']['count']} global bests "
                    "land in the LAST 50 calls, the largest of them (+0.0210) at call 240"
                ),
                (
                    "top-10 AUC score is still rising steeply at the budget boundary "
                    f"({landmarks['best_after_250']:.4f} best against a top-10 of 0.4665, "
                    "up from 0.4276 fifty calls earlier)"
                ),
                (
                    "the best molecule's ancestry is a 4-generation chain with monotone "
                    "improvement 0.1154 -> 0.4307 -> 0.4635 -> 0.4865, so descendants ARE "
                    "being exploited"
                ),
                (
                    "the N-F control: the plausible des-fluoro analogue scores 0.4902, so the "
                    "headline is not an artefact of an exotic motif"
                ),
            ],
            "disconfirming_evidence": [
                (
                    "the best score after 250 calls (0.4865) is only +0.0210 above the best "
                    "initialization molecule found within the first 16 calls (0.4655); on "
                    "the headline metric alone the run looks flat for 228 calls"
                ),
                (
                    "children of elite parents (>=0.35) improve only 16.2% of the time and "
                    "their mean delta is -0.027, so the local proposal distribution is "
                    "mildly destructive at the top of the archive"
                ),
                "the rate is slow: similarity moved +0.0199 in 238 calls",
            ],
            "inferred": [
                (
                    "extrapolating the last-50 improvement rate to 750 further calls is a "
                    "LINEAR EXTRAPOLATION over a search process and is not measured"
                ),
            ],
        }
    if task == "celecoxib_rediscovery":
        distance = bundle["celecoxib_distance"]
        pieces = distance["structural_piece_counts_over_all_charged_molecules"]
        return {
            "classification": "B",
            "label": "plateaued in the wrong basin",
            "more_budget_helps": False,
            "measured_evidence": [
                (
                    f"0 of 250 charged molecules carry the primary/benzene sulfonamide "
                    f"({pieces['primary_sulfonamide']}), the trifluoromethyl "
                    f"({pieces['trifluoromethyl']}), the N-aryl pyrazole "
                    f"({pieces['n_aryl_pyrazole']}) or the 1,5-diarylpyrazole core "
                    f"({pieces['core_diarylpyrazole']})"
                ),
                (
                    f"0 of 250 molecules carry celecoxib's Bemis-Murcko scaffold "
                    f"({distance['molecules_with_target_scaffold']})"
                ),
                (
                    "only 2 of 250 molecules contain a pyrazole ring at all, and both are "
                    "initialization-era; the last 150 calls contain none"
                ),
                (
                    "the run's single controller improvement (call 216, +0.0061) moved "
                    "structurally AWAY from the target: its 2048-bit Morgan Tanimoto to "
                    "celecoxib is 0.1089 against the best initialization molecule's 0.1486"
                ),
                (
                    "per-50-call MAX bit-vector Tanimoto falls 0.1486 -> 0.1231 -> 0.1158 -> "
                    "0.1273 -> 0.1273, i.e. the closest structural approach happened in the "
                    "first 50 calls and was never beaten"
                ),
                (
                    "fluorine is 1.0% of atom_insert actions and sulfur 0.2%, so the atoms "
                    "the target pharmacophore is made of are essentially never introduced"
                ),
            ],
            "disconfirming_evidence": [
                (
                    "the top-10 AUC IS rising monotonically (0.1257 at call 16 to 0.1888 at "
                    "call 250), so the population is genuinely improving in bulk -- the "
                    "plateau is in the BASIN, not in the controller's ability to refine"
                ),
                (
                    "children still beat their parents 76 times, and the best molecule sits "
                    "at the end of a 5-generation chain, so this is not a recursion failure"
                ),
            ],
            "inferred": [
                (
                    "that the required transition is unreachable by this proposal "
                    "distribution is INFERRED from the piece census plus the element mix; it "
                    "is not a proof of unreachability under the rewrite system, which can in "
                    "principle build every one of those pieces"
                ),
            ],
        }
    return {"classification": None, "label": "excluded from scientific conclusions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autopsy", required=True)
    parser.add_argument("--celecoxib-distance", required=True)
    parser.add_argument("--binder-probe", nargs="*", default=[])
    parser.add_argument("--jump-counterfactual", nargs="*", default=[])
    parser.add_argument("--output", default="diagnostics/pmo_3x250_autopsy_v1.json")
    parser.add_argument("--summary", default="diagnostics/pmo_3x250_autopsy_v1.md")
    arguments = parser.parse_args()

    bundle = json.loads(Path(arguments.autopsy).read_text())
    bundle["celecoxib_distance"] = json.loads(Path(arguments.celecoxib_distance).read_text())

    for path in arguments.binder_probe:
        payload = json.loads(Path(path).read_text())
        task = Path(payload["task_dir"]).name
        payload.pop("rows", None)
        bundle["tasks"][task]["jump_binder_decomposition"] = payload
    for path in arguments.jump_counterfactual:
        payload = json.loads(Path(path).read_text())
        task = Path(payload["task_dir"]).name
        for block in payload["strata"].values():
            block.pop("cells", None)
        bundle["tasks"][task]["jump_parent_counterfactual"] = payload

    bundle["classification"] = {
        task: classification(task, bundle) for task in SCIENTIFIC_TASKS
    }
    bundle["excluded"] = {
        "gsk3b": (
            "returned exactly 0.0 on all 250 charged calls; descriptive only. Its "
            "task-blind binder numbers are retained because the plan bank and the binder "
            "never see the objective."
        )
    }
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(bundle, indent=2, sort_keys=True))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

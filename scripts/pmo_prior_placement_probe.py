"""Does the learned law penalise ELEMENT PLACEMENT, not just element choice?

The construction lane draws its chain uniformly from an intent's element
vocabulary, which produces polyperoxides and N-O linkers: valid, executable,
RDKit-parseable, and not molecules.  Narrowing the element SET cannot fix that,
because the damage is in WHERE the elements go relative to each other.

The model's `atom_insert` mark carries `neighbors` as well as `atom_type`, so it
scores placement.  This probe asks the question directly: over real production
parents, compare the model's probability for inserting an oxygen onto an oxygen
(making a peroxide) against inserting the same oxygen onto a carbon.

Zero oracle calls.  No property calculator is used to select anything; the
element/neighbour labels come from the graph.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from compose_v4.chem.molecular_graph import IDX_TO_ELEMENT
from compose_v4.control.learned_successor_prior import LearnedSuccessorPrior
from compose_v4.rewrite.trace_shard import decode_state

PARENTS = Path("/Users/rmaganti/compose_pmo_replay_data/production_parents_v1.json")
CHECKPOINT = Path("/Users/rmaganti/compose_fragment_ckpt/ringcore_a7546e2_best.pt")
SCOPE = "3721d69851110fdd"


def _element(state, vertex: int) -> str:
    """Element SYMBOL of a graph vertex.

    `state.atom_types` already holds an ELEMENT index, so it maps straight
    through `IDX_TO_ELEMENT`.  `ORGANIC_VOCABULARY.element_of` is the different
    thing -- it maps a (element, valence) CLASS index to an element index -- and
    confusing the two silently mislabels every bucket.
    """
    return IDX_TO_ELEMENT[int(state.atom_types[int(vertex)])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--require-element", default=None,
                    help="restrict to parents containing this element symbol")
    ap.add_argument("--min-states", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_tracelet_rollouts import load_factorized_rollout_checkpoint

    model, _ = load_factorized_rollout_checkpoint(CHECKPOINT, expected_scope_hash=SCOPE)
    model.eval()
    prior = LearnedSuccessorPrior(model, cache_entries=4096)

    payload = json.loads(PARENTS.read_text())
    rows = sorted([r for t in payload for r in payload[t]["parents"]],
                  key=lambda x: (-x["uses"], x["key"]))[: args.sample]

    # mass[(inserted_element, neighbour_element)] -> list of within-state shares
    buckets: dict[tuple[str, str], list] = collections.defaultdict(list)
    states_used = 0
    for row in rows:
        source = decode_state(row["state"])
        if source.n_real_atoms < 5:
            continue
        if args.require_element is not None and not any(
            _element(source, v) == args.require_element
            for v in range(source.n_atoms)
            if int(source.atom_types[v]) != 0
        ):
            continue
        law = prior._marked_law(source)
        inserts = [m for m in law.marks if m.executor_rule_name == "atom_insert"]
        if len(inserts) < 10:
            continue
        total = sum(m.probability for m in inserts)
        if total <= 0:
            continue
        local: dict[tuple[str, str], float] = collections.defaultdict(float)
        counts: dict[tuple[str, str], int] = collections.defaultdict(int)
        for mark in inserts:
            # `action.atom_type` is an ELEMENT index, like `state.atom_types`.
            new_element = IDX_TO_ELEMENT[int(mark.action.atom_type)]
            for neighbour, _order in mark.action.neighbors:
                key = (new_element, _element(source, neighbour))
                local[key] += float(mark.probability) / total
                counts[key] += 1
        for key, mass in local.items():
            # share of insert mass relative to the share a UNIFORM draw over the
            # same insert marks would give this placement class
            uniform = counts[key] / len(inserts)
            buckets[key].append((mass, uniform))
        states_used += 1

    report = {}
    for key, pairs in buckets.items():
        if len(pairs) < int(args.min_states):
            continue
        model_share = [p[0] for p in pairs]
        uniform_share = [p[1] for p in pairs]
        ratios = [m / u for m, u in pairs if u > 0]
        report[f"{key[0]}_onto_{key[1]}"] = {
            "states": len(pairs),
            "median_model_share": statistics.median(model_share),
            "median_uniform_share": statistics.median(uniform_share),
            "median_enrichment_vs_uniform": statistics.median(ratios) if ratios else None,
        }

    out = {
        "schema_version": "pmo_prior_placement_probe_v1",
        "oracle_calls_spent": 0,
        "states_used": states_used,
        "require_element": args.require_element,
        "min_states_per_class": args.min_states,
        "STABILITY_WARNING": (
            "Enrichment values are NOT stable across parent subsets: the same "
            "class can move several-fold between an all-parent sample and an "
            "element-restricted one. The ORDERING of classes is what reproduces; "
            "treat a single enrichment number as indicative, not measured, and "
            "never quote a class supported by only a handful of states."
        ),
        "reading": (
            "enrichment < 1 means the learned law puts LESS mass on that placement "
            "class than a uniform draw over the same legal insert marks would. "
            "O_onto_O is the peroxide class the construction lane emits."
        ),
        "placement_classes": dict(sorted(report.items())),
        "prior_statistics": prior.statistics,
    }
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"states {states_used}")
    print(f"{'placement':18s} {'states':>7s} {'model':>9s} {'uniform':>9s} {'enrichment':>11s}")
    for name, m in sorted(report.items(), key=lambda kv: kv[1]["median_enrichment_vs_uniform"]):
        print(f"{name:18s} {m['states']:7d} {m['median_model_share']:9.4f} "
              f"{m['median_uniform_share']:9.4f} {m['median_enrichment_vs_uniform']:11.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

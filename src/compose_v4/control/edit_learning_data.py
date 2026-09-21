"""Informative edit contrasts and small exploration niches, separate from scores."""

import math

import numpy as np

from compose_v4.control.docking_value import graph_kernel, identity, molecular_features
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.control.program_mutation import mutate_parameter
from compose_v4.rewrite.trace_shard import decode_state


def mutation_contrast_panel(
    entry,
    *,
    attachment,
    parameter_move,
    parameter_choice,
    eligibility,
    max_primitives=32,
    max_blocks=8,
):
    """Lock parent, attachment-only, parameter-only and joint reconstructions.

    All four start from the same constructor source; x is its measured completed
    parent. Only completed endpoints are gated. An ineligible part never vetoes
    the joint program, and no unscored part receives a synthetic reward.
    """
    source = decode_state(entry["source_state"])
    original = EditProgram.from_payload(entry["program"])
    if not isinstance(parameter_choice, (tuple, list)) or len(parameter_choice) != 2:
        raise ValueError("parameter contrast requires the recorded two-field choice")
    variant = mutate_parameter(original, parameter_move, tuple(parameter_choice))
    arms = {}
    for name, program, binding in (
        ("parent", original, tuple(entry["assignment"])),
        ("attachment", original, tuple(attachment)),
        ("parameter", variant, tuple(entry["assignment"])),
        ("joint", variant, tuple(attachment)),
    ):
        try:
            _, trace = execute_program_graph(
                source,
                compile_program_graph(program),
                binding,
                max_primitives=max_primitives,
                max_blocks=max_blocks,
            )
        except ValueError as error:
            arms[name] = {"status": "execution_rejected", "reason": str(error)}
            continue
        properties = eligibility({"smiles": trace["endpoint"]})
        if type(properties.get("oracle_eligible")) is not bool:
            raise ValueError("contrast requires explicit endpoint eligibility")
        arms[name] = {
            "status": "eligible" if properties["oracle_eligible"] else "ineligible",
            "endpoint": trace["endpoint"],
            "program": program.payload(),
            "assignment": list(binding),
            "trace": trace,
            "properties": properties,
        }
    if arms["parent"].get("endpoint") != entry["endpoint"]:
        raise ValueError("contrast parent fails exact construction identity")
    body = {
        "schema_version": "within_parent_edit_contrast_v1",
        "parent_entry_id": entry["entry_id"],
        "source_group": entry["source_group"],
        "oracle_protocol": entry["oracle_protocol"],
        "source_state": entry["source_state"],
        "arms": arms,
        "mutation": {
            "parameter_move": parameter_move,
            "parameter_choice": parameter_choice,
            "attachment": list(attachment),
        },
        "new_oracle_calls": 0,
    }
    return {**body, "panel_sha256": identity(body)}


def observed_interaction(panel, observations, *, task):
    if identity({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("contrast panel identity mismatch")
    if task.oracle_protocol != panel["oracle_protocol"]:
        raise ValueError("contrast oracle protocol mismatch")
    means, missing, counts = {}, [], {}
    for name, arm in panel["arms"].items():
        if arm["status"] != "eligible":
            missing.append({"arm": name, "reason": arm["status"]})
            continue
        rows = [r for r in observations if r["endpoint"] == arm["endpoint"]]
        if not rows:
            missing.append({"arm": name, "reason": "not_evaluated"})
            continue
        receipts = {}
        for row in rows:
            if row["oracle_protocol"] != task.oracle_protocol or not row["receipt_id"]:
                raise ValueError(
                    "interaction requires actual identified same-protocol observations"
                )
            utility = task.utility(row["score"])
            if row["receipt_id"] in receipts and receipts[row["receipt_id"]] != utility:
                raise ValueError("interaction receipt rebound to another score")
            receipts[row["receipt_id"]] = utility
        means[name] = float(np.mean(list(receipts.values())))
        counts[name] = len(receipts)
    if missing:
        return {
            "status": "incomplete",
            "missing": missing,
            "interaction": None,
            "observed_means": means,
            "repeat_counts": counts,
        }
    return {
        "status": "observed",
        "interaction": means["joint"] - means["attachment"] - means["parameter"] + means["parent"],
        "observed_means": means,
        "repeat_counts": counts,
        "distinct_endpoints": len({a["endpoint"] for a in panel["arms"].values()}),
        "interpretation": "descriptive scoring interaction; not a biological mechanism or confidence interval",
    }


def exploration_niches(endpoints, utilities, *, max_niches=4, per_niche=2):
    """A small deterministic max-min partition, preserving locally strong parents.

    This does not replace the incumbent archive or change the benchmark objective.
    Parameters are search settings, not established optima.
    """
    if len(endpoints) != len(utilities) or not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("niches need aligned unique endpoints and measured utilities")
    if not all(math.isfinite(u) for u in utilities) or any(
        type(v) is not int or v < 1 for v in (max_niches, per_niche)
    ):
        raise ValueError("finite utilities and positive bounded niche dimensions required")
    features = [molecular_features(s)[1] for s in endpoints]
    distance = 1 - graph_kernel(features, features)
    first = max(range(len(endpoints)), key=lambda i: (utilities[i], endpoints[i]))
    centers = [first]
    while len(centers) < min(max_niches, len(endpoints)):
        remaining = [i for i in range(len(endpoints)) if i not in centers]
        chosen = max(remaining, key=lambda i: (min(distance[i, centers]), endpoints[i]))
        if min(distance[chosen, centers]) < 1e-12:
            break
        centers.append(chosen)
    groups = [[] for _ in centers]
    for i in range(len(endpoints)):
        groups[int(np.argmin(distance[i, centers]))].append(i)
    selected = [
        sorted(group, key=lambda i: (-utilities[i], endpoints[i]))[:per_niche] for group in groups
    ]
    return {
        "centers": [endpoints[i] for i in centers],
        "members": [[endpoints[i] for i in group] for group in groups],
        "selected": [[endpoints[i] for i in group] for group in selected],
        "incumbent": endpoints[first],
        "objective_unchanged": True,
    }


def evidence_niches(endpoints, utilities, *, max_niches=4, per_niche=2):
    """Max-min partition whose centres must be BOTH distant and promising.

    `exploration_niches` picks every centre after the first by distance alone, so the
    molecules furthest from the incumbent found the niches -- and the thing furthest
    from a drug-like lead is a one-atom fragment. Measured on a completed 3x250 PMO run,
    3 of 4 centres were 1-3 heavy-atom fragments (`[SH4]`, `CCl`, `NN(N)F`, `P`, `NF`,
    `O=C=[SH4]`), the drug-like niche held 189-203 of 234 endpoints and contributed 2
    admitted parents, and the three fragment niches contributed 6 scoring 0.000-0.12
    against incumbents at 0.22-0.50.

    Niching is supposed to preserve distinct PROMISING basins. Scaling the separation by
    the endpoint's own min-max normalised utility says exactly that: a centre earns its
    niche by being far from the others AND by carrying evidence. The normalisation is
    taken from the archive's own observed range, so there is no task constant and no
    absolute threshold; with no spread in utility it degrades to the pure max-min rule.

    `exploration_niches` is deliberately left untouched -- live T4 arms allocate through
    it, and changing it would move a running campaign.
    """
    if len(endpoints) != len(utilities) or not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("niches need aligned unique endpoints and measured utilities")
    if not all(math.isfinite(u) for u in utilities) or any(
        type(v) is not int or v < 1 for v in (max_niches, per_niche)
    ):
        raise ValueError("finite utilities and positive bounded niche dimensions required")
    features = [molecular_features(s)[1] for s in endpoints]
    distance = 1 - graph_kernel(features, features)
    values = np.asarray(utilities, dtype=float)
    span = float(values.max() - values.min())
    merit = (values - values.min()) / span if span > 0 else np.ones(len(values))
    first = max(range(len(endpoints)), key=lambda i: (utilities[i], endpoints[i]))
    centers = [first]
    while len(centers) < min(max_niches, len(endpoints)):
        remaining = [i for i in range(len(endpoints)) if i not in centers]
        chosen = max(
            remaining, key=lambda i: (min(distance[i, centers]) * merit[i], endpoints[i])
        )
        if min(distance[chosen, centers]) < 1e-12:
            break
        centers.append(chosen)
    groups = [[] for _ in centers]
    for i in range(len(endpoints)):
        groups[int(np.argmin(distance[i, centers]))].append(i)
    selected = [
        sorted(group, key=lambda i: (-utilities[i], endpoints[i]))[:per_niche] for group in groups
    ]
    return {
        "centers": [endpoints[i] for i in centers],
        "members": [[endpoints[i] for i in group] for group in groups],
        "selected": [[endpoints[i] for i in group] for group in selected],
        "incumbent": endpoints[first],
        "objective_unchanged": True,
    }

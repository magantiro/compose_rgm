"""Selected-path verification for the resumable frontier's first oracle round."""

from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.control import graph_geometry as GG
from compose_v4.control.continuation import _tilted
from compose_v4.control.docking_value import DockingValue, snapshot_equivalence
from compose_v4.control.frontier_search import payload_hash
from compose_v4.control.molecular_search_codec import (
    decode_search_state,
    encode_search_state,
)
from compose_v4.control.molecular_task_search import MolecularHierarchy
from compose_v4.control.option_continuation import (
    EXECUTABLE_PRODUCT_GATE,
    OptionContinuationKernel,
)
from compose_v4.control.option_selector import GENERIC_OPTION, balanced_option_prior
from compose_v4.control.task_search import SearchRow
from compose_v4.experiments.continuation_profile import ExecutorMeter, state_payload
from compose_v4.experiments.t4_endpoint_selection import (
    acceptable_endpoint,
    calculate_properties,
)
from compose_v4.experiments.t4_frontier_search import SCHEMA, FrontierConfig
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def matching_replayed_successor(successors, expected_payload):
    """Resolve one recorded augmented state from an aliased primitive mark.

    A physical descriptor mark can be compatible with several option-progress
    branches. The event's exact saved option state disambiguates those branches;
    accepting the mark requires exactly one match, not exactly one total branch.
    """
    expected = payload_hash(expected_payload)
    matches = [
        successor
        for successor in successors
        if payload_hash(state_payload(successor)) == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected primitive does not uniquely recover its recorded augmented successor"
        )
    return matches[0]


def verify_preparation(result, warm, system):
    """No learned-law re-enumeration: replay the selected marks through option contracts.

    Full-row numerical correctness is covered by the production kernel tests.
    Here its recorded reference is checked against the cached row and selected
    exact-state path. Only this explicit audit can emit the new oracle-lock schema.
    """
    saved = result["checkpoint"]
    config = FrontierConfig(**saved["config"])
    body = {k: v for k, v in saved.items() if k != "checkpoint_sha256"}
    if (
        saved["schema_version"] != SCHEMA
        or payload_hash(body) != saved["checkpoint_sha256"]
        or saved["round"] != warm["round"] + 1
        or saved["archive_prefix"] != [payload_hash(row) for row in warm["archive"]]
        or saved["prior_oracle_attempts"] != warm["oracle_attempts"]
        or saved["new_oracle_calls"] != 0
        or saved["oracle_authorized"] is not False
        or saved["retired_lineages"]
        or len(saved["frontier"]) != config.lineages
        or [u["lineage_index"] for u in saved["frontier"]]
        != list(range(config.lineages))
    ):
        raise ValueError("frontier preparation hash, prefix, round or lineage mismatch")
    model = DockingValue.fit(
        warm["archive"],
        before_round=saved["round"],
        source_sha256=saved["source_sha256"],
    )
    if model.payload != saved["value_snapshot"]:
        raise ValueError(
            "frontier task snapshot is not the complete prior-round archive"
        )
    source_snapshots = result.get("source_value_snapshots", [saved["value_snapshot"]])
    if not source_snapshots or len(
        {p["snapshot_sha256"] for p in source_snapshots}
    ) != len(source_snapshots):
        raise ValueError("frontier source task snapshots are missing or duplicated")
    snapshot_audit = [
        snapshot_equivalence(
            model.payload,
            source,
            probe_smiles=(candidate["smiles"] for candidate in result["pool"]),
        )
        for source in source_snapshots
    ]
    allowed_snapshot_ids = {
        model.payload["snapshot_sha256"],
        *(p["snapshot_sha256"] for p in source_snapshots),
    }
    old_states = {payload_hash(r["state"]) for r in warm["archive"]}
    previous = {r["smiles"] for r in warm["archive"]}
    endpoints, options, decisions = {}, Counter(), Counter()
    with ExecutorMeter(None).instrument() as meter:
        meter.phase = "selected_path_verification"
        for unit in saved["frontier"]:
            root = decode_search_state(unit["root"])
            if payload_hash(encode_state(root.graph)) not in old_states:
                raise ValueError("frontier parent is not an exact warm-archive state")
            node, origin_lineage = root, None
            planner = unit["planner"]
            if planner is None:
                raise ValueError("frontier did not prepare its lineage")
            if (
                payload_hash(
                    {k: v for k, v in planner.items() if k != "checkpoint_sha256"}
                )
                != planner["checkpoint_sha256"]
                or planner["snapshot_id"] not in allowed_snapshot_ids
            ):
                raise ValueError("planner checkpoint hash mismatch")
            cached = {
                payload_hash(planner["states"][r["node"]]): r for r in planner["rows"]
            }
            for index, event in enumerate(unit["path"]):
                if (
                    event["index"] != index
                    or payload_hash(event["source"])
                    != payload_hash(encode_search_state(node))
                    or event["round"] != saved["round"]
                ):
                    raise ValueError("frontier path discontinuity")
                d, selected = event["decision"], event["selected_index"]
                row = SearchRow(
                    tuple(range(len(d["core"]))),
                    tuple(d["labels"]),
                    tuple(d["core"]),
                    tuple(d["floor"]),
                    d["epsilon"],
                )
                p, q = row.reference, np.asarray(d["probabilities"])
                recorded = cached[payload_hash(event["source"])]
                if (
                    recorded["core"] != d["core"]
                    or recorded["floor"] != d["floor"]
                    or recorded["epsilon"] != d["epsilon"]
                    or recorded["labels"] != d["labels"]
                    or not 0 <= selected < len(p)
                    or planner["states"][recorded["children"][selected]]
                    != event["product"]
                ):
                    raise ValueError(
                        "decision differs from its complete cached reference row"
                    )
                tilted, _, _ = _tilted(row.core, d["values"], kappa=1, exploration=0)
                expected = (1 - row.epsilon) * tilted + row.epsilon * np.asarray(
                    row.floor
                )
                if (
                    q.shape != p.shape
                    or not np.isfinite(q).all()
                    or np.any(q < 0)
                    or not np.isclose(q.sum(), 1, rtol=0, atol=1e-10)
                    or not np.allclose(d["reference"], p, rtol=0, atol=1e-12)
                    or not np.allclose(q, expected, rtol=0, atol=1e-12)
                    or np.any(q < row.epsilon * np.asarray(row.floor) - 1e-12)
                    or np.any((q > 0) & (p <= 0))
                    or q[selected] <= 0
                    or d["labels"][selected] != event["selected_label"]
                    or d["snapshot_id"] not in allowed_snapshot_ids
                ):
                    raise ValueError(
                        "invalid frontier decision probability, support or snapshot"
                    )
                live = q > 0
                kl = float(np.sum(q[live] * np.log(q[live] / p[live])))
                if kl > 1 + 1e-10 or not np.isclose(kl, d["kl"], atol=1e-10, rtol=0):
                    raise ValueError("frontier decision KL mismatch")
                if config.guidance == "post_hoc" and (
                    not np.allclose(q, p, atol=1e-12, rtol=0) or any(d["best_witness"])
                ):
                    raise ValueError("post-hoc arm used generation-time task guidance")
                decisions[f"{node.stage}:{d['status']}"] += 1
                product = decode_search_state(event["product"])
                if node.stage == "where":
                    expected_row = MolecularHierarchy(None).row(node)
                    if (
                        tuple(d["labels"]) != expected_row.labels
                        or not np.allclose(
                            p, expected_row.reference, atol=1e-12, rtol=0
                        )
                        or payload_hash(
                            encode_search_state(expected_row.successors[selected])
                        )
                        != payload_hash(event["product"])
                    ):
                        raise ValueError(
                            "WHERE changed the frozen local/global reference"
                        )
                elif node.stage == "what":
                    if (
                        GENERIC_OPTION not in d["labels"]
                        or row.epsilon != 0.1
                        or not np.allclose(
                            row.core,
                            balanced_option_prior(tuple(d["labels"]), exploration=0),
                            atol=1e-12,
                            rtol=0,
                        )
                    ):
                        raise ValueError(
                            "WHAT lost generic or the balanced option prior"
                        )
                    hierarchy = MolecularHierarchy(None)
                    active = hierarchy.option_state(node, event["selected_label"])
                    if (
                        payload_hash(state_payload(active))
                        != payload_hash(state_payload(product.active))
                        or product.budget != node.budget
                        or product.lineage != node.lineage
                    ):
                        raise ValueError("WHAT option origin or horizon mismatch")
                    origin_lineage = node.lineage
                    options[active.option] += 1
                else:
                    mark = event["mark"]
                    codec = (
                        action_codec_v4 if mark["schema_version"] == 4 else action_codec
                    )
                    rule, action = codec.decode_action(mark)
                    if event["selected_label"] != f"{rule}:{action!r}":
                        raise ValueError("selected primitive label mismatch")
                    # Reuse the actual option filter/progress/executor machinery,
                    # but admit only the selected witness, not a new proposal row.
                    witness = OptionContinuationKernel(
                        lambda _, rule=rule, action=action: (
                            (rule,),
                            (action,),
                            (1.0,),
                        ),
                        system,
                        max_executor_applications=None,
                        product_gate=EXECUTABLE_PRODUCT_GATE,
                    ).row(node.active)
                    try:
                        following = matching_replayed_successor(
                            witness.successors, event["option_product"]
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"lineage {unit['lineage_index']} event {index}: "
                            "selected primitive fails executor or option contract"
                        ) from error
                    expected_product = MolecularHierarchy._successor(node, following)
                    if payload_hash(
                        encode_search_state(expected_product)
                    ) != payload_hash(event["product"]):
                        raise ValueError(
                            "primitive exact product or option progress mismatch"
                        )
                    if product.stage == "where":
                        local = GG.structural_displacement(
                            node.active.origin,
                            product.graph,
                            origin_lineage,
                            product.lineage,
                        )
                        cumulative = GG.structural_displacement(
                            root.graph, product.graph, root.lineage, product.lineage
                        )
                        endpoints[(root.root_id, index)] = (
                            event,
                            node,
                            local,
                            cumulative,
                        )
                node = product
            if payload_hash(unit["current"]) != payload_hash(
                encode_search_state(node)
            ) or (node.budget and unit["status"] != "no_admissible_action"):
                raise ValueError(
                    "preparation stopped before its declared primitive horizon"
                )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(warm["archive"][0]["smiles"]))
    pool = result["pool"]
    if len(pool) != len({c["smiles"] for c in pool}):
        raise ValueError("canonical duplicate in frontier pool")
    for c in pool:
        event, source, local, cumulative = endpoints[(c["root_id"], c["event"])]
        if (
            c["state"] != event["product"]["graph"]
            or c["smiles"] != canonical_state_key(decode_state(c["state"]))
            or c["smiles"] in previous
            or not c["program_complete"]
            or c["bundle_id"] != source.active.bundle_id
        ):
            raise ValueError("candidate is not a new committed option completion")
        expected = {
            "r_release": source.region.released_fraction,
            "r_coherent": local["largest_changed_fraction"],
            "r_change": local["changed_fraction"],
            "d_cycle_rank": local["d_cycle_rank"],
            "d_ring_systems": local["d_ring_systems"],
            "d_heavy": local["d_heavy"],
        }
        props = calculate_properties(
            Chem.MolFromSmiles(c["smiles"]),
            seed_fp=seed_fp,
            generator=generator,
            sa_scorer=sascorer.calculateScore,
            delta=0.4,
            qed_min=0.6,
            sa_max=4,
        )
        if (
            any(not np.isclose(c[k], v) for k, v in {**expected, **props}.items())
            or c["cumulative_change"] != cumulative
            or c["oracle_eligible"] is not acceptable_endpoint(c)
            or not np.isclose(c["predicted_docking"], model.predict([c["smiles"]])[0])
        ):
            raise ValueError("candidate geometry, feasibility or task-score mismatch")
    take = result["proposed_for_audit"]
    pool_by_smiles = {c["smiles"]: c for c in pool}
    if (
        len(take) > config.oracle_batch
        or len({c["smiles"] for c in take}) != len(take)
        or len({c["allocated_bundle_id"] for c in take}) != len(take)
    ):
        raise ValueError("oracle lock exceeds budget or repeats molecules/bundles")
    for c in take:
        if (
            {k: v for k, v in c.items() if k != "allocated_bundle_id"}
            != pool_by_smiles.get(c["smiles"])
            or not acceptable_endpoint(c)
            or c["allocated_bundle_id"] not in c["origin_bundle_ids"]
        ):
            raise ValueError(
                "oracle lock contains an unverified or ineligible candidate"
            )
    return {
        "schema_version": "t4_frontier_oracle_lock_v1",
        "round": saved["round"],
        "checkpoint_sha256": saved["checkpoint_sha256"],
        "pool_sha256": payload_hash(pool),
        "source_sha256": saved["source_sha256"],
        "input_sha256": saved["input_sha256"],
        "code_revision": saved["code_revision"],
        "config": saved["config"],
        "prior_oracle_attempts": warm["oracle_attempts"],
        "take": take,
        "audit": {
            "selected_path_executor_calls": meter.calls,
            "pool_count": len(pool),
            "eligible_count": sum(c["oracle_eligible"] for c in pool),
            "selected_options": dict(sorted(options.items())),
            "decisions": dict(sorted(decisions.items())),
            "source_value_snapshot_equivalence": snapshot_audit,
        },
    }

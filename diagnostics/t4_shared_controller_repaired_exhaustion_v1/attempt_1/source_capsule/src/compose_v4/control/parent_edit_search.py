"""Connect same-pool parent/edit allocation to the persistent program engine."""

import numpy as np

from compose_v4.control.parent_edit_model import (
    MUTATION_RECIPE,
    ParentEditFeatures,
    select_parent_edits,
)


def prepare_query_batch(
    search, task, *, count, seed, model=None, model_factory=None, diagnostic=False
):
    """Generate complete candidates, then lock a query subset before any oracle.

    The entire generated pool and its work ledger remain in the query receipt.
    Caller supplies measured outcomes to the existing observe_batch method.
    A learned model is deliberately unavailable for non-diagnostic deployment.
    """
    if search.oracle_protocol != task.oracle_protocol:
        raise ValueError("search/task oracle identities disagree")
    if search.config.score_direction != ("minimize" if task.kind == "t4" else "maximize"):
        raise ValueError("search parent direction disagrees with task utility")
    if type(count) is not int or count < 1:
        raise ValueError("query count must be a positive integer")
    if model is not None and model_factory is not None:
        raise ValueError("supply a fitted model or a lazy model factory, not both")
    if (model is not None or model_factory is not None) and not diagnostic:
        raise ValueError("learned parent-edit allocation is not prospectively qualified")
    pool = search.propose_batch(task.endpoint_evaluator())
    candidates = pool["candidates"]
    if not candidates:
        return pool
    if len(candidates) <= count:
        selection = {
            "mode": "direct",
            "reason": "pool_does_not_exceed_query_allowance",
            "selected_ids": [c["candidate_id"] for c in candidates],
            "task_id": task.task_id,
            "model_sha256": None,
        }
        locked = search.lock_query_subset(pool["batch_id"], selection["selected_ids"], selection)
        return {
            **locked,
            "proposal_seconds": pool["proposal_seconds"],
            "work_cache": pool.get("work_cache"),
        }
    if model_factory is not None:
        model = model_factory()
    unique = sorted({r["endpoint"] for r in search.observations.values()})
    utilities = [
        float(
            np.mean(
                [
                    task.utility(r["score"])
                    for r in search.observations.values()
                    if r["endpoint"] == endpoint
                ]
            )
        )
        for endpoint in unique
    ]
    predicted = np.zeros(len(candidates))
    if model is not None:
        feature = ParentEditFeatures()
        rows = []
        for candidate in candidates:
            parent = search.entries[candidate["provenance"]["entry_id"]]
            scores = [
                task.utility(r["score"])
                for r in search.observations.values()
                if r["endpoint"] == parent["endpoint"]
            ]
            if model.payload["recipe"] == MUTATION_RECIPE:
                rows.append(
                    feature.with_mutation_context(
                        candidate,
                        parent_state=parent["trace"]["states"][-1],
                        parent_scores=scores,
                        parent_record=parent,
                    )
                )
            else:
                rows.append(
                    feature(
                        candidate, parent_state=parent["trace"]["states"][-1], parent_scores=scores
                    )
                )
        predicted, _ = model.predict(rows, oracle_protocol=task.oracle_protocol)
    selected = select_parent_edits(
        [c["candidate_id"] for c in candidates],
        predicted,
        utilities,
        k=task.top_k,
        count=min(count, len(candidates)),
        seed=seed,
        mode="learned" if model is not None else "score_blind",
        diagnostic=diagnostic,
    )
    selected["task_id"] = task.task_id
    selected["model_sha256"] = None if model is None else model.payload["model_sha256"]
    locked = search.lock_query_subset(pool["batch_id"], selected["selected_ids"], selected)
    return {
        **locked,
        "proposal_seconds": pool["proposal_seconds"],
        "work_cache": pool.get("work_cache"),
    }

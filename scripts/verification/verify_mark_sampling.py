"""Is sampling a MARK the same as sampling a CANONICAL SUCCESSOR?

The claim behind the rollout optimisation is exact, not approximate:

    P(canonical successor y | x) = sum over marks m reaching y of P(m | x)

so drawing a mark in proportion to its probability and canonicalising only that
one mark samples the canonical successor law -- without applying and
canonicalising the other ~500 marks.

This checks the identity deterministically rather than by Monte Carlo: for each
state it rebuilds the mark->canonical map, sums the mark probabilities per
canonical key, and compares against the probabilities the full kernel reports.
A Monte-Carlo agreement check would only bound the error by sampling noise; this
either matches to machine precision or it does not.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/rmaganti/compose_v2_work/src")
H = Path("/Users/rmaganti/compose_trainset_backup")
LOCAL = H / "localprep/artifacts/editing_v2"
A8 = LOCAL / "process_v2_active8/8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb"
# No hardcoded scratchpad path: those belong to a single session and become
# dead references the moment it ends. The materialized scorer is fetched from
# the durable volume with
#   modal volume get compose-v4-artifacts \
#       editing_v2/r_theta_run/materialized_scorer <dir>
# and its location passed in.
SP = Path(os.environ.get("COMPOSE_SCRATCH", "")).expanduser()
if not SP or not (SP / "matscorer").is_dir():
    raise SystemExit(
        "set COMPOSE_SCRATCH to a directory containing 'matscorer' "
        "(fetch it from the volume: editing_v2/r_theta_run/materialized_scorer)")

from compose_v4.data.corpus_training_library import load_corpus_training_library  # noqa: E402
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (  # noqa: E402
    open_process_v2_t1_source,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (  # noqa: E402
    build_process_v2_score_revised_scratch_runtime,
    load_materialized_scorer_state,
)
from compose_v4.experiments.production_successor_kernel import (  # noqa: E402
    _default_rewrite_system,
    canonical_successor_result,
    enumerate_factorized_marked_law,
)
from compose_v4.rewrite.kernel import canonical_state_key  # noqa: E402

started = time.perf_counter()
source = open_process_v2_t1_source(
    A8,
    gate_zero_decision_path=LOCAL / "process_v2_gate_zero_v7/DECISION.json",
    artifact_root=LOCAL.parent,
    repo_root=Path("/Users/rmaganti/compose_v2_work"),
)
mat = load_materialized_scorer_state(SP / "matscorer")
runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
    source, materialized_state=mat)
model = runtime.model
system = _default_rewrite_system(model)

resolution = json.loads(
    (H / "modal_upload_stage/run_inputs/editing_v2_split_precedence_resolution.json").read_text())
library = load_corpus_training_library(
    [H / "consolidated_train"],
    excluded_sources=resolution["excluded_source_keys"]["train"],
    verify_state_roundtrip=False)
import gzip  # noqa: E402
reserve = json.load(gzip.open(
    "/Users/rmaganti/compose_v2_work/diagnostics/"
    "editing_v2_matched_validation_reserve_ids.json.gz", "rt"))
states, _f, _e = library.inputs_for(list(reserve["reserve_entry_ids"])[:8])
print(f"[{time.perf_counter()-started:6.1f}s] ready", flush=True)

worst_gap = 0.0
worst_virtual = 0.0
for i, state in enumerate(states[:5]):
    result = canonical_successor_result(model, state, 0.5)
    exact = {s.key: float(s.probability) for s in result.batch.successors}

    law = enumerate_factorized_marked_law(model, state, 0.5)
    rebuilt: dict[str, float] = {}
    virtual_mass = 0.0
    for mark in law.marks:
        successor = system.apply(state, mark.executor_rule_name, mark.action)
        key = canonical_state_key(successor)
        if key == law.source_key:
            virtual_mass += float(mark.probability)
            continue
        rebuilt[key] = rebuilt.get(key, 0.0) + float(mark.probability)

    missing = set(exact) ^ set(rebuilt)
    gap = max((abs(exact[k] - rebuilt.get(k, 0.0)) for k in exact), default=0.0)
    worst_gap = max(worst_gap, gap)
    worst_virtual = max(worst_virtual,
                        abs(virtual_mass - float(result.batch.virtual_mass)))
    print(f"  state {i}: {len(law.marks):4d} marks -> {len(exact):4d} successors | "
          f"support mismatch {len(missing):3d} | max |P_exact - P_summed| {gap:.3e} | "
          f"virtual mass gap {abs(virtual_mass - float(result.batch.virtual_mass)):.3e}",
          flush=True)

print(f"\nworst probability gap : {worst_gap:.3e}")
print(f"worst virtual-mass gap: {worst_virtual:.3e}")
print("\nVERDICT:", "EXACT -- mark sampling is canonical-successor sampling"
      if worst_gap < 1e-12 and worst_virtual < 1e-12
      else "NOT EXACT -- do not use the rollout shortcut")

"""Where do the ~12 seconds of one successor enumeration actually go?

TIMING ONLY, against the LOCAL Active8. The counts here are not the corpus
counts (RUN_PATHS pins the volume's gate-zero), but the per-phase COST split is
what transfers, and that is all this is for.

The question that matters for C0: a rollout step needs ONE sampled successor,
not the full canonical partition. If the cost is dominated by the per-mark
apply/canonicalise loop rather than by the model forward, then sampling a single
mark and canonicalising only that one is the same distribution for a small
fraction of the work -- because P(canonical successor y) is exactly the sum of
the mark probabilities that reach y, so sampling a mark in proportion to its
probability and mapping it to its canonical key samples the canonical law.
"""
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

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

import torch  # noqa: E402

from compose_v4.data.corpus_training_library import load_corpus_training_library  # noqa: E402
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (  # noqa: E402
    open_process_v2_t1_source,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (  # noqa: E402
    build_process_v2_score_revised_scratch_runtime,
    load_materialized_scorer_state,
)
from compose_v4.experiments.production_successor_kernel import (  # noqa: E402
    canonical_successor_result,
    enumerate_factorized_marked_law,
)

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
print(f"[{time.perf_counter()-started:6.1f}s] runtime built", flush=True)

resolution = json.loads(
    (H / "modal_upload_stage/run_inputs/editing_v2_split_precedence_resolution.json").read_text())
library = load_corpus_training_library(
    [H / "consolidated_train"],
    excluded_sources=resolution["excluded_source_keys"]["train"],
    verify_state_roundtrip=False)

# Pick drug-sized states, matching the C0 source band.
import gzip  # noqa: E402
reserve = json.load(gzip.open(
    "/Users/rmaganti/compose_v2_work/diagnostics/"
    "editing_v2_matched_validation_reserve_ids.json.gz", "rt"))
ids = list(reserve["reserve_entry_ids"])[:40]
states, _f, _e = library.inputs_for(ids)
print(f"[{time.perf_counter()-started:6.1f}s] {len(states)} states", flush=True)

# ---- phase split -------------------------------------------------------
rows = []
for i, state in enumerate(states[:6]):
    t0 = time.perf_counter()
    law = enumerate_factorized_marked_law(model, state, 0.5)
    t_law = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = canonical_successor_result(model, state, 0.5)
    t_full = time.perf_counter() - t1

    n_marks = len(law.marks)
    n_succ = len(result.batch.successors)
    rows.append((n_marks, n_succ, t_law, t_full))
    print(f"  state {i}: {n_marks:5d} marks -> {n_succ:4d} successors | "
          f"marked law {t_law:6.3f}s | full {t_full:6.3f}s | "
          f"apply+canon {t_full - t_law:6.3f}s "
          f"({100*(t_full-t_law)/t_full:4.1f}%)", flush=True)

import statistics as st  # noqa: E402
law_share = st.mean(r[2] / r[3] for r in rows)
print(f"\nmarked-law share of total: {law_share:6.1%}")
print(f"apply+canonicalise share : {1-law_share:6.1%}")
print(f"mean marks per state     : {st.mean(r[0] for r in rows):.0f}")
print(f"mean full enumeration    : {st.mean(r[3] for r in rows):.3f}s")
print(f"\nIf a rollout step only needs ONE sampled successor, its cost floor is")
print(f"the marked law: {st.mean(r[2] for r in rows):.3f}s vs "
      f"{st.mean(r[3] for r in rows):.3f}s -- "
      f"a {st.mean(r[3] for r in rows)/max(st.mean(r[2] for r in rows),1e-9):.1f}x saving.")

# ---- hot functions -----------------------------------------------------
profiler = cProfile.Profile()
profiler.enable()
canonical_successor_result(model, states[0], 0.5)
profiler.disable()
buffer = io.StringIO()
pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(18)
print("\n" + "\n".join(buffer.getvalue().splitlines()[4:30]))

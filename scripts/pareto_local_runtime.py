"""Build the frozen R_theta runtime LOCALLY, from artifacts pulled off the volume.

WHY THIS EXISTS. A parallel implementation that produces claim-bearing numbers
may not ship on fixture parity alone. Synthetic fixtures are the right tool for
finding races; they cannot show that the real model, on the real successor
fibers, takes the identical actions. That requires replaying the committed
12-source smoke against the real checkpoint, which requires the real checkpoint
here.

WHAT MUST COME FROM THE VOLUME, NOT FROM THE REPO. `RUN_PATHS.json` carries an
explicit warning that the repo's local Active8 copy DIFFERS from the volume's:

    "v6 is the decision whose source_index_sha256 matches the Active8 stream ON
     THE VOLUME. The locally generated v7 matches the LOCAL Active8 copy, which
     differs -- the corpus was compiled on Modal, so the volume's Active8 is
     authoritative here."

So a parity replay built on the repo's Active8 would authenticate against the
wrong stream and could differ from the run it claims to reproduce. Every path
below points at the volume-sourced copy under `local_runtime/`, and
`open_process_v2_t1_source` re-verifies the whole chain (Active8 completion,
plan, Gate-0 PASS, frozen policy) on every construction, so a mismatched
artifact fails loudly rather than producing a plausible wrong answer.

WHY THIS FAILS OFF-VOLUME, AND WHY THAT IS NOT A CORPUS PROBLEM
--------------------------------------------------------------
`source_index_sha256` is computed over a body whose FIRST FIELD is
`active8_run_root` -- the ABSOLUTE FILESYSTEM PATH
(`editing_v2_process_v2_gate_zero.py:593`), and the same path is re-derived
inside `reduce_gate_zero`. So the index pins the MOUNT POINT as well as the
content, and a byte-identical mirror at a different path cannot reproduce it.

Measured, so nobody rediscovers it the hard way:

    decision                       b5d042a06c27be4a...
    local mirror, local path       ed51268a155877d4...   <- looks like a mismatch
    local mirror, container path   b5d042a06c27be4a...   <- MATCH

Every content field already agreed: completion sha, sentinel sha, contracts
binding, all 270 shard entries, eligible task identities, role census, sealed
role metadata. The local Active8 is CONTENT-IDENTICAL to the authoritative one.
There is nothing to fetch and nothing to fix.

Run where the volume is mounted at its own path; the chain validates there
unchanged. Do not modify the lineage system to accommodate a mirror.

Layout produced by the download step:

    runs/run_v2_01/R_THETA_CHECKPOINT.pt      84 MB
    local_runtime/materialized_scorer/        49 MB
    local_runtime/active8/                   751 MB, 1641 files
    local_runtime/gate_zero_v6/DECISION.json
    local_runtime/pareto_control_smoke/       the SERIAL BASELINE, 12 shards
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: The 884 MB of volume-sourced runtime is gitignored and lives in ONE place, so
#: a git worktree running this script must be told where. Set COMPOSE_RUNTIME_HOME
#: rather than editing a path in, which is what created a worktree-local variant
#: of this file that then drifted from the committed one.
HOME = Path(os.environ.get("COMPOSE_RUNTIME_HOME", REPO))

LOCAL = HOME / "local_runtime"
ACTIVE8 = LOCAL / "active8" / "8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb"
GATE_ZERO = LOCAL / "gate_zero_v6" / "DECISION.json"
MATERIALIZED = LOCAL / "materialized_scorer"
CHECKPOINT = HOME / "runs" / "run_v2_01" / "R_THETA_CHECKPOINT.pt"
SERIAL_BASELINE = LOCAL / "pareto_control_smoke"


def build_local_model():
    """The frozen R_theta, eval mode, grad disabled. Identical to the app's."""
    import torch

    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )

    for path in (ACTIVE8, GATE_ZERO, MATERIALIZED, CHECKPOINT):
        if not path.exists():
            raise SystemExit(
                f"missing {path}\n"
                "Run the volume download first; see this module's docstring.")

    source_obj = open_process_v2_t1_source(
        ACTIVE8,
        gate_zero_decision_path=GATE_ZERO,
        # The app passes the container's /artifacts; the plan's run_artifact_root
        # is resolved against it and must land back on ACTIVE8, so the local
        # mirror root is the parent of the process_v2_active8 tree.
        artifact_root=LOCAL / "artifact_root",
        repo_root=REPO,
    )
    bundle = load_materialized_scorer_state(MATERIALIZED)
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source_obj, materialized_state=bundle)
    model = runtime.model
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model

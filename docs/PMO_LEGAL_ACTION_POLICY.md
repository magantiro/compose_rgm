# PMO legal WHERE/HOW action gate

## Scientific question

The previous dependency-region comparison improved held-family ranking when a
complete candidate was supplied, but none of the 559 generic complete endpoints
recovered a teacher transformation. This gate tests the next missing capability:
can a target-free policy retain and rank the observed next molecular successor
inside the fixed generic Active8 legal fiber, conditional on the observed
generic executor rule? Rule selection remains a separate upstream problem.

For each exact teacher state, the runner enumerates one canonical-successor
fiber using the fixed broad-organic vocabulary and the production Editing-V2
executor. Padding-slot aliases are quotiented by canonical molecular identity.
The teacher is never added when absent.

The comparison holds out complete PMO task families before fitting any numeric
preprocessing or ranker statistics. Within the teacher rule's legal fiber, it
compares deterministic uniform ranking with a diagonal graph-conditioned
contrastive successor ranker trained on same-state legal negatives. Training
examples are balanced by task family, lineage and decision.

## Claim boundary

This is a one-step WHERE/HOW coverage and rank test conditional on the teacher's
generic rule. It is not a rule-selection result, complete-program generation,
autonomous route recovery, PMO objective optimization or evidence that a scored
pilot will improve. Runtime checkpoints contain numeric parameters and fixed
generic support only.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m tools.pmo_legal_action_policy \
  --output diagnostics/pmo_legal_action_policy/attempt_1
```

Focused verification:

```bash
.venv/bin/python -m pytest tests/test_pmo_legal_action_policy.py
.venv/bin/python -m ruff check \
  src/compose_v4/experiments/pmo_legal_action_policy.py \
  tools/pmo_legal_action_policy.py \
  tests/test_pmo_legal_action_policy.py
```

The runner is zero-oracle, refuses overwrite, verifies all sealed inputs and
publishes result and runtime checkpoints atomically.

## Development attempt 1

The first complete development execution evaluated 6,143 held-family next-step
events from 184 routes and made zero oracle calls. Within routes that fit the
unchanged 32-primitive runtime support, the fixed Active8 rule fiber contained
the teacher successor with source-balanced coverage 0.9887. Conditional on the
observed generic rule, the learned WHERE/HOW ranker improved weighted mean rank
from 65.76 to 14.52, top-10 recall from 0.3220 to 0.6283, and top-32 recall from
0.5621 to 0.8974. Precision among covered events was 1.0 because missing
teachers were never injected.

The run took 813.9 wall seconds on one arm64 CPU process. Shared exact-fiber
preparation consumed 783.0 seconds, fold fitting 5.3 seconds, and held-fold
evaluation 20.7 seconds. Peak resident memory was 1.22 GB. An earlier
prepublication attempt was terminated after 223 seconds with zero oracle calls
when profiling identified redundant cross-fold fiber enumeration; its abort
receipt is preserved separately.

Attempt 1 is a development preview, not the authoritative sealed result,
because its implementation files were not yet contained in the recorded Git
revision. Preserve it and repeat the unchanged computation from the eventual
clean committed revision before using the metrics as authoritative evidence.
Regardless of provenance, the result remains conditional on the observed rule
and does not authorize a scored pilot. The next scientific problem is generic
rule sequencing and stopping in a complete-program decoder.

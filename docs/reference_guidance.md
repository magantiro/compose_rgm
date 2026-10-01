# Frozen-reference program selection

This interface combines a task controller's candidate probabilities with a
frozen molecular reference. It does not retrain the reference or change the
executor. The task runner records the reference identity and selection policy
before any objective call.

## Scope and acceptance

The problem is to reuse learned molecular-edit preferences when selecting complete
executable programs. The output is a distribution over the caller's already
eligible, deduplicated candidate panel. The proposed benefit is better selection.
Implementation tests do not establish an optimization improvement.

The reference scores exact persistent-slot actions and intermediate states. Its
support is the loaded checkpoint's declared vocabulary and native action fiber,
with at most 40 active atoms for the supplied editing checkpoint. A legal program
need not have a score under that checkpoint. Neither endpoint SMILES nor a trace
from a different proposed transformation can substitute for its executed trace.

The implementation acceptance checks are:

- off mode does not load or evaluate a reference.
- shadow mode preserves baseline probabilities, selections and random state.
- active mode changes selection probabilities when scores differ.
- inference leaves reference parameters unchanged and uses CPU float32.
- malformed traces, checkpoint identity drift and unexpected numerical errors
  fail explicitly.
- missing native-mark coverage is reported and never silently removes candidates.
- the interface works without network access, oracle calls or a cloud account.

The comparison baseline is the same controller and candidate generator with
guidance off. Shadow mode is a compatibility diagnostic, not evidence of guidance.
Any subsequent scientific comparison must freeze strengths and candidate/oracle
budgets before evaluation and report quality, diversity, coverage and compute.

## Selection rule

For baseline probabilities `p`, program scores `s`, strength `beta`, and a positive
log-weight cap `c`, active mode uses

```
adjustment[i] = clip(beta * (s[i] - max(s)), -c, 0)
q[i]         = p[i] * exp(adjustment[i]) / sum(p * exp(adjustment))
```

The reference score is the mean log probability of each executed canonical
successor under the checkpoint's productive-jump kernel, at a declared fixed
progress coordinate. Equivalent legal marks are grouped by molecular outcome.
The mean is a finite-panel preference, not a path likelihood or an exact Doob
transform. Fragment panel selection uses native mark scores from the same
checkpoint, while the PMO and T4 selectors use executed-program scores. The
cap limits changes in selection odds, not changes in benchmark performance.
Zero baseline probabilities stay zero.
The reference cannot make an ineligible candidate eligible.

Off and shadow return the original baseline probabilities without renormalizing
them. Shadow only records scores. Active mode requires a positive strength and
cap. If any candidate is outside native-mark scoring coverage, the default policy
retains the entire baseline panel and records the reason. A strict policy can
instead require complete coverage and stop. Unexpected errors are never converted
into a fallback.

The explicit `preserve_mass` coverage policy handles mixed panels. It keeps each
unscored candidate's baseline probability. It reweights scored candidates within
their original total probability mass. This requires at least two scored
candidates with positive baseline probability. Otherwise it returns the baseline.
The receipt reports coverage and marks mixed guidance as `active_partial`.
This rule preserves first-draw mass. It does not preserve marginal inclusion
probabilities in a multi-candidate batch and cannot guarantee improved scores.
Missing traces and unsupported marks remain visible with no invented score.

## Integration boundary

### PMO and T4 exploration adapters

The selector integration uses the controllers' existing non-deterministic slots.
PMO keeps its online-value top-k allocation. The local T4 program-only runner
keeps its fixed allocation floors and fits no program-value model. The remaining
eligible T4 candidates are drawn uniformly in off mode or reweighted by the
frozen reference in active mode. Every admitted program executes through
the shared primitive rewrite system.

The implementation must preserve the exact baseline RNG calls in off and shadow
modes, the selected batch size, candidate order before sampling, and all hard
allocation floors. Active sampling is without replacement, with strictly
positive probability for every remaining candidate. Record first-draw weights
separately from sequential conditional probabilities. Neither is a marginal
batch-inclusion propensity. Capture exact executed programs at the producer,
never recover their persistent-slot states by reparsing endpoint SMILES.

Off and shadow modes preserve the controller's RNG behavior. Active mode changes
probabilities only inside the non-floor draw. PMO model-ranked selections and
hard allocation quotas remain unchanged.

Pass the controller's actual pre-lock selection probabilities, not an invented
uniform distribution. A deterministic top-k or contextual batch allocator needs
an explicit adapter. This interface must not silently replace it. Preserve the
candidate identities, eligibility checks, proposal construction and oracle budget.

PMO records containing exact `source_state`, `trace` and `endpoint` can be checked
directly. T4's `expand(..., include_realized_actions=True)` compiles accepted
structural goals through the primitive executor and retains their exact source
and actions. A compiled program is checked by replay from its measured source.
Compilation abstentions keep the candidate unscored and leave it in the pool.
The route lane instead captures
the source and actions directly from complete-region execution through
`t4_route_proposals.expand_route`. Both paths use the same reference scorer and
T4 selector.

`ProgramPanelGuidance` joins each remaining controller row to its exact program
by ID. Pass it as `reference_guide` and persist a `reference_receipts` list with
the query lock when calling PMO's `RewardAdaptiveProgramController.acquire` or
T4's `select_batch`. For a deduplicated T4 pool, use canonical `smiles` as both
the program ID and `identity_field`. A run must freeze the checkpoint hash,
strength, cap, coverage policy and trace compiler settings before selection.
Those settings must not change silently on resume.

The implementation checks establish that the reference can change selection
probabilities. They do not establish an optimization gain. That requires a
matched task evaluation with the selected policy and budget fixed in advance.

## Run the offline example

Use the Python 3.11 / RDKit 2024.3.5 environment in `requirements/core.txt`.
The example compares two fixed one-edit programs from ethane. Its uniform
baseline is a toy selector, not the PMO or T4 allocator.

```bash
PYTHONPATH=src python examples/reference_guidance.py \
  --mode off --output runs/reference_guidance/off.json

PYTHONPATH=src python examples/reference_guidance.py \
  --mode shadow --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/shadow.json

PYTHONPATH=src python examples/reference_guidance.py \
  --mode active --strength 0.25 --log-weight-cap 1 \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/active.json

# The same fixed programs through the actual controllers' cold-start selectors.
PYTHONPATH=src python examples/reference_guidance.py \
  --selector pmo --mode active --strength 0.25 \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/pmo-active.json

PYTHONPATH=src python examples/reference_guidance.py \
  --selector t4 --mode active --strength 0.25 \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/t4-active.json
```

The controller examples use two fixed programs, one slot and no reserved floors.
They do not run a PMO or docking campaign. The example strength is illustrative,
not a selected benchmark setting. Outputs
include exact traces and hashes, checkpoint identity, configuration, probabilities,
software versions, source hashes, and the zero-oracle accounting. Existing output
files are never overwritten. Off mode needs no checkpoint. Shadow and active
require the independently verified fragment asset:

```
SHA-256: c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8
Ring catalog: 639ff6078c32d43c
```

Do not load untrusted pickle checkpoints. The shared checkpoint is tracked with
Git LFS and its bytes are verified by the asset manifest.

## Library use and checks

Load once with `compose_v4.model.reference_checkpoint.load_frozen_reference`.
Wrap it with `FrozenProgramReference`, adapt the exact candidate records with
`pmo_program_input` or `t4_program_input`, then pass the lazy score call to
`guide_panel`. The caller samples from the returned probabilities using its own
existing RNG and locks the selected candidates before any objective call.

```python
panel = guide_panel(candidate_ids, controller_probabilities,
                    config=GuidanceConfig(mode="active", strength=0.25),
                    score=lambda: reference.score(programs))
selected = rng.choice(len(candidate_ids), p=panel.probabilities)
receipt = panel.receipt()
```

Run the focused tests without a checkpoint, or set `COMPOSE_REFERENCE_CHECKPOINT`
to include the real-asset checks. An explicitly supplied but missing, altered, or
incompatible checkpoint fails. Only an unconfigured external asset is skipped.

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_reference_guidance.py \
  tests/test_program_reference.py tests/test_reference_guidance_example.py \
  tests/test_reference_selection.py tests/test_t4_reference_traces.py
```

For macOS installations requiring the documented OpenMP guard, also set
`KMP_DUPLICATE_LIB_OK=TRUE`. No test calls an oracle or starts a remote job.

## Generated T4 panels

Use `examples/t4_reference_panel.py` to check the complete path from proposal
generation to reference-aware selection, without docking. Supply a lead, a
similarity threshold and explicit generation settings. For example:

```bash
python examples/t4_reference_panel.py \
  --lead 'COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34' --delta 0.6 \
  --draws 8 --horizon 3 --proposal-seed 11 --selection-seed 17 \
  --mode active --strength 0.25 --coverage preserve_mass \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/t4-panel.json
```

The output contains every eligible candidate, exact compiled program, scoring
failure, selection probability and random state. The parent score is a synthetic
zero because no docking is performed. The selector is in cold-start mode with
one slot and no reserved allocation floors. An empty panel is recorded as `empty_panel`.
These settings check implementation behavior, not optimization performance.

To use the route-template proposal lane, supply a separate program template
prior and its SHA-256. This input constructs molecular programs. The neural
reference checkpoint remains fixed. The loader verifies both the file identity
and the internal payload hash. The PARP1 prior in this example is bundled with
the T4 task files:

```bash
python examples/t4_reference_panel.py \
  --lead 'COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34' --delta 0.6 \
  --proposal-lane route_complete_region \
  --program-template-prior experiments/t4/assets/parp1_checkpoint.json \
  --program-template-prior-sha256 73f25dcddb71a3645b7bd45252e37814f837c6f5b1b8d59f5f4e08940439d311 \
  --route-pool-size 8 --route-realizations 8 --route-beam-width 8 \
  --route-expansion-width 8 --route-bindings 2 --compiler-expansions 64 \
  --mode active --strength 0.25 --coverage preserve_mass \
  --checkpoint local_assets/fragments/r_theta_nll.pt \
  --output runs/reference_guidance/t4-route-panel.json
```

The output uses `compose.t4_reference_panel` schema version 3 and records the
program template prior's identity, construction settings and candidate
accounting. Construction is deterministic for these inputs.
The selection seed controls the subsequent draw. `--draws` and `--horizon` apply
only to the shallow lane and are rejected for route proposals. These bounded work
settings are for an offline integration check, not a benchmark configuration.

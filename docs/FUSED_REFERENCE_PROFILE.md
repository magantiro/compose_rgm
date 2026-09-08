# Frozen-model fused reference audit

Initial prospective decision, 2026-09-07. The user approved checking relevant verification
failures and then one capped learned-model audit. This authorizes neither a
guided comparison nor docking, training, or promotion of the option into T4.

## Question and relation to local/global control

COMPOSE remains an executable stochastic molecular rewrite process. This audit
asks whether its frozen learned marked law, conditioned by the existing opt-in
stateful fused option, has an executable five-step path on PARP1 seed0 at an
acceptable measured proposal cost. The primary output is one exact-state
primitive trajectory with support, validity, probability, and cost receipts.

The complete controller still factorizes into region selection Q(M), applicable
option selection Q(o | M), and executable primitive control. Its conserved
allocation unit is (parent, region, option), not each intermediate. Generic
remains available; the default option registry and T4 execution are unchanged.
The new fused option is restricted to eligible edges inside M. Large intended
scope does not imply large realized change. Record both separately.

This diagnostic fixes the option and selects the smallest necessarily applicable
region by a frozen source-only rule, before looking at learned probabilities.
It is NOT a draw from Q(M) or Q(o), a test of the global/local allocation policy,
or a docking-score comparison. It samples the existing reference row exactly,
with zero lookahead and no terminal objective. A later guided comparison must
retain this same-generator reference and match total executor budgets. No claim
of guide superiority or completion probability follows from one attempt.

Support is the opt-in channel documented in FUSED_OPTION_INTEGRATION.md: four
single-neighbor carbon births and one closure, adding a six-carbon aromatic
cycle across an existing eligible carbon ring edge. The production 15-class
vocabulary, 40-active-atom ceiling, charge policy, executor, and inherited 300/20
proposal support cap are unchanged. This narrow option does not establish
coverage of heterocycles, other ring sizes, pendant rings, spiro or bridged
systems. Inspected winner molecules and docking outcomes are not inputs.

## Frozen inputs, census, and stop conditions

The authoritative self-hashed contract is
`configs/fused_reference_profile_contract.json`, semantic SHA-256
`fefbcbdcbb1bbded2a44506c14aab57244ab0b7b5c12b3e21da48e6273c221f9`
(revision 2; revision-1 evidence and its contract are retained in attempt_1).
It binds exact source-manifest, run-path, and checkpoint hashes; source index 0;
seed 1000; time 0.5; the deterministic applicability-only region selection; and
the unchanged macro conditioning parameters. This is an already inspected
development source, not a sealed test or molecular-performance panel.

One task, one CPU, one thread, 6 GiB, no GPU, no retries, at most five sampled
edits and 2,000 ALL public executor calls, including calls inside law enumeration.
The function stops after 900 seconds. This is a limit, not an ETA; no current
five-row runtime estimate is available. Configured function compute is capped
at $0.023778 using the price assumptions in the contract, excluding build,
startup, storage, and platform premiums. There are zero docking calls.

Persist exact raw laws, augmented rows, sampled states and RNG after every step.
Executor exhaustion publishes an abstention and every entered-call receipt, never
a partial probability row. Unexpected failures publish a traceback. A platform
timeout with no result is incomplete evidence. Twenty-second heartbeats expose
the current phase and completed units. A partial existing run cannot be retried
automatically. Review support and cost before authorizing any further run.

## Inventory and verification relevance

Read-only Modal inventory confirmed the bound RUN_PATHS.json, the 83.7 MiB
R_THETA_CHECKPOINT.pt, and the existing materialized scorer receipt and tensors.
CLI volume paths start at `/editing_v2/...`; `/artifacts` is the container mount
prefix, not a volume subdirectory. Every bound input is verified again remotely.

The old failed continuation profile has 28 completed raw laws and rows. Its
program, region, and augmented states differ. It does not supply a complete
matching runtime/software receipt for certified cross-run reuse of the possibly
overlapping root law. Rather than blindly trusting that overlap or constructing
a new cross-run cache framework, this bounded path measures at most five rows.
The old evidence is retained; its exhaustive tree is not replayed.

The repository-wide suite at implementation commit 1be2706772b9bc576fa3df84592fa8f7770abe20
is NON-GREEN: 4,423 passed, 47 failed, 58 errors, two skips, one xfail. Its complete
receipt and comparison to the preceding run are in
`diagnostics/fused_option_audit/verification.json`. There were zero new failing
test identities; matching identities alone do not prove matching causes or waive
their relevance. All 19 new fused-option/audit tests passed in that run.

Local catalog failures used RDKit 2026.03.6, PyTorch 2.14.0, NumPy 2.5.3, and
Python 3.12.9 on arm64. The existing production image pins RDKit 2024.3.5,
PyTorch 2.4.0, NumPy 1.26.4, and Python 3.11. Environment dependence is a
hypothesis, not an established explanation. Model, catalog, executor, and runtime
source files are unchanged relative to the earlier successfully initialized
production profile. Existing Modal application functions also remain unchanged;
only the independent audit function is added.

The bounded remote task must pass the unchanged production input, materialized
state, source admission, catalog, and strict checkpoint-loading gates before
enumerating any model law. It writes a separate runtime gate receipt. Failure
stops this attempt; no gate hash or threshold will be modified to make it pass.
Thus the local full suite is not represented as passing, and this work is not
declared a completed repository milestone. Focused tests cover the new driver,
its dependencies, slot-safe experiment surface, failure boundary, and both
launcher routes without network access. No repository-wide rerun is required
merely for this isolated profile and operational wrapper; scientific runtime
dependencies remain frozen.

## Launch and evidence handoff

Use a fresh clean committed worktree to avoid unrelated or ignored package files
entering the image. Bind its exact revision and physical image manifest. Run:

```sh
python3 tools/preflight.py --strict
modal deploy modal_apps/genmol_t4_opt_app.py
python3 tools/t4_launch.py --fused-reference-profile
```

This spawns only `fused_reference_profile` into the deployed app. It does not
launch `t4_population_cell`. The receipt names the content-addressed volume
directory and call ID. Monitor that directory's progress, heartbeat, runtime
gate, and final result or failure receipt. Publish the audited result and its
interpretation separately from the implementation. Do not push without approval.

## Authorized boundary repair, revision 2

The user approved fixing the numerical interface and proceeding with one new
capped test. Attempt 1 passed all frozen runtime gates, then failed the audit's
stricter duplicate mass check before any option row or sampled edit. It has no
complete law to reuse; all its receipts remain unchanged.

Read the normalization tolerance directly from the unchanged production
evaluator's signature and bind it in the contract. Do not renormalize raw
weights in the audit. Reject nonfinite, negative, misaligned, or out-of-contract
mass and save diagnostic values separately from valid probability rows. Persist
runtime environment observations before initialization and on failure as well
as success. All source, seed, option, model, controller, chemistry-support,
executor-budget, and timeout choices remain unchanged.

Focused regression and dependency checks are the repair gate. Do not repeat
unaffected training, molecular preparation, or the historical non-green full
suite for this isolated reporting/interface repair. Use the existing deployment
and spawn procedure, with a new immutable run ID. This does not authorize a
docking campaign or alteration of scientific acceptance thresholds.

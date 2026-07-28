# Deviation Register — Program Coherence Audit (HEAD 4204c9d)

Accumulated from the read-only mapping. Severity: CRITICAL (different scientific process) · HIGH
(train/eval/sample incompatibility) · MEDIUM (reproducibility/stale-artifact) · LOW (naming/maintenance).

| ID | Sev | Layer | Canonical behavior | Divergent behavior | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|--|
| D1 | MEDIUM | operator naming | one op = one name per namespace | `cycle_insert`/`cycle_attach` are BOTH live model-slots AND dead legacy executor-rule names | No (legacy execs never invoked under enable_cycle_ops) | naming hazard; public exposure prohibited | naming consolidation (separate proposal) | OPEN (intentional, documented) |
| D2 | LOW-MED | metrics/logging | metric name = public op name | `teacher_examples_cycle_insert` logs internal slot, not `cycle_close`/`bond_insert` | Yes (eval logs) | log-readability trap | alias metric names to public (fix-phase) | OPEN |
| D3 | LOW | operator naming | slot name matches semantics | slot-6 `cycle_attach` legacy=add, now holds cycle_open=remove | No | nominal only | doc note | OPEN (nominal) |
| D4 | LOW | teacher scoring | one action type per slot | slots 5/6 route both Bond* and legacy Cycle* actions | No (unreachable in RingCore) | latent only | assert in fix-phase | OPEN (unreachable) |
| D5 | LOW | code hygiene | import MARK_RULE_NAMES | griddd_conditional.py:682 hard-codes a copy | Yes (that script) | DRY drift if reordered | import the constant (fix-phase) | OPEN |

## From §4/§5 (model builders + capabilities)
| ID | Sev | Layer | Divergence | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|
| H1 | HIGH | model construction | no canonical model factory — trainer (`train_tracelet_cnof_gate.py:3005`) + loader (`evaluate_tracelet_rollouts.py:180`) duplicate the ~15-arg constructor; consistent TODAY but hand-maintained | Yes | a new/renamed flag on one side silently mis-loads a checkpoint | one `build_model_from_metadata()` factory both call | OPEN (structural, not live) |
| H2 | HIGH | reward-FT sampler | `trajectory_log_prob` (`griddd_reward_finetune.py:53`) builds batches with NO ring_catalog + NO capability flags → editing/ring teachers score −inf | Yes but EXPERIMENTAL lane (reward-FT, weak/unshipped; tested on toy acyclic only) | any ring/editing mark in a reward-FT trajectory → inf/nan | pass model.operator_capabilities + ring_catalog | OPEN (experimental, not RingCore-production) |
| M1 | MEDIUM | capabilities | `model.operator_capabilities` consumed by NO production batch-builder; gate/loader/train-loader each rebuild flags | Yes | the single-source object can't enforce the contract it documents | route production through the object (fix-phase) | OPEN |
| M2 | MEDIUM | capabilities | gate's `_eval_capabilities` is a hand-built twin from `args` (model not yet built) | Yes | drifts from `model.operator_capabilities` if coupling changes; unchecked | assert equality once model exists | OPEN (equal today) |
| M3 | MEDIUM | eval batch | `sample_factorized_mark_batch(capabilities=None)` defaults to de-novo enumeration (editing off) | Yes (only gate calls it, with caps) | a forgetful caller builds a de-novo batch → invariant trips | make capabilities required / no unsafe default | OPEN (caught by invariant) |
| L1 | LOW | fingerprint | `OperatorCapabilities` omits `enable_cycle_ops` + `enable_heteroatom_scan`; RingCore capability objects (`program_contract.py`, `build_ring_core_v1_manifest.py`) omit cycle-ops — the defining RingCore feature | Yes | capability FINGERPRINT under-identifies; eval-cache key blind to cycle-ops (mitigated: RingCore uses fresh zero-mixture validation, no cache) | add cycle_ops/heteroatom to the contract fingerprint | OPEN (my contract has enable_cycle_ops as a top-level field, but the capability sub-fingerprint is blind) |

## From §8/§9 (executor + successor + probabilistic object) — §8 CONFIRMED, §9 CONFIRMED
| ID | Sev | Layer | Divergence | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|
| E1 | LOW | executor default | generic trace/compile utils (`trace.py:57` execute_trace, `compiler.py:235,298`, `trace.py:255` inverse_step) DEFAULT to `default_rewrite_system()` (NO connectivity constraint) when `system=` omitted | latent (all production callers override with de_novo_rewrite_system) | a future caller forgetting `system=` could commit a disconnected state | make de_novo the default / require system= | OPEN (footgun, not live) |
| E2 | LOW | successor identity | dual graft grouping keys: colored-tree key (`:1568`, de-novo) vs `canonical_state_key` (cyclic, `:1289`) — both grouping-only, never committed | Yes | two mechanisms for "same successor"; equivalence documented not runtime-asserted | assert equivalence in a test | OPEN (low, mis-group at worst) |
| E3 | LOW | canonicalization | `_canon` inline re-impl in the SMC controller (`griddd_value_guided_smc_controller.py:77`) instead of `canonical_state_key` | Yes | functionally identical for valid states | import canonical_state_key | OPEN |

**§8 result: ONE executor (`RewriteSystem.apply` kernel.py:51) + ONE canonical successor (`canonical_state_key` kernel.py:95) confirmed across compile/corruption/replay/sampling/rollout. No alternative committed-state executor, no silent RDKit repair.**
**§9 result: probabilistic levels (head→complete-mark→embedded-chain→CTMC-rate) cleanly separated + converted; hazard retained in the Bregman/GM loss; editing=embedded chain, de-novo=rates; graft logsumexp realizes the canonical molecular kernel. No level mixing.**

## From §3 (entrypoint inventory)
| ID | Sev | Layer | Divergence | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|
| DEV-1 | LOW | gate backend guard | edit flags don't require `--training-backend factorized_marks`; gate default is legacy `exact_fiber` (:927) → would build legacy `TraceletRateModel` w/o enable_* | latent (recipes all set factorized_marks) | edit flags silently dropped if backend unset | add symmetric guard | OPEN |
| DEV-2 | MEDIUM | rollout eval reference | `evaluate_rollout_shards.py:457` scores V/U/N+FCD vs a CNOF-neutral reference | Yes, but a de-novo evaluator (RingCore uses ring_core_rollout_panel instead) | distribution mismatch on broad-organic ckpt (understates novelty, inflates FCD) | broad-organic reference or scope-guard the shard app | OPEN (not RingCore path) |
| DEV-3 | LOW | scope baseline | only `max_atoms=48` in tree (`benchmark_lead_scope_coverage.py:39`) — intentional old-CNOF baseline | Yes (that benchmark) | none (not a live config) | doc note | OPEN (intentional) |
| DEV-4 | LOW | trainer multiplex | gate multiplexes 3 model builders + corpus-marginal fit; only factorized is production | Yes | maintainability; legacy reachable in same 3976-line file | split legacy branches (separate proposal) | OPEN |
| DEV-5 | LOW | legacy scripts | 7 "pancake quotient" scripts runnable | not from production recipe | reader confusion re: live backbone | legacy/ quarantine or delete | OPEN |

## From §14 (legacy + parallel paths) — headline: no reachable-and-silently-wrong legacy path
| ID | Sev | Layer | Divergence | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|
| G1 | MEDIUM (priority) | gate guard | no assertion `cycle_op_mix ⇒ disable_ring_grow_macro` (gate:1538 guards only reverse) | not from canonical (recipe passes flag) | a launch omitting the flag runs grow+cycle → different model | one-line gate assertion | OPEN (fix candidate) |
| G2 | MEDIUM | rollout eval | `evaluate_tracelet_rollouts:373` CNOF-hardcoded reference, main load :326 no scope-hash guard | not from canonical (—train-only skips; crashes on zero-mixture) | standalone run on broad ckpt scores vs CNOF ref | scope-hash guard + organic branch | OPEN |
| G3 | LOW | dead code | `build_corrupted_prior_dataset.py` CNOF corruption builder, imported nowhere | No | none | delete/quarantine | OPEN |
| G4 | LOW | footgun | `make_edit_pair` CNOF vocab default | No (callers override) | trap for new callers | make required | OPEN |
| G5 | LOW | doc | freeze-manifest recipe string omits `--scaled-manifest`/`--max-atoms 40` | No (Modal wires both) | copy-paste footgun | fix the doc string | OPEN |

## From the bounded A100 run failure (device handling)
| ID | Sev | Layer | Divergence | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|
| DEV-CUDA | HIGH | warm-start device | `_cksum` (gate:266) called `.numpy()` on a CUDA tensor without `.cpu()` — worked on the CPU dry-launch (device=cpu), threw on the A100 (device=cuda) BEFORE any training step | Yes (every A100 warm-start with row-table recording) | bounded A100 run crashed at semantic-transfer | add `.cpu()` before `.numpy()` | FIXED |

**LESSON: a CPU-only dry-launch structurally cannot validate the CUDA path. Device-specific bugs (`.numpy()`
on cuda, `.item()`/host-sync differences, autocast/bf16 numerics) require a minimal GPU smoke — the CPU
dry-launch proves orchestration + math, not device portability.**

## From §15/§10 (paper-code + RingCore definition) — papers careful; NO HIGH-risk archetypes present
| ID | Sev | Layer | Divergence | Reachable? | Consequence | Required fix | Status |
|--|--|--|--|--|--|--|--|
| P1 | MEDIUM | paper data claim | papers say base = "GuacaMol C/N/O/F **with charged N⁺/O⁻**"; loader (`cnof.py:199`) is CNOF **neutral-only** (base=50k CNOF-neutral) | paper text | data-provenance overstatement (grammar vs trained data) | say "C/N/O/F, neutral" | OPEN (paper) |
| P2 | MEDIUM | paper support claim | "finite catalog does not define legality / all ring sizes possible" — true for the UNRELEASED compositional core, NOT the released catalog-bounded base B | paper text | support-scope overstatement (hedged in limitations) | scope the claim to RingCore | OPEN (paper) |
| P3 | LOW | paper terminology | "SMC over the rewrite CTMC" — editing controllers run a fixed-step EMBEDDED jump chain (SYSTEM_CONTRACT:271 already flags this) | paper text | terminology precision | say "embedded jump chain over a fixed edit budget" | OPEN (paper) |
| P4 | LOW | context drift | learnings.md/CLAUDE.md cite stale max_atoms=48 numbers (hash e59fb098, 490466/98.1%) — papers correctly say 40 | context files only | future paper number from 48-scope would be wrong | update context to 40/3721d698/466483 | OPEN |
| R1 | MEDIUM | RingCore catalog role | `ring_system_delete` (`ring_delete`, 256 targets) IS catalog-bounded via `enumerate_clean_ring_system_deletes(state, catalog)` — a LIVE catalog dependency. "macros disabled / catalog acceleration-only" is only PARTIALLY realized | Yes (production) | "no catalog / macros off" understates a live dependency; ring DELETION is catalog-bounded (5 seed topologies) while ring ADDITION is compositional | document precisely: generation=compositional, deletion=catalog-bounded | OPEN (doc precision) |

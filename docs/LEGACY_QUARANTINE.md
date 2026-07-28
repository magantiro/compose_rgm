# Legacy Quarantine — Program Coherence Audit (HEAD 4204c9d)

**Headline: no legacy path is reachable-and-silently-wrong from the canonical RingCore-V1 recipe.** Every
legacy/parallel object is flag-gated off, dead (imported nowhere), a version-gated migration loader, an
explicit ablation, or fails loudly. Confirmed: NO `edit_transport`/`edit_flow` path (native-GM framing held),
NO second `make_edit_pair`, NO carbon-tree editing initialization, `sample_factorized_mark_batch` updated
in-place (not a parallel legacy).

| Legacy object | file:line | Reachable from production? | Disposition |
|--|--|--|--|
| `load_cnof_corpus_split` (train gate else-branch) | gate:1682 | No (`--organic-vocabulary` → :1666) | retain-as-ablation |
| `load_cnof_corpus_split` (rollout eval) | evaluate_tracelet_rollouts:373; evaluate_rollout_shards:457 | No (—train-only skips in-app eval; crashes on zero-mixture ckpt) | quarantine + add scope-hash guard |
| `build_corrupted_prior_dataset.py` | whole file | No (imported nowhere) | delete/quarantine |
| `make_edit_pair(vocabulary=CNOF default)` | source_corruption:291 | No wrong-default reach (all callers override) | make vocabulary required |
| **legacy grow macro `enable_ring_grow_macro` default True** | model:1662; dead-mask :2530 | **Disabled in canonical** (`--disable-ring-grow-macro`); NO gate enforcement `cycle_op⇒disable` | retain for de-novo B; **ADD gate assertion** |
| 5-seed catalog / 4096 templates | gate:1859; ring_system_fiber | **YES — PRODUCTION** (defines head shapes; distinct from the disabled grow macro) | retain (production) |
| strict warm-start `--initialize-from-source-checkpoint` | train_tracelet_gm:2387; gate:3335 | No (canonical uses compatible; crashes loudly on 4→15) | retain-as-migration-loader |
| eval-batches v1 + `--allow-legacy-evaluation-cache` | gate:106,540 | No (v2 current) | retain-as-migration-loader |
| `rate_factorization` superposed/quotient_energy | gate:955 | Only if explicitly selected (canonical=hierarchical) | retain-as-ablation |
| pancake analytic-zero / quotient-distillation | canonical_successor_distillation, griddd_conditional, qualify_*_pancake_* | No (not imported by training gate) | retain (separate conditional lineage) |
| `TraceletRateModel`/`PriorTiltedTraceletRateModel` | gate:3051,3061 | No (RingCore requires factorized_marks) | retain-as-ablation |
| max_atoms=48 (benchmark_lead_scope_coverage:39 + 3 diagnostics) | — | No (all production builders max_atoms=40; would crash if mixed) | retain (diagnostics) |
| `legacy_v0` electronic-role zeroing | evaluate_tracelet_rollouts:212 | version-gated migration loader | retain |

## Key deviation (elevated)
**G1 (MEDIUM, priority fix) — no gate assertion `--cycle-op-mix ⇒ --disable-ring-grow-macro`.** gate:1538
guards only the reverse. A `--cycle-op-mix` launch omitting `--disable-ring-grow-macro` runs the legacy
whole-ring grow macro IN PARALLEL with compositional `cycle_close` → a DIFFERENT model (two ring-addition
mechanisms). NOT reachable from the canonical recipe (which passes the flag — the running preflight is correct),
and `ring_core_identity.py:147` detects the drift post-hoc, but there is no PREVENTION. Fix: assert the
implication in the gate (a one-line guard, fix-phase).

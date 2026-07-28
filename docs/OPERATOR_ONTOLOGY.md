# Operator Ontology — RingCore-V1 (audit §6, HEAD 4204c9d)

**Verdict: coherent.** No name refers to different operations within a single active namespace; no two names
ambiguously map to one operation within the model. 5 deviations, all LOW/MEDIUM (naming homonyms + DRY risk).

## The two namespaces (the whole key)
- **Executor rule registry** — `de_novo_rewrite_system()` / `default_rewrite_system()` (`rewrite/kernel.py`): 16
  rules keyed by `rule_name`, dispatched by `RewriteSystem.apply(state, rule_name, action)`.
- **Model dense-family slots** — `MARK_RULE_NAMES` (`model/factorized_tracelet_rate_model.py:96`): exactly 10
  slots, the `family_head` classification axis. A teacher/sampled mark carries an **executor** `rule_name`; the
  model maps it to a **family slot** for scoring. Identical names for most families; the exception is the
  cycle-op repurposing.

## The 10 model families (public · slot(idx) · teacher rule · builder · executor · enable flag · RingCore)
| # | Public | Slot (idx) | Teacher rule | Candidate builder | Executor | enable flag | RingCore |
|--|--|--|--|--|--|--|--|
|1|atom_insert|atom_insert(0)|atom_insert|`_factorized_candidates`|atom_insert|— |ENABLED|
|2|atom_delete|atom_delete(1)|atom_delete|`_factorized_candidates`|atom_delete|— |ENABLED|
|3|atom_restate (bioisostere)|atom_restate(2)|atom_restate|`_factorized_candidates`|atom_restate|`enable_heteroatom_scan`|ENABLED|
|4|bond_reorder|bond_reorder(3)|bond_reorder|`_factorized_candidates`|bond_reorder|— |ENABLED|
|5|**graft** (cyclic)|bond_reroute(4)|bond_reroute|`enumerate_pendant_graft_actions`|bond_reroute|`enable_cyclic_graft`|ENABLED|
|6|**cycle_close**|cycle_insert(5)|**bond_insert**|`build_cycle_op_records`|**bond_insert**|`enable_cycle_ops`|ENABLED (SUPPORTED_BY_CORE)|
|7|**cycle_open**|cycle_attach(6)|**bond_delete**|`build_cycle_op_records`|**bond_delete**|`enable_cycle_ops`|ENABLED (SUPPORTED_BY_CORE)|
|8|ring_grow_macro|ring_system_grow(7)|ring_system_grow|(grow decoder)|ring_system_grow|`enable_ring_grow_macro` (dflt True)|**DISABLED** (masked dead, params retained)|
|9|ring_delete (clean)|ring_system_delete(8)|ring_system_delete|`enumerate_clean_ring_system_deletes`|ring_system_delete|`enable_ring_opening`|ENABLED|
|10|ring_aromaticity_restate|ring_system_restate(9)|ring_system_restate|`enumerate_ring_system_restate_actions`|ring_system_restate|`enable_ring_restates`|ENABLED|

**Executor-only rules (NOT model families):** `bond_insert`/`bond_delete` (`kernel.py:110-111`) are executor
rules + `factorized_fiber` candidate families but NOT dense slots; `source_corruption.FORBIDDEN_FAMILIES`
forbids them as micro-teachers. They go live only as the executors for cycle_close/cycle_open. `cycle_delete`,
`cycle_detach`, `ring_ear_insert`, `ring_ear_delete` are legacy null-prior ops, dead. `ring_ear_insert` /
`ring_spiro_attach` are the public `REQUIRES_STRUCTURED_P2_MODE` deferrals.

## Cycle-op resolution
`cycle_insert`/`cycle_attach` are **internal engineering slot names** (repurposed dead slots 5/6);
`cycle_close`/`cycle_open` are the **public/paper names** (explicit: `freeze_ring_core_v1.py:52,183` — the
internal names MUST NOT appear in paper/checkpoint metadata). `_CYCLE_OP_EXECUTOR_TO_FAMILY =
{"bond_insert":"cycle_insert","bond_delete":"cycle_attach"}` is applied at **every** teacher→slot site
(verified: the only two `MARK_RULE_TO_INDEX[teacher]` lookups — rate-model `:3892` + conditional `:894` — both
alias; every other use is a hard-coded literal). **The aliasing is complete and internally consistent.**

## Deviations (fed to DEVIATION_REGISTER)
- **D1 (MEDIUM):** cross-namespace homonym — `cycle_insert`/`cycle_attach` are simultaneously live model-slot
  names and dead legacy executor-rule names. Deliberate dead-slot repurposing (keeps `family_head` width 10 +
  warm-start strict-safe); safe at runtime; the freeze registry itself flags it. Naming hazard, not a bug.
- **D2 (LOW–MEDIUM):** training metric names expose the internal slot (`teacher_examples_cycle_insert`) not the
  public name (`cycle_close`) and not the teacher rule (`bond_insert`) — a log-readability trap. Within the
  letter of the exposure policy (logs ≠ paper/checkpoint metadata).
- **D3 (LOW):** slot-6 semantic-valence inversion — legacy `cycle_attach` *adds*, repurposed slot now holds
  `cycle_open` which *removes*. Nominal (legacy op dead).
- **D4 (LOW, unreachable):** latent action-type overload at slots 5/6 (`BondInsert`+`CycleInsert` both →
  `logits["cycle_insert"]`). Unreachable in RingCore (no de-novo tracelet teachers mixed with `enable_cycle_ops`).
- **D5 (LOW):** `griddd_conditional.py:682` hard-codes a copy of the `MARK_RULE_NAMES` 10-tuple instead of
  importing it — silent DRY drift if the constant is reordered.

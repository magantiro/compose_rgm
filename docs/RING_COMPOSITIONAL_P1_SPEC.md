# P1 implementation spec — compositional cycle_close / cycle_open (coupled core)

Authorized ring-support redesign. Support = compositional `cycle_close`/`cycle_open` (defines reachability);
`ring_system_*` = macro cache. This spec is the **coupled** part of P1 (scoring → GM-loss → quotient →
sampler), to be implemented as one consistent, well-tested unit. Everything is gated by `enable_cycle_ops`
(default False) → **B / current B-edit byte-identical when off** (the slot overrides never activate).

## Done (safe, landed)
- Increment 1 `d842069`: `enable_cycle_ops` flag + `cycle_close_head` (Linear h→3, pair × bond order 1/2/3)
  + `cycle_open_head` (Linear h→1, per non-bridge edge). Added only when on → byte-identical off; 101 tests.
- Design `0fd1584`: OVERRIDE dead slots 5 (`cycle_insert`) / 6 (`cycle_attach`) when `enable_cycle_ops`
  (keep `MARK_RULE_NAMES`=10, keep dead params → byte-identical; no family_head widening / no rip-out).

## Enumeration (no new feature mask needed)
- `cycle_open` targets = `batch.cycle_edge_mask` (already computed: all edges − `nx.bridges`, §6 bridge-based,
  not SSSR). Per **edge** (a<b, bonded, non-bridge).
- `cycle_close` targets = **nonbonded real pairs** computed at scoring time: `real_pair & (batch.bonds==0) &
  upper_triangle`. Same-component is automatic for connected states. Per **pair × bond order** (1/2/3).
  *(Every teacher target is a real ring bond, which in the opened precursor is a nonbonded same-component
  pair → always inside this mask; no −inf loss.)*

## The coupled changes (all in `forward_mark_batch` + its helpers, gated by `enable_cycle_ops`)
1. **Scoring override** (~3667–3687): when `enable_cycle_ops`, replace the slot-5/6 entries:
   - `logits["cycle_insert"] = cycle_close_head(pair)` shape `(B, n, n, 3)`; `masks["cycle_insert"] =`
     closeable-pair mask broadcast over the 3 orders. **Mirror `bond_reorder`'s pair layout exactly.**
   - `logits["cycle_attach"] = cycle_open_head(pair).squeeze(-1)` shape `(B, n, n)`; `masks["cycle_attach"] =
     batch.cycle_edge_mask`. Mirror a per-pair single-score family.
   - Keep the template `cycle_logits/cycle_mask` path for the `else` (off) branch unchanged.
2. **`action_log_z` / masked-logsumexp**: the per-family normalizer already iterates the dict by family; it
   must accept the pair shape for slots 5/6 when on (it already does for `bond_reorder`). Verify the
   `_selected_mark_log_probability` family accessor (~3888) returns the pair `logits/masks` for slots 5/6.
3. **Teacher-scoring dispatch** (~3293): add branches — a teacher `bond_insert` mark scores against
   `logits["cycle_insert"][pair, order]`; a teacher `bond_delete` against `logits["cycle_attach"][edge]`.
   The teacher record's `rule_name` is `bond_insert`/`bond_delete` (executor), mapped to slot 5/6.
4. **Family→executor-rule map**: sampler + apply use `bond_insert`/`bond_delete` (not `cycle_insert/attach`)
   when the sampled slot-5/6 action is a `BondInsert`/`BondDelete`. Add a small
   `_family_executor_rule(slot, action)` shim used by the sampler, teacher builder, and quotient.
5. **Canonical-successor quotient** (~3888): a `cycle_close` on symmetric pairs (e.g. two equivalent closure
   sites) can map multiple marks to one canonical successor — dedup like the other per-pair families;
   `cycle_open` of symmetric edges likewise. Aggregate mark→successor mass before control.
6. **Sampler** (~3160): draw slot 5/6 → sample a pair (+order for close) from the masked head → build the
   `BondInsert`/`BondDelete` action → apply via the mapped executor rule.

## Supervision (P3, separate) — do NOT reuse the whole-ring grow-inverse
Regenerate from real molecules: spanning-tree decomposition (verified 400/400) yields, per non-tree ring
bond, a `cycle_open` teacher (trim) and a `cycle_close` teacher (grow-inverse); multiple valid spanning trees
per molecule avoid one arbitrary basis; scaffold-split. Balance by ring-topology stratum (cap benzene-like).

## Tests (P1 acceptance)
- byte-identical off (state_dict + a fixed forward) — regression on every commit.
- enumeration ⊆ legal: every enumerated `cycle_close`/`cycle_open` executes; cycle-rank ±1; ≤40 atoms.
- teacher-in-mask: every real-ring-bond teacher scores FINITE (no −inf); GM Poisson–Bregman finite.
- exact executor match: sampled mark's successor == quotient successor.
- quotient: Σ mark mass over a canonical successor == the family→successor rate (no over/under-count).
- round-trip: open→close and close→open recover exactly.
- tiny-overfit: cycle_close/cycle_open rows receive finite nonzero gradient + a nonzero update.

## Then P2–P6
P2 ear/spiro macros; P4 K≈64 macro cache with compositional fallback + mass aggregation on shared successors;
P5 two-level ring-mass calibration; P6 full re-audit (fuzz, inverse, canonical-successor, tiny-overfit) +
regenerate manifest/caches + warm-start/zero-mixture/throughput → `GO_FULL_RING_SUPPORT_PREFLIGHT`.

**Guardrails:** no Stage-7 optimizer updates, no final cache compilation, no manifest/catalog regeneration
until implemented + re-audited.

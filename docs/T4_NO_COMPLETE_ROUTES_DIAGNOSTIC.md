# T4 complete-route and dynamic-synthesis diagnostic

## Identity and scientific question

- **Problem:** determine whether COMPOSE can recover the frozen T4 controller's
  useful search behavior without initializing from stored complete
  winner-route programs.
- **Primary output:** complete score-versus-call curves for one 69-program arm
  and one route-free dynamic-synthesis arm on each of 5HT1B seed 0, BRAF seed 1
  and JAK2 seed 1, paired with the already completed full-146 replicate-0
  result.
- **Claim under test:** runtime composition of generic parameterized modules
  can recover useful docking improvement supplied by the 77 stored complete
  routes. The 69-only arm diagnoses how much survives mechanical removal alone.
- **Setting:** T4 delta 0.4, strict endpoint similarity, QED and synthetic
  accessibility gates, the production QuickVina protocol, one matched search
  and docking seed per cell and at most 1,000 newly docked candidates per cell.
- **Primary comparison:** full shared 146-program controller versus the exact
  69-program ablation versus dynamic route synthesis with no initial route
  archive.
- **Support:** connected exact 48-slot molecular states with at most 40 active
  heavy atoms, the existing supported atom and bond vocabulary and the
  production exact executor. Stereochemical and formal-charge editing remain
  outside the editing support.

The retained 69 programs are still development artifacts derived from T4
material. Arm B therefore only isolates the additional contribution of the 77
stored complete routes. Arm C loads no row from the 146-entry library. Its
generic module grammar was informed by development experience, so this remains
benchmark-informed method development rather than a held-out T4 result.

## Frozen pairing

The full-146 arm is not rerun. Its completed replicate-0 results in the frozen
official run are the authoritative comparison. Both new arms copy the
benchmark settings below. Arm C changes only its proposal source as declared in
the next section.

Three source files changed after the full-arm contract was sealed. The derived
contract records both old and current hashes and the exact inactive differences:
an unused zero-valued retrieval default, a rescue-only missing-observation status,
and a local read-only status-tool change. No diagnostic worker activates the
retrieval default or rescue status, and workers do not import the status tool.
This is a declared decision-equivalent compatibility repin, not byte identity.

| Setting | Frozen value |
| --- | --- |
| Cells | `5ht1b_0`, `braf_1`, `jak2_1` |
| Controller seed | 20260913 |
| Docking seed | 1701 |
| Cold-start seed | 20260913 |
| Per-cell ceiling | 1,000 distinct candidate dockings |
| Controller | program-only, score-blind, cache128 |
| Proposal channels | mutation 7/9, recombination 2/9, broad 0 |
| Candidate work | 128 attempts, up to 16 candidates and 45 seconds per batch |
| Program work | at most 32 primitives and eight blocks |
| Stored-program composition | disabled in B; Arm C uses its declared dynamic composition |
| Reference inference | disabled |
| Endpoint gate | similarity > 0.4, QED > 0.6, SA < 4.0 |
| Early stop | the existing competitive plateau rule after at least 500 calls |

The derived library must equal the full library after applying exactly this
predicate:

```text
remove(row) := len(row.program.blocks) == 1
               and row.program.blocks[0].label
                   == "compiled_complete_transformation"
```

No program may be edited, renamed, reordered or reserialized individually to
change its semantics. The lock records the identity of every retained and
removed program.

## Dynamic-only arm

Arm C begins every unit with an empty complete-route archive and never loads a
program row from the 146-entry source file. A bounded sampler selects one to
three modules, binds every module to the exact state produced by the previous
module, compiles the complete composition into the production edit-program
representation, and exact-replays it before endpoint eligibility is checked.
No intermediate molecule is docked or selected by task score.

The frozen generic module vocabulary is:

```text
segment_grow, segment_shrink, segment_replace, substituent_delete,
append_ring, fuse_ring, functionalize, carbonyl_insert,
heteroatom_substitute, bond_reroute, cycle_open,
cycle_close, ring_system_restate
```

Sites, segment lengths from one to eight atoms, C/N/O/F identities, ring size and composition,
electronic pattern, and compatible attachments are chosen at runtime from the
current exact graph. These are parameterized operators, not stored destination
molecules or complete T4 routes. Each complete proposal is limited to 32
primitives, eight blocks, three modules, the existing 40-heavy-atom endpoint
support, 128 attempts, 16 eligible candidates and 45 proposal seconds per
round.

After a dynamically synthesized endpoint receives a genuine docking result,
its exact program may enter that unit's online archive. Later proposals may
mutate, recombine or compose only those self-discovered routes, mixed with
fresh dynamic synthesis under a prospectively sealed policy. The route archive
is empty again at the start of every independent unit.

For sources with at least 36 active heavy atoms, one shared capacity-aware
policy increases proposal mass on contraction and replacement modules and on
two-module compositions. It uses no target identity, winner route or docking
score, preserves nonzero support for every generic module, and does not inspect
intermediate task values. Its near-capacity cold start uses 16 single-deletion
and 32 two-deletion attempts, each rebound to the exact predecessor, before
online archive search begins. This is a shared near-capacity rule, not a
cell-specific route. The 48-attempt deterministic cap is below, not above, the
common 128-attempt and 45-second proposal ceilings.

## Outcome and interpretation

For lower-is-better docking, define the descriptive recovered-improvement
fraction for each cell as

$$
R = \frac{S_0-S_A}{S_0-S_F},
$$

where $S_0$ is the separately measured source-molecule score, $S_F$ is the
completed full-146 replicate-0 best score and $S_A$ is either the 69-only or
dynamic-only best score.
The report must identify the exact receipt and physical docking conditions for
$S_0$. If its formal oracle-domain identity differs from the official search
domain, report $R$ as descriptive rather than protocol-identical. A missing or
zero denominator is an abstention, not a value to patch numerically.

Absolute score, query count, proposal yield, termination, failed calls and the
full curve remain primary evidence. $R$ does not convert one noisy docking run
into a causal or generalization claim.

## Query, compute and recovery contract

- At most 6,000 new docking calls total, 1,000 per new arm/unit.
- At most six concurrent single-CPU workers; no GPU.
- No new full-146 call and no automatic confirmation call.
- No automatic oracle retry. A started call without a durable result is charged
  and blocks automatic resume until a separately authorized recovery audit.
- Use an independent artifact namespace. Never read or write official-run
  checkpoints as optimizer state for the ablation arm.
- Persist candidate locks before docking, one result receipt per charged call,
  30-second heartbeats, end-of-round summaries and full score curves.

The official frozen T4 rescue remains operationally independent. This
diagnostic may reuse completed official results read-only, but it may not alter,
restart or finish an official unit.

## Preflight packaging repair

The first remote structural-preflight invocation stopped before any docking
query because the Modal image contained the mechanically derived 69-program
library but did not contain the unchanged source 146-program library needed to
verify the exact partition. The failure is preserved at
`diagnostics/t4_no_complete_routes/attempt_1/prequery_failure_0001.json`.

The bounded repair adds that hash-bound source file to the remote image solely
for read-only partition verification. It does not expose the source library to
the Dynamic-only optimizer, alter either proposal law, change an endpoint gate,
or authorize an oracle retry. The derived contract records the old commit,
contract and input hashes, the failure-artifact hash and the added remote
material. The same zero-oracle preflight must pass before launch.

## Acceptance

Before launch:

1. Verify that the source library has 146 entries, exactly 77 satisfy the
   removal predicate and the output has 69 entries.
2. Verify semantic equality and order preservation for every retained row.
3. Verify that the three selected units copy their source state, controller
   seed, docking seed, oracle domain, budget, endpoint gates, plateau rule and
   controller configuration from the frozen contract.
4. Generate the first 69-only and dynamic-only batch twice on each exact source
   state, record yield and deterministic identity, and exact-replay every
   returned candidate. Verify that the dynamic arm's initial route archive is
   empty and that no source-library program identity is loaded. Zero yield is a
   valid negative diagnostic, not permission to change the controller.
5. Verify every dynamic proposal contains one to three declared generic modules,
   no intermediate oracle observation and no more than the sealed primitive and
   block limits.
6. Run focused tests, lint/format checks and the launch-boundary verification
   required for touched code. Launch only from a clean committed and pushed
   revision after a remote zero-oracle image/input preflight.

The result is complete when all six new units terminate or fail under the
frozen rules, every charged call is accounted for, the completed full-146
comparators are hash-bound, and a machine-readable three-arm report records
absolute outcomes, curves, recovery fractions or abstentions and limitations.
No wider benchmark or deeper dynamic search follows automatically.

# Scored two-option continuation comparison

Completed answer-informed development diagnostic, not a qualified general controller.
The new search pool improves the observed perindopril-MPO best from
**0.5222329678670935 to 0.5392403377381663**, still below the public example's
**0.8088297765764039**. The improved molecule belongs to the shared planning pool;
it is not evidence that the future arm outperformed the immediate arm.
The public structure was manually transcribed and independently rescored in the
preceding diagnostic. This is not a matched IVG AUC or no-prescreen-run comparison.

Five exact saved parents were evaluated. Both policies saw the same four first
option draws and four continuations per successful first draw. The future policy
backed up best witnessed outcomes. The immediate policy used first endpoints.
The policies changed one of five actual first choices with paired acting randomness.

| Parent | Immediate: fresh mean / best | Future: fresh mean / best |
| --- | ---: | ---: |
| Current best | 0.472662 / 0.498273 | 0.472662 / 0.498273 |
| Original root 0 | 0.385682 / 0.400381 | 0.385682 / 0.400381 |
| Original root 1 | 0.110899 / 0.150849 | 0.110899 / 0.150849 |
| Original root 2 | 0 / 0, abstained | 0 / 0, abstained |
| Original root 3 | 0.207396 / 0.316228 | 0.122999 / 0.243432 |

The aggregate fresh mean is 0.235328 for immediate and 0.218448 for future.
For the changed root, lookahead preferred an endpoint at 0.083461 over one at
0.106198 because its witnessed continuation reached 0.276289. Fresh continuations
did not reproduce that advantage. The observed fresh-mean difference on that
root is -0.0843974. Four fresh draws per selected parent and five parent decisions
do not support a general statistical conclusion about lookahead.

The fresh test measures the selected state's yield under **new reference draws**.
It does not evaluate a closed-loop planner that retains and executes its witnessed
successful continuation. A max-witness backup is not an estimate of mean reference
return. The diagnostic therefore does not justify rejecting future-aware control,
nor does its small pool improvement validate this control recipe. No future-value
network was trained. The known public routes were not injected into these proposals.

Both policies selected the same failed first proposal at root 2. The protocol
retains failed draws as explicit zero-return abstentions, not fabricated molecules.
This is an operationally undesirable policy outcome, not a hidden missing row.
It remains in the denominator. Any changed treatment needs a separately named
policy; these outcomes must not be rewritten to remove it.

## Chemistry and cost

| Phase | Attempted / completed | Unique canonical products | Increased cycle rank | Increased ring systems |
| --- | ---: | ---: | ---: | ---: |
| First options | 20 / 19 | 19 | 3 | 3 |
| Lookahead options | 76 / 72 | 71 | 23 | 15 |
| Fresh options | 20 / 19 | 19 | 2 | 2 |

The lookahead products include completed parameterized fused and pendant ring
programs, cyclization, ring opening, restatement and ordinary edits. Generic remains
active. Every admitted complete program was replay-verified by the unchanged
production executor. Full per-candidate intended release, realized displacement,
primitive count, option identity and topology deltas are in `audit.json`.

The new best is a one-primitive `rebuild` connectivity rearrangement from an
iodinated variant of the old best. Its intended release is 0.15, realized largest
changed fraction 0.05, and cycle-rank and ring-system deltas are both zero. Its
SMILES is:

```text
CCOn1nc(C(=O)O)c(CC(C)C(=O)NC2CC(c3ccc(I)cn3)C(C(C)=O)C(C3CCCC3)C2)c1CO
```

The new best has 40 heavy atoms, cycle rank four and four ring systems. The public
target has 38 heavy atoms, cycle rank four and two ring systems. Equal cycle rank
does not imply the same connectivity or ring organization. Target topology was
computed from its previously verified exact witness endpoint, not a reconstructed
replay state.

There were 29 four-draw workers, at most 20 concurrent. They produced 110 completed
products and 107 unique canonical products in total. The run used 105 new physical
oracle calls (19 first-stage, 68 lookahead, 18 fresh), alongside 292 historical
calls. The separate public-target/parity diagnostic consumed two earlier calls.
No docking or prescreen was performed. The physical ledger and arm-logical attempts
remain separate: each arm had 16 fresh attempts after its abstention, with shared
work for identical choices.

Driver wall time was 584.001 seconds (9m44s), excluding the 94-second deployment.
Summed worker time was 1042.108 seconds, including 823.405 seconds of learned-law
enumeration. There were 215 fresh laws and 21,158 executor applications. Maximum
worker times in the three phases were 54.062, 96.637 and 38.489 seconds. Do not
confuse summed parallel worker time with wall time.

## Decision

Do not scale this sparse max-backup/fresh-reference recipe unchanged. The next
controller work should retain successful continuation paths and use successful
trajectories, including the authorized public-target development routes, to improve
proposal allocation. This requires distinguishing witnessed achievable value from
typical continuation value and conditioning on available edit budget and program
phase. Preserve generic exploration, primitive resolution and the existing
local-to-global region geometry. A public-route-informed recovery is a development
result; independent targets/tasks are still required for generalization claims.

## Provenance and checks

Clean producer commit: `69d6b01fe735`.
Run: `8b29ca08e292179230af1006651c72235b27f890487f5854e9124f0000ec8d1c`.
Modal call: `fc-01M27692MF2JTY05ZQV194YABR`.
Volume prefix: `compose-v4-artifacts/pmo_continuation_choice/<run>`.
Authoritative sealed result SHA-256:
`7097feb4ce5948104bcf105bb83125c56a09916d56d1f49629ede6805e57a753`.

The result binds full configuration, input hashes, seeds, software, hardware,
oracle receipt hashes, all decisions, worker accounting and timestamps. The three
scored-phase files preserve exact saved states and candidate metadata. `audit.json`
binds its inputs and reporting implementation; its new report script was uncommitted
when the deterministic reduction ran. The remote scientific producer was clean.

Fourteen focused policy/adapter/planner tests passed in 2.54 seconds; touched-code
Ruff lint/format and preflight passed. Sealed inputs and the generated report were
inspected. No repository-wide suite was run. This is a bounded development result,
not completion or release qualification of the full controller milestone.

With the pinned chemistry environment:

```sh
python tools/pmo_continuation_report.py diagnostics/pmo_continuation_choice
```

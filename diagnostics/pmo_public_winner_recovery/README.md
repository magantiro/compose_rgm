# Answer-known public target recovery

The locked manual transcription of IVG's first plotted perindopril-MPO example
scores **0.8088297765764039**, matching the plot's rounded 0.81 label. The same
production evaluator reproduces the recorded current best **0.5222329678670935**
exactly. These are individual molecule scores, not top-ten AUC or docking scores.
Manual transcription remains a provenance limitation; the public plot is not
bound to a particular no-prescreen run.

All five locked starts reach the same supported 2D target through replay-verified
executor edits:

| Exact saved start | Primitive steps | Search seconds | Minimum active atoms |
| --- | ---: | ---: | ---: |
| Current best | 55 | 2.409 | 15 |
| Original root 0 | 38 | 5.710 | 26 |
| Original root 1 | 41 | 0.839 | 12 |
| Original root 2 | 40 | 1.432 | 16 |
| Original root 3 | 41 | 1.039 | 14 |

The five searches took 11.43 seconds total. Two new objective calls took 0.0019
seconds altogether, excluding imports and evaluator construction. The subsequent
route audit makes no objective calls. Nothing entered optimizer training or its
archive. There are five correct endpoints out of five recovered witnesses and
five attempted sources, with all 215 primitive transitions replayed exactly.

## Interpretation

This example is executable within the frozen charge-preserving, achiral,
40-active/48-slot representation. It has 38 heavy atoms and SSSR ring sizes
5, 5, 5, 6, including two aromatic rings. Both the current best and target have
cycle rank four, so increasing ring count alone does not describe the gap.

The current-best route deletes 25 atoms, inserts 23, closes four cycles and
changes three bond orders. Original root 0 instead uses one deletion, 12
insertions, ten atom restatements, eight bond reorders, three cycle openings and
four closures. These are found routes, not shortest routes or proof that these
operations are necessary. Substantial replacement and retyping are relevant,
not only ring addition.

The compiler knows the full target. These witnesses do not establish blind
controller recovery, macro decision counts, option/region applicability, or
probabilities under deployed learned action tables. We have a capability result,
not a new autonomous-controller result.

The current PMO proposal worker calls `execute_option` with adaptation disabled;
its HOW step calls `hierarchy.sample_reference`. `BranchPolicy` learns endpoint
pair preferences, not a remaining-budget future-value function. Existing
`TaskSearch` and `FrontierSearch` prototypes contain return backups, but are not
the current repeated-feedback PMO execution path. The next controller repair
should be evaluated as actual in-loop future guidance, not relabelled endpoint
ranking. A task-aware representation may help, but these results do not establish
that molecular fingerprints are the cause of the score gap.

## Provenance and verification

Producer `bb0f3f9` was clean at numerical execution. `scores.json` contains the
input/implementation hashes, exact saved starting states, full configuration,
public source identity, software/hardware, and two-call ledger. `result.json`
binds every path artifact. `route_audit.json` replays and describes each path,
recording its own implementation hash and local worktree status; its newly added
audit script was not yet committed when it ran.

Local macOS arm64, Python 3.12.9, RDKit 2024.03.5, NumPy 1.26.4, PyTDC 0.3.6,
SciPy 1.13.1. This differs from the remote Python 3.11 runtime. Eight focused
witness tests passed in 4.66 seconds; touched-code Ruff checks passed. All five
actual routes passed exact replay. The repository-wide suite was not run: this
is not release qualification or a completed autonomous-controller milestone.

With pinned evaluator dependencies on `PYTHONPATH`:

```sh
python tools/pmo_public_winner_recovery.py --assets /path/to/verified/ivg/assets
python tools/pmo_target_route_audit.py
```

Compatible score/path artifacts are reused; incompatible ones fail rather than
silently trigger rescoring. Public PDF/PNG assets reside in the main workspace;
their expected hashes and public URL are in the configuration. MCS uses a
one-second wall-time limit, so correspondence selection can vary across hardware.
Stored witnesses nevertheless replay exactly. No shortest-path claim is made.

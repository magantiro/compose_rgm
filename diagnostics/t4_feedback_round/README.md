# Feedback-round development record

## Authorized run, 2026-09-08

The user approved equal 2,500-public-executor-call shares for eight parents,
within the existing 20,000 total ceiling. The single feedback round completed
on deployed `genmol-t4-opt`, with no automatic next round:

- Scientific code: `bdb933948ec6b9f5bbfd0b1c14cce26b7a6a8ef6`.
- Call: `fc-01M2100VP8YHZP8DJMTSCMBBMC`.
- Volume: `compose-v4-artifacts`.
- Prefix: `t4_feedback_round/7b3ff467f806ff38058d520a9aee8273caabe3be0a9e164678f7428ed2f323ff`.
- Contract self-hash: `9fdf7b46c149e49b36eb6196700129af5c89bb2f973304b49dd0f7924ade90c8`.
- Spawn receipt: `attempt_1/spawn.json`, SHA-256
  `3388164b3f463a32a3bf81a364e39910bbcc5dba434446e6d16dedf9821bfce3`.

All three remote source files matched the contract's physical hashes before
launch. No feedback namespace existed. Strict preflight passed in the clean
worktree, followed by deployment and `tools/t4_launch.py --feedback-round`.
The launcher's own clean-source check also passed. Unrelated concurrent
scaffold/decoder work was excluded and preserved in the shared workspace.

Launch verification at the scientific revision: 77 passed in 8.59 seconds,
covering parent-budget cancellation, retained exact outputs, feedback import,
warm continuation, matched-pilot arithmetic, endpoint selection and saved
docking. `launch_focused.xml` SHA-256:
`3ff0eef6f4c90212d8432272f4217c585d099aec3ef6b23fbfc57b7df7857fb6`.
Ruff checks on touched standalone modules/tests and diff checks passed.
The unrelated non-green full suite was not repeated.

The offline auditor now starts after the archive's actual event, so it does
not require a fictitious completed round 2. Three focused auditor tests passed
in 2.20 seconds, recorded in `audit_tests.xml`. This reporting-only change is
not part of the serialized scientific image and does not regenerate proposals.

## Completed result

The best observed feasible docking score improved from **-9.3 at 33 calls to
-9.6 at 46 calls**. All 13 new dockings completed without failure and passed
the frozen QED, SA, and original-seed similarity constraints. The round stopped
normally. Unused allowance is not authority for another round.

`review.json` binds every downloaded JSON by SHA-256 and checks exact-state
topology, sampled probabilities, the frozen KL bound, executor accounting,
candidate-lock chronology, docking receipts, and deterministic archive reduction.
The raw receipts are under `attempt_1/7b3ff467f806ff38058d520a9aee8273caabe3be0a9e164678f7428ed2f323ff/`.
The 19.55 MB executor ledger remains on disk and at the same Modal prefix,
hash-bound in the review, but is excluded from Git to avoid duplicating it.

| Quantity | Observed result |
|---|---:|
| Parent units checkpointed | 8; 3 exhausted their individual shares |
| Region/option draws | 24; no unselected region draws |
| Unique compressed pool | 36 molecules, 23 bundles, 13 feasible |
| Docked candidates | 13 unique molecules from 11 bundles |
| Public executor calls | 12,885 of the 20,000 ceiling |
| Ledger statuses | 11,849 executed; 1,035 invalid; 1 budget-interrupted |
| Law enumerations / sampled transitions | 69 / 117 |
| Proposal / docking time | 566.014 s / 33.057 s |
| Total driver wall time | 840.615 s, including 225.008 s initialization |
| Pool / docked mean pairwise Morgan distance | 0.695 / 0.619 |
| Docked intended release, min/median/max | 0.036 / 0.150 / 0.435 |
| Docked realized coherent change, min/median/max | 0.045 / 0.091 / 0.143 |

Selected options were generic (8), append (3), rebuild (2), restate (2), and
one each of aromatize, build_ring_system, cyclize, decorate, grow, local, open,
scaffold_extend, and small_ring. The fused program was applicable in 8 of 24
draws but selected zero times. The one pendant program reached step 9 of 11
before its parent's share expired; its mandatory two restatement steps did
not complete, so it emitted no oracle endpoint. Share exhaustion is
finite-budget censoring, not chemical invalidity or a completed program.

Two pool candidates gained a cycle relative to their parent; one was docked.
That `small_ring` endpoint scored -9.4 and added a three-membered ring to a
previously fused-ring-built parent. Its cycle rank is five, two above the seed.
All 13 docked endpoints have zero RDKit bridgehead and spiro atoms. This does
not establish medicinal quality or synthetic feasibility.

The -9.6 best instead came from `generic` refinement of a -9.2 parent descended
from the earlier pendant-ring construction. It retained cycle rank four and
two ring systems, while its aromatic-ring count increased from two to three.
This is an observed build-then-refine ancestry, not another ring addition.
The 0.3 best-score gain is not a causal algorithm improvement: docking is
unseeded, this is an inspected development cell, and no matched control ran.
Completing all eight parent units under bounded shares is demonstrated; a
speed improvement over the prior interrupted run is not demonstrated.

### What the IVG comparison actually says

Reuse all five `ivg_snapshot_*` records in
`../t4_winner_comparison/comparison.json`; its old COMPOSE record is superseded
by this round's best. The source CSV, winner snapshot, analysis script, and
ring-taxonomy hashes were checked against that artifact on 2026-09-08.

| Structural descriptor | New COMPOSE best | Five saved IVG winners |
|---|---:|---:|
| Heavy atoms | 29 | 30–33 |
| Graph cycle rank | 4 | 5–6 |
| Aromatic rings | 3 | 3 |
| Largest SSSR ring | 7 | 8–9 |
| Ring systems | 2 | 2–3 |

The missing chemistry is not simply more aromatic rings. These IVG endpoints
have different ring arrangements, more nonaromatic cycles, and a larger
central scaffold ring. These are endpoint descriptions, not atom-mapped
expansion paths or proof that a particular edit improves docking. The saved
winner snapshot also lacks complete upstream revision/license provenance;
it is not a matched-budget benchmark or an untouched final evaluation.

The next useful question is whether existing executor paths can efficiently
change ring size and arrangement while retaining feasible candidates, alongside
continued refinement. Do not turn these winners into templates or similarity
rewards, tune macro weights to their scores, or assume that more rings are
always better. No additional scientific run was launched for this comparison.

The final auditor additionally reports parent scores and ring descriptors and
reconciles the complete executor ledger. Two focused tests passed in 2.97 s;
Ruff lint, formatting, and diff checks passed. `final_audit_tests.xml` SHA-256:
`b7e1a08d9ef1a5a435199cda855061b73627914772b3c6c59d6bcc4ae600a551`.
The added test initially expected a zero-count status key for an empty
synthetic ledger; that fixture assertion was corrected without changing the
auditor or acceptance rule. No repository-wide pass is claimed.

## Earlier preparation record

Implementation revision: `01c17cf8ebeb` (full identity available in Git).
The single twenty-call round was authorized in chat. Its intra-round executor
allocation is still awaiting the user's choice. The contract is deliberately
unsealed; no deployment, proposal, or docking has been launched for this task.

Implemented exact-state import of all 33 evaluated molecules plus the original
seed. Preserve their scores and ancestry, distinguish the partial diagnostic
from a completed optimizer round, and use a deterministic, source-hash-derived
fresh-episode RNG. Reuse the existing locked-round runner, frozen inputs,
controller, constraints, and original-seed similarity. One round only, at most
20 new calls, no implicit retries or automatic continuation.

Clean-worktree verification at the implementation revision:

```sh
PYTHONPATH=src:. /Users/rmaganti/compose_rgm_git/.venv/bin/python -m pytest -q \
  tests/test_t4_feedback_round.py tests/test_t4_warm_continuation.py \
  tests/test_t4_matched_pilot.py tests/test_t4_endpoint_selection.py \
  --junitxml=/Users/rmaganti/compose_rgm_git/diagnostics/t4_feedback_round/focused.xml
python3 tools/preflight.py --strict
```

60 passed in 6.68 seconds. JUnit artifact SHA-256:
`d5af9cc0a9492c3298d37bb4ffcc068b1992e533e5f3719147c7caf3128f06d3`.
Ruff check and format check passed on the new module, shared continuation
runner, launcher, and new tests. `git diff --check` passed. The legacy Modal
app was not reformatted. No full-suite or remote-runtime pass is claimed.

The clean worktree excluded concurrent scaffold-construction and ring-fiber
edits. Those edits were preserved in the shared workspace. Local commits are
unpushed. The prior best score remains -9.3 after 33 calls; these tests are
software verification, not new optimization evidence.

Pending choice: retain the shared 20,000-call proposal ceiling or explicitly
allocate 2,500 calls per parent, retaining committed valid outputs and marking
unfinished trajectories. The latter would change finite-budget search effort,
not the executor, admissible molecular support, or primitive transition law
at completed decisions. It is not implemented or implicitly authorized.

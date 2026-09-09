# First IVG winner reconstructed with whole-ring planning

One answer-known PARP1 seed0, delta=0.4 development winner was reconstructed
exactly in the pinned RDKit 2024.03.5 runtime. The first declared plan succeeded;
its 21 primitive transitions and every intermediate exact state passed a separate
replay. This is 1/1 requested endpoint coverage and 1/1 exact witness correctness,
not a blind discovery result or a shortest-path proof.

| Execution program | Primitive edits | Existing controller interface |
| --- | ---: | --- |
| Remodel the linker | 4 | Three shrink primitives and one grow; no compound linker option |
| Add pendant benzene | 7 | Existing parameterized pendant aromatic C6 ring contract |
| Fuse nonaromatic six-membered ring | 5 | Existing parameterized fused C6 ring contract |
| Add ring carbonyl oxygen | 1 | Ordinary grow primitive |
| Insert carbonyl-bearing atom into the original core ring | 4 | Diagnostic primitive program, not existing `expand_ring` contract |

Thus the result is **five execution programs, two of them existing whole-ring
options**, totaling 21 primitives. It is not evidence that the production
controller already makes five supported compound-option selections. The two
ring requests belong to the default ring menu, but the diagnostic also supplies
their attachment, atom order and bond pattern. Those supplied choices are not
drawn from R_theta.

The older target-informed witness used 23 primitives. The declared direct core
program works without its two electronic preparation edits, so the fallback
six-action core variant was not attempted. No new witness search was needed.
The old witness's local RDKit2026 results remain separate; the new complete
trace was executed and replayed under RDKit2024.03.5.

The exact endpoint is:

```text
O=C1CCCc2cc(CCc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21
```

## Work and verification

- Compilation plus execution: 21 executor calls, 0.113173 seconds.
- Separate exact-state replay: 21 executor calls, 0.094776 seconds.
- Total measured audit section: 0.666562 seconds, including source-closure
  hashing, excluding interpreter/import startup and final output serialization.
- All 42 executor attempts succeeded. No law enumeration, docking, GPU, remote
  job, model training, candidate filtering or exclusions.
- One CPU worker, arm64; Python 3.12.9, NumPy 1.26.4, RDKit 2024.03.5;
  integer molecular states. Timing is one instrumented local observation, not
  a matched autonomous-search speed comparison or an amortized benchmark.
- Scientific source: `b2d60737aa5d2f79a7cff8843fd3d7c3e4b2bfa5`, clean detached
  worktree `/private/tmp/compose-whole-ring.GGBq3I/source`. Preflight: zero
  mounted-source drift. Existing unrelated changes were excluded.
- 19 focused tests passed, zero failures/errors/skips, 5.349 seconds in that
  clean source: whole-ring compiler, existing ring programs and winner paths.
  Ruff lint and formatting passed for the three new Python files;
  `git diff --check` passed. No repository-wide suite was run under the bounded
  T4 development policy. This is not a release milestone.

`result.json` contains all exact source/intermediate states, actions, stage
boundaries, descriptor contracts, executor attempts, input hashes, 73 source
hashes and operational metadata. SHA-256:
`252dccee4785d3a8e972df30c4a075e7e98bdf02e120300990bcb1de9d21cee9`.
The verification receipt binds the focused-test XML retained at the recorded
temporary path. The plan and interpretation are in `docs/T4_WHOLE_RING_PLAN.md`.

## Controller implication

Observed: a supplied complete ring request executes cheaply and coherently
through the existing primitive machinery. Inferred, narrowly: primitive
growth/closure is not intrinsically an expensive obstacle for this route.
Unproven: autonomous proposal coverage, learned-law probability, region
applicability, task-guided ranking, docking improvement and generalization.

The next integration should plan and assess complete program candidates, with
target-independent attachment/electronic/composition choices. It must retain
generic edits and local-to-global region selection, and declare any changed
proposal law explicitly. Linker and carbonyl-bearing core remodeling require
general program interfaces, not copying this winner recipe into the optimizer.
No production controller behavior or frozen R_theta/kappa was changed here.

Reproduce from the scientific source with the pinned chemistry environment:

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/t4_whole_ring_plan.py --audit /Users/rmaganti/compose_rgm_git/diagnostics/ivg_winner_paths/audit.json --output /path/to/new/result.json
```

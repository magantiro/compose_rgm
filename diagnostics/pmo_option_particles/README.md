# Complete-option particles: best-score null, modest immediate top-ten gain

Run `30fb08d196021d7f8a8d524c86b17c74532b72feac92cebaa1c003322e96441f`
completed on 2026-09-11 in 299.99 seconds, excluding 98.31 seconds of deployment.
It used 74 new deterministic PMO calls, 78 unique worker tasks, and no training.
This is one warm, winner-informed development comparison, not a PMO benchmark.

| Arm | Best queried score | Top-ten mean | Canonical queried archive |
| --- | ---: | ---: | ---: |
| Independent reference trajectories | 0.522233 | 0.490682 | 53 |
| Immediate-score SMC | 0.522233 | 0.500620 | 48 |
| Achieved-return SMC | 0.522233 | 0.490682 | 52 |

Every arm retains all evaluated candidates, including those not resampled. The
top-ten values are endpoint means, **not** query-indexed PMO AUC. Initial states
and historical labels are exposed development assets. The future model had
already been trained on known winning paths; no target is supplied to the
proposal generator in this experiment.

## What the run demonstrates

- Reference and future arms each completed 48 options. The immediate arm
  completed 43 of 44 attempted options; its one failed fused program leaves an
  absorbed empty particle slot at later boundaries, not five separate failures.
- Actual new primitive depths reach 19, 14, and 19 in reference, immediate, and
  future arms, respectively. Six option opportunities are not 66 realized edits.
  The separately demonstrated 43-59-step winner paths are not shortest-path
  lower bounds, so these counts do not prove that a shorter successful route is
  impossible. They do show that this test did not exercise long reconstruction.
- Ring construction is happening. Reference trajectories include five pendant
  constructions and one fused construction. One five-membered aromatic pendant
  raises an original-start descendant from 0.143969 to 0.354943; the same option
  type on another molecular context lowers 0.406638 to 0.121369. Ring addition
  alone is not a task-aware choice of scaffold, site, or composition.
- Immediate guidance resamples at boundary 1 and thereafter all completed
  products have 39-40 heavy atoms. It concentrates near the size-saturated
  incumbent. The future head resamples only at boundary 5 and produces no final
  top-ten advantage over the independent arm. This does not establish that SMC
  cannot help; it rejects a performance claim for this particular head/recipe.
- Particle ancestry and resampling decisions replay exactly. Floating weights
  agree within 4.45e-15 (diagnostic tolerance 1e-12 across ARM/x86 exp/log).
  Every emitted complete option was primitive-replayed in its producer.

## Compute and decision

The 74 oracle calculations consumed 0.109 seconds in total. Proposal wall time
across six boundaries was about 231 seconds. Driver volume commits took 84.61
seconds summed across 40 commits, partly overlapping proposal work through
heartbeats; do not add these times as disjoint wall-clock components.

Keep the particle engine and saved experiment as a baseline, not as a solution
to the performance gap. Do not rerun the same model with more calls and call it
a repair. The next intervention must address state-specific multi-edit proposal
and continuation quality, including replacement at the size limit, rather than
adding another ring template or changing kappa. Use the positive-support route
certificates and actual completed-option score deltas to supervise or evaluate
that intervention under a separate bounded development contract. Its training
exposure and prospective evaluation must remain explicit.

`report.json` binds input hashes, executed source inventory, analysis code,
software, arithmetic replay, option counts, intended/realized structural scale,
ring-system/cycle-rank deltas, diversity, and top-ten SMILES. Full remote artifact:
`pmo_option_particles/<run>/result.json` on `compose-v4-artifacts`.
Source bytes are preserved beside the downloaded result under
`/private/tmp/compose-pmo-option-particle-runs/<run>/source_snapshot.tar.gz`.

Verification: 14 focused particle, batch-accounting, value, and fake-chemistry
orchestration tests passed (13 in 2.35 s and one in 5.83 s). Ruff and diff
whitespace checks passed. No full repository suite, final milestone, official
PMO superiority, or held-out generalization claim is made. Git operations remain
deferred under the user's instruction.

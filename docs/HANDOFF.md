# COMPOSE RGM handoff

This repository is the lossless working handoff for COMPOSE: Rewrite Generator
Matching for validity-closed molecular generation. It contains the executable
model, tests, experiment recipes, paper draft, research plans, and the exact
diagnostic artifacts used to make current design decisions.

Read [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the dated distinction between
completed work, the active compiler gate, queued training, and legacy evidence.

## Scientific contract

COMPOSE learns a continuous-time Markov generator over executable molecular
graph rewrites. Every state is a complete, connected, RDKit-valid molecule;
every transition is selected from the state-dependent legal rewrite fiber.
The neural model observes the current molecule and time, never its paired data
endpoint. Inference is target-free ancestral CTMC simulation, not beam search.

The current production path uses:

- a directly samplable, target-independent degree-bounded carbon-tree source;
- flexible-size atom insertion and deletion;
- atom retyping and bond-order changes;
- subtree **Graft** (the backward-compatible code name is `bond_reroute`);
- coordinated whole-ring-system grow/delete actions, including monocyclic,
  fused, bridged, spiro, aromatic, and non-aromatic systems; and
- Generator Matching of the time-dependent rates of legal marked rewrites.

Graft and complete-ring rewrites are derived operators: they compress useful
multi-edit programs without changing the validity-closed state space. The
topology catalog is proposal support, not fragment assembly and not a finite
vocabulary of fully typed molecular fragments.

## Research documents

The three HTML documents in [`research_plans/`](research_plans/) are the
canonical strategy documents for the handoff:

1. [`paper1_compose_methods.html`](research_plans/paper1_compose_methods.html)
   — methods-paper thesis, evidence plan, positioning, and benchmarks.
2. [`paper2_compose_lipid.html`](research_plans/paper2_compose_lipid.html)
   — translational COMPOSE-Lipid plan and prospective in-vivo evidence chain.
3. [`compose_two_paper_execution_plan.html`](research_plans/compose_two_paper_execution_plan.html)
   — dependency-aware execution plan for both papers.

They preserve the agreed strategy: establish a sufficient unconditional
generator and then make conditional generation, optimization, and controllable
editing the methods paper's decisive evidence. Paper 2 inherits the generator
but is written as a biotechnology paper, not a second ML-theory paper.

## Current evidence boundary

As of 2026-07-19:

- Validity-closed execution, flexible-size tree transport, Graft, complete-ring
  actions, factorized marked-rate training, and ancestral sampling are
  implemented.
- Graft aliases are quotiented by canonical molecular successor: self-successor
  Grafts are removed and rates of distinct marks reaching the same molecule are
  summed. Path-cache format was bumped so stale pre-quotient paths cannot be
  silently reused.
- Exact semantic ring support is shared between teacher and sampler. The saved
  128-example teacher audit has zero failures; see
  [`audits/semantic_ring_teacher_audit_v4.json`](audits/semantic_ring_teacher_audit_v4.json).
- The old step-6,250 checkpoint predates the Graft successor quotient and the
  final exact ring-support compiler. Its 2,000-sample metrics and trajectories
  are diagnostics, not evidence for the corrected model.
- Objective-aware exact-support and resource preflights are complete. The
  production choice is A100 + 32 CPU cores + 24 data workers; its final
  preflight interval reached 54.52 examples/s and all 16 rollouts were valid,
  connected, non-null, unique, and novel. The 7/16 small-ring warning remains
  a required early-checkpoint diagnostic, not a final result.
- The 30,000-update quotient-correct run, 100-sample visual inspection, and
  frozen 2,000-sample FCD evaluation remain required. Validation early stopping
  and the credible-checkpoint preview are automatic; final sampling/FCD runs on
  CPU only after the selected checkpoint releases the A100.
- Full invariance of learned rates to arbitrary atom-slot permutations remains
  a formal audit obligation. Canonical successor aggregation fixes the measured
  Graft gauge churn; it is not by itself a proof of every presentation-level
  neural invariance claim.

## Historical failure diagnostic

The large “pancaking” trajectory is preserved at
[`trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32.png`](trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250/full_trajectory_step6250_index32.png).
It records 118 events, including 103 Grafts, but only 30 unique chemical states:
89 events were canonical molecular self-transitions. There were zero strict
nontrivial `A -> B -> A` two-cycles. The four paginated PNGs and JSON replay sit
beside it. Trajectory 75 is retained as an efficient comparator.

This distinction matters: the old model was mostly changing atom-slot
presentations of the same molecule, not repeatedly undoing substantive chemical
edits. The successor quotient addresses exactly that measured failure mode.

## Reproducing local checks

```bash
python -m pip install -e ".[dev,cloud,eval]"
pytest -q
python scripts/audit_factorized_teacher_support.py --help
cp -R docs/trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250 \
  /tmp/compose-rgm-trajectory-rerender
python scripts/diagnostics/rerender_saved_compose_trajectories.py \
  /tmp/compose-rgm-trajectory-rerender
```

The rerender utility intentionally rewrites PNGs and JSON in its target
directory. Run it on a scratch copy, as above, because RDKit depiction can vary
across versions even when the molecular trajectory is identical.

Cloud entry points and production recipes live in [`modal_apps/`](../modal_apps/)
and [`recipes/`](../recipes/). Generated checkpoints, serialized path caches,
rollout tensors, full training datasets, and credentials are intentionally
excluded from Git. One frozen 5,000-SMILES held-out reference subset and small
JSON/CSV/PNG evidence records are intentionally retained because they are
needed to interpret and reproduce the archived stage-1 evaluation.

## Next gated sequence

1. Commit and deploy from a commit-bearing immutable run label; confirm that
   its Modal artifact directory does not already exist.
2. Launch the fresh A100/32-CPU/24-worker flexible-size, quotient-correct
   unconditional run. Validation runs every 250 updates after a 500-update
   warmup, with six evaluations of early-stopping patience.
3. At the first checkpoint with at least 50% validation-loss improvement and
   family accuracy at least 0.60, render 100 ancestral samples and audit
   event rates, atom-count trajectories, ring phenotypes, Graft self-transitions,
   validity, connectivity, uniqueness, and novelty.
4. Let early stopping freeze the validation-selected `checkpoint.pt`; after the
   A100 exits, evaluate 2,000 samples on CPU for FCD and its mean-versus-
   covariance decomposition.
5. Add the valid-rewrite recovery objective, then run the matched QED and
   multi-objective guidance comparisons required by Paper 1.

Validity makes conditional generation unusually actionable because the oracle,
reward, or guidance signal can be evaluated on a real molecule at every jump
and can tilt only executable successors. Better conditional sample efficiency
is therefore a central testable hypothesis, not an automatic consequence that
should be claimed without the matched filtering/repair ablations in the plans.

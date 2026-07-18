# COMPOSE RGM handoff

This repository is the lossless working handoff for COMPOSE: Rewrite Generator
Matching for validity-closed molecular generation. It contains the executable
model, tests, experiment recipes, paper draft, research plans, and the exact
diagnostic artifacts used to make current design decisions.

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

As of 2026-07-18:

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
- A fresh quotient-correct training run, early checkpoint evaluation, 100-sample
  visual inspection, and frozen 2,000-sample FCD evaluation remain required.
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
python -m pip install -e ".[dev,eval]"
pytest -q
python scripts/audit_factorized_teacher_support.py --help
python scripts/diagnostics/rerender_saved_compose_trajectories.py \
  docs/trajectory_diagnostics/legacy_prequotient/full_trajectories_step6250
```

Cloud entry points and production recipes live in [`modal_apps/`](../modal_apps/)
and [`recipes/`](../recipes/). Generated checkpoints, serialized path caches,
rollout tensors, datasets, and credentials are intentionally excluded from Git;
their small JSON/PNG evidence records are retained when they are necessary to
interpret a scientific decision.

## Next gated sequence

1. Finish the exact ring-support performance gate without weakening semantic
   support or introducing rejection sampling.
2. Run the full local suite and a fresh H100 preflight.
3. Launch the fresh flexible-size, quotient-correct unconditional run with
   early stopping and checkpointed validation.
4. At the first credible checkpoint, render 100 ancestral samples and audit
   event rates, atom-count trajectories, ring phenotypes, Graft self-transitions,
   validity, connectivity, uniqueness, and novelty.
5. Freeze one checkpoint and evaluate 2,000 samples for FCD and its mean versus
   covariance decomposition.
6. Add the valid-rewrite recovery objective, then run the matched QED and
   multi-objective guidance comparisons required by Paper 1.

Validity makes conditional generation unusually actionable because the oracle,
reward, or guidance signal can be evaluated on a real molecule at every jump
and can tilt only executable successors. Better conditional sample efficiency
is therefore a central testable hypothesis, not an automatic consequence that
should be claimed without the matched filtering/repair ablations in the plans.

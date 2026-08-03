# COMPOSE Editing V2: start here

Status date: 2026-08-03

This is a navigation document. It does not authorize a scientific job and does
not replace a self-hashed contract, decision artifact, or `AGENTS.md`.

## Scientific objective

COMPOSE learns an executable, trans-dimensional stochastic rewrite process,
pushes its mark law to a canonical molecular-successor kernel, and uses that
kernel for exact and dynamic molecular design.

The current primary validation lane is source-conditioned molecular editing
under a fixed embedded-jump budget. Timed unconditional generation is a
separate lane with its own checkpoint, hazard requirements, and gates.

## Current implementation boundary

Preserve and reuse:

- the production executor and legal rewrite fibers;
- exact persistent-slot molecular states and codecs;
- canonical molecular identity and successor aggregation;
- the production candidate evaluator;
- the frozen corpus, split, role, and evidence-lane definitions;
- the model and canonical-successor objective.

The Wave-2B and Wave-2C Active8 and Gate-0 orchestration is diagnostic evidence,
not the production execution path. The minimal vertical rebuild starts from the
stable scientific base `0e3f24a20963937fd33b173dbf7fe916d3e8601e`.

The rebuild must implement only:

1. chunk-parallel Active8 decisions, with candidate evidence and capability
   cells derived once at write time;
2. a bounded Active8 release sentinel;
3. a lightweight deterministic Gate-0 reduction over authenticated Active8
   evidence, with no second molecular-data pass, successor re-enumeration,
   per-trace point lookup, or separate distributed subsystem;
4. a native Process-V2 T1 capacity lane;
5. a thin Process-V2 P50 prelaunch and execution wrapper.

## Gated execution sequence

The current sequence is:

1. materialize and verify the content-addressed Process-V2 derivatives;
2. run whole-trace Active8 admission;
3. run structural Gate 0 on the exact admitted corpus;
4. run the unique-state canonical-successor T1 capacity gate;
5. run P50 only after T1 returns `GO`;
6. advance to P500 and P2000 only through their registered decisions;
7. authorize a long run only after the staged gates pass.

The presence of launcher or trainer code does not authorize a later stage.

## Active8 development scope

The bounded development families are:

- `atom_insert`
- `atom_delete`
- `atom_restate`
- `bond_reorder`
- `bond_reroute`
- `cycle_insert`
- `cycle_attach`
- `ring_system_restate`

`ring_system_delete` and `ring_system_grow` are disabled in the bounded pilot.
Active8 is a development freeze, not a final production-support claim.

## Authoritative reading order

1. [`../AGENTS.md`](../AGENTS.md), including the COMPOSE-specific contract.
2. [`HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md`](HANDOFF_COMPOSE_TRACEABILITY_2026-07-29.md).
3. [`EDITING_V2_FORENSIC_STATUS_2026-07-31.md`](EDITING_V2_FORENSIC_STATUS_2026-07-31.md).
4. The current self-hashed files under `configs/` and their bound decision
   artifacts.
5. [`PAPER1_FRAMING_AUTHORITATIVE.md`](PAPER1_FRAMING_AUTHORITATIVE.md) and the
   applicable portions of [`PAPER_MASTER_PLAN.md`](PAPER_MASTER_PLAN.md).
6. [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md), revalidated against current artifacts
   before manuscript use.

## Historical documents

The following remain available for provenance but are not current execution
instructions:

- [`HANDOFF.md`](HANDOFF.md), the 2026-07-19 unconditional handoff;
- [`CURRENT_MODEL.md`](CURRENT_MODEL.md), the earlier whole-ring model contract;
- [`HANDOFF_RINGCORE_V1_POSTRUN.md`](HANDOFF_RINGCORE_V1_POSTRUN.md), the
  completed 16,000-step diagnostic run;
- Wave-specific Process-V2 reports, which describe the revision named in each
  report and do not authorize later work.

## Before any remote scientific job

Require all of the following:

- a clean, pushed, exact source commit;
- current prerequisite and authorization artifacts;
- a genuine molecular end-to-end test with nonzero exact assignments;
- a production-sized benchmark for the actual hot path;
- focused tests, Ruff, `git diff --check`, and the required unloaded full suite;
- content-addressed outputs with complete input and implementation provenance.

Failed, missing, stale, `NO_GO`, or merely hash-valid prerequisites do not
authorize training.

# Winner-informed Perindopril program curriculum

## Identity and claim

Scientific problem: determine whether COMPOSE can turn a published high-scoring
Perindopril molecule into a small, executable program curriculum whose completed
outputs exceed the reported InVirtuoGen (IVG) top-ten area under the curve (AUC)
for this task.

Primary output: exact complete edit programs, bindings, primitive traces,
completed supported molecular graphs and a counted oracle ledger.

Claim under test: from four declared exact roots, a frozen answer-known program
curriculum can produce ten or more unique supported endpoints early enough that
the official 10,000-query, counted-only top-ten AUC with `finish=True` exceeds
IVG's repository-reported no-prescreen result of 0.645. The prescreen result,
0.753, is a secondary reported comparator with a different information regime.

This is a winner-informed Perindopril development result. It is not a held-out
PMO evaluation, a claim that the generic controller discovered the public
molecule, or evidence of general PMO superiority. The released molecule and its
task score were inspected before this recipe was frozen.

## Inputs and support

- Public source: IVG's rasterized Perindopril top-molecule figure, already bound
  by PDF SHA-256 `b034d0cc3292e08036de4f8933e133cb54373830421fec57dd219c371a2eab65`.
- Exact public transcription and score parity: the existing
  `pmo_public_winner_recovery` configuration and result.
- Construction supervision: five existing target-known recovery witnesses. The
  four original-root witnesses are eligible roots for the scored curriculum;
  the historical COMPOSE-best witness is structural evidence only.
- Representation: neutral, charge-preserving 2D graphs in the broad-organic
  15-class vocabulary, at most 40 active atoms in 48 persistent slots. No
  stereochemical or formal-charge claim.
- Oracle: PyTDC 0.3.6 and RDKit 2024.03.5 Perindopril MPO, using the pinned
  production source hash already recorded by the recovery result.

The curriculum may compile the public endpoint and a bounded local family made
only through supported executor operations. Every candidate must replay from a
declared exact root. A SMILES string may identify the desired endpoint during
answer-known compilation, but it may not replace the saved exact state or
executor trace.

## Retrospective development finding

Before this contract was written, a local deterministic probe found that the
published endpoint scores 0.8088297765764039. Replacing its distal bromine with
each of F, Cl, I, O, N, C, S, P or B produced nine distinct parseable molecules
with the same computed score. A shorter connecting segment produced
0.810643483378. These values selected the curriculum and therefore are not a
prospective test. The locked run verifies program execution, accounting and the
official metric.

The identical-score family is a task-metric plateau, not nine independent
chemical improvements. The output must preserve this distinction.

## Frozen bounded procedure

1. Verify every material input hash and the pinned chemistry environment.
2. Replay all five existing public-target recovery witnesses.
3. Use the exact endpoint state of original root 0 to compile supported suffixes
   for the nine element substitutions and one shorter-segment variant.
4. Extract eleven complete programs from original root 0: one public-endpoint
   program and ten prefix-plus-suffix programs. Replay each extracted program
   from the original exact source binding and lock its canonical endpoint.
5. Require at least ten unique endpoints, all within support. Record program
   failures and abort before scoring if this structural gate fails.
6. Score the four original roots first, then the eleven locked program outputs
   in deterministic endpoint order. Deduplicate canonically. The hard ceiling is
   15 expected and 16 allowed calls; no retry or replacement is allowed.
7. Compute the official PMO counted-only top-ten AUC with frequency 100,
   `finish=True`, and denominator 10,000. All root scores and all new endpoint
   scores count.

The run is local CPU work. It uses no GPU, frozen-reference inference, learned
surrogate, hidden reward component or uncharged prescreen. It does not launch
Modal or alter any T4 artifact.

## Baselines and decision rule

Primary external comparator: IVG no-prescreen Perindopril top-ten AUC 0.645,
reported by its official results repository. Secondary comparator: IVG
prescreen 0.753, which used a different initialization regime.

Internal causal ablation: the prior fast program-only controller without this
public-target curriculum remains the same-generator control. Its completed
Albuterol run is not a Perindopril numeric baseline and must not be presented as
one. A matched Perindopril control is future work unless separately frozen.

Positive result: replay coverage is eleven of eleven with zero invalid outputs,
at least ten unique scored endpoints are retained, and counted-only AUC exceeds
0.645. Exceeding 0.753 is recorded separately. Negative or partial outcomes are
preserved without changing the endpoint family, query order or metric.

## Required artifacts and verification

Publish under `diagnostics/pmo_winner_program_curriculum/`:

- `curriculum.json`: source/input hashes, full program payloads, bindings,
  suffix compilation receipts, endpoint identities, support descriptors and
  the disclosed evidence roles;
- `result.json`: complete query ledger, top-ten trajectory, official AUC,
  comparison table, software/hardware, code revision and interpretation;
- `README.md`: concise measured outcome, limitations and runnable command.

Focused tests must cover input mismatch, exact route replay, program round trip,
endpoint uniqueness/support, deterministic ordering and the official AUC
calculation. Run touched-code formatting/lint, `git diff --check`, repository
verification and the full test suite at the launch boundary. Inspect artifacts
and final diff before commit.

## Pre-launch verification status, 2026-09-13

The focused curriculum and executor dependency suite passes 28 tests. Touched
Python files pass Ruff lint and formatting, and `git diff --check` passes.

The repository-wide suite is currently blocked during collection by an existing
editing-V2 identity mismatch. The semantic-capability registry records process
identity `6c4721f0dd37132aae657e7aa5f1bfc01cef270662f228171c4587eb7dd48491`,
while the current process implementation reports
`0a10a2fae24d51853dc30674e31842ea11313571d991e822f23dab7e1124445b`.
No curriculum code changes either file. The mismatch is retained as a failing
gate rather than silently re-pinning the registry or skipping its test. The
scored curriculum does not launch until this separate lineage issue is resolved
under its own authorization.

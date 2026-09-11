# Contextual successful-edit replay, bounded first experiment

The preceding full-pool audit found no proposal above the current 0.603023
incumbent. Refit endpoint preferences did not produce constructive improvement.
This experiment changes the proposal distribution before a molecular program is
chosen: reuse beneficial replacements at analogous removal/attachment contexts.

Project output and claim: source-conditioned executable molecular edit programs,
with broad primitive chemistry and local-to-global changes. The optional replay
channel is a nonparametric task-trained proposal, not a future-value head or an
exact Doob transform of the frozen neural reference. It changes neither executor
nor atom, charge, size, slot, or canonicalization semantics. Its finite history
is not a predefined whole-ring vocabulary and does not define the full model's
support. Full-controller integration must retain generic reference editing.

## Frozen before new proposals or labels

Training inputs are the isolated donor probe and the first two completed donor
comparisons only. Later archive/program-choice/pool-audit labels are available
for charged lookup but do not fit the replay policy. Deduplicate by canonical
parent/product before fitting and retain every origin and nonpositive exclusion.
Use actual saved persistent-slot source/donor states and bridge operands, never
reconstruct a replay source from SMILES. Verify each positive edge's proposed
graph identity against its recorded scored endpoint without redoing its search.

For every positive edge e, store the removed component and the retained boundary's
rooted radius-2 Weisfeiler-Lehman features. They encode atom identity, charge,
hydrogens, bond order and the cut port. Labels are deterministic hashes, not a
learned preprocessing vocabulary. Normalize positive score gains within each
parent, multiply by exp(10 times endpoint score), then normalize across entries.
This prevents repeated trajectory aliases or heavily sampled parents from
dominating the experience distribution before explicit task-quality weighting.

For each available cut c on a new parent x, compute smoothed count-Tanimoto
similarities for retained context and removed component. Replay mass on (c,e)
is proportional to the entry weight times the square of their product, with
similarities smoothed as 0.05 + 0.95 J. Draw a cut and an experience together;
use the experience's donor and donor cut, then compile under the same executor.
This is resemblance-based transfer, not evidence that a previous gain is causal
or guaranteed to transfer. No endpoint predictor screens the proposed program.

Baseline is the existing uniform donor/uniform oriented-cut proposal. Treatment
mixes 80% contextual replay with 20% that same unrestricted donor law. Thus the
entire original donor proposal support stays positive, including larger changes.
Neither arm is the whole COMPOSE controller. Do not call mark/cut probabilities
representation-invariant canonical molecular probabilities.

One fresh fixed batch: same 16 current exact high-scoring parents (best 0.603023),
four draws per parent per arm, seed 20261001. At most 128 attempted programs and
128 new physical PMO calls, all unique completed endpoints queried, not ranked.
Both arms share the original 100 donors, compiler cap 64 steps/128 expansions,
and endpoint oracle. Failed, self and repeated proposals get no replacement.
Lock every draw and arm membership before revealing new outcomes. Previously
charged labels can be reused without importing an unrequested molecule into an
arm's candidate set. All 249455 prescreen calls and 669 prior development calls
remain in accounting; no public winner or new reference training is used.

Measure best and top-ten candidates, positive offspring fraction, full-pool
coverage and diversity, intended release/actual topology change, actual repeat
and new queries, feature/preparation/compiler/oracle time. A new best over 0.603023
and concurrent baseline earns fresh-seed replication followed by integration.
Damage avoidance without a new best is not success. Low applicability or repeat
coverage is a proposal failure, not permission to drop generic chemistry. Null
results stop this unchanged recipe. The benchmark goal remains separate.

Compute: local qualified RDKit 2024.03.5/PyTDC 0.3.6, one CPU, no neural R_theta
calls, GPU, Modal deployment or remote spend. Expected 30-90 seconds after input
preparation; 300-second invocation limit with per-draw receipts and no implicit
retry of unresolved oracle calls. Focused checks only. Preserve source hashes,
deterministic seed derivation, exact operands/witnesses, split identities, score
locks and oracle receipts. No full-panel or scientific-release completion claim.

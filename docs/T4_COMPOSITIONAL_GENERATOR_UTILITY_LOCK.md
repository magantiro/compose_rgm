# T4 compositional-generator prospective utility lock

## Scope

This milestone prepares a zero-oracle endpoint panel for a possible later
comparison of baseline-uniform, baseline-learned, expanded-uniform and
expanded-learned compositional structural-subgoal generation. It uses exactly
`5ht1b_0`, `braf_1`, `jak2_1`, `parp1_0` and `fa7_0`.

The self-hashed prelock contract was committed before the expanded candidate
lock was available. It fixes the cell and arm set, endpoint gate, selection,
tie-break, deduplication and abstention behavior. A separate self-hashed input
binding must bind both immutable committed candidate locks and the committed
selection implementation before either candidate payload is opened.

## Selection

Every in-scope candidate is exact-replayed and audited under RDKit 2024.03.5.
Eligibility requires a valid connected non-null, non-self endpoint, exact active
and heavy atom agreement, at most 40 active atoms, source Morgan radius-2/2,048-bit
similarity strictly greater than 0.4, quantitative estimate of drug-likeness
(QED) strictly greater than 0.6 and synthetic accessibility (SA) strictly less
than 4.0.

Within each cell and arm, eligible candidates are canonically deduplicated. The
lowest-rank representative is locked, with canonical SMILES and candidate
identity as deterministic tie-breaks. Each cell/arm selects at most one molecule.
After independent selection, identical target, canonical molecule, docking seed
and evaluator identities are merged across arms while retaining every membership.
There is no replacement, backfill or reselection after a merge or abstention.

## Artifacts and interpretation

The command publishes a complete compressed candidate ledger, selected membership
lock, deduplicated request lock, all-20-unit abstention ledger, result and concise
interpretation under
`diagnostics/t4_compositional_generator_utility_lock/attempt_1`.

Candidate generation scores are retained only as labelled proposal provenance.
No teacher endpoint, transformation, metric, task score, docking score or
evaluation manifest enters selection. Baseline and expanded generator artifacts
are read-only.

This lock can establish only score-blind endpoint eligibility and coverage. It
does not establish docking utility, teacher recovery, route recovery, autonomous
optimization or an InVirtuoGen comparison.

## Launch boundary

The request lock is not launch authority. A future run requires a separate,
explicit authorization naming the exact request-lock physical and payload SHA-256,
the exact request count as its call ceiling, and zero retry, replacement or
backfill. This milestone never calls docking or another oracle, launches Modal or
accesses a live run.

# Winner-blind completed-option search, 2026-09-09

Measured: the four fixed searches completed 118/120 option attempts, with two
support dead ends and zero new docking calls. Every completed option was replayed
through the existing executor. No IVG molecule, path or winner-derived target
entered generation. The reference law, executor, region/option priors, kappa and
shared saved task-value snapshot were unchanged.
Both support dead ends were the legacy build_ring_system option after eight
committed growth primitives, one in each seed arm, not the explicit whole-ring
construction channels.

| Root / arm | Completed / attempted | New eligible molecules | New eligible with added cycles | Best predicted docking among new eligible | Proposal seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original seed / post-hoc | 29/30 | 8 | 5 | -8.098 | 881.5 |
| Original seed / guided | 29/30 | 11 | 6 | -8.117 | 662.3 |
| Best archive parent / post-hoc | 30/30 | 0 | 0 | unavailable | 795.6 |
| Best archive parent / guided | 30/30 | 0 | 0 | unavailable | 795.4 |

New means absent from the canonicalized 51-call development archive; eligibility
requires QED >= 0.6, SA <= 4, original-seed similarity >= 0.4 and the existing
medicinal-chemistry screen. Counts are distinct canonical complete-option
endpoints harvested across all four levels, not just the final three beam states.
Each seed arm also recovered one already observed eligible molecule. These are
one paired random seed at two roots, not reliable success-rate estimates.

## Finding

Whole-ring construction is available in autonomous search. The seed searches
executed pendant five- and six-membered ring options, carbonyl operations,
generic edits, growth, shrinkage and restatement. One four-option chain was
aromatize, pendant aromatic five-ring, pendant aromatic six-ring, pendant
nonaromatic five-ring. Its 20 replayed primitives added three cycles and three
ring systems, but the endpoint was not oracle-eligible. The archive-root searches
also executed fused-ring options. They each completed eight cycle-adding options.
No teleporting or whole-ring executor operation was introduced.

Guidance changed seed-root allocation: the two candidate sets overlap on 20 of
38 distinct molecules. It retained one eligible final-depth endpoint versus none
in the post-hoc arm. Its best predicted score improved by only 0.019 relative to
post-hoc, and that preferred molecule was a non-ring structural modification.
This does not establish a docking advantage or a general benefit of guidance.

The archive-root comparison locates a concrete failure. The observed -9.7 parent
starts at similarity 0.417910. All 30 generated endpoints fall below 0.4 (maximum
0.380282); the median is 0.257832. All candidate desirabilities at the three
retention decisions are therefore zero, so the guided first-slot distribution
is exactly uniform and both arms produce identical attempts. Eighteen endpoints
also fail QED and thirteen fail SA, with overlapping reasons. Twenty-six pass
the separate medicinal-chemistry screen; four fail its isolated 11-membered-ring
rule. Those molecules remain in the evidence, not in an oracle queue.

Inferred next target: constraint-aware selection between completed options.
The current terminal utility cannot distinguish temporarily infeasible states
by their potential to recover, and the beam does not retain an unchanged good
parent. Preserve the incumbent and test informative constraint-recovery ranking
without changing final benchmark gates. Merely adding more ring options does
not address the measured failure. This follow-up is proposed, not implemented
or launched by this diagnostic.

## Local/global accounting

Intended region release spans 0.0333 to 0.8947 in both seed searches. Median
per-option realized coherent change is 0.1053, with maximum 0.3158. Across new
eligible endpoints, cumulative coherent change from the seed has median 0.3158
in both arms and maximum 0.4737 post-hoc or 0.4211 guided. Each eligible
ring-changing endpoint adds one cycle and one ring system. Multi-ring outputs
are generated but not counted as eligible successes.

For the archive root, release spans 0.0286 to 0.75; median per-option realized
coherent change is 0.0667 and maximum 0.2333. These quantities are kept separate:
a large permitted region does not imply a correspondingly large realized edit.
The machine-readable summary retains complete option counts, chains and deltas;
the locks retain exact regions, primitive witnesses and all selection rows.

## Runtime, repair and provenance

Four one-core CPU searches ran with width three, three bundles per retained
parent and four completed-option decisions. No GPU or training was used. Total
worker times including startup were 1041.8, 967.7, 768.8 and 966.8 seconds in case
order. They made 211 fresh marked-law enumerations and 16,842 public executor
calls, including replay. Two exact laws were reused from the compatible prior
cache after checking 116 source dependencies and all frozen runtime inputs.
Runtime differences are observations, not a controlled throughput advantage.

The first launch's seed-root workers failed before proposals: the archive seed
SMILES was noncanonical, although its canonical identity matched the saved
exact graph. The repair compares identities without reconstructing persistent
slots and canonicalizes archive identities for post-lock novelty accounting.
Only cases 0 and 2 were relaunched. The two failures consumed 101.3 and 104.1
seconds and are preserved under attempt_1. Archive-root cases were not rerun.
The reducer verifies identical serialized dependencies outside the repaired
driver, unchanged driver AST outside the documented input/accounting repair,
and unchanged novelty classifications for the actual reused candidate sets.

Generation revisions: 7e337d28f19630743f98bf12aedc68aa15f153f1 (archive roots)
and ac7de60c26c0806252f2abf7ee6e98a6bc22e304 (repaired seed roots). Exact run IDs,
call IDs, configuration, software, input hashes and source manifests are in the
launch/result artifacts. Bulk laws and executor receipts remain on Modal volume
compose-v4-artifacts under `/t4_macro_beam/case_<case>/<run_id>/`.

Reproduce the deterministic reduction in the pinned RDKit 2024.03.5 environment:

```sh
PYTHONPATH=src:. python3 diagnostics/t4_macro_beam/summarize.py
```

The summary binds every material input and the reducer hash, checks paired
first-level generation, zero-oracle accounting, exact value-snapshot identity,
canonical accounting, complete-option retention and probability floors.
Eight focused checks passed on the first clean commit and nine on the repair.
Strict preflight and focused lint/format checks passed. Whole-file legacy app
lint still reports the same 17 pre-existing findings, none in the new wrapper.
No repository-wide suite ran; this is bounded development evidence, not a full
milestone qualification. No next search or docking job was launched.

# Second-generation measured-history comparison

All 49 distinct locked candidates were evaluated. Twenty fresh confirmations
followed, totaling **69 calls**, no oracle failures, **101.9983 driver seconds**
and **293.4748 summed worker seconds**. Identical JAK2 and 5HT1B arm champions
saved four of the maximum 24 confirmation calls. No further docking is active.

Source commit: `66bec5cf631a`; call: `fc-01M2BW7XCWAB9PCK4J2AQDNTFW`.
Volume: `compose-v4-artifacts`, prefix:
`t4_second_generation/f190cc525ea42d159c5674289e3aae3b5bbe8c6a2038619278b2e1f5ecd7774a`.
The launch used a clean clone. Deployment took 67.685 seconds, separately from
the driver timing. Dollar spending was not obtained from provider billing.

## First evaluations, lower docking score is better

| Cell | Previous incumbent | Ranked best new | Score-blind best new |
| --- | ---: | ---: | ---: |
| JAK2 seed1 | -10.3 | -10.9 | -10.9 |
| FA7 seed0 | -9.2 | -9.4 | -9.3 |
| BRAF seed1 | -11.1 | -10.3 | -11.2 |
| 5HT1B seed0 | -13.1 | -12.7 | -12.7 |

Ranked FA7 retains its seven-candidate pool; the other pools contain eight.
Cross-arm shared molecules received one physical first evaluation. Each arm's
curve follows its own prelocked candidate order and retains the old incumbent.
These are development comparisons from the same paid histories, not matched
external benchmark wins. Historical workshop results are not overwritten.

## Fresh confirmations only

| Cell | Incumbent repeats | Ranked champion repeats | Score-blind champion repeats |
| --- | --- | --- | --- |
| JAK2 | -9.9, -9.8 | -10.7, -9.6 | same molecule, shared repeats |
| FA7 | -9.1, -9.2 | -9.3, -9.3 | -9.3, -9.2 |
| BRAF | -10.9, -11.1 | -9.9, -10.3 | -11.1, -11.1 |
| 5HT1B | -13.2, -13.9 | -7.5, -12.7 | same molecule, shared repeats |

The ranked repeat-mean differences in favor of the challenger are +0.30,
+0.15, -0.90 and -3.45, respectively. Score-blind differences are +0.30,
+0.10, +0.10 and -3.45. These are descriptive two-repeat means, not confidence
intervals or significance claims. The selected first score is excluded from
each repeat-only mean.

## Decision

Keep the coordinated program mechanism and all genuine scores, but **do not
promote score-ranked parent allocation as superior**. It ties the control on
two cells, offers only a small FA7 gain and loses BRAF substantially. New
first-score improvements on three cells are not proof that stronger parent
ranking caused them. Preserve broad exploration and the existing 5HT1B
incumbent; do not spend another batch on this unchanged allocation rule.

The BRAF score-blind winner's locked provenance records a -10.6 parent,
`extend_segment` followed by attachment mutation, 31 -> 32 atoms relative to
the measured parent, and 39 -> 32 relative to the original benchmark seed.
This is an observed program example, not a general causal rule. It suggests
that choosing productive mutations conditional on parent context deserves
more attention than simply ranking parents by their existing scores.

The new 5HT1B champion's -7.5/-12.7 repeats have identical prepared ligand
hashes (`l.mol` and `l.pdbqt`) but different docking seeds and output pose
hashes. Thus this particular swing is not attributable to different prepared
ligand inputs. It exposes search variability in the existing exhaustiveness-1
docking protocol. No seed was dropped, no score replaced and no oracle setting
changed. Further oracle qualification would require a distinct declared assay.

## Durable outputs and checks

`remote_result.json` binds the clean launch, runtime inputs, all scores, failures,
pose hashes, seeds, timing and first/repeat separation. `review.json` verifies
the launch identity, locked candidates, arm membership, query counts and
confirmation selection; it hashes all material inputs and eight updated
arm-specific archives. Each archive consumes its own locked candidates and
compatible confirmation receipts. No new proposal or model fit was performed
during ingestion.

Four focused runner tests passed, covering own-arm selection, shared repeats,
failure/budget integrity, exact replay, completed-result reuse and ambiguous
start rejection. Ruff and whitespace checks passed. Existing controller tests
were not rerun for unchanged controller code; full-suite release verification
was not performed and the overall controller milestone is not complete.

Four calls remain unused in this 73-call allocation, not authorization for a
new generation. The next safe step is to analyze the now-scored mutation
contrasts and docking variability before specifying another bounded experiment.

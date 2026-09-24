# Held fragment-interface qualification: failed gate, no promotion

The frozen single-interface policy **did not qualify**. All 54 predeclared
prompt/arm units completed (1,080 attempts), but the runner correctly refused
to publish `summary.json`: on the first held motif prompt, the candidate's
attempt records differed from the global-release control despite the frozen
contract demanding identity. The candidate also lowered motif full-task
success from 96/120 under permanent restriction to 76/120. It must not be
promoted or tuned against these held prompts.

## Identity and accounting

- Frozen source commit: `526e30a3852211df315b73740522d3af4f16eea8`.
- Predeclared contract SHA-256:
  `1b348d3278c953a7fc0e086fbb251fbdb1420919ba1feceb10a45dd36cdcb314`.
- Held analysis: `quality_denominators.json`, SHA-256
  `38a2e74ad1a52dd4a2de76a03f5f5c1eb7fe3b3a8b74ae2110f71465297f931c`.
  It embeds hashes of every atomic unit, source implementation,
  checkpoint/input identity and the pinned IVG evaluator blobs. The evaluator
  is IVG commit `b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`;
  `in_virtuo_gen/train_utils/metrics.py` SHA-256 is
  `3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099`.
- 54 units × 20 attempts = 1,080 exact attempt records, 973 committed endpoints,
  898 full-task emissions. The denominator auditor verified that each
  attempt-aligned list reconstructs the saved committed/emitted lists and that
  official quality agrees with a fresh per-molecule QED/SA recount.
- Every committed endpoint was parseable, connected, and chemically valid:
  973/973. This is the COMPOSE chemical-validity invariant, not the task-success
  percentage. Independent RDKit substructure containment was 969/973, not
  100%; four non-containing committed endpoints are retained below.

The official [IVG evaluator](https://github.com/invirtuolabs/InVirtuoGen_results/blob/main/in_virtuo_gen/train_utils/metrics.py)
counts **distinct valid** molecules with QED ≥ 0.6 and SA ≤ 4, divided by the
number of attempted samples, not by valid commits. The released
[GenMol fragment runner](https://github.com/NVIDIA-BioNeMo/genmol/blob/main/scripts/exps/frag/run.py)
also deduplicates, then applies QED ≥ 0.6 and SA ≤ 4, divides by configured
`num_samples=100` per prompt, and averages prompt ratios. The audit reports
the attempt, commit, and unique-valid denominators independently. Its
"official-like emitted quality" uses only full-task emissions, because our
strict harness withholds committed molecules that fail the supplied fragment
condition. It should **not** be called a direct published comparator result.

## Held outcomes

Each family/arm has six held prompts × 20 attempts = 120 attempts. `Qe` is
the number of distinct QED+SA-qualified *emitted* molecules, summed within
prompts, divided by 120 for official-like quality. `Qc` applies the same rule
to *committed* molecules, including task failures; the latter is diagnostic.

| Family / arm | Full task success | Valid commits | Unique valid commits | Qe / attempts | Qc / commits | Qc / unique valid | Mean QED / SA on commits | Mean emitted uniqueness / diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif, permanent restriction | 96/120 | 99/99 | 84 | 22/120 | 23/99 | 23/84 | .617 / 3.681 | 81.2% / .515 |
| Motif, global release | 92/120 | 97/97 | 96 | 23/120 | 26/97 | 26/96 | .554 / 3.908 | 100% / .730 |
| Motif, single-interface policy | **76/120** | 95/95 | 93 | 23/120 | 26/95 | 26/93 | .575 / 3.727 | 100% / .710 |
| Decoration, permanent restriction | 93/120 | 109/109 | 108 | 27/120 | 33/109 | 33/108 | .592 / 4.050 | 100% / .559 |
| Decoration, global release | 91/120 | 107/107 | 107 | 19/120 | 25/107 | 25/107 | .566 / 4.233 | 100% / .692 |
| Decoration, single-interface policy | 93/120 | 109/109 | 108 | 27/120 | 33/109 | 33/108 | .592 / 4.050 | 100% / .559 |
| Superstructure, all three arms | 119/120 each | 119/119 each | 119 each | 46/120 each | 46/119 each | 46/119 each | .584 / 3.453 each | 100% / .726 each |

Motif Qc per *attempt* was 23/120 strict, 26/120 global release, 26/120
candidate. Thus the apparent committed-only quality gain changes denominator
when the number of commits changes, and the one-count emitted-quality gain
does not establish intrinsic improvement: candidate mean QED fell and SA rose.
The released policies made outputs more unique and diverse but reduced the
rate at which a valid committed endpoint satisfied the supplied placement
constraint. Decoration has the opposite tradeoff: the global-release arm
increased diversity but lowered both QED/SA quality and official-like quality.
The frozen candidate is exactly the strict control on multi-interface
decoration and all three arms are exactly identical on zero-interface
superstructure, as intended.

### Per-prompt result: full success / qualified emitted (each out of 20 attempts)

| Family / held prompt | Permanent restriction | Global release | Single-interface policy |
| --- | ---: | ---: | ---: |
| Motif / CYCLOTHIAZIDE | 20 / 3 | 19 / 4 | 16 / 3 |
| Motif / ELIGLUSTAT | 20 / 13 | 20 / 13 | 20 / 14 |
| Motif / FUTIBATINIB | 15 / 1 | 14 / 2 | 11 / 4 |
| Motif / LESINURAD | 10 / 5 | 9 / 3 | 6 / 1 |
| Motif / LOVASTATIN | 11 / 0 | 12 / 0 | 9 / 0 |
| Motif / SPIRAPRIL | 20 / 0 | 18 / 1 | 14 / 1 |
| Decoration / CYCLOTHIAZIDE | 17 / 0 | 15 / 0 | 17 / 0 |
| Decoration / ELIGLUSTAT | 17 / 10 | 17 / 7 | 17 / 10 |
| Decoration / FUTIBATINIB | 19 / 3 | 18 / 4 | 19 / 3 |
| Decoration / LESINURAD | 20 / 12 | 20 / 7 | 20 / 12 |
| Decoration / LOVASTATIN | 13 / 0 | 15 / 0 | 13 / 0 |
| Decoration / SPIRAPRIL | 7 / 2 | 6 / 1 | 7 / 2 |
| Superstructure / CYCLOTHIAZIDE | 19 / 0 | 19 / 0 | 19 / 0 |
| Superstructure / ELIGLUSTAT | 20 / 13 | 20 / 13 | 20 / 13 |
| Superstructure / ERLOTINIB | 20 / 7 | 20 / 7 | 20 / 7 |
| Superstructure / FUTIBATINIB | 20 / 10 | 20 / 10 | 20 / 10 |
| Superstructure / LESINURAD | 20 / 7 | 20 / 7 | 20 / 7 |
| Superstructure / LIOTHYRONINE | 20 / 9 | 20 / 9 | 20 / 9 |

## Why the predeclared gate failed

The candidate condition checks the *predecessor* coverage. It keeps the
undeclared-core-growth ban while the required interface is unsatisfied, then
releases it. Global release does not keep that ban even during the
coverage-increasing action. Consequently they can admit different actions and
consume different trajectories before coverage, so attempt identity was an
incorrect frozen expectation. The runner caught this on CYCLOTHIAZIDE and
published no summary. The result is a failed gate, not grounds to rewrite the
expectation after looking at held outcomes. The candidate also had 19 final
motif constraint failures versus 3 under strict restriction, consistent with
the observed placement-success loss. This is not chemical invalidity.

Four distinct committed molecules failed an independently constructed RDKit
fragment query, although they were chemically valid. Their SMILES and exact
prompt/attempt IDs are in the atomic units; two representative witnesses are
`CCN(C)C(=O)C1CCn2c1ccc(C)c2=O` (FUTIBATINIB motif, global release, attempt 8)
and `C=CCOC(=O)c1sc2nnc(Br)n2c1COC` (LESINURAD motif, global release,
attempt 19). New ring closures/aromatic perception are plausible causes, but
the stored endpoint alone does not establish the precise primitive cause;
full graph traces would be needed. Do not report 100% RDKit fragment
preservation from the pathwise locked-graph invariant.

## Comparator and decision boundary

The historical document quotes GenMol V1, but the current official
[GenMol README](https://github.com/NVIDIA-BioNeMo/genmol) reports a stronger
V2 comparator: motif validity 99.4%, quality 49.0%, diversity .626;
decoration validity 99.2%, quality 39.7%, diversity .571;
superstructure validity 99.7%, quality 39.0%, diversity .551. The immutable
GenMol revision/blob SHA could not be retrieved in this environment, so these
README values are **reported but not yet pinned**. This 20-attempt, one-seed
held qualification is not the full 100-attempt official suite, and no claim
of beating IVG or GenMol follows from it. The de novo GenMol V1/V2 README
commands also use different released sample-count configs (1000 versus 100),
so their quality difference should not be read as a matched-sample ablation.

Decision: **reject promotion of the single-interface policy from this gate**.
Keep the finished units and failed-gate artifact immutable. A future attempt
would require a new predeclared contract and development evidence, not a
post-held edit to this contract or a retroactive passing summary. The linker
prompt-join defect remains a separate, unaddressed harness issue.

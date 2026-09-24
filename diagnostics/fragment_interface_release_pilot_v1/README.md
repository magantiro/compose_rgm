# Fragment interface-release pilot: measured decision

Status: zero-oracle **development pilot only**, not an official fragment-suite
result, not a model selection, and not a GenMol/IVG superiority claim. The
predeclared protocol is in `docs/FRAGMENT_INTERFACE_RELEASE_PILOT_2026-09-23.md`.
The complete machine-readable result is `summary.json` (SHA-256
`bb2c9fcb1954afa36dc0b92383239fa4d61e4c496cbd4bb737e1cc5fec4145db`).
The per-molecule decomposition is `qualitative_descriptors.json` (SHA-256
`66a83116a5a59f14a3094941530007587c2e942c1816cdee9a4656ca042629f7`).
Each of the 20 prompt-arm units contains every attempt record and molecule,
bound to the checkpoint, prompt, official evaluator, implementation hashes,
and the pinned Python/RDKit/NumPy/Torch environment.

## Attempt and committed-endpoint denominators

| Task | Permanent restriction | Release after coverage | Committed endpoint check |
| --- | ---: | ---: | ---: |
| Motif extension | 68/80 full successes; 74 commits | 70/80; 73 commits | 147/147 valid, connected, and fragment-preserving |
| Scaffold decoration | 77/80; 80 commits | 77/80; 80 commits | 160/160 valid, connected, and fragment-preserving |
| Superstructure | 39/40; 39 commits | 39/40; 39 commits | 78/78 valid, connected, and fragment-preserving |

An independent no-censoring check found 20 units x 20 attempted trajectories =
400 attempts, 385 committed endpoints, and 370 emitted task successes. The
attempt-aligned records reconstruct the separate committed and emitted lists
exactly. All 385 committed endpoints parse, sanitize, and are connected;
385/385 retain the fragment by the independent containment audit. The two
superstructure arms are attempt-by-attempt identical, as required when the
prompt declares no interface.

## Official-formula quality/diversity on committed molecules

These use the pinned IVG evaluator implementation but only 20 attempts per
prompt and one seed. For diversity, the unweighted mean of the four **within-
prompt** values is relevant; pooling different drug scaffolds obscures the
effect.

| Task | Arm | Quality (%) | Uniqueness (%) | Within-prompt diversity |
| --- | --- | ---: | ---: | ---: |
| Motif | Permanent | 7.50 | 82.72 | 0.6963 |
| Motif | Release | 17.69 | 97.50 | 0.7881 |
| Decoration | Permanent | 40.00 | 100.00 | 0.6210 |
| Decoration | Release | 31.25 | 100.00 | 0.7108 |

The decoration result is a **negative quality tradeoff**, not a clean win.
Quality fell in three of four decoration prompts and was unchanged in one;
diversity rose in all four. The official quality numerator counts *unique*
qualifying molecules (QED >= 0.6 and SA <= 4.0), divided by the number of
submitted molecules. Motif quality rose mainly because duplicate qualifying
molecules collapsed: 5 unique qualifying molecules under permanent restriction
versus 12 under release, even though mean QED and SA did not improve.

The committed-molecule decomposition makes the tradeoff explicit:

| Task | Arm | Mean QED | Mean SA | Mean heavy atoms | QED pass | SA pass | Joint pass (attempts) | Unique joint pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motif | Permanent | 0.443 | 3.375 | 15.4 | 13/74 | 59/74 | 13/74 | 5 |
| Motif | Release | 0.430 | 3.527 | 17.0 | 12/73 | 52/73 | 12/73 | 12 |
| Decoration | Permanent | 0.585 | 3.316 | 25.4 | 44/80 | 62/80 | 32/80 | 32 |
| Decoration | Release | 0.567 | 3.445 | 26.3 | 39/80 | 56/80 | 25/80 | 25 |

The first eight committed structures per arm were rendered without selecting
by quality: `baricitinib_motif_first8.png` (SHA-256
`9df04e60005c2c6ea4487b7f19c87a172d84df5a211c8485750e7f415ef99654`),
`baricitinib_decoration_first8.png` (SHA-256
`51fce766c90aa99e50733ec228f1b6d24143c30dbd0bfa1f3e758172c07e8d13`),
and `erlotinib_decoration_first8.png` (SHA-256
`98605cb583472781ecb03b98d829ebf32f0bb5d7f7c46dc29f1e60dc40810aa1`).
Visual inspection agrees with the descriptors: release explores more varied
peripheral growth, sometimes creating bulkier or unusual side chains. Both
arms still contain chemically valid but visually questionable molecules; this
pilot does not establish improved medicinal-chemistry realism.

## Decision and next test

Do **not** promote unrestricted post-coverage growth globally. It qualifies as
a useful diversity mechanism, but its decoration-quality cost is real on the
committed denominator. The prompt specification gives a generic way to separate
the regimes without a task/drug lookup: every released motif prompt has exactly
one declared interface, every decoration prompt has two to six, and
superstructure has none. A future, separately frozen test may release
post-coverage core growth only for a **single-interface** specification and
retain permanent restriction for multi-interface specifications. That is a
conditioning-level hypothesis, not a result from this pilot. It must be tested
at the full matched protocol before any paper claim.

The older benchmark document quotes GenMol V1 values. GenMol V2 is a different,
stronger comparator and must be bound to its exact public artifact/revision and
reported separately before headline comparison. No comparator claim is made
here.

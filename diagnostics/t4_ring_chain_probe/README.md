# Ring, modify, ring: short empirical capability probe

All **8/8 prescribed trajectories** completed both ring programs and the
intervening bond-order edit: 12 committed edits each, **96/96 selected edits
exactly replayed**. One of eight final two-ring products passed all endpoint
screens. This is capability evidence under a model-free descriptor law, not
autonomous learned-controller recovery or docking improvement.

| Source | Completed chains | Accepted first-ring completions | Accepted after modification | Accepted two-ring endpoints |
| --- | ---: | ---: | ---: | ---: |
| Original PARP1 seed0 | 4/4 | 4/4 | 3/4 | 1/4 |
| Current best feasible archive molecule, docking -9.7 | 4/4 | 0/4 | 0/4 | 0/4 |

Each completed ring program added one graph cycle. All final products therefore
have cycle-rank delta +2 relative to their source. There were eight distinct
final molecules, 22 distinct molecules among 24 option-completion records, and
seven distinct accepted molecules among eight accepted completion records.
Every completion had zero RDKit bridgehead and spiro atoms. Existing seven-rings
were retained in the sources; the newly requested rings were five/six-membered.

The negative result is mainly endpoint feasibility. Among all 24 completion
records, 16 failed original-seed similarity >=0.4, six failed QED >=0.6 and eleven
failed SA <=4 (overlapping counts). Four records also had cumulene flags after
the random local bond-order edit. These were retained for diagnosis and were
not treated as acceptable returned candidates. No endpoint threshold changed.

## Useful candidate pair

The original-seed, second-sequence, RNG-0 trajectory remained feasible while
adding a pendant six-ring, modifying its bond order, then adding a fused
five-membered N-containing ring. Exact states and all marks are in
`units/0-1-0.json`; canonical SMILES are metadata, not replay state.

| Endpoint | Heavy atoms | Cycle rank | QED | SA | Seed similarity | Predicted docking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| After first ring and modification | 25 | 4 | 0.936642 | 3.223640 | 0.532258 | -8.090621 |
| After second ring | 28 | 5 | 0.736337 | 3.485932 | 0.422535 | -8.423683 |

Second-ring endpoint:

`CN(C)Cc1ccc2c(c1)CNC(=O)c1c(C3=COCCC3)c3cc[nH]c3n1-2`

These are frozen predictor outputs, **not measured docking scores**. The lower
prediction for the two-ring molecule is a useful ranking hypothesis; it does not
establish improved binding or calibrated confidence. Neither endpoint was docked.

## Cost and provenance

- Producer: `54fa85fa9a7466314005d5210227593e84ebe19b`, clean worktree
  `/private/tmp/compose-winner-paths.W1n7sh`, strict preflight passed.
- Scope: `docs/T4_ANYTIME_OPTION_CREDIT.md` at that revision, SHA-256
  `40de95237bffd8762ce0bd786f24fbd15fd159eb13272ea1a19bf0e6d4a6a16c`.
- Result SHA-256: `f3a04caaa08ac729058f2751cdc756f73d34af1450632846b1e2746ddef6b189`.
- `result.json` binds the exact 51-call archive and previous production lock,
  75 local implementation files, complete source/sequence/seed configuration,
  software and SA assets. Per-trajectory receipts are payload-hashed and reusable.
- Molecular work including scoring and receipt publication: **5.5458565 seconds**.
  This excludes interpreter imports, input validation and runtime parity checks.
  375 option-kernel executor applications plus 96 selected-action replay calls;
  these are not a count of every nested internal micro-executor invocation.
- No R_theta enumeration, fitting, accelerator allocation or docking. One local
  process, integer exact graph arrays and float64 weights/predictions.
- Isolated overlay `/private/tmp/compose-t4-chemistry.hizM8Y`: RDKit 2024.03.5,
  NumPy 1.26.4, SciPy 1.13.1, Pillow 12.3.0; Python 3.12.9, macOS arm64.
  The project environment was not modified. This is not full Modal equivalence.
- Runtime checks matched all 51 saved training-feature rows exactly and all ten
  previous-production candidate property/prediction records within absolute
  tolerance 1e-12, before generating the new trajectories.
- Initial execution stopped during imports because the project's newer SciPy
  requires NumPy 2. No molecule was generated. Adding compatible SciPy only to
  the isolated overlay resolved this environment failure; no gate was relaxed.

Reproduction uses the committed script with positional arguments for the hashed
archive, previous lock and a new output directory, and
`PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:.` with the existing
`.venv/bin/python`. The old lock and archive remain in their existing diagnostic
directories. Do not duplicate generation to produce a different report.

## Decision

Stop treating repeated ordinary ring construction as the primary missing
capability. This probe demonstrates that capability on two inspected sources,
not universal applicability or useful learned probabilities. The next search
question is selection of parent, attachment site and follow-up edits while
preserving benchmark feasibility. These four warm-source trials do not prove
all continuations from the current best molecule fail.

A useful next fresh-oracle diagnostic is the feasible one-ring/two-ring pair
above, with both candidates locked before either score is observed and two
explicitly accounted calls. That experiment has not been launched. The separate
anytime-credit implementation has passed focused tests, but this prescribed
probe does not validate its autonomous learned-guidance performance. IVG winner
structures/paths/labels were not inputs to the probe, descriptors or scorer.

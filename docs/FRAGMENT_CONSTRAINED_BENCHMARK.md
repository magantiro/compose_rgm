# Fragment-constrained generation protocol, GenMol and InVirtuoGen

## Status and boundary

This document binds the zero-oracle COMPOSE adapter to the primary released
SAFE/GenMol benchmark inputs and the GenMol and InVirtuoGen evaluation
protocols. It also records a five-prompt, deterministic proposal-support smoke.
It does not authorize or report a scored or full 100-sample generated-molecule
experiment. It is isolated from T4 docking and PMO optimization.

The scientific problem is conditional molecular generation from retained
fragments. The primary output is a set of complete connected molecules that
contain the supplied fragment cores and complete their declared open attachment
sites. The claim to test later is whether one shared COMPOSE program controller
can generate valid, diverse, high-quality completions under these constraints.
The direct baselines are SAFE-GPT, GenMol, and InVirtuoGen. The declared input
support is the released ten-drug, five-task benchmark and COMPOSE's current atom,
bond, size, and connected-molecule support. No broader fragment-generation claim
is implied.

## Exact task

The benchmark uses ten drugs:

1. Baricitinib
2. Cyclothiazide
3. Eliglustat
4. Erlotinib
5. Futibatinib
6. Lesinurad
7. Liothyronine
8. Lovastatin
9. Maribavir
10. Spirapril

Each drug supplies inputs for five task labels:

| Task | Released constraint |
| --- | --- |
| Linker design | Two side-chain fragments, each with an open attachment point; generate a connected completion containing both. |
| Scaffold morphing | GenMol and InVirtuoGen reuse the linker-design inputs and, for their own reported rows, the same generated results. |
| Motif extension | One motif with an open attachment point; retain and extend the motif. |
| Scaffold decoration | One larger scaffold with one or more open attachment points; retain it and complete the declared sites. |
| Superstructure generation | One core substructure; generate a molecule containing the core. |

Attachment points are dummy atoms such as `[1*]`, `[11*]`, or `[17*]`.
Their isotope numbers are serialization labels, not atom addresses for
COMPOSE. The adapter removes the dummies to form an address-free retained-core
query, remembers their neighboring core roles, and requires every declared site
to gain an external heavy-atom neighbor. Linker fragment embeddings must be
non-overlapping. Superstructure inputs contain no fixed dummy site in the
released GenMol asset, so the adapter requires core containment and does not
invent an attachment position.

The released budget is 100 generations per drug and task. GenMol and
InVirtuoGen report the mean and standard deviation across three runs/seeds.
Thus one reported task row summarizes ten prompts and 1,000 attempted molecules
per run. This is not an expensive-oracle budget.

## Metrics

The published metrics are:

- **Validity:** parseable molecular outputs divided by 100 attempts for each
  drug, then averaged across drugs.
- **Uniqueness:** unique valid canonical molecules divided by valid molecules.
- **Diversity:** mean pairwise Tanimoto distance over unique valid molecules,
  using radius-2, 2,048-bit Morgan fingerprints without chirality in the TDC
  evaluator used by GenMol.
- **Quality:** unique valid molecules satisfying QED >= 0.6 and SA <= 4,
  divided by all 100 attempts. This denominator is important.
- **Central distance (GenMol):** mean Tanimoto distance from the original drug
  to the unique valid completions, using radius-2, 1,024-bit Morgan
  fingerprints. InVirtuoGen's main comparison omits this column.

Both released generators assume fragment satisfaction because conditioning is
forced during decoding. They do not report it as a separate metric. COMPOSE can
propose outside the condition, so the adapter additionally reports chemical
validity, constraint validity conditional on chemical validity, and strict
benchmark validity (chemical validity plus constraint satisfaction). Headline
COMPOSE validity must use the strict value; otherwise failure to retain the
prompt would be silently rewarded.

## Published comparator values

The table below transcribes InVirtuoGen Table 4, which deliberately uses the
updated **one-step** GenMol linker result rather than GenMol's original
two-step/rejection-filtered linker result. Validity, uniqueness, and quality
are percentages; diversity is a fraction. Values are mean +/- standard
deviation across three runs.

| Task | Method | Validity | Uniqueness | Quality | Diversity |
| --- | --- | ---: | ---: | ---: | ---: |
| Motif extension | SAFE-GPT | 96.10 +/- 1.90 | 66.80 +/- 1.20 | 18.60 +/- 2.10 | 0.562 +/- 0.003 |
|  | GenMol | 82.90 +/- 0.10 | 77.50 +/- 0.10 | 30.10 +/- 0.40 | 0.617 +/- 0.002 |
|  | InVirtuoGen | 68.97 +/- 0.759 | 96.83 +/- 0.290 | 39.27 +/- 1.078 | 0.620 +/- 0.005 |
| Linker design | SAFE-GPT | 76.60 +/- 5.10 | 82.50 +/- 1.90 | 21.70 +/- 1.10 | 0.545 +/- 0.007 |
|  | GenMol one-step | 16.70 +/- 0.20 | 97.80 +/- 0.50 | 4.30 +/- 0.40 | 0.530 +/- 0.002 |
|  | InVirtuoGen | 60.37 +/- 0.573 | 84.76 +/- 1.620 | 22.33 +/- 1.250 | 0.520 +/- 0.004 |
| Scaffold morphing | SAFE-GPT | 58.90 +/- 6.80 | 70.40 +/- 5.70 | 16.70 +/- 2.30 | 0.514 +/- 0.011 |
|  | GenMol one-step | 16.70 +/- 0.20 | 97.80 +/- 0.50 | 4.30 +/- 0.40 | 0.530 +/- 0.002 |
|  | InVirtuoGen | 60.37 +/- 0.573 | 84.76 +/- 1.620 | 22.33 +/- 1.250 | 0.520 +/- 0.004 |
| Superstructure | SAFE-GPT | 95.70 +/- 2.00 | 83.00 +/- 5.90 | 14.30 +/- 3.70 | 0.573 +/- 0.028 |
|  | GenMol | 97.50 +/- 0.90 | 83.60 +/- 1.00 | 34.80 +/- 1.00 | 0.599 +/- 0.009 |
|  | InVirtuoGen | 75.70 +/- 0.898 | 99.41 +/- 0.157 | 27.43 +/- 0.953 | 0.730 +/- 0.001 |
| Scaffold decoration | SAFE-GPT | 97.70 +/- 0.30 | 74.70 +/- 2.50 | 10.00 +/- 1.40 | 0.575 +/- 0.008 |
|  | GenMol | 96.60 +/- 0.80 | 82.70 +/- 1.80 | 31.80 +/- 0.50 | 0.591 +/- 0.001 |
|  | InVirtuoGen | 90.70 +/- 0.616 | 88.58 +/- 1.130 | 36.37 +/- 1.096 | 0.560 +/- 0.003 |
| Average | SAFE-GPT | 85.00 +/- 1.788 | 75.48 +/- 1.773 | 16.26 +/- 1.031 | 0.550 +/- 0.006 |
|  | GenMol | 62.08 +/- 0.242 | 87.88 +/- 0.436 | 21.06 +/- 0.263 | 0.570 +/- 0.002 |
|  | InVirtuoGen | 71.22 +/- 0.399 | 90.87 +/- 0.445 | 29.55 +/- 0.813 | 0.590 +/- 0.001 |

GenMol's own original Table 2 reports a two-step linker/scaffold-morphing
configuration with 100% validity, 83.7% uniqueness, 21.9% quality, and 0.547
diversity. InVirtuoGen explains that this two-step result uses rejection
filtering and therefore selects the one-step GenMol row above for its main fair
comparison. These two GenMol regimes must not be mixed.

## Primary sources and frozen assets

| Evidence | Version / commit | Material asset and SHA-256 | License |
| --- | --- | --- | --- |
| GenMol paper, arXiv:2501.06158v3 | v3 HTML cached locally | `diagnostics/genmol_paper/p.html`: `4060f49b536b07f5dbd2594b3e3dd42e553bb605eaf50333e01c2d6211656a10` | Paper terms |
| GenMol official repository | `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` | upstream `data/fragments.csv` and vendored copy: `a4fb8357d0f1102cbdc8d79d802e15f66a59a9722c0b7125ce693fe7a29872a9` | Code Apache-2.0; SAFE-DRUGS CC BY 4.0 |
| InVirtuoGen paper, arXiv:2509.26405v2 | v2 | Table 4 and Sec. 3.2 | Paper terms |
| InVirtuoGen official repository | `b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb` | `references/frags_downstream.csv`: `dceeec39f389dda928a210694449948645be73d12da0ad93bc843ba1f2bb5975`; `evaluation/downstream.py`: `ab34f586b41f1ad6e21cd933b08e7ff86eb44711c9c9a4c31c29f894cc9c7b50` | CC BY-NC-SA 4.0 |
| SAFE official repository | `d162d23e3e8c8a77b54c559e88a35d08b3b1d1b5` | `DATA_LICENSE`: `87e3070926ffc1c1b4b7c97efe3b6882aeadd2adc82f6f085721b86665cc9a90` | CC BY 4.0 |

Primary URLs:

- https://arxiv.org/abs/2501.06158
- https://github.com/NVIDIA-BioNeMo/genmol
- https://arxiv.org/abs/2509.26405
- https://github.com/invirtuolabs/InVirtuoGen_results
- https://arxiv.org/abs/2310.10773
- https://github.com/datamol-io/safe

## Reproducibility caveats found in the released code

1. InVirtuoGen creates superstructure attachment-point prompts with Python
   `random.choice` before setting its per-seed random seeds. The exact prompts
   used in the paper are not persisted in the repository, so byte-exact replay
   of that model's superstructure conditioning is not possible from the release
   alone. The underlying ten core substructures are static and are what this
   adapter binds.
2. A caption/comment in the released InVirtuoGen evaluator says scaffold
   decoration shares linker results. The executable code actually copies the
   linker result to **scaffold morphing**, consistent with the paper table and
   task definitions. This adapter follows the executable behavior and paper.
3. The current InVirtuoGen repository contains result JSON files whose numbers
   differ from the paper's Table 4. Published comparator claims must use the
   paper table above, not those later repository runs.
4. The two GenMol prompt CSVs and the InVirtuoGen translation are not
   byte-identical. They encode the same static non-superstructure constraints
   with different dummy numbering, stereochemistry handling, and row order.
   The adapter uses the original named GenMol asset as the canonical manifest.

## What is implemented, and what is not

`compose_v4.benchmark.fragment_constrained` now provides:

- typed loading of all 50 explicit drug/task prompts;
- fragment parsing independent of absolute dummy labels;
- strict candidate constraint checks;
- official metric denominators and fingerprint settings;
- a separate constraint-validity diagnostic required for COMPOSE;
- fail-closed handling of schema drift and sample-count drift.

`compose_v4.benchmark.fragment_constrained_runner` now provides the smallest
prompt-conditioned COMPOSE proposal gate needed to exercise all five labels.
It conditions only on task, released fragment strings, and a generic variant
index. The original full drug remains an evaluation reference and never enters
proposal construction.

For linker design and scaffold morphing, the runner caps both fragments with H,
selects the larger retained component as a valid connected source, and compiles
a generic one-to-three-carbon bridge plus the second retained component as one
dependency-aware edit program. It never commits the disconnected pair. Every
primitive is executed through the existing Editing-V2 executor, and every
committed state is checked for validity and connectedness. Each retained
fragment is rechecked in every state after its construction/attachment lock.
Motif extension, scaffold decoration, and superstructure generation preserve
their one retained core and add generic carbon extensions at declared sites (or
the first deterministic hydrogen-bearing site for superstructure inputs, which
declare no site).

The runner is deliberately not a learned controller and does not evaluate QED,
SA, similarity, docking, or another objective. It may abstain on active-atom,
primitive, block, transient-valence, or source-support limits; those limits are
not relaxed after seeing a prompt outcome. A full benchmark still requires a
self-hashed scored-launch contract, 100 attempts per prompt, three declared
runs, and the frozen evaluator above.

The bounded smoke loads all 50 prompts, freezes the first manifest prompt for
each task before execution, and attempts exactly five proposals:

```bash
PYTHONPATH=src ../../.venv/bin/python \
  tools/fragment_constrained_proposal_smoke.py
```

Its authoritative output is
`diagnostics/fragment_constrained_proposal_smoke_v1/result.json`. This is a
zero-oracle executor/constraint support check, not a comparator result.

The first frozen all-prompt audit then attempted one deterministic proposal for
each of the 50 manifest prompts under the same 40-atom, 32-primitive, and
eight-block support:

```bash
PYTHONPATH=src ../../.venv/bin/python \
  tools/fragment_constrained_proposal_panel.py \
  --contract configs/fragment_constrained_proposal_panel_v1.json
```

It completed 49/50 proposals, with 49/49 constraint-valid completed endpoints
and 49/50 exact-valid execution yield. The single preserved abstention was the
Futibatinib superstructure prompt: the v1 site selector chose an atom bearing an
explicit chiral hydrogen and its generic extension was over-valent. This is a
runner defect found by the frozen panel, not evidence that the prompt itself is
unsupported. The v1 contract, full receipts, per-task metrics, and failure
detail are under `diagnostics/fragment_constrained_proposal_panel_v1/`.

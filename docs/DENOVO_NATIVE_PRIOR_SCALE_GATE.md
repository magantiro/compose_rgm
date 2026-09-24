# COMPOSE-native de novo prior scaling gate

Status of this document: preparation only. It does not itself authorize model
training, a scored benchmark, checkpoint promotion, or a change to a live
campaign. The 2026-09-23 user request to pursue a stronger molecular prior
authorized the split-first preparation and implementation checks below.

Subsequent status, without changing this preparation gate: the separate
self-hashed training contract is
`configs/denovo_native_prior_scale_training_v1.json` (payload SHA-256
`62eea12e463b81384ca07c8a5ffa09c2eedbdbdad172546026d91a5f5d06979d`).
The two CPU-only path-compilation launches and their exact runtime handles are
recorded in `diagnostics/denovo_native_prior_scale_v1/compile_launches.json`.
They are preparation for the matched 50,000-versus-216,149 training comparison,
not trained checkpoints or evidence of improved quality. The contract, rather
than this preparation document, governs any later support compilation, training,
evaluation and promotion decision.

## Identity

- Scientific problem: the post-ring-marginal C/N/O/F de novo checkpoint emits
  valid and diverse molecules but rarely passes the official joint QED/SA
  quality criterion. Its training distribution and sampling process are much
  smaller than the pretrained priors used by GenMol and InVirtuoGen.
- Primary output: a state- and time-conditioned marked rate law over complete
  legal molecular rewrites; exact execution commits only connected supported
  molecular graphs. This is not SAFE or token generation.
- Claim under test: additional split-clean molecular training data improves
  the quality-diversity frontier and held-out chemical-space coverage while
  preserving exact validity. This is a hypothesis, not an observed result.
- Setting: first compare nested 50,000 versus larger C/N/O/F-neutral training
  sets under identical model architecture, source prior, executor, support,
  optimizer schedule, candidate budget, and held-out molecules. Only after
  that causal scale gate may a separate broad-organic de novo model be proposed.
- Data/support: the source is the hash-bound GuacaMol subset
  `guacamol_subset_500000_seed0.smiles`, SHA-256
  `70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791`.
  The first comparison is neutral, connected C/N/O/F molecular graphs with at
  most 40 active atoms, no stereochemical claim and no charge-changing action.
  The historical corpus census reports 225,149 eligible molecules; preparation
  must independently verify the count and preserve every excluded source row.

## Evidence and reason for a new split

The historical Lineage B checkpoint used 50,000 C/N/O/F-neutral molecules and
the selected step-1,000 weights. The post-ring-marginal C1 sample reports
17/150 joint-quality successes, 150/150 valid committed endpoints, 150/150
unique endpoints and diversity 0.8896577814. These are a small diagnostic
sample, not an official benchmark result. A matched 24-pair F-restatement
pilot failed its predeclared signal and cannot replace a learned prior.

`load_cnof_corpus_split` shuffles eligible molecules and slices training,
validation and test consecutively. Changing its `train_size` therefore changes
the held-out identities. The scaling comparison must instead freeze one ordered
training reservoir, one validation set, one IID test set, and one disjoint
scaffold-group test set before selecting either training size. The 50,000 arm
must be an exact prefix/subset of the larger arm. All ring catalogs, source-size
statistics, template frequencies and model preprocessing must be derived from
the selected arm's training molecules only. The held-out tests must not guide
architecture, sampler temperature or checkpoint selection.

## Preparation acceptance criteria

1. Read the exact source asset and verify its physical SHA-256 before parsing.
   Preserve source line numbers, first-seen canonical identity, canonical
   duplicate reconciliation, and explicit exclusion reason codes.
2. Freeze one deterministic seed derivation, group policy and ordered split.
   No canonical identity may cross a partition. No scaffold key used for the
   scaffold test may appear in either training arm or validation. If the
   acyclic scaffold or another dominant group prevents the requested split,
   fail and report its count rather than split the group silently.
3. Serialize a versioned manifest with source, code, RDKit, seed, scope,
   counts, exclusions, partition digests and a payload hash. Verify byte-stable
   regeneration on a small fixture and fail on a source-hash mismatch.
4. Training must consume the prepared artifact directly. It may not rescan or
   repartition the source corpus on a GPU worker. The same validation and test
   identities must be proven by hashes in both arm checkpoints.
5. Report training-family/scaffold imbalance and choose weighting before fitting.
   Do not silently drop rare but supported topology to reduce memory or cost.

After the preparation implementation is committed, the CPU-only materializer is:

```bash
PYTHONPATH=src python scripts/prepare_frozen_cnof_prior_split.py \
  /guacamol/guacamol_subset_500000_seed0.smiles \
  diagnostics/denovo_native_prior_scale_v1/frozen_split.json \
  --expected-source-sha256 70526d92f1f08d8e292cb31218f81b6924a2182f772c43348015110669d47791 \
  --source-access-basis 'user-controlled GuacaMol corpus volume; redistribution not assumed' \
  --expected-eligible-unique 225149
```

The source path above is a mounted-volume path from the historical corpus
contract, not a local file in this worktree. The command must fail if that
asset is missing or differs physically. The trainer's opt-in
`--frozen-cnof-split-manifest` path consumes this artifact directly and records
the manifest, source, training-prefix and held-out partition hashes in its
checkpoint. On an accelerator host, a different mount path for the corpus is
accepted only after its full physical SHA-256 equals the frozen source hash;
the runtime path and observed hash are recorded separately. The default
historical loader remains unchanged. The later request to learn a comparable
COMPOSE-native prior authorizes preparing the two matched recipes, but neither
this split nor those recipes authorizes an accelerator launch without the
separate self-hashed training contract.

For the matched scale comparison, both arms derive the empirical carbon-tree
source-size law from the identical first 50,000 frozen training molecules.
The larger arm changes target training density, not its starting-size law. The
common source-prior prefix digest is recorded in both checkpoint identities.

## Prospective model and evaluation gates

- Compare the new matched 50,000 arm with the larger arm at matched architecture,
  optimizer updates, source coupling, batch, validation cadence and sampling
  budget. Also report data-exposure-adjusted and wall/accelerator-time views.
- Select checkpoints only from validation canonical-successor likelihood and
  registered rollout safeguards, never from test or raw mark-level loss alone.
- Sweep a small predeclared sampler setting grid on development molecules.
  Select the operating point by official joint quality subject to frozen
  committed validity 1.000, uniqueness and diversity floors, plus FCD,
  scaffold/topology coverage and molecule-grid safeguards. Preserve the
  unchanged sampler as a causal control.
- Evaluate the frozen selection once on IID and scaffold-held-out tests. Persist
  every attempt, endpoint, seed, exact trajectory and failure reason. Report
  all-step validity separately from committed validity, joint QED/SA quality,
  diversity, novelty, FCD, held-out recall and relevant ring/host marginals.
- Do not promote a model on quality alone if diversity or topology coverage
  collapses. A non-improving scale result is a negative result and directs the
  next diagnosis toward source/transport or model capacity, not a relaxed gate.

## Comparator boundary

GenMol and InVirtuoGen use pretrained fragment-string models on roughly a
billion molecular examples and use empirical length and sampler controls.
Their reported de novo quality-diversity frontiers are external references,
not training labels. Import neither their SAFE representation nor their
post-decode repair/filtering into COMPOSE's exact committed-state guarantee.
Fragment-constrained generation remains a separate exact-interface problem;
the failed single-interface held gate is preserved and cannot be called solved
by de novo pretraining.

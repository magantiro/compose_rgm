# T4 Compositional Structural-Subgoal Generator, Attempt 1

## Outcome

The frozen zero-oracle gate passes at K=128 on improved generated component
support, with important negative results. The learned joint generator raises
source-balanced granular held-component coverage from 46.25% to 59.77% and
produces a mean 92.67 novel patches per source versus 76.60 for the uniform
grammar. Both arms emit 128 unique endpoints on all 15 held sources and every
accepted candidate exactly realizes.

Neither policy recovers a complete teacher endpoint or a radius-2 teacher
transformation at K=8, K=32 or K=128. The learned arm recovers 2/147 exact
structural patch components, while uniform recovers none. Learned coverage is
worse at K=8 and K=32, improves only at K=128, loses 8.64 percentage points of
attachment-edge coverage at K=128, and has lower legal compile coverage (93.52%
versus 99.90%). These negatives are preserved in `result.json`.

## Artifacts

- `candidate_lock.json.gz`: teacher-free autonomous lock, 15 sources, 30 pools,
  128 candidates per pool, zero shortfall.
- `result.json`: complete fold-resolved, source-balanced, instance-weighted,
  family-level, work and frozen-gate metrics.
- `fold_*/runtime_checkpoint.json.gz`: numeric fold-specific model statistics
  and generic support only.
- `fold_*/source_manifest.json`: held source inputs without teacher fields.
- `fold_*/evaluation_manifest.json.gz`: separately sealed held teachers and
  fold-specific training whole-patch vocabulary.
- `fold_*/fit_report.json`: training counts, balance summaries, conversion
  coverage and abstentions.

## Provenance and scope

- Contract payload SHA-256:
  `a91272c148d7b6421f4209a8aadae031fd52d783ac8904e8ac4544d79262c151`
- Candidate lock SHA-256:
  `f533e70d940ff191df007b28107f12eed9bce6abb43f44f7afc91d47422afd28`
- Result SHA-256:
  `5d084456e5eb9e749751a201045c6d753430753193fa61d4e681b42ba4bb2452`
- Scientific implementation revision:
  `7ecc138373978421feaab27cd3ae7a5e6a6ebea4`
- Sealed realizer SHA-256:
  `c803e3c3118e262a8b6fc4df1cfb9f8fed2934d2dc6fb47431ee4e319b64ec5f`
- Sealed extractor SHA-256:
  `ba603dbfa82a3130761146498510160d9dfcb74d3aa5b44298afe6bfefe59823`
- Cost: zero oracle calls, zero docking calls, zero Modal launches, one CPU
  worker.
- First candidate-lock wall time: 3,080.16 seconds.
- Independent candidate-lock wall time: 3,082.24 seconds.
- Evaluation wall times: 372.94 and 368.65 seconds.
- Determinism: the complete independent candidate-lock rerun was byte-identical
  to the original and had the same physical SHA-256.

The exact local-fiber optimization changes only when nonlocal actions are
discarded. The optimized fiber is covered by a decision-equivalence regression;
all scientific support and budgets remain unchanged. Full byte-for-byte
candidate-lock determinism verification passed. Final repository checks are
reported with the milestone handoff.

This result is offline held-source generation evidence. It is not molecular
utility, docking improvement, route recovery, autonomous optimization or an IVG
comparison. No scored pilot was launched.

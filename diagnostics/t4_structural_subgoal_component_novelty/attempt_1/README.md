# T4 Structural-Subgoal Component-Novelty Result

## Outcome

The grouped-source audit rejects whole-patch retrieval as a transfer mechanism
but supports a compositional graph generator. Across the three frozen folds,
none of the 147 held structural subgoals had an exactly isomorphic whole patch
in its training-fold vocabulary. Small generic constituents transferred much
more often: bond-attribute tokens covered 100.0% of held instances, dependency
tokens 99.7%, target-topology graphlets of at most three vertices 98.8%, atom
transition tokens 96.4%, source graphlets 84.6%, target attributed graphlets
82.4%, and attachment-edge tokens 82.3%.

This is not a pure recombination result. Only 11 of 147 held subgoals had every
granular component observed in the corresponding training fold. The other 136
contained at least one unseen granular component. The evidence therefore favors
a joint conditional generator that can synthesize new local graphs and
attachments from transferable low-level structure. It does not support a larger
finite template bank or independent marginal classifiers.

## Primary T4 measurements

All counts hold out complete source groups before constructing vocabularies.
Every one of the 15 source groups is held exactly once.

| Component family | Familiar / held instances | Coverage | Held unique coverage |
| --- | ---: | ---: | ---: |
| Whole patch | 0 / 147 | 0.0% | 0.0% |
| Whole source-region motif | 10 / 147 | 6.8% | 7.4% |
| Source graphlets, at most 3 vertices | 2,794 / 3,301 | 84.6% | 62.7% |
| Whole target topology | 45 / 147 | 30.6% | 14.7% |
| Target-topology graphlets, at most 3 vertices | 3,614 / 3,657 | 98.8% | 89.8% |
| Whole atom-attribute bundle | 0 / 147 | 0.0% | 0.0% |
| Atom-attribute tokens | 1,636 / 1,697 | 96.4% | 84.1% |
| Whole bond-attribute bundle | 9 / 147 | 6.1% | 6.3% |
| Bond-attribute tokens | 1,761 / 1,761 | 100.0% | 100.0% |
| Whole attachment pattern | 54 / 147 | 36.7% | 14.8% |
| Attachment-edge tokens | 181 / 220 | 82.3% | 73.5% |
| Whole dependency motif | 27 / 147 | 18.4% | 15.6% |
| Dependency tokens | 717 / 719 | 99.7% | 92.3% |
| Target radius-1 fragments | 703 / 1,180 | 59.6% | 38.3% |
| Target radius-2 fragments | 165 / 1,180 | 14.0% | 10.1% |
| Target attributed graphlets, at most 3 vertices | 3,015 / 3,657 | 82.4% | 53.0% |

The per-subgoal all-component rates provide an additional boundary on the
coverage interpretation:

| Granular family | Subgoals with every emitted component familiar |
| --- | ---: |
| Atom-attribute tokens | 100 / 147 |
| Bond-attribute tokens | 147 / 147 |
| Attachment-edge tokens | 119 / 147 |
| Dependency tokens | 145 / 147 |
| Source graphlets, at most 3 vertices | 18 / 147 |
| Target-topology graphlets, at most 3 vertices | 123 / 147 |
| All six granular families simultaneously | 11 / 147 |

No component family was marked degenerate by the preregistered rule requiring a
single training class to exceed 90% of instances. Coverage and train-vocabulary
transfer precision remain separate in `result.json`; high held coverage does
not imply every training component transfers.

## Auxiliary conversion

The remaining 69 Full-146 programs supplied 69 origin applications. Fifty-five
had an exact context binding, executed, and converted into 72 structural
subgoals. Their structural conversion precision was 55/55. Fourteen applications
abstained because no exact origin-context binding was found.

The PMO dependency-region corpus contains 184 routes, including 106 within the
32-primitive runtime limit. The unchanged extractor converted 85 routes into
115 structural subgoals; all 85 reconstructed their target exactly. Ninety-nine
routes abstained on a known `TypeError` while ordering structurally equivalent
retained and deleted roles. This is an extractor coverage limitation, not
evidence that those PMO transformations are chemically incompatible.

Within that 85-route subset, adding PMO components to each T4 training fold
increased instance-weighted held T4 coverage as follows:

| Family | T4 train only | T4 + converted PMO |
| --- | ---: | ---: |
| Source graphlets, at most 3 vertices | 84.6% | 87.2% |
| Target-topology graphlets, at most 3 vertices | 98.8% | 99.4% |
| Atom-attribute tokens | 96.4% | 98.4% |
| Attachment-edge tokens | 82.3% | 84.5% |
| Target radius-1 fragments | 59.6% | 64.0% |
| Target radius-2 fragments | 14.0% | 15.3% |
| Target attributed graphlets, at most 3 vertices | 82.4% | 86.0% |

This is measured compatibility and coverage only. It is not a joint-policy fit
or evidence of autonomous PMO-to-T4 transfer.

## Interpretation and next decision

Measured evidence supports the following diagnosis:

1. Whole held patches are outside whole-template support by construction.
2. Most primitive attributes and small topology/dependency motifs are familiar.
3. Complete source regions, joint attachments, and larger attributed target
   neighborhoods often remain novel.
4. A future generator should select an address-free source region and jointly
   generate target topology, atom/bond attributes, attachments, and dependency
   structure. Independent marginal heads would ignore the measured coupling
   problem.
5. PMO is compatible with the same representation for the exactly converted
   subset and modestly expands held T4 component coverage. Full use requires a
   separately authorized extractor repair and split-first joint-training
   contract.

The next authorized scientific gate should measure autonomous, held-source
generation of exactly realizable structural patches, with transformation recall,
diversity, uniqueness, and realization precision. Teacher-route identity remains
a diagnostic. Molecular utility requires a separate score-blind candidate lock
and scored launch.

## Provenance and verification

- Result: `result.json`
- Result SHA-256: `bdd2acaac63b274dc35346ce14bc6a203b62316037056ddbda83b18fb910be0a`
- Payload SHA-256: `bc2549f5f376586d64881969503e91c17ee29a7c00a85e5f59dc7f028b1619f8`
- Implementation revision: `fcc3486b83457f04255ef74eb23a095861a50812`
- Contract payload SHA-256: `d596975381ed99109f80c4f263f3560ecc36b8dbd93fe15bb5da9a2fa4464122`
- Cost: zero oracle calls, zero docking calls, zero Modal launches, zero model fits, one CPU worker
- Measured wall time: 103.0 seconds
- Determinism: an independent clean-worktree rerun at the same revision was
  byte-identical, with the same result SHA-256.
- Focused verification before launch: 21 tests passed; Ruff passed; Black
  formatted the new files; `git diff --check` passed.
- Repository-wide verification was attempted once at the milestone boundary.
  It encountered numerous failures and errors outside the focused audit tests
  and was stopped at 58% at the user's direction rather than spending further
  time enumerating an already-red unrelated suite. It is not recorded as
  passing, and the repository-wide definition-of-done gate remains unsatisfied.

The first direct-file invocation failed before corpus loading because the
repository root was absent from Python's module path. It wrote no result and
made no oracle call. The authoritative run used the module entry point:

```text
PYTHONPATH=src:. <python> -m tools.t4_structural_subgoal_component_novelty \
  --pmo-corpus <hash-bound-pmo-corpus> \
  --pmo-result <hash-bound-pmo-result> \
  --output diagnostics/t4_structural_subgoal_component_novelty/attempt_1/result.json
```

No compiler, sealed structural realizer, T4 artifact, PMO source artifact, or
live run was modified.

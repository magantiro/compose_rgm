# Submitted ICLR 2027 paper: evidence map

This directory identifies the owner-supplied 35-page submitted PDF by SHA-256
and maps its main experimental results to immutable Git artifact revisions. It
does **not** contain a substitute PDF, launch authorization, or an assertion
that a fresh clone can reproduce every experiment. The exact identities and
claim boundaries are in [manifest.json](manifest.json).

Check the local PDF and all source-artifact bytes without a network call:

```bash
python3 tools/verify_submission_lineage.py \
  --paper /path/to/19340_COMPOSE_Molecular_Genera.pdf
```

This command verifies *identity*, not the arithmetic or scientific validity of
the results. It needs a full local Git history containing the referenced
revisions. A source export or shallow clone cannot pass the historical-object
checks until the pinned small reductions are packaged directly. A missing
historical commit is reported as missing, never replaced by a similarly named
file in the current tree. The PDF is supplied by the user and need not be
distributed with the code.

## Reproduction status

| Result | What is locally available | What remains |
| --- | --- | --- |
| Reference likelihood | Stored summary | Raw transition corpus and checkpoint access |
| Fragment benchmark and ablations | Hash-verified saved-result reducer in [`experiments/fragments/`](../fragments/) | Full-generation assets and evaluator access |
| QED editing | Final reduction on a historical revision | Bring the reduction into the release path; external trajectories and checkpoint |
| PMO-1K | Per-seed metric file on a historical revision | Raw-receipt verification, producer path, and JNK3 reconciliation |
| PMO A/B | Original 14-completed-cell reduction on a historical revision | Packaged reducer and raw-receipt provenance; preserve failed and unfinished runs |
| T4 docking | Frozen table artifact on a historical revision | Packaged reducer, raw-receipt provenance, docking assets, and zero-call fallback labeling |

The 14-cell A/B comparison is the one printed in the submitted paper. The
historical reduction's status is provisional because other campaigns were
still unfinished at the time. A later 16-cell result is a different analysis,
not a silent replacement for the submission. Likewise, the later T4 replicate
campaign is not the producer of submitted Table 3.

This map is deliberately task-specific rather than a generic `reviewer/`
package. Its next release gate is to provide offline reductions and exact
asset-access instructions under each experiment directory, run them from a
fresh clone or source export, and record any row that cannot be independently
reproduced. No scored experiment or manuscript edit is implied by this map.

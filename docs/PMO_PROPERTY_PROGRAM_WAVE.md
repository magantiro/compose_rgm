# PMO learned-property program wave

## Problem, output and claim

This wave tests whether the executable winner-informed program recipe transfers
from target-defined PMO objectives to three learned-property objectives. The
primary output is a locked set of exact COMPOSE edit programs plus a complete,
counted task-oracle ledger.

The proposed claim is narrow: starting from the same unrelated exact molecular
root, COMPOSE can execute fifteen unique programs around one publicly reported
high-scoring structure per task, and the resulting early top-ten area under the
curve (AUC) can be compared with IVG's reported values. The public structures
are answer-known development supervision. This is not blind optimization or a
held-out PMO evaluation.

## Tasks, source evidence and oracle identity

The wave contains `qed`, `gsk3b` and `jnk3`. Each anchor is transcribed from the
first structure in the corresponding official IVG `top_smiles_cuda:0.pdf` panel.
The contract pins each panel and both IVG comparison CSVs by SHA-256 at official
repository commit `b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`.

QED uses the pinned PyTDC 0.3.6 implementation. GSK3B and JNK3 use the existing
flat-tree numpy evaluators because their scikit-learn 0.21.3 pickles cannot be
safely loaded directly on the modern runtime. During preparation for this wave,
the official TDC downloads had SHA-256 values
`18d1cc9bb9498e4bae0755842080558cb2d0444ecb51beffa0ef58a6d760b74b`
and `5c7548343996a22d9a7aae8a91f34e4bc2e5e6e5a6d9c2decce3b164459af52b`.
These exactly match the pickle identities recorded by the frozen GSK3B and JNK3
extractions. The scored implementation verifies the extraction manifest and
flat-tree parameter hashes before evaluation.

DRD2 is deliberately excluded. Its newly downloaded TDC pickle hash is
`bf6d92d9bc1bef7efa688e14855c707e6ab7b43ec2c2f2ef28deabdc2a53480a`,
whereas the local frozen SVM manifest records
`dbc473fca922c834dbaee6eaba832caaff26d4f891734078fb1af359a111100f`.
No current parity artifact proves these are decision-equivalent. That gap must
be resolved before a DRD2 result is called authoritative.

The first scoring prelaunch failed before reserving a query because the pinned
PyTDC source was imported before its audited `rdkit.six` compatibility adapter.
The replacement implementation installs compatibility first and consumes
`curriculum_v2.json`; the original zero-call curriculum remains preserved as
`curriculum.json`.

## Frozen procedure

1. Load the same previously charged, unrelated neutral COMPOSE root and verify
   its physical hash.
2. Compile and replay a complete route from that root to the published anchor.
3. Without task-oracle calls, enumerate deterministic supported one-edit anchor
   variants using the same neighborhood rule as the completed exact-target wave.
4. Lock the first fifteen unique exact-replayed program endpoints, including the
   published anchor.
5. Abort before scoring unless all 45 programs replay and are unique within task.
6. For each task, score the unrelated root followed by its fifteen locked
   endpoints. Reserve and complete each query once. Never retry an ambiguous
   started query or replace a failed endpoint.
7. Compute official PMO top-ten AUC with frequency 100, denominator 10,000 and
   `finish=True`. Report exact margins against both IVG regimes.

The ceiling is 48 calls. Support remains neutral charge-preserving graphs with
at most 40 active atoms in 48 persistent slots. Candidate construction uses no
learned reference, surrogate selector, hidden prescreen or task score.

## Baselines, decision rule and limitations

The primary external baselines are the means of IVG's official three-run
no-prescreen and prescreen CSVs. A task beats a comparator only when its measured
official AUC is strictly higher. Best endpoint and final top-ten values are
reported separately.

The same deterministic neighborhood rule is the causal consistency control
against the completed exact-target wave. This wave does not isolate the value of
the public anchor from the executor, because the anchor is intentionally the
development supervision. Failure to compile fifteen candidates, oracle errors,
ties and negative margins remain first-class outcomes.

The anchor neighborhoods are narrow and answer-known. A positive result would
show executable recovery and local transfer around public solutions, not that a
general controller independently discovered those solutions. DRD2, isomer,
median, scaffold-hop and remaining MPO tasks require separately qualified waves.

## Measured result, 2026-09-13

All 48 locked queries completed with no retries or replacements. The result was
mixed: JNK3 exceeded the IVG no-prescreen comparator, but no task exceeded the
prescreen comparator.

| Task | COMPOSE AUC | IVG no-prescreen | Margin | IVG prescreen | Margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSK3B | 0.863308800000 | 0.951550908576 | -0.088242108576 | 0.987755000000 | -0.124446200000 |
| JNK3 | 0.882293600000 | 0.825432740059 | +0.056860859941 | 0.897834849848 | -0.015541249848 |
| QED | 0.930267966546 | 0.942328202796 | -0.012060236250 | 0.943446030900 | -0.013178064354 |

The exact anchors reproduced their panel rewards: GSK3B 1.00, JNK3 0.98 and QED
0.948442. Generic one-edit neighborhoods did not preserve those values reliably.
GSK3B's final top-ten mean was 0.864, JNK3's was 0.883 and QED's was 0.931013.
The result therefore rejects the hypothesis that one high-scoring public anchor
plus a generic local neighborhood is sufficient to beat the stronger IVG
prescreen results on these learned-property objectives.

The next justified development step is to use the complete public top-molecule
panels as multiple structurally distinct anchors, or to derive task-appropriate
program variations from their common features. It is not justified to call the
generic one-edit rule successful on GSK3B/QED or to hide the negative margins.
The authoritative payload SHA-256 is
`28f7c6e7703051bf40e45e25535c5a53b5cd9fbdca85d5d4ffa5cb9c9a96b642`.

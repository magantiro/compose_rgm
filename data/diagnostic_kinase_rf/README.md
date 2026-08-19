# Kinase RF actives — DIAGNOSTIC EVIDENCE ONLY

Rescued from an ephemeral session scratchpad on 2026-08-19. A session directory
had already been deleted once that day, and `scripts/task3_offmanifold_check.py`
still points at that (now unreproducible) path.

`kinase.tsv` — 104,258 rows, targets `jnk3` (50,923; 923 active) and `gsk3b`
(53,334), columns `target / is_active / is_train / smiles`. sha256 in
`kinase.tsv.sha256`.

## THIS FILE MUST NOT ENTER THE InversionGNN BENCHMARK LANE

Not as training data, not as initialization, not as a seed or a filter.

The InversionGNN task-training regime is **ZINC molecules + oracle labels, on a
fixed budget**. These are curated, experimentally-determined kinase actives.
Using them anywhere in that lane would inject exactly the biological prior the
comparison is meant to measure, and the resulting number would be meaningless
while still looking plausible.

Legitimate uses: oracle orientation/parity checks (does a JNK3-active actually
score high?), MOLLEO basin-geometry diagnostics, and sanity panels. It was used
this way for the MOLLEO substrate work, which is documented and is not a
benchmark claim.

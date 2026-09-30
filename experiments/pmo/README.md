# Submitted PMO structured-proposal ablation

The submitted paper compares COMPOSE's structured proposer (A) with
length-matched uniform legal edit chains (B) on **14 completed matched
objective-seed cells**. This task entry point recomputes the saved-cell
arithmetic for Tables 18, 19, and 21 and the descriptive statistics in Table
20. It does not refit controllers, rerun oracles, or verify raw query receipts.

The immutable input is `diagnostics/pmo_abc_ablation_v1/reduction_v1.json`
at Git revision `3e01fed6bcea4a517aa5dc4fcb59be7d6444fe5a`. Its exact SHA-256
is pinned in [`experiments/paper/manifest.json`](../paper/manifest.json). With
that historical object available locally, run:

```bash
git show 3e01fed6bcea4a517aa5dc4fcb59be7d6444fe5a:diagnostics/pmo_abc_ablation_v1/reduction_v1.json \
  | PYTHONPATH=src python3 tools/reduce_submitted_pmo_ablation.py --input -
```

The same command accepts `--input /path/to/reduction_v1.json` if the pinned
artifact is supplied separately. It fails on a hash mismatch, an altered
protocol, duplicate or missing cells, or a changed completed-pair set. Both
metrics use the **same 14 fully completed pairs**; partial Top-10 values from
unfinished runs are excluded. Each cell has equal weight, so objectives with
more completed replicates contribute more to the aggregate. The original
artifact labels its overall status provisional because other B campaigns were
not yet finished when it was written. A later 16-pair analysis is not silently
substituted for the submitted table.

The 1,008 charged-call campaign budget and first-1,000-resolved-receipt metric
prefix are checked from the source. The output reports the four excluded cells
and preserves the source status. The saved reduction does not bind all raw
receipt hashes or deployed image identities, so this command proves table
arithmetic on the saved cells, **not** campaign replay. The exact producer
paths and external Modal artifacts still need a separate access and provenance
gate before a fresh reviewer clone can reproduce the scored search.

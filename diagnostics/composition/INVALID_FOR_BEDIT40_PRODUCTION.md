# INVALID_FOR_BEDIT40_PRODUCTION

The artifacts below were generated under the **max_atoms=48** broad-organic scope
(`scope_hash e59fb09801459470`). The production B-edit scope is now **max_atoms=40**
(`scope_hash 3721d69851110fdd`, `BROAD_ORGANIC_V1`), matching base B's architecture. These 48-atom
artifacts are **INVALID for B-edit-40 production** and MUST fail compatibility checks against the 40-atom
training configuration. They are retained ONLY as diagnostic artifacts, with their original scope hash +
provenance intact, for the 48→40 distributional comparison.

Do NOT combine the 48-atom pool with the 40-atom pool, and do NOT produce the 40 pool by filtering these.

| Artifact | Scope | Status |
|---|---|---|
| `broad_mining_full_summary.json` | max_atoms=48 (`e59fb098…`) | diagnostic only |
| `broad_mining_validation_shard_summary.json` | max_atoms=48 | diagnostic only |
| `scaled_edit_data_manifest.json` | max_atoms=48 | diagnostic only |
| `/artifacts/edit_mining_full_broad/edit_pool_full.jsonl` (Modal volume, 347,793 rows) | max_atoms=48 | diagnostic only |
| `cold_vocab_audit.json`, `benchmark_lead_scope_coverage.json` | scope-independent / 48-era | re-verify at 40 |

The clean 40-atom re-mine regenerates the corpus census, MMP + scaffold + corruption traces, dedup/caps,
characterization, curriculum bins, and the immutable manifest from source under
`scope_hash 3721d69851110fdd`.

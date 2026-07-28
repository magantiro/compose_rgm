# Checkpoint Lineage — Program Coherence Audit (HEAD 4204c9d)

Classification by run-label pattern + the loader-rejection guarantee (no exhaustive per-run metadata fetch;
the safety property is proven by the identity gate, not by inspecting each of ~40 historical runs).

## Key checkpoints
| Run label | Class | Notes |
|--|--|--|
| `compose-v4-stage3-flexible-graft-3k-1ac6f19-v1` | **BASE_B** | SHA `c9d927…`, CNOF de-novo, max_atoms=40, non-EMA; the warm-start source |
| `compose-v4-ringcore-v1-bounded-4204c9d-v1` | **RINGCORE_PREFLIGHT** / NON_SCIENTIFIC_PREFLIGHT | the bounded run (failed on DEV-CUDA; being relaunched from the fix commit) |
| `compose-v4-ringcore-v1-dry-*` | RINGCORE_PREFLIGHT | CPU dry-launch runs (NON_SCIENTIFIC_PREFLIGHT) |
| `compose-v4-bedit-broad-a07da93-v1` | STALE_PRE_FIX | mining/compile only, NO completed checkpoint (B-edit never trained e2e) |
| `compose-v4-stage3-*`, `compose-v4-full-ring-*`, `compose-v4-support-gate250-*`, `compose-v4-*-preflight-*`, `compose-v4-*-smoke-*` | LEGACY / DIAGNOSTIC | historical de-novo/preflight/smoke runs; not RingCore edit checkpoints |

## Loader-rejection guarantee (PROVEN by tests, not by per-run inspection)
`ring_core_identity.verify_checkpoint_identity` + `load_factorized_rollout_checkpoint` reject incompatible
checkpoints, verified by `tests/test_ring_core_identity_gate.py` (10 negative tests): base-B-as-edit, wrong
max_atoms(48), stale operator registry, cycle-ops disabled, ring-macros enabled, ring_system_grow enabled,
wrong scope, missing metadata. The eval-cache is v2 + capability-hash-keyed → v1 caches never reused.

## NON_SCIENTIFIC_PREFLIGHT quarantine
All RingCore preflight checkpoints/rollouts/summaries are tagged NON_SCIENTIFIC_PREFLIGHT (in the rollout
harness `STATUS`/`QUARANTINE` + the manifest). Paper-result scripts consume only committed `diagnostics/`
JSONs, never a preflight checkpoint. No checkpoint remains UNKNOWN (every run classified by pattern).

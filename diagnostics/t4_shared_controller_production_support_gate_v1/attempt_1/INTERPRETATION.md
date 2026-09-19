# Shared-controller production support gate

Decision: **PASS**.

This deterministic, zero-oracle replay used the same runtime-admission and candidate-union helpers as the shared production controller. The admission order was original-root Fiber validation, canonical self-event exclusion, endpoint fingerprinting, bound stale-query filtering, cross-expert canonical deduplication, and archive exclusion.

- BRAF seed 0 at delta 0.6: 53 immutable proposal records entered the replay. Four satisfied the original-root Fiber, the exact 20-entry stale-query denylist removed three retained-core candidates, and one unseen large route candidate remained. Its canonical SMILES SHA-256 is `96e3828141b0db1c6d681771bf0c87a2e5ceee400344e11af084a834b3feaf37`, with route rank 38 and 16 realized primitives.
- 5HT1B seed 2 at delta 0.6: 828 immutable exact protonation-aware proposal records entered the replay. Fourteen satisfied the original-root Fiber, all fourteen survived stale filtering, deduplication, and archive exclusion, and all were unseen. The survivors comprised one charge-only program and thirteen charge-plus-structural programs.

The run made zero docking calls, zero oracle calls, zero Modal launches, and zero live-run reads. It establishes candidate support and runtime-admission equivalence only. It does not establish docking utility or authorize a scored campaign.

Authoritative machine-readable result: `result.json`, physical SHA-256 `ffc48740523613d7d0a11ba59a1265e54553a0952f8535d594cbf1a1dd733ec9`, payload SHA-256 `a587a846b031493077c31de97f63fdc18d2a33b2a7d2bf4b359da2d7a05b020d`. The result is bound to implementation commit `7c3d18f77dc71d294c229d9de72b2ae56b89b94a` and records physical and payload hashes for every material input.

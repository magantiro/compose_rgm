# Fragment-constrained proposal panel v1

This frozen zero-oracle panel attempted one deterministic COMPOSE proposal for
each of the 50 released prompts. The prospective contract is
`configs/fragment_constrained_proposal_panel_v1.json`, with payload SHA-256
`77c04325ca7b438da899e2074517a69122eb8ed55cd3477a73a17d80288915af` and
physical SHA-256
`2feeaa6be98ce1fec126d00b931de985b0610e3fbeadbfbac5488a54400e6081`.

## Computed result

- Proposal coverage: 49/50 (0.98).
- Constraint precision conditional on a completed proposal: 49/49 (1.00).
- Exact-valid execution yield: 49/50 (0.98).
- Unique endpoints: 39 among 49 completed endpoints. The ten linker-design
  endpoints are intentionally repeated by scaffold morphing because the
  released protocol reuses those prompts, so this overall count is not a
  benchmark uniqueness estimate.
- Abstentions: one, for Futibatinib superstructure generation, reason
  `candidate_chemistry_unsupported`. The first hydrogen-bearing atom selected
  by the frozen runner carried an explicit chiral hydrogen, so adding a bond
  without removing that explicit hydrogen produced an over-valent candidate.
- Unexpected failures: zero. Constraint failures among completed proposals:
  zero. Oracle, scoring, and Modal calls: zero.

Every completed proposal independently rechecked exact receipt consistency,
all committed states for chemical validity and connected-or-null structure,
and the released fragment constraint. Coverage and conditional constraint
precision are intentionally separate.

The authoritative machine-readable artifact is `result.json`, SHA-256
`509951e64f628c05ef1039f54dc9bf9dfd1ed96b5fc63dc1bdf51420257cdc31`.
This v1 result is preserved as the negative regression-discovery audit. It is
not the 100-sample, three-run scored benchmark and contains no objective or
quality evidence.

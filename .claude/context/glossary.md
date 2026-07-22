# Glossary (RGM domain terms)

- **RGM (Rewrite Generator Matching)** — Generator Matching on an executable stochastic rewrite
  system; learns marginal rule-match firing rates over a molecular-graph CTMC.
- **Committed state / fiber** — a complete valid connected molecule; the *legal-event fiber*
  `A(G)` is the set of legal marked rewrites from `G` (validity-closed by construction).
- **Mark / rewrite** — a typed `(rule, match, payload)` event; `T_a(G)` its executed successor.
- **Tracelet** — the factorized rate model / sampler family (`FactorizedTraceletRateModel`).
- **Graft / ring-system rewrites** — verified derived operators (subtree move; whole-ring grow/delete).
- **Pathwise constraint** — a predicate that holds at *every* committed state, not just the endpoint
  (Paper 1's spine); rule-closed ones are exact invariants (delete violating marks from the fiber).
- **Value-guided SMC** — twisted SMC over the CTMC for property steering; V0 (immediate reward) beat
  V1 (learned twist) / V2 (lookahead) empirically.
- **Lineage B** — the base checkpoint (flexible-Graft + whole-ring-system, GuacaMol-trained, step-1000).
- **CNOF** — C/N/O/F + charged N+/O- chemistry (matches GrIDDD at the element level).

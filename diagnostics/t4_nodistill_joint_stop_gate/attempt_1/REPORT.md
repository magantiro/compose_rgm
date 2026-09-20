# T4 NoDistill joint STOP gate, attempt 1

## Decision

The predeclared standalone gate passed. The authoritative machine-readable
result is `result.json` with SHA-256
`379a74f54af693b32f9c55f0c4829f86d7e570ce29b4241c167dff03b93c384a`.
All four locks were published before the teacher audit and independently
reproduced byte for byte.

## Primary result

On `5ht1b_0` at delta 0.4, the coverage-preserving joint arm emitted 32 complete
STOP candidates. All 32 replayed exactly. Six passed the actual endpoint
admission rule, were endpoint-unique, and occupied six distinct joint strata.
The otherwise matched independent marginal emitted 32 exact candidates but
none passed endpoint admission. Valid complete STOP yield therefore improved by
6 at unchanged exact replay precision of 1.0.

The proposed lock occupied 32 joint strata across scale, heavy-atom change,
cycle-rank change, retained-interface class, and rewrite mode. The marginal lock
occupied two. After locking, the proposed arm supported all six historical rows
under the declared coarse and semantic diagnostics; the marginal supported
none. Those six rows comprise two unique coarse signatures, and the proposed
arm covered both. Row counts must not be interpreted as six independent
signatures.

## Contrast and negative result

On `parp1_0` at delta 0.4, the joint arm produced 8 unique valid complete STOP
candidates in 8 valid joint strata, compared with 1 for the marginal. Both arms
retained exact replay precision 1.0. Neither arm covered the contrast cell's
single historical coarse teacher signature, so coarse and semantic support are
both zero. This null result is retained without weakening the gate or changing
the support.

## Scope and limitations

Generation consumed only the source graph, arm, seed, and frozen generic
settings. It received no task or cell name, route or template identity, teacher
action or endpoint, objective or docking score, absolute route address, or
learned weight. The run made zero oracle and docking calls. Observed candidates
used at most 16 primitives, 4 blocks, and 40 active atoms under the unchanged
32-primitive, 8-block, 40-active-atom support.

This is zero-oracle development-source support evidence. It is not a scored
objective result, held-out generalization result, or production-controller
integration. Historical teacher support is a post-lock coarse structural
diagnostic, not endpoint or exact-route recovery. The historical rows lack a
settled retained-interface label, so none was inferred.

## Verification

Focused gate tests passed (4 tests), Ruff passed, and Black reported all four
Python files unchanged. The reused executor and Dynamic-v1 test selection had
15 passes and one unrelated frozen-input failure: the clean branch `HEAD`
version of `dynamic_program_synthesis.py` has SHA-256
`4f254eb6b0a18cddb43181dadcc7e02ce4a3bd84c47f62506a93b9a38a8f8f8e`,
while the older Dynamic-v1 contract expects
`80fcc5f64a04306613fd84955dbed0b4c4ce5d512a3ed2c2ddd119515b899257`.
The frozen contract and shared production file were not changed.

`verification.json` binds the implementation, inputs, locks, result, exact
commands, observed limits, and the preserved verification limitation.

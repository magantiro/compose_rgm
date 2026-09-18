# JAK2 shallow versus progressive-structured FiberControl support

Decision: **FAIL** the predeclared promotion gate. This experiment made zero
oracle calls.

At the frozen equal budget of 512 proposal draws per lane, the unchanged shallow
lane produced 140 unique eligible endpoints and the progressive-v1 structured
lane produced 138. Structured increased eligible amide endpoints from 4 to 6,
but neither lane produced an eligible diamine-ring endpoint or a complete
amide-plus-diamine basin endpoint.

| lane | attempts with eligible output | eligible records | unique eligible | amide | diamine ring | basin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shallow | 56 / 64 | 247 | 140 | 4 | 0 | 0 |
| structured | 50 / 64 | 253 | 138 | 6 | 0 | 0 |

The progressive structured lane therefore does not earn docking calls in this
revision. The result is narrower than a rejection of structured programs: 37
eligible structured endpoints came from `construct_substituted_ring` programs
and 22 from `ring_path_remodel`, so those capabilities execute and survive the
endpoint gate. What remains absent is the required coupled retained-role
replacement, ring construction and functionalization.

The authoritative result is `result.json`, payload SHA-256
`1ed5c820d23ec24f78354f26c2c188dbd650d208dd3c338c8ba2f27164b2620c`.
Its compressed attempt ledger has SHA-256
`15bd324c07ddc64df62e43770a3eee97273ef1137af257ae23f82fcbd3c2bfc2`.
The successful Modal app was `ap-9zpLv8tIpaQU4LVYTsKtDv`.

An earlier launch, `ap-LlZ3q0LFvG4D7lDIujRHXa`, failed in the local entrypoint
because `src` was absent from `PYTHONPATH`. It created no remote worker, proposal
or oracle call. The successful launch changed only that process environment; the
frozen code, contract, seeds and budgets were unchanged.

This is generated zero-oracle support evidence, not docking utility and not an
autonomous T4 performance result.

# Target-held-out constructive-prior production probe

Decision: **KILL THIS INJECTION; ZERO ORACLE CALLS**.

The matched JAK2 seed-1, delta-0.6 probe completed 128 independent programs per
arm. Both arms exact-executed all 128 attempts and produced 128 unique structurally
valid endpoints. Neither arm produced a benchmark-eligible endpoint.

| Zero-oracle outcome | Uniform same pool | Route prior |
| --- | ---: | ---: |
| Complete / attempted | 128 / 128 | 128 / 128 |
| Unique raw endpoints | 128 | 128 |
| Eligible endpoints | 0 | 0 |
| Ester broken | 25 | 52 |
| Amide | 0 | 0 |
| Diamine ring | 5 | 0 |
| Amide plus diamine-ring basin | 0 | 0 |
| Similarity pass | 0 | 0 |
| QED pass | 0 | 0 |
| SA pass | 2 | 7 |

The target-held-out additive prior therefore changed the chemistry, doubling ester
breakage and increasing SA-passing endpoints, but moved away from the required ring
factor and never coupled an amide with a diamine ring. It satisfies the frozen kill
criterion: zero raw basin endpoints and no strict improvement in either amide or
diamine-ring count. Docking this exact proposal injection is not authorized or
scientifically justified.

This is a negative result for the coarse `(site, mode-counts)` prior attached to the
four shallow constructive families at fixed depth 5 to 8. It is not a negative result
for FiberControl, the previously verified progressive v1 structured support, or a
route-derived proposal model whose output includes complete dependency-region
chemistry. In particular, all raw endpoints from this probe failed similarity and QED,
so selection or docking feedback could not rescue them.

Authoritative artifacts:

- `jak2_1_uniform_same_pool.json`, SHA-256
  `b256d1341abb2f4977ad4ce567c2b19780859cac7bd4bea4c55bca6b0a8513c9`
- `jak2_1_route_prior.json`, SHA-256
  `4ae0ecad94dade05b9dcc7340a76e4e9f92373654f8546ca725772897cef3a6a`
- uniform ledger, SHA-256
  `1ac34e277526fe900411fd30e9e57a927c7db6fabfef90edc145e52a02c2ceaf`
- route-prior ledger, SHA-256
  `ba6adb52a4bc7a25aae99e5e12113b70495390d6f6fa4e77a98f15776d39fd14`
- contract payload, SHA-256
  `b03ddf40b61649d8e42613acba0ea2ccdfb47a11496faf9aa8e98e203a9c01eb`
- code revision recorded by both results:
  `f9d3a007060607711f720c33d78b5668c2ee5122`

The next bounded test should use the already verified progressive v1 structured lane
as an additive proposal source beside unchanged v0, filter both before allocating any
docking budget, and measure JAK2 basin and eligible yield before a scored pilot.

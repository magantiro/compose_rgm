# T4 program-retrieval diagnostic

`attempt_2/result.json` is the authoritative sealed result from clean commit
`d906b49a8d9a3d2eef6d4f14833ee819ddb5cad2`. Its complete per-arm batches retain
every exact proposal attempt, eligible endpoint, binding, execution trace and
input hash.

| Measure | Frozen cold-start control | Bounded direct retrieval |
| --- | ---: | ---: |
| Eligible endpoints, 15 cells | 157 | 178 |
| Cells with an additional public-winner reconstruction | 0 | 13 |
| Additional public winners reconstructed | 0 | 26 |
| Summed proposal seconds | 53.14 | 51.46 |

The predeclared structural gate passed. This is a zero-oracle, answer-known
development result. Public winner overlap measures whether the controller can
execute known useful programs without first corrupting them; it is not autonomous
benchmark discovery and the external scores are not new COMPOSE observations.

The first preliminary execution produced scientifically identical control and
retrieval batch identities for all fifteen cells, but its provenance flag counted
its own newly written output as a dirty worktree. It is therefore excluded as the
authoritative artifact. Attempt 2 repaired the recorder, required a clean source
tree, and reproduced every scientific batch identity.

Important residuals:

- 5HT1B seed 2 remained at one eligible endpoint in both arms.
- BRAF seed 2 decreased from thirteen to eleven eligible endpoints, although it
  reconstructed two public winners absent from the control.
- A successor scored run needs its own frozen launch contract. These files do not
  alter the live benchmark at commit `c272b88` and make zero docking calls.

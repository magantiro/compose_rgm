# T4 utility data-acquisition launch, attempt 1

The exact four-request utility data-acquisition lock was scored once under the
frozen QuickVina evaluator. All four reservations reached one successful
terminal state. Exactly four first-score calls were charged. No retry,
replacement, backfill, confirmation, candidate regeneration, or new generator
call occurred.

## Outcome

| Cell | Request | Candidate | Score |
| --- | --- | --- | ---: |
| `parp1_2` | `16684461…ce0` | `fd4d7541…36c` | -7.8 |
| `parp1_2` | `29b8aab0…44b` | `6ab68c2d‥0fb` | -8.3 |
| `fa7_1` | `a72ddc72…b34e` | `d7c7b562…3345` | -6.9 |
| `fa7_1` | `e6bd0f87…72db` | `705e30be…af12` | -8.0 |

Lower scores are better. The PARP1 pair has mean -8.05 and absolute spread
0.5 kcal/mol. The FA7 pair has mean -7.45 and absolute spread 1.1 kcal/mol.
Both exact cells therefore contain two finite, non-tied labels, and the frozen
reducer reports `conditional_data_gate_ready=true`.

This is prospective training-data acquisition, not prospective evaluation of
a utility model or controller. The result only establishes that the unchanged
utility-data support gate may now be reevaluated using the four frozen labels.
It does not itself establish that the gate passes, that a fitted selector
generalizes, or that either candidate in a pair is superior outside this bound
single-seed evaluator.

## Immutable identities

- Source revision: `9bf71eff393bd7e7b706a0d08d005b8ea7e25015`.
- Launch implementation revision:
  `83e8092f97b8d1f784e7cbc82f462948c02e30d3`.
- Launch-contract physical SHA-256:
  `fc4e0ce03f7c11973910d1a522864de2299726cea661ffa2f6270ef54b3230a5`;
  payload SHA-256:
  `129f9ad786edaecfea7cfeeb78410c92a659845b4c70c4996c363e412070e016`.
- Candidate-lock physical SHA-256:
  `d790389696d729f19ce617b3dcf4afddf0f3ce06110882a1b628889e0c354249`;
  payload SHA-256:
  `8b5acdbfca42a122d6a882d48e6bbe00468052c913f79e806b6dbb7164685471`.
- Request-lock physical SHA-256:
  `e11441244c98d919d42315dcafbdede300bbb6232bdc5a911896bdb3d1010d05`;
  payload SHA-256:
  `f78985599a7ff5e533384ac7d61c2cc77f7c6cf94e064f6a6ec0e3ed692bc279`.
- Reduced-result physical SHA-256:
  `42539073e6a62b470d6e00c2565f35aa47bc44aa48fad3e8d1ee2ecbb816b639`;
  payload SHA-256:
  `47646fc8ea9016098ee98615f08b9f3b673333c50a269e41a420934460b069ba`.
- Review physical SHA-256:
  `d35a0a64bcca4e52ac32a017ab7c6d898e7b93b633cdf2948ebe4cc95f0b916e`;
  payload SHA-256:
  `626aaa696c2d55ed7f14319d68c7a9788badf1ef39b4817cd617e2a2425a412e`.
- Run ID:
  `2d8bc03cd6b55b0a0453ed80241ca0829b233dc7db3a4b124f06d952b2ad6726`.
- Zero-score preflight call: `fc-01M2KV39E36Y7WGAZDERWH9J2M`.
- Worker calls: `fc-01M2KV3E9CWBM3CRS9A7RVPNET`,
  `fc-01M2KV3EWH7FRFMWQPSZM5H6FF`,
  `fc-01M2KV3F9NME289WE2E7FSB7CX`, and
  `fc-01M2KV3FS613F2HZZHDQF9WZ9P`.
- Zero-docking reducer call: `fc-01M2KV5Y1XJ39B68YE51EQQAN3`.

The remote preflight verified RDKit 2024.03.5, Open Babel 3.1.1, the bound
QuickVina executable, both receptor hashes, the launch revision, and the exact
four-request census before worker spawn. The self-hashed `result.json` is the
authoritative execution ledger. Each request directory preserves its immutable
reservation, start, terminal result, and pose files. `review.json` records the
claim boundary and acquisition summary.

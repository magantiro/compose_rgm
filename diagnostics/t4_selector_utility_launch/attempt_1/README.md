# T4 selector utility launch, attempt 1

The exact two-request selector lock was scored once under the frozen QuickVina
evaluator. Both reservations reached one successful terminal state. Exactly two
first-score calls were charged. No retry, replacement, backfill, confirmation,
candidate regeneration, or new generator call occurred.

## Outcome

| Cell | Selector rank | Selector score | Raw-order counterpart | Source reference |
| --- | ---: | ---: | ---: | ---: |
| `jak2_1` | 1 | -8.3 | -6.8 | -8.0 |
| `parp1_0` | 2 | -7.8 | -6.8 | -7.3 |

Lower scores are better. The selector endpoints improved on their exact
previously scored raw-order counterparts by 1.5 and 1.0 kcal/mol, respectively,
and on their cell source references by 0.3 and 0.5 kcal/mol. The mean selector
score was -8.05, compared with -6.8 for the raw-order counterparts and -7.65 for
the source references.

This is the intended positive development signal: offline complete-macro
selection found useful endpoints even though exact teacher recovery was not its
utility objective. The evidence is small and selected. Only two of the five
declared development cells produced eligible selector requests, and selector
architecture choice used offline teacher and component diagnostics. The result
therefore does not establish final-benchmark generalization, route recovery,
autonomous optimization, or superiority to IVG.

## Immutable identities

- Contract commit: `2cf70d8`; contract payload SHA-256:
  `f0e507b475c66d01de5cbd11909f92daa383555dee6a226e79f1e60c5620d4a4`.
- Launch implementation commit:
  `eb275132298188edb809566832762a959e1cf335`.
- Request-lock physical SHA-256:
  `9802da3b356ae5f41460a29377640c9a0e3fb866f2e82c46ea581624074ba046`.
- Request-lock payload SHA-256:
  `6d49e2a397b662a4425a4a5ee0d17a6223d64c4ac3e0effa594dc8d8c1c52b29`.
- Reduced-result physical SHA-256:
  `c7d63593ed4e90b37a7e37a70c288aeee6bc8300dfdf1e97af1d92e315c77f8e`.
- Reduced-result payload SHA-256:
  `0dffec4a76f86f8fe1a72076e47bdbe966780d5ff5d0c4cba97a6fb34d23071c`.
- Review physical SHA-256:
  `f082acd1faa6bbdc4782f84041420c8d2bac7699c59e0df3c7078259ec3fb512`.
- Review payload SHA-256:
  `67fcb8fb304d336e0036c24cc160fa41846237d0a6d1008f60a14f830e99c0c1`.
- Raw-counterpart result revision:
  `977e4bdc624a92a868f70e586327326c9b2508b0`; physical SHA-256:
  `c44d9168591996a0008eeeef617c349f17b83389780d522a3aca62b61ad03b02`;
  payload SHA-256:
  `3e9a73281ccc3505dad7059ab06609ebf5470328ec08183aea4a4fe67dbbdb1d`.
- Run ID:
  `9f0c5f2cc25117aa76e235f5905e095cf120832a641aa74e0386642e8c354571`.
- Modal app ID: `ap-KTVGX5XP17A4zBUD2ONVkQ`.
- Zero-score preflight call: `fc-01M2KNKX5J7V6M11DJRRAZ1RX0`.
- Worker calls: `fc-01M2KNM1PYRMZXADPFPA7MSPJG` and
  `fc-01M2KNM2DDHCD10FPM5VFMSW78`.
- Zero-docking reducer call: `fc-01M2KNQWWS5E3ACRZBS2DXBJXW`.

The remote preflight verified RDKit 2024.03.5, Open Babel 3.1.1, the bound
QuickVina executable, both receptor hashes, the contract identity, and the exact
two-request census before worker spawn. The self-hashed `result.json` is the
authoritative execution ledger; `review.json` records the bound comparisons and
claim boundary.

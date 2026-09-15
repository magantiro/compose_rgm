# T4 compositional generator utility launch, attempt 1

The exact four-request score-blind lock was executed once under the frozen
QuickVina evaluator. All four reservations reached one terminal state. Two
requests returned scores and two returned no score. No retry, replacement,
backfill, candidate regeneration, or new generator call occurred.

## Outcome

| Cell | Policy memberships | Request | Terminal outcome |
| --- | --- | --- | --- |
| `jak2_1` | baseline-learned, expanded-learned | `4b148133...` | complete, -6.8 kcal/mol |
| `parp1_0` | baseline-learned, expanded-learned | `c8208357...` | complete, -6.8 kcal/mol |
| `jak2_1` | baseline-uniform, expanded-uniform | `26ebbab7...` | failed-no-retry, bound wrapper returned no score |
| `parp1_0` | baseline-uniform, expanded-uniform | `31be6702...` | failed-no-retry, bound wrapper returned no score |

The successful-score yield is 2/4 unique requests and 4/8 preserved arm
memberships. The first-score call census is exactly 4/4, with zero incomplete
reservations and zero automatic retries.

## Immutable identities

- Contract commit: `ea4dfcf`; contract payload SHA-256:
  `4175d28ad705166cfda7cdc05b71ae74339592820df2d2ad6134d21df96706f3`.
- Launch implementation commit: `26b9e030a9149244aa0816a1cf04d8b9e78242ef`.
- Request-lock physical SHA-256:
  `3d4a1712cda185d13e5eb66e4b8bc47db7e85960854b4132bb8710bdf7cf4469`.
- Request-lock payload SHA-256:
  `420a2b063f1309493c7b283559e517b183a40e9f982fafc12b7093570b3b5247`.
- Reduced-result physical SHA-256:
  `c44d9168591996a0008eeeef617c349f17b83389780d522a3aca62b61ad03b02`.
- Reduced-result payload SHA-256:
  `3e9a73281ccc3505dad7059ab06609ebf5470328ec08183aea4a4fe67dbbdb1d`.
- Run ID:
  `527e98d7a3459a620a10a5590a2c06e80d75208e3013508df013204da71413ef`.
- Modal app ID: `ap-VfGbftudLfgX5ZnqXAsD1z`.
- Zero-score preflight call: `fc-01M2KGH44B4HT1JMN8GPN3JCRP`.
- Worker calls: `fc-01M2KGJ2KS99A4BD40K9X9V2FG`,
  `fc-01M2KGJ346GHQJ9PQGY5MKVAD3`,
  `fc-01M2KGJ4CKG8JQJSNDR3MJDH9B`, and
  `fc-01M2KGJ50E7JPZNWEW23B1DN0F`.
- Zero-docking reducer call: `fc-01M2KGP8TV4CWJ83AMSN9WCNYB`.

The remote preflight verified RDKit 2024.03.5, Open Babel 3.1.1, the bound
QuickVina binary, both receptor hashes, the contract identity, and the exact
four-request lock before workers were spawned.

## Interpretation boundary

This panel establishes endpoint docking outcomes only under the bound
QuickVina evaluator. Baseline and expanded arm memberships collapse to the same
four molecules, so these results cannot estimate an expansion effect. They do
not establish teacher or route recovery, autonomous optimization, or superiority
to the iterated virtual screening baseline.

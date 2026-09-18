# Adaptive versus reward-blind selection on identical pools

`jak2` delta=0.6, 112 charged calls per arm, batch 8, 480 raw programs per parent, program depth 3, seed 20260918.
Shared root `COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34` at -8.10.

## Best so far against charged calls

| calls | pool | multi | adaptive | blind |
| ----: | ---: | ----: | -------: | ----: |
| 9 | 141 | 0 | -9.30* | -8.80* |
| 17 | 217 | 2 | -9.30 | -9.60* |
| 25 | 286 | 10 | -9.50* | -9.60 |
| 33 | 213 | 2 | -9.80* | -9.60 |
| 41 | 242 | 5 | -9.80 | -9.60 |
| 49 | 196 | 14 | -10.00* | -9.70* |
| 57 | 183 | 8 | -10.00 | -9.70 |
| 65 | 186 | 9 | -10.20* | -9.70 |
| 73 | 157 | 8 | -10.20 | -9.90* |
| 81 | 186 | 5 | -10.30* | -9.90 |
| 89 | 152 | 6 | -10.30 | -9.90 |
| 97 | 147 | 9 | -10.30 | -9.90 |
| 105 | 104 | 0 | -10.30 | -10.00* |

## Matched rounds: both arms picked eight from the same list

| round | adaptive batch best | blind batch best | adaptive - blind |
| ----: | ------------------: | ---------------: | ---------------: |
| 1 | -9.30 | -8.80 | -0.50 |
| 2 | -9.30 | -9.60 | +0.30 |
| 3 | -9.50 | -8.80 | -0.70 |
| 4 | -9.80 | -8.70 | -1.10 |
| 5 | -9.50 | -9.30 | -0.20 |
| 6 | -10.00 | -9.70 | -0.30 |
| 7 | -9.50 | -9.70 | +0.20 |
| 8 | -10.20 | -9.20 | -1.00 |
| 9 | -10.20 | -9.90 | -0.30 |
| 10 | -10.30 | -9.10 | -1.20 |
| 11 | -10.20 | -9.60 | -0.60 |
| 12 | -9.80 | -9.50 | -0.30 |
| 13 | -9.80 | -10.00 | +0.20 |

Mean paired difference **-0.423** (negative favours adaptive, because a lower docking score is better).
Sign test over 13 decided rounds: adaptive better in **10**, blind better in 3, ties 0, two-sided p = 0.0923

## Where each arm's picks landed among what was docked

Ranks are over that round's docked molecules only. This is a selected sample, so the `best docked` column is a LOWER BOUND on the pool ceiling and says nothing about whether stronger candidates went unchosen.

| round | best docked | adaptive best rank | blind best rank |
| ----: | ----------: | -----------------: | --------------: |
| 1 | -9.30 | 1 of 15 | 4 of 15 |
| 2 | -9.60 | 2 of 16 | 1 of 16 |
| 3 | -9.50 | 1 of 16 | 4 of 16 |
| 4 | -9.80 | 1 of 16 | 6 of 16 |
| 5 | -9.50 | 1 of 16 | 3 of 16 |
| 6 | -10.00 | 1 of 16 | 2 of 16 |
| 7 | -9.70 | 3 of 15 | 1 of 15 |
| 8 | -10.20 | 1 of 16 | 4 of 16 |
| 9 | -10.20 | 1 of 16 | 3 of 16 |
| 10 | -10.30 | 1 of 16 | 7 of 16 |
| 11 | -10.20 | 1 of 16 | 4 of 16 |
| 12 | -9.80 | 1 of 15 | 4 of 15 |
| 13 | -10.00 | 3 of 16 | 1 of 16 |

## Top-tail ranking skill

A controller does not need to fit the pool; it needs to be right about the few molecules worth a call. So this reports where the round's ACTUAL best molecule sat in the model's pre-docking ranking of the whole pool, which is answerable because the pool is shared and the ranking is recorded before any score comes back.

| round | pool | rank of the round's best | percentile | round best | regret |
| ----: | ---: | -----------------------: | ---------: | ---------: | -----: |
| 1 | 141 | 2 | 1.4% | -9.30 | +0.00 |
| 2 | 215 | 55 | 25.6% | -9.60 | +0.30 |
| 3 | 275 | 117 | 42.5% | -9.50 | +0.00 |
| 4 | 204 | 2 | 1.0% | -9.80 | +0.00 |
| 5 | 237 | 8 | 3.4% | -9.50 | +0.00 |
| 6 | 185 | 15 | 8.1% | -10.00 | +0.00 |
| 7 | 166 | 34 | 20.5% | -9.70 | +0.20 |
| 8 | 175 | 4 | 2.3% | -10.20 | +0.00 |
| 9 | 144 | 1 | 0.7% | -10.20 | +0.00 |
| 10 | 168 | 1 | 0.6% | -10.30 | +0.00 |
| 11 | 136 | 2 | 1.5% | -10.20 | +0.00 |
| 12 | 125 | 2 | 1.6% | -9.80 | +0.00 |

Median percentile of the round's best molecule: **2.3%**. In the model's top decile on **9 of 12** rounds, and in its top ten candidates on 8.
`regret` is the adaptive arm's own best minus the best molecule docked that round by either arm, so 0.00 means it did not leave anything on the table among what was actually scored.

## The model's picks against its own random quota

- model-selected picks: **-9.074** mean over 78 calls
- random-quota picks: **-8.669** mean over 26 calls
- difference **-0.405** (negative favours the model, since a lower docking score is better)
This is the cleanest within-arm read available: the same round, the same pool, the same parents, six picks chosen by the ranking against two chosen at random.

## Calibration of the adaptive arm's own predictions

Over 104 counted calls, corr(predicted endpoint, realized) = **+0.262**. Positive means the model orders candidates in the right direction; near zero means selection is effectively blind whatever the rule says.

## Where improvements came from

- **adaptive**: 6 improving rounds, 6 single-region, 0 multi-region; families base 5, retained_element 1.
- **blind**: 5 improving rounds, 5 single-region, 0 multi-region; families retained_element 4, scale 1.

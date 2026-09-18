# Adaptive versus reward-blind selection on identical pools

`jak2` delta=0.6, 112 charged calls per arm, batch 8, 480 raw programs per parent, program depth 3, seed 20260918.
Shared root `COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34` at -8.10.

## Best so far against charged calls

| calls | pool | multi | adaptive | blind |
| ----: | ---: | ----: | -------: | ----: |
| 9 | 139 | 0 | -9.00* | -8.80* |
| 17 | 267 | 1 | -9.00 | -9.30* |
| 25 | 144 | 3 | -9.30* | -9.30 |
| 33 | 156 | 3 | -9.80* | -9.40* |
| 41 | 276 | 7 | -9.80 | -10.30* |
| 49 | 249 | 11 | -10.10* | -10.30 |

## Matched rounds: both arms picked eight from the same list

| round | adaptive batch best | blind batch best | adaptive - blind |
| ----: | ------------------: | ---------------: | ---------------: |
| 1 | -9.00 | -8.80 | -0.20 |
| 2 | -8.80 | -9.30 | +0.50 |
| 3 | -9.30 | -9.30 | +0.00 |
| 4 | -9.80 | -9.40 | -0.40 |
| 5 | -9.40 | -10.30 | +0.90 |
| 6 | -10.10 | -9.50 | -0.60 |

Mean paired difference **+0.033** (negative favours adaptive, because a lower docking score is better).
Sign test over 5 decided rounds: adaptive better in **3**, blind better in 2, ties 1, two-sided p = 1.0000

## Where each arm's picks landed among what was docked

Ranks are over that round's docked molecules only. This is a selected sample, so the `best docked` column is a LOWER BOUND on the pool ceiling and says nothing about whether stronger candidates went unchosen.

| round | best docked | adaptive best rank | blind best rank |
| ----: | ----------: | -----------------: | --------------: |
| 1 | -9.00 | 1 of 13 | 2 of 13 |
| 2 | -9.30 | 4 of 16 | 1 of 16 |
| 3 | -9.30 | 1 of 15 | 1 of 15 |
| 4 | -9.80 | 1 of 16 | 2 of 16 |
| 5 | -10.30 | 5 of 15 | 1 of 15 |
| 6 | -10.10 | 1 of 16 | 4 of 16 |

## Calibration of the adaptive arm's own predictions

Over 47 counted calls, corr(predicted endpoint, realized) = **+0.129**. Positive means the model orders candidates in the right direction; near zero means selection is effectively blind whatever the rule says.

## Where improvements came from

- **adaptive**: 4 improving rounds, 4 single-region, 0 multi-region; families scale 1, base 1, retained_element 1, element 1.
- **blind**: 4 improving rounds, 4 single-region, 0 multi-region; families element 2, retained_deletion 1, base 1.

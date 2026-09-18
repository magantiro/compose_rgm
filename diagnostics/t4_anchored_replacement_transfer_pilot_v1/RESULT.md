# JAK2 seed-0 anchored-transfer docking result

The prospective transfer gate failed. All 12 score-blind locked endpoints docked
successfully, but the best score was -9.3. The contract's published IVG delta-0.6
mean for this cell is -9.7, so zero candidates met or beat the comparator. The
observed scores ranged from -9.3 to -7.3.

This result separates proposal support from utility. The frozen generic anchored
operator generated 218 unique eligible seed-0 endpoints, 53 more than shallow at
the matched free-proposal budget, but this prospectively locked diverse sample did
not transfer the IVG-level utility observed on seed 1.

The sealed result's inherited `summary` field uses labels and a promotion sentence
from the earlier seed-1 pilot, including an `ivg_mean_minus_10_4` key. That label is
not the seed-0 comparator and must not be used here. The raw candidate scores are
valid. This report applies the preregistered seed-0 criterion (-9.7) from
`configs/t4_anchored_transfer_pilot_v1.json` without changing or rerunning the
sealed result.

- Successful calls: 12/12
- Best score: -9.3
- Candidates at or below -9.7: 0/12
- Automatic retries or replacements: 0
- Candidate-lock ID: `d6314db30bcbc94e1b333145f182bbf4401ceff367e04125055b1df94a146f1a`
- Result payload SHA-256: `3f26eabf33c04456a7bbc1b50ccf775bfaf8ef81f155d7226f662339f90212d0`
- New oracle calls: 12

The evidence supports anchored replacement as a useful proposal expert on seed 1
and a high-yield but not utility-validated expert on seed 0. It does not justify a
universal JAK2 operator or credit FiberControl or route distillation. The next
controller should combine independent experts, allow empty-output abstention, and
use objective feedback rather than allocating a fixed oracle share to anchored
replacement.

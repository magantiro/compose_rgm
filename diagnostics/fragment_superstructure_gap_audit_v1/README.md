# Frozen superstructure validity decomposition

This is a retrospective, zero-sampling audit of the 30 frozen
`fragment_official_suite_v2` superstructure shards (10 prompts x 3 seeds x
100 attempts). `result.json` records every input SHA-256 and verifies the
locally cached InVirtuoGen evaluator against its pinned upstream hash before
checking each shard's validity numerator against its 100-attempt denominator.

The corrected COMPOSE chemical-output figure is **2,809/3,000 = 93.63%**.
All 2,809 committed endpoints were chemically valid. The exact official
evaluator validity was **2,806/3,000 = 93.53%**: 191 attempts committed no
endpoint, and 3 chemically valid commits failed required-fragment containment.
Among commits, independently reported containment was 2,806/2,809 = 99.893%.

The no-commit gap is highly concentrated: SPIRAPRIL accounts for 156/191;
LOVASTATIN 23, CYCLOTHIAZIDE 7, and FUTIBATINIB 5. The saved shards also
report 224 trajectories hitting the bounded rejection budget, but that count
is **not** a disjoint reason category: trajectories can exhaust their budget
after one or more accepted events and still commit a molecule. The old shards
retain only aggregate budget counts and mean events, not attempt-aligned
events/refusals or all SMILES, so the exact zero-event versus exhaustion split
of the 191 cannot be recovered. Do not relabel all 191 as rejection exhaustion.

Scientific interpretation: the ~6.4 percentage-point shortfall from 100%
chemical output per attempt is primarily a sampler/output-efficiency problem,
not invalid emitted chemistry. The three containment misses are a separate
locked-chemistry problem. The frozen baseline must not be retroactively changed;
future runs must retain attempt-level event and refusal records so this split
can be measured exactly.

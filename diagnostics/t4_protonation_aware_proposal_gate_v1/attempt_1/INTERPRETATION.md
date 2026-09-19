# Protonation-aware proposal gate, attempt 1

## Outcome

The sealed zero-oracle gate **passed**. On the predeclared primary root,
5HT1B seed 2 at delta 0.6, the task-blind Editing-V3 expert produced **14
unique fully eligible endpoints** spanning **8 Murcko scaffolds**. One endpoint
was the charge-only deprotonation product and 13 combined that same admitted
deprotonation with an unchanged generic shallow/local structural proposal.
Eligible program length ranged from 1 to 10 primitives (median 2).

Across all three roots, **1,766/1,766 attempted in-budget programs exact-replayed**
from their original root. Every published action list began with exactly one
`atom_protonation_restate`. No docking, oracle, Modal, live-run, endpoint or
teacher input was used.

The authoritative machine-readable result is `result.json`, payload SHA-256
`69c718b442509324bf28f1acc46b6b7bf49ced0554f70cc06d3a979339c59b11`.
It was generated from code revision `88b54d2690f53a2f20eea48a04daa7cd5255a35c`.

## Primary funnel

The expert enumerated all three admitted protonation actions and attempted 1,078
in-budget complete programs: 3 charge-only, 768 shallow/local, 45 retained-core
prune and 262 route-complete-region programs. All 1,078 exact-replayed, yielding
828 unique exact endpoints after endpoint deduplication. Of those unique
endpoints:

- 755 passed the structural-validity check;
- 128 passed similarity at delta 0.6;
- 361 passed QED at 0.6;
- 145 passed SA at 4.0;
- 14 passed all endpoint gates.

The primary root completed in 384.109 seconds on the recorded CPU environment.
The route-complete and retained-core lanes added no unique fully eligible result
after deterministic endpoint deduplication. The result therefore localizes the
smallest useful generic composition to protonation restatement plus the existing
one-module shallow/local mechanism.

## Contrasts and abstentions

The neutral-tertiary-amine contrast independently exercised the reverse charge
transition. It attempted 690 programs, abstained from two route tails whose
combined program exceeded 32 primitives, exact-replayed all 688 in-budget
programs, and yielded 460 unique endpoints. Two were fully eligible and belonged
to two Murcko scaffolds. The contrast completed in 206.162 seconds.

The no-admitted-site contrast enumerated zero protonation actions, published zero
candidates and explicitly abstained in 0.0004 seconds. It did not fall back to an
older charge-preserving support version.

## Preflight evidence preserved separately from the sealed result

Before the contract was executed, bounded local preflights isolated the cheapest
composition on the primary root. Protonation followed by the existing route lane
gave 0 eligible endpoints among 161 exact route candidates; protonation followed
by anchored replacement gave 0 among 128 exact candidates. The retained-core
lane found one eligible endpoint. A 256-draw, one-module shallow/local probe gave
256/256 exact executions and 16 unique eligible endpoints in 118.42 seconds.
Those preflights used development seeds/configurations and are not authoritative
gate results; they motivated no relaxation of the already sealed endpoint gate.

## Claim boundary

This result establishes autonomous executable proposal support and strict
endpoint feasibility only. It does not establish docking utility, FiberControl
selection quality, exact teacher recovery as an objective, or T4 benchmark
improvement. Editing-V2 and historical V4 proposal paths remain unchanged; the
new action support is active only when this Editing-V3 expert is explicitly
selected.

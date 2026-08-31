# BUILD_RING_SYSTEM — frozen contract

macro_engine.py sha256[:16] = `2d63103f0ba807c7`
qualification artifact: `diagnostics/exact_macro.json`
cell: 5ht1b_s7 · seed `Cc1ccc(N2CC[NH2+]CC2)cc1C(F)(F)F`

## Result

8/8 anchors pass the full qualification: ring systems +1,
aromatic rings +1, and true T4 feasibility (QED>=0.6, SA<=4, sim>=delta) on the
RETURNED molecule. 6 distinct products across 8 anchors.

| anchor | product | sim | QED | SA | T4 d0.4 | T4 d0.6 |
|---|---|---|---|---|---|---|
| a0 | `FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1-c1ccccc1` | 0.524 | 0.904 | 2.97 | yes | no |
| a1 | `FC(F)(F)c1ccc(-c2ccccc2)c(N2CC[NH2+]CC2)c1` | 0.524 | 0.904 | 3.0 | yes | no |
| a2 | `FC(F)(F)c1cc(-c2ccccc2)cc(N2CC[NH2+]CC2)c1` | 0.615 | 0.904 | 2.99 | yes | yes |
| a3 | `FC(F)(F)c1cccc(N2CC[NH2+]C(c3ccccc3)C2)c1` | 0.581 | 0.904 | 3.41 | yes | no |
| a4 | `FC(F)(F)c1cccc(N2CC[NH2+]C(c3ccccc3)C2)c1` | 0.581 | 0.904 | 3.41 | yes | no |
| a5 | `FC(F)(F)c1cccc(N2CC[NH2+]CC2)c1-c1ccccc1` | 0.432 | 0.904 | 3.07 | yes | no |
| a6 | `FC(F)(F)c1cccc(N2CC[NH2+]CC2c2ccccc2)c1` | 0.545 | 0.904 | 3.42 | yes | no |
| a7 | `FC(F)(F)c1cccc(N2CC[NH2+]CC2c2ccccc2)c1` | 0.545 | 0.904 | 3.42 | yes | no |

## Contract

    A_m(x) = A(x) INTERSECT C_m(x)

The macro restricts support; it never invents it. C_m is computed from the
tracked `ring_frontier` before any action is applied, then intersected with the
executor's exact legal support A(x). Frozen R_theta ranks only the survivors.
An empty intersection returns UNSAT with no fallback.

Nothing here modifies `production_successor_kernel.py`. The frozen kernel is
called, not changed: touching it would change `process_identity_sha256` and
invalidate every Modal run.

## Realised trace

    grow x6  ->  bond_insert(new, new)  ->  ring_system_restate

Both closure endpoints are atoms this macro grew, established from tracked
state-space slots, never by rematching the seed against the product.

R_theta ranks over the 8 qualifying runs:

| step | rank range |
|---|---|
| chain growth (atom_insert) | r6–r28 |
| closure (bond_insert) | r47–r169 |
| aromatisation (ring_system_restate) | r2–r5 |

The closure is the one step R_theta ranks poorly, and it is exactly the step
descriptor compilation locates without reference to rank. At the six-carbon
precursor the closure fiber holds ~150 legal actions and the macro examines 2.

## Applicability

Bounded by seed headroom, not by the macro. 9 of the 15 T4 seeds start below
QED 0.6 before any edit, and 5ht1b_s8 starts at SA 4.69; on those cells adding
an aromatic carbocycle moves the endpoint further from feasibility. A local
sweep over all 15 seeds x 8 anchors built the ring 120/120 times with similarity
0.53-0.90 throughout, and QED was the only binding constraint. Growth macros
belong to the small QED-rich seeds; the rest need QED repair under a similarity
floor, which is a different macro.

## Support is not portable

The local primitive fiber and the production marked law are different supports
and need different descriptors for the same chemical step:

| | local fiber | production law |
|---|---|---|
| closure | 127 bond_insert, 0 cycle_close | bond_insert and cycle_close both present |
| aromatisation | bond_reorder x3 (Kekule walk) | ring_system_restate, one correlated event |
| bond_reorder enabled | yes | no |

`RingSystemRestate` carries no atom-scoped field; its payload is `changes`, a
tuple of `BondOrderChange(a, b, new_order)` applied as one event. Filtering it
on `.v` matched nothing and returned UNSAT while the correct actions sat at
ranks 2-5. Any claim about reachability must be measured on the production
side.

# Exact admission-mask memoization: 6.28× / 3.24×, qualified bitwise, and blocked

**The optimization works and is exact. It cannot be deployed on the frozen line,
because the two files it touches are inside the Process-V2 identity hash.**

## What was done

Profiling put 90% of a law call in two admission-mask computations. Inside them
the cost was not the legality decision but recomputation of state-invariant
facts, once per candidate:

    is_valid_state(SOURCE)          984 calls/state   313.6 ms
    canonical_state_key(SOURCE)     984 calls/state   200.1 ms
    _exact_state_identity(SOURCE)   900 calls/state   146.0 ms
    is_connected_or_null(SOURCE)    984 calls/state    71.7 ms
                                    pure redundancy   731.5 ms = 46.4%

plus 960 Kekule alias instantiations per state drawn from **2.4 distinct
selections**, so the alias validity / connectivity / canonical-key screen ran
about 400 times per distinct alias. Atom-restate had the same shape:
`resonance_invariant_bond_classes` recomputed once per `(vertex, target class)`
— 15 classes times every real slot, ~375 times per state — and an alias set that
depends only on the vertex's aromatic component, rebuilt for all 15 classes.

Three transformations, both files:

1. **Source-global hoist.** `prepare_*_context` refuses to build a context
   unless `is_valid_state` and `is_connected_or_null` hold, so a context
   belonging to this exact state already certifies both, and `canonical_state_key`
   is a pure function of the state. A non-matching context falls through to the
   original path and still raises, after the action gate, exactly as before.
2. **Cheap exact identity guard.** `np.array_equal` on the four state arrays in
   place of rebuilding four Python tuples per candidate — same value-equality
   predicate, and the bond block alone is 2304 elements on a 48-slot state.
3. **Alias memo.** Cached per selection tuple (cycle-close) and per component
   index (atom-restate). The screen is cached PER ALIAS rather than as one
   verdict, because the early return carries `target_compatible_alias_count`,
   which accumulates across the loop; collapsing it would change that count when
   a later alias fails.

No legality predicate was rewritten, no candidate is filtered by a heuristic, no
capability flag changed. Only the two `prepare_*` functions construct these
contexts anywhere in the tree, so the hoist's precondition holds universally.

## Qualification

`scripts/verification/admission_mask_parity.py`, 600 distinct molecules (the 64
dev-panel sources plus a deterministic ZINC250k stride), comparing four exact
digests each: both admission masks and both ORDERED enumerator outputs.

    cycle-close     1178.0 ->  187.7 ms    6.28x
    atom-restate     783.6 ->  241.6 ms    3.24x

    BITWISE PARITY: PASSED  (600 states, 2400 digests identical)

Ordered output is compared, not just the masks: a mask collapses ordering and
multiplicity, but the sampler draws by index, so a permutation with identical
content would still change which molecule a given random draw returns.

Projected on the container's measured law-call budget — close 2747 ms, restate
1626 ms, everything else 487 ms of 4860 ms — a law call would fall to ~1426 ms,
about **3.4x**, on top of the banked 3.22x.

## Why it is blocked

`editing_v2_process_identity.py` builds "one full-SHA process identity whose
change invalidates artifacts" over the source bytes of nineteen files. Two of
them are the two this optimization edits:

    src/compose_v4/rewrite/semantic_atom_restate.py     CHANGED
    src/compose_v4/rewrite/semantic_cycle_close.py      CHANGED
    ...the other seventeen                              clean

Any edit to those files — a comment would do it — changes
`implementation_source_sha256`, therefore `process_identity_sha256`, and the
frozen chain stops authenticating:

    process_identity_sha256 is '0c938177a348...', which is not the live
    Process-V2 identity b41640fa612c...

This is not a lint. `open_process_v2_t1_source` raises, and the production SMC
app calls it during startup, so a run on the optimized code does not start. The
mechanism is working exactly as designed: a content hash cannot know that a
change is behaviour-preserving, so it treats every change as invalidating.

Deploying would therefore mean re-deriving the Active8 corpus and the Gate-0
PASS under a new identity, and re-running everything banked against the old one.
That is wildly disproportionate to a speed change, and it would invalidate the
coverage results this sprint exists to serve.

**Do not attempt to route around the hash.** Installing the memo by monkeypatch
from an unhashed module would leave a process whose declared identity did not
describe the code that ran. The provenance chain is worth more than the 3.4x.

## Disposition

Preserved on `perf/exact-admission-mask-memoization`, not merged. The frozen
line keeps the original files and the original identity. If the corpus is ever
recompiled for an independent reason — a genuine process version change — this
should ride along, and its qualification harness and 600-state oracle are
committed so it can be re-verified in about five minutes without redoing any of
the analysis.

The retained, deployable levers are the ones that live OUTSIDE the hashed files:
the existing law cache (2.28x, already banked) and the persistent cross-attempt
cache (34% overlap measured, not yet built).

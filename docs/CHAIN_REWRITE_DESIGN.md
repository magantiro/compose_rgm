# Learned chain-continuation rewrite (`AlkylGraft`) — lipid-native COMPOSE extension

## Why
COMPOSE/RGM generates atom-by-atom. That is the right granularity for drug-like
small molecules (every atom high-information, entangled with ring topology). It is
the *wrong* granularity for lipid **tails**: a C8–C18 alkyl chain is a long,
self-similar, low-information run, and asking an atom-level model to emit it as N
independent `AtomInsert`s makes each step a fresh chance to drift (spurious N/O,
stray C=C). Empirically: de-novo-from-stub gave 1/12 clean tails; seed-from-lead
(tails pre-supplied) gave 10/10.

The fix is **not** deterministic/parametric tails (that is a menu, not a generator).
The fix is to **match generation granularity to the object**: atom-level where
information is dense (the ionizable head — a small-molecule-sized problem the RGM
is already good at), **chain-level where it is self-similar (the tail)**. The tail
distribution stays fully **learned and continuous** — we just let the model emit it
in one high-level move instead of 16 fragile ones.

This makes the generative problem *factorize*:
`P(lipid) = P(head) · P(tail1) · P(tail2 | attachment) · P(linker)` — a
small-molecule head model × low-dimensional learned tail models. Small molecules
give no such factorization; lipids hand it to us. Hence "lipids are easier."

## The rewrite family
Add `AlkylGraft` (and its CTMC reverse `AlkylPrune`) to the rewrite algebra.

```
AlkylGraft(
    anchor: int,              # existing carbon to grow from (chain terminus or branch point)
    length: int,              # new backbone carbons added, 1..L_MAX (e.g. 18)
    unsat: tuple[int, ...],   # 1-indexed cis-C=C positions along the new run; () = saturated
    branch_at: int = 0,       # backbone position of an alkyl branch (0 = none)
    branch_len: int = 0,      # branch carbons if branch_at > 0
)
AlkylPrune(anchor: int, run: tuple[int, ...])   # remove a pendant alkyl run (reverse move)
```
**v1 scope:** `unsat=()`, `branch_at=0` (saturated linear). v2 adds cis-unsaturation
+ mono-branch (covers oleyl/linoleyl/2-hexyldecyl). Everything is a *distribution
the model learns*, never hard-coded.

## Where it plugs into COMPOSE (integration surface)
1. **operators/tracelets** — define `AlkylGraft`/`AlkylPrune` dataclasses + `apply_*`
   (add `length` NULL-slot carbons as a chain from `anchor`, single bonds, terminal
   CH3; update implicit-H/valence). Index-stable like the other micro-ops.
2. **fiber (`_candidate_actions`)** — enumerate: for each eligible carbon anchor
   (open valence), for length ∈ 1..L_MAX, emit a graft. Reverse: prune pendant runs.
3. **factorized rate model** — new factorized head predicting the graft rate,
   factorized as `rate(anchor) · P(length | anchor, ctx) · P(unsat) · P(branch)`.
   The `P(length|…)` head is where the model *learns the tail-length distribution*.
4. **teacher-path decomposition** — when compiling a corpus lipid's teacher program,
   emit tails as **single `AlkylGraft` moves** (identify linear alkyl runs off the
   head/linker via region labels) instead of atom-by-atom, so training data uses the
   new move and the model learns to generate via grafts. Reverse-consistent with the
   CTMC (graft ↔ prune) for detailed-balance / Poisson-Bregman loss.
5. **sampler** — no change beyond recognizing the new action (rate → exponential
   waiting time → apply), same as every other rewrite.

## Correctness / framework fit
- CTMC balance: `AlkylGraft` is paired with `AlkylPrune` so the generator/teacher
  compilation stays reversible, exactly like insert/delete pairs today.
- Legality lives in the executor (open-valence anchor, valence-valid result), so the
  sampler cannot emit illegal chains — same contract as the existing operators.
- Nothing about the head or linker machinery changes; this is purely an additional
  legal move. Region-awareness already tells the model *where* a tail is.

## Payoff
- **De-novo tails become reliable** (clean chains by construction, few moves, no drift).
- **Full-distribution coverage** of tail length/saturation/branching, learned — not
  discrete enumerated points.
- **Efficiency**: a C16 tail = 1 move, not 16 — training and sampling both cheaper.

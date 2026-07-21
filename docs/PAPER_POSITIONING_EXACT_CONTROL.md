# Paper positioning: pathwise-meaningful, validity-closed molecular control

**(Corrected framing — supersedes the earlier "exact controllability" draft, which
overstated the guidance-exactness and the diffusion-can't-do-constraints claims.)**

**The reframe:** the point of a valid molecule at *every* step is not "valid
endpoints" (everyone gets those). It is **pathwise chemical semantics** — the
process's committed states are *complete, chemically valid, connected molecular
graphs*, and its jumps are *executable chemical rewrites*.

## Headline claim (defensible)
> COMPOSE learns a flexible-size, non-monotone Markov generator directly over valid
> connected molecular graphs. Because every transition is an executable chemical
> rewrite, molecular constraints and property oracles can act on meaningful states
> throughout generation — enabling pathwise-feasible editing, anytime candidates, and
> closed-loop molecular control in a single model.

## What is genuinely strong — three *native* capabilities
1. **Well-defined intermediate evaluation** — QED, fingerprints, substructure/similarity,
   many learned oracles evaluable after *every committed rewrite*.
2. **Structural constraints by construction** — scaffold preservation, required/forbidden
   motifs, connectivity, atom/bond budgets, enforced through the *legal event set*.
3. **Anytime molecular editing** — every intermediate is a usable candidate; sampling
   can be inspected, stopped, redirected, or resumed from a real input molecule.

**The differentiation is a *conjunction*, not an exclusivity.** No single existing
method has *all* of {complete-valid-**and-evaluable** committed states + flexible
non-monotone size + executable-rewrite editing + rule-closed constraints} native in
one process: ConStruct/PRODIGY preserve constraints but intermediates aren't complete
evaluable molecules; CoCoGraph guarantees validity but isn't a flexible editor; GrIDDD
is flexible-size but its intermediates are latents. Our edge = the combination is
native to one executable-rewrite process, not bolted on via projection/clamping/
rejection/a separate optimizer.

## What we DO NOT claim (honest scoping)
- **NOT exact conditional generation.** Exact Doob h-transform needs the true harmonic
  function (generally intractable); learned/SMC guidance is approximate; twisted-SMC's
  asymptotic correctness is *relative to the learned model + likelihood*, not the real
  molecular distribution; and twisted SMC is **not exclusive to us** (diffusion uses it).
- **NOT "diffusion can't do hard constraints."** PRODIGY (projection), ConStruct
  (constraint-preserving trajectories), CoCoGraph (guaranteed valid) already can.
- **NOT "real molecules."** Valence-valid connected 2D graph ≠ stable/synthesizable/
  meaningful. Say **"complete, chemically valid connected molecular graphs."**

## The exact statement (what a reviewer cannot knock down)
- **Rule-closed structural constraints are enforced EXACTLY by construction** (legal
  event set) — scoped to rule-expressible constraints (scaffold, substructure,
  connectivity, size). *Provable, and ours cleanly.*
- **Property-conditioned distributions are learned or guided — NOT claimed exact**
  without an exact h-transform or a valid sampling correction.

## Decisive experiments
1. **Pathwise hard-constraint satisfaction** — fixed scaffold, required/forbidden
   substructures, connectivity, size/composition budgets. Ours: exact by construction
   (already MEASURED 100% for similarity; scaffold constraint now implemented). Matched
   controls: **GrIDDD/FreeGress (direct conditional), proposal+rejection, projection/
   constrained-diffusion (ConStruct/PRODIGY)** — not just vanilla DiGress.
2. **Usable-oracle efficiency** = useful evals / all attempted evals. Favors us: we
   reject infeasible successors *in the fiber before scoring*, so ~all oracle calls land
   on valid feasible molecules. (Efficiency, not correctness.)
3. **Anytime performance** — oracle-calls/time to first feasible hit; best feasible
   reward as sampling proceeds.
4. **Editing behavior** — start from a lead, revise non-monotonically, stop at any
   intermediate candidate.
5. **Exactness demonstration ONLY where justified** — on a small *exhaustive* state
   space, compute the true h and verify the transformed generator numerically. Present
   *real* molecular guidance separately as approximate learned / finite-particle SMC.
6. **E1 unconditional** — the generator works (validity/uniqueness/novelty + marginals;
   FCD non-SOTA acceptable). B + ring-fix.

## Honest framing (top-tier, hard to knock down)
Framework + native-conjunction capability + **rule-closed exact constraints (provable,
scoped)** + competitive optimization at matched budgets + **honestly-approximate learned
guidance, verified exact only on a tractable toy.** No claim of exact conditional
generation, no claim that constraints are impossible elsewhere.

# Conditional controller design — value-guided SMC over the rewrite CTMC

**Lane:** `claude/generator-cond-uncond` · **Base:** Lineage B (native quotient,
clean dynamics; see `docs/HANDOFF_LIPID_GENERATOR_FROM_GENERATORS.md` §0.5 and the
`unconditional-base-is-lineage-b` memory) · **Status:** design (Task #9).

This is the deployable conditional method. The `griddd_valid_fiber_controller`
best-first search is only a **ceiling estimator + label generator** (it collapses
diversity); it is not the paper method. The paper method is a diversity-preserving
value-guided Sequential Monte Carlo (twisted SMC / Feynman–Kac) controller.

## 1. Why this method, and what is genuinely novel

Grounded in the deep-research survey (twisted SMC / FK-steering; Doob h-transform
& stochastic control for jump processes; DEFT/DRAKES/RTB reward fine-tuning;
Nisonoff discrete guidance). The **RGM-specific novelty** — defensible at a
top-tier venue — is that our CTMC is **validity-closed**: every intermediate
state is a complete, real, oracle-scoreable molecule. That uniquely enables:

1. **Exact per-step value on real molecules** — the twist/potential can call the
   true oracle at any state, no noisy surrogate over corrupted/partial graphs
   (unlike diffusion, where intermediate states are non-molecules).
2. **Hard feasibility in the legal action fiber** — similarity/scaffold/linker
   constraints are enforced by *removing violating rewrites from the fiber*, not
   a soft penalty; every particle is always feasible.
3. **Resampling over real molecules** — SMC resampling moves probability mass
   between genuine candidates, so diversity is preserved by construction and the
   scheme is asymptotically exact for the reward-tilted target.

## 2. Target distribution

Reward-tilted base law over trajectories terminating in valid molecules:
`π(x) ∝ p_B(x) · exp(r(x)/α) · 1[x ∈ F]`, where `p_B` is Lineage B's ancestral
CTMC law, `r` the oracle (QED / pen-logP / DRD2), `α` the temperature, and `F`
the hard feasibility set (e.g. Tanimoto(x, lead) ≥ 0.40). This is the DRAKES /
Feynman–Kac target; the Doob h-transform gives the exact steered generator
`Q*_xy = Q^B_xy · h(y)/h(x)` with `h(x)=E_B[exp(r(X_T)/α)·1_F | X_t=x]`.

## 3. Algorithm (particle SMC / FK)

Particles = molecular states. `N` particles, base rates from B.

```
init N copies of the lead (or B-sampled starts)
for t in 1..T (or until horizon):
  for each particle x_i:
    draw candidate successor y_i ~ B.base_rates(x_i)      # ancestral, canonical
    restricted to the HARD fiber F (illegal rewrites removed before sampling)
    weight w_i *= G_t(x_i -> y_i)                          # FK potential
  # resample when ESS < N/2 (systematic); reset weights
  x <- resample(y, w)
return the population (dedup by canonical SMILES) + best-feasible per lead
```

**FK potential `G_t` (three interchangeable value backends, in build order):**
- **V0 — reward-difference potential (start here):**
  `G_t = exp((r(y_i) - r(x_i))/α)`. Uses the exact oracle at each real
  intermediate. No training. Immediately shows the SMC diversity advantage over
  best-first at matched oracle budget.
- **V1 — learned twist / Doob value (the result):** amortize `h` with a small
  value net `V_θ(x) ≈ log h(x)`, trained via DEFT (regress toward bootstrapped
  future reward) or Relative Trajectory Balance on B-rollouts. Training labels
  already exist: the valid-fiber controller emits `(canonical_smiles, qed, sim)`
  per scored molecule. `G_t = exp(V_θ(y)-V_θ(x))` cuts oracle calls at inference.
- **V2 — short-rollout lookahead (ablation/oracle-rich regime):** estimate `h(y)`
  by a few B-rollouts from `y`; most accurate, most expensive.

**Temperature `α`:** anneal high→low (explore→exploit). Report an α sweep.

## 4. Constraints & diversity

- **Hard fiber F:** compute the legal successor set at each state, drop any whose
  canonical form violates the constraint *before* sampling — so proposals are
  never wasted on infeasible edits (this is where B's clean dynamics pay off:
  no self-Graft no-ops to burn budget on, unlike pancake).
- **Diversity:** population + systematic resampling + ESS control; optional
  reward-tempering / kernel penalty to avoid mode collapse. Report internal
  diversity (mean pairwise Tanimoto), #modes, novelty vs the training corpus.

## 5. Benchmarks, baselines, metrics (paper)

- **Tasks:** ZINC constrained QED optimization at Tanimoto ≥ {0.4, 0.6};
  penalized-logP (constrained); DRD2. Constrained QED is the headline.
- **Baselines:** GrIDDD (45.1% @ sim≥0.4), GCPN, MARS, GraphAF, GFlowNet(-TB).
- **Metrics:** success rate (≥ target at sim floor), **diversity**, **novelty**,
  under a **matched oracle-call budget** (reuse `griddd_conditional`'s exact
  accounting). Failure modes to pre-register and watch: oracle exploitation,
  diversity collapse, applicability-domain escape.

## 6. Repo interfaces to reuse (no new base training needed)

- Base: `load_factorized_rollout_checkpoint` on B; native sampler (clean, likely
  **zero calibration** — pending the calibration ablation).
- Legal fiber / canonical successors: `rewrite.kernel.de_novo_rewrite_system`,
  the CNOF fiber enumerator, `molecular_graph_to_smiles`.
- Oracle accounting + constraints: `experiments/griddd_conditional.py` (matched
  budget), `experiments/guided_rewrite_sampling.py` (rate tilting).
- Qualify B in the **`canonical_successor_native`** lane (not the pancake-pinned
  analytic path) — see Task #13.

## 7. Build phases

1. **SMC scaffold + V0** on B, hard fiber, matched budget → measure success +
   diversity vs the 33% best-first ceiling and vs GrIDDD. (No training.)
2. **V1 learned twist** (DEFT/RTB) trained on rollout labels → the reported
   result; α sweep, particle-count and SMC-vs-best-first diversity ablations.
3. Full task suite (pen-logP, DRD2) + baselines + oracle-budget curves.

## 8. What this is NOT (claim discipline)

Valid RDKit graph ≠ realistic chemistry; a higher success rate ≠ a better
molecule without diversity/novelty/AD checks. Report all-attempt denominators and
matched budgets. Asymptotic exactness is a property of the SMC target, not a
guarantee at finite N — report ESS and resampling diagnostics.

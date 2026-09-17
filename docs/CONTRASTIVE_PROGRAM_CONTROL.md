# Finite-budget contrastive control of stochastic graph-rewrite programs

Formulation only. No controller is implemented against this yet, and no oracle call has
been spent on it. Every quantity marked MEASURED is a number from this branch; every
quantity marked ASSUMED is a modelling choice that has not been tested; every quantity
marked OPEN is not yet derived.

The organising claim is that for a scarce-reward stochastic generator, control requires
tilting two different processes at once: the law over molecular programs, and the law
over which programs are paid for. COMPOSE makes the second tractable because its path is
an explicit sequence of manipulable, protected graph-program decisions.

---

## 0. What the evidence forces

Four measurements on this branch constrain the formulation and are carried through it.

**(M1) Discovery, not continuation.** Over eleven replayed T4 runs, the branch that
produced the run's best molecule ranked 3rd to 12th of 8 to 29 by reward history at the
moment the decision had to be made, on 8 of 11 runs. The three exceptions are
continuations of the already-leading branch. So a controller whose value depends only on
observed branch rewards is information-limited before discovery, and the outer objective
must value what an experiment TEACHES, not only what it finds.

**(M2) Protected options are necessary.** The strong JAK2 transformation is a single
coordinated program of median 14 primitive edits with median 0 ancestral edits, whose
internal prefixes violate the eligibility gate and which returns to eligibility only on
completion. Intermediate molecular states therefore must not be decision points.

**(M3) Program size is not the objective.** Autonomous programs of 16-17 primitives exist
and score -7.2 to -7.8, while bank programs of the same size reach -11.7;
corr(size, score) is -0.059 autonomous against -0.352 for the bank. Nothing in the
formulation may give credit to program length as such.

**(M4) The intervention algebra is partial.** Measured over 147 teacher subgoals: element
/ charge / hydrogen / degree resample independently at rate 0.810, bond order at 1.000,
and `output_count` (scale), topology, attachment presence and dependency parent NEVER
resample independently. Binding is a separate coordinate whose support is
target-dependent: 74% of BRAF patches admit more than one legal binding, and **0% of JAK2
patches do**.

M4 is the load-bearing one. It means the option cannot be factored into independent
scalar decisions, and any formulation that writes one is describing a machine we do not
have.

---

## 1. Inner process: a semi-Markov law over protected programs

### 1.1 State, options, and time

The inner path is

    tau = (G_0, O_0, G_1, O_1, ..., G_K),    G_{t+1} = T(G_t, O_t).

`O_t` is a COMPLETE PROTECTED PROGRAM: a `CompleteRegionProgram` that opens with one
patch, optionally continues, and commits only at STOP. Its internal primitive actions and
intermediate graphs are *not* part of the path, by M2. `T` is the exact production
executor and is deterministic, so

    K(G_{t+1} | G_t, O_t) = delta_{T(G_t, O_t)}.

Three clocks must be kept apart, and only the third is scarce:

| clock | unit | governs |
| --- | --- | --- |
| primitive time | one executor action | internal to an option; never a decision point |
| decision time `t` | one option | the inner path measure |
| oracle time `b` | one docking call | the outer budget and the real objective |

An option of 2 primitives and an option of 20 primitives cost the same one oracle call.
This is the sense in which the process is semi-Markov: options have variable duration in
primitive time and unit duration in oracle time. No discounting over primitive time
appears anywhere, which is what stops M3 from re-entering through the back door.

### 1.2 Where the randomness lives, and the reference law (ASSUMED)

    P_ref(tau) = mu(G_0) * prod_t q_ref(O_t | G_t)

with no executor term, because `T` is deterministic and unchanged under control. This is a
real structural simplification relative to diffusion fine-tuning, where the transition
kernel is itself stochastic and controlled: here the executor factor is a delta on both
sides and cancels, leaving only the option law in any likelihood ratio.

**Support is a hard requirement, not a detail.** Control can only reweight what the
reference already emits: if `q_ref(O|G) = 0` for a legal option, no reward tilt of any
strength reaches it, and the option is unreachable for the entire run. A purely
route-derived law is exactly the kind that would assign zero mass to legal novel options,
because it is fitted on 77 trajectories. The reference is therefore a mixture

    q_ref  =  (1 - eps) * q_route  +  eps * q_block( . | O_cur , G ),

and the second component is NOT a uniform draw over the grammar. MEASURED: the uniform
law `q_legal`, which draws every token from the fiber `token_domain` returns, assigns
strictly positive mass to every admitted intervention sibling -- so absolute continuity
holds and there are no literal zeros -- at a median log-probability of -196.8, which is
10^85 expected draws against a budget of 10^3. Per target that is 10^80 for JAK2 and
10^108 for PARP1. A teacher patch itself sits at -105.2, or 10^46.

    q_legal:  support everywhere, accessibility nowhere.

So the exploration floor has to be LOCAL IN PROGRAM SPACE. `q_block` conditions on the
current patch, picks one semantic coordinate, and resamples that coordinate's minimal
closure from its own fiber while holding the compatible complement fixed. MEASURED: a
median log-probability of -3.18, i.e. 24 expected draws, with probability 0.744 of being
seen within 32 draws. It is a genuine stochastic process over the legal fiber, so

    pi << q_ref   whenever   pi only reweights options reachable by some closure,

and it is reachable inside an actual budget.

STRUCTURAL CONSEQUENCE, not a detail: `q_block` is a KERNEL over patches, not a marginal
law over options given `G`. The reference process is therefore patch-local, and the inner
path measure is over (current patch, intervention) pairs rather than over freshly drawn
options. This is the formal price of buying reachability, and it should be carried
explicitly wherever the path measure is written.

Note the convergence. The intervention closure was derived to make contrasts
IDENTIFIABLE, by cancelling parent terms. It turns out to be the only tractable
broad-support reference as well. One object, two independent justifications.

`eps` is a support floor and must NOT be tuned against T4 scores; tuning it on the
benchmark would make a reachability guarantee a fitted quantity. Report sensitivity
separately.

STILL OPEN: `q_route` is not yet fitted, so the mass it places on admitted siblings is
unmeasured. That audit cannot invalidate the siblings -- they demonstrably exist and
execute -- it can only show that the route law alone cannot reach them, which is what the
mixture exists to fix.

### 1.3 The option factorisation the measurement licenses

Write `O = (H, theta, beta)`:

- `H` -- structural hypothesis (region and operation family);
- `theta` -- realisation parameters, which do NOT factor into independent scalars;
- `beta` -- the binding, i.e. where the patch attaches, enumerated by
  `attachment_bindings` and drawn from a target-dependent support.

By M4, `theta` splits into a free part and coupled blocks:

    theta = ( theta_free , theta_blocks )
    theta_free   = ( element/charge/hydrogen/degree , bond order )     independent
    theta_blocks = ( scale , topology , dependency structure )          move together

so the reference law factors as

    q_ref(O|G) = q_ref(H|G)
               * q_ref(theta_blocks | H, G)
               * prod_{c in free} q_ref(theta_free,c | H, theta_blocks, G)
               * q_ref(beta | H, theta, G).

This is the honest version of "hierarchical credit". Credit flows to `H`, to each BLOCK,
and to each free coordinate -- not to fifteen primitives, and not to a factorial grid
over coordinates that cannot move independently.

### 1.4 Reward tilt and the likelihood ratio

For terminal utility `U` and inverse temperature `beta_T`,

    P*(tau)  =  (1/Z) P_ref(tau) exp( beta_T * U(G_K) ),

    h(G)     =  E_{P_ref}[ exp( beta_T * U(G_K) ) | G ],

    pi*(O|G) ∝ q_ref(O|G) * h(T(G,O)) / h(G).

Because `T` is deterministic, `h(T(G,O))` is an ordinary function evaluation rather than
an expectation over successors -- the Doob ratio is computable wherever `h` is. The
Radon-Nikodym derivative is

    dP_pi/dP_ref (tau) = prod_t [ pi(O_t|G_t) / q_ref(O_t|G_t) ]

and, by 1.3, factors over `H`, blocks, free coordinates and binding. That factorisation
is the formal statement of hierarchical credit.

**Path cost (OPTIONAL).** Since many option sequences reach the same `G*`, terminal
reward alone leaves their relative probability inherited from `q_ref`. A path cost
`U(G_K) - lambda C(tau)` can penalise fragile execution or gratuitous destruction. By M3,
`C` must NOT be program length.

---

## 2. Outer process: a finite-budget experimental path

### 2.1 State and filtration

    S_t = ( F_t , A_t , Pi_t , U*_t , b_t )

- `F_t` frontier of executed, eligible, undocked molecules;
- `A_t` measured archive of (molecule, score) pairs;
- `Pi_t` posterior over structural effects (section 3);
- `U*_t = max_{x in A_t} U(x)` incumbent;
- `b_t` remaining docking calls.

The filtration is generated by ORACLE OUTCOMES ONLY:

    calF_t = sigma( E_0, Y_0, ..., E_{t-1}, Y_{t-1} ).

Everything computable without the oracle -- exact execution, eligibility (QED, SA,
similarity), structural prior probability, diversity, the legal fiber `token_domain`,
count of legal continuations -- is `calF_t`-measurable at zero cost and may be used
freely. Nothing else about an undocked molecule is known. This is the formal content of
"do not train a fake docking oracle": a surrogate would inject non-`calF_t` information
into the belief and the pretence would not survive the terminal objective.

### 2.2 The objective is an extreme value

    J(S_B) = max_{x in A_B} U(x).

Not an average. M1 is what makes this consequential: a branch producing `0,0,0,0,+3` beats
one producing `+.1,+.1,+.1,+.1`, and any Bernoulli-rate statistic ranks them the other
way. The rate-based allocator built on this branch failed exactly here, losing to the
historical policy on 6 of 7 runs.

### 2.3 An intervention bundle

An experiment is not a set of molecules; it is a DESIGN:

    E = ( G , H , theta_base , coordinate-or-block c , { v_1, ..., v_m } )

realised as options `O^(i)` that share `G, H` and all of `theta_base` except `c`, which
takes value `v_i`. Admissibility follows M4: `c` may be a free coordinate (element, bond
order), a coupled block resampled coherently, or the binding `beta` -- and on JAK2 the
binding offers no alternatives, so bundles there must contrast free coordinates or blocks.
`|E| = m` oracle calls.

### 2.4 Information arriving from a call

`Y_E = (Y_1,...,Y_m)`, `Y_i = U(T(G,O^(i))) + noise`. The update is

    S' = Update(S,E,Y_E):   A' = A ∪ {(x_i,Y_i)},  U*' = max(U*, max_i Y_i),
                            Pi' = posterior(Pi | E, Y_E),  b' = b - m.

`Pi'` is where a matched bundle pays: see 3.2.

### 2.5 Tilted search law and the Bellman recursion

With reference search law `rho_0(E|S,b)` and search path `Xi`,

    Q*(Xi) ∝ Q_0(Xi) exp( beta_S * J(S_B) ),
    H_b(S) = E_{Q_0}[ exp( beta_S * J(S_B) ) | S, b ],

    rho*(E|S,b) ∝ rho_0(E|S,b) * E_{Y_E}[ H_{b-|E|}( Update(S,E,Y_E) ) ] / H_b(S).

The risk-neutral form is the finite-horizon recursion

    V_b(S) = max_E  E_{Y_E}[ V_{b-|E|}( Update(S,E,Y_E) ) ],      V_0(S) = U*.

`rho*` is the Gibbs solution of `max_rho E_rho[Q_b] - (1/beta_S) KL(rho || rho_0)`, which
is why the KL-regularised and Doob forms coincide.

**The expectation over `Y_E` is the whole point.** The molecules have not been docked, so
choosing `E` must integrate over outcomes not yet seen. A bundle earns its place both by
possibly containing a winner and by changing where later calls go. M1 says the second term
is not optional: on 8 of 11 runs no amount of exploitation of observed branch rewards
would have found the winner.

---

## 3. The belief model, and why matching is required

### 3.1 Hierarchical effects (ASSUMED)

    Delta U  =  theta_region + theta_H + theta_block + sum_c theta_free,c
                + interactions + epsilon,        epsilon ~ docking noise.

Levels share strength hierarchically, so one outcome updates `H`, its block and its
coordinate by different amounts. With ~20 observations this must be Bayesian with honest
posterior width; a point estimate here is the mistake that produced a `q_theta` ranking
elite JAK2 constructions at median 156 and poor ones at 11.

### 3.2 Why the bundle must be matched (this is the derivation, not a preference)

For two options from the SAME parent differing only in coordinate `c`:

    Y_i - Y_j = ( theta_c(v_i) - theta_c(v_j) ) + ( eps_i - eps_j ).

Every parent-level term -- `theta_region`, `theta_H`, and the parent's own quality --
cancels exactly. For two UNMATCHED observations it does not, and `theta_c` is confounded
with the parent effect. This is not hypothetical: the soft allocator on this branch scored
AUC 0.693 on an unmatched label and collapsed to 0.556-0.579 once stratified, because
roughly two thirds of its apparent skill was the confound.

**Corollary (MEASURED consequence).** Historical archive pairs are OBSERVATIONAL. The
search chose what to generate, so a "matched-looking" historical pair does not license the
cancellation above. Archive contrasts may initialise `Pi` and may be used for predictive
validation; only prospective matched bundles identify `theta_c`.

---

## 4. An implementable approximation

The exact recursion is intractable. A one-step lookahead gives

    Q_b(S,E) ≈ E_{Y_E}[ max(U*, max_i Y_i) - U* ]      (discovery)
             + lambda_b * I( Theta ; Y_E | calF )       (learning)

The first term is expected improvement of an extreme value over the bundle -- correct for
section 2.2 and different from per-molecule expected improvement. The second is the mutual
information between the structural effects and the outcomes, which a matched bundle
maximises per call precisely because of 3.2.

**Status of `lambda_b`: OPEN.** It should be INDUCED by the horizon -- information is worth
only what it changes about the remaining `b - |E|` calls, so `lambda_b -> 0` as `b -> 0`
and grows with `b`. A principled surrogate is the gap between the two-step and one-step
values. It must not be a hand-tuned exploration constant: an arbitrary constant of exactly
this kind (0.35) dominated the quality term in the replay allocator and drove two of three
continuation runs to zero pre-discovery share. Deriving `lambda_b` is a named deliverable,
not a detail.

Global-to-local behaviour is then a consequence rather than a schedule: large `b` favours
informative global blocks, small `b` favours exploiting the best basin. BRAF (local
refinement compounding), JAK2 (one protected coordinated program) and 5HT1B (right family,
unresolved binding) should all be expressible by the same rule at different `b` and
different `Pi`.

**Progressive widening** is the resolution ladder over section 1.3: open `H`, then its
block, then free coordinates, then binding -- and only once a level has earned evidence.
By M4 the ladder has a target-dependent floor: on JAK2 the binding rung does not exist
(0% of patches admit an alternative), so JAK2 resolution must come from blocks. The
5HT1B-style descent into binding is available there (31%) and on BRAF (74%).

---

## 5. What the route corpus supplies

The 77 routes compress to 147 structural subgoals. They give the CO-ORDINATE SYSTEM and
the REFERENCE DYNAMICS, never the answers.

| object | source | status |
| --- | --- | --- |
| which decisions form one protected option | dependency-region decomposition; M2 | MEASURED |
| which coordinates are free vs coupled blocks | `token_domain` + decode; M4 | MEASURED |
| legal counterfactual family for a coordinate | `token_domain` frozen fiber | MEASURED |
| binding support per target | `attachment_bindings`; M4 | MEASURED |
| `q_route(H | G)` and `q_route(theta | H,G)` | route corpus, context-relative roles | TO FIT |
| prior over `Theta` (structural effects) | archive contrasts, observational | TO FIT |
| widening order | hierarchy in 1.3 | DERIVED |

The distinction that must survive into every artifact:

    route-informed DIAGNOSTIC     evaluated route available -- debugging only
    route-informed DEVELOPMENT    other routes available, this cell's route withheld
    held-route QUALIFICATION      no route from this target contributes to q_route

A result from the first is never reported as autonomous discovery.

---

## 6. What is not yet established

1. `lambda_b` is not derived (section 4).
2. The BLOCK-INTERVENTION GENERATOR does not exist. M4 says scale, topology and
   dependency need coherent re-decode, and on JAK2 those carry most of the available
   contrast because binding offers none. Until it exists, a JAK2 bundle can vary only
   element identity and bond order.
3. `q_route(H|G)` has not been fit at the hypothesis level; the fitted `q_theta` was at
   `(site, mode)` level and was measured to be value-blind.
4. No evidence yet that a first contrastive batch improves the targeting of a second.
   That is the gate before any scaled run, and it needs prospective calls -- a replay
   over passively collected history cannot supply it, because the counterfactual scores
   an active design requests were never measured.

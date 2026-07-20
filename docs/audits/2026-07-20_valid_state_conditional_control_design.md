# Valid-State Conditional Control for COMPOSE RGM

**Status:** architecture decision for the first production QED experiment; the
literature and implementation audits remain additive, not permission to weaken
the execution gates in the GrIDDD contract.

## Decision

The semantic object being conditioned is the rate of a **canonical molecular
successor**, not a padded action slot and not an unquotiented rewrite alias.
Rewrite families remain useful for scalable factorization, proposal allocation,
and interpretation. They must not create a second, inconsistent conditional law.

For valid connected state `x`, condition `z`, time `t`, family `c`, and canonical
successor `y`, write the qualified unconditional generator as

```
q0_c(x, y, t) = sum of the rates of executable marks in family c reaching y
lambda0_c(x, t) = sum_y q0_c(x, y, t)
p0_c(y | x, t) = q0_c(x, y, t) / lambda0_c(x, t).
```

The conditional residual is separated without double counting:

```
qz_c(x, y, t) = q0_c(x, y, t)
                 * exp(u_c(x, t, z))
                 * exp(v_c(x, y, t, z) - log E_p0_c[exp(v_c)]).
```

`u_c` changes the total productive hazard of a family. The centered `v_c`
redistributes that mass among its canonical successors while leaving the family
hazard unchanged. Consequently,

```
lambdaz_c = lambda0_c * exp(u_c)
qz(x, y, t) = sum_c qz_c(x, y, t).
```

This gives the model both decisions the task requires:

1. **what kind of revision is useful now** (grow, shrink, retype, Graft,
   bond/ring revision); and
2. **where and how it should occur** among executable successors.

Because the within-family residual is centered, the same preference is not
silently counted once in the family head and again in the mark head. Aliases
that reach the same molecule are aggregated before the molecular generator is
interpreted.

## What every-valid-state chemistry uniquely makes practical

At each committed state, COMPOSE can compute real molecular quantities rather
than predictions on a masked, disconnected, or chemically invalid object. The
conditional adapter can therefore consume:

- the requested target `z`;
- the current measured property `r(x)`;
- the signed target gap `z - r(x)`;
- progress/time and remaining event budget;
- exact feasibility margins required by the task, such as similarity to the
  lead and scaffold preservation;
- optionally, a learned value-to-go rather than only the immediate property
  difference.

The target gap is potentially important. A fixed QED target alone asks the
network to infer from the graph whether it is below, near, or above the target.
Supplying the exact current value is Markovian—it is a deterministic function
of the valid state—and makes upward, downward, and stop/revise behavior easier
to learn. It is nevertheless a state-feedback property query. The primary
native-direct GrIDDD protocol therefore uses the target, valid graph, lead, and
time only. Exact current QED, exact target gap, current Tanimoto margin, or
successor values belong in a separately labeled state-feedback/controller
ablation with every query charged. A learned graph-only surrogate is allowed in
the direct model if it was trained without evaluation-time oracle calls.

The principled non-myopic control is a generalized Doob transform. If
`h(x,t,z)` estimates the probability or exponentiated value of eventually
satisfying the target, then ideally

```
qh(x, y, t | z) = q0(x, y, t) * h(y,t,z) / h(x,t,z).
```

An immediate QED delta is only a local approximation to this value ratio. It
can become greedy, damage lead similarity, or prefer a short-term improvement
that blocks a better later rewrite. COMPOSE's valid intermediates let us train
or evaluate `h` directly on real molecules along the path.

## Efficient production architecture

### Primary direct-conditioned model

- Warm-start the qualified unconditional encoder and productive generator.
- Add small conditional residual adapters for family hazards and centered
  within-family canonical-successor preferences.
- Condition the primary native-direct model on target, valid current graph,
  lead representation, and time/progress. Add exact current property, target
  gap, and constraint margins only in the separately charged state-feedback
  ablation.
- Use classifier-free condition dropout during training, but report native
  direct-conditioned sampling as the primary GrIDDD-comparable arm.
- If classifier-free amplification is evaluated, apply it separately to the
  family-hazard residual and centered within-family residual. Do not stack
  family temperatures, hand-set channel budgets, and successor-oracle tilts
  into the headline direct arm; that would obscure which generator is being
  tested.
- Permit the direct model to change total and family hazards. Flexible-size
  targeting requires learning when insertion, deletion, or topology revision
  should occur, not merely resampling locations under a fixed clock.

### Valid-successor controller

- Draw a small proposal set from the qualified base/direct generator.
- Execute, canonicalize, and deduplicate the proposals.
- Score only the unique valid successors with QED and lead constraints.
- Resample with a positive score/value tilt and report every oracle call.
- Preserve the base total hazard in the matched controller experiment so the
  mechanistic comparison isolates successor selection from event-clock changes.

This finite-proposal controller is an auditable Monte Carlo approximation, not
an exact Doob transform. It is valuable because it tests the valid-state
advantage before investing in a learned value network.

Lead similarity is a terminal/global task condition, not automatically a hard
barrier at every intermediate state. The residual/value model should see the
lead, scaffold, current similarity, and progress. A hard all-state similarity
cut can make useful temporary detours unreachable. Terminal feasibility and an
anytime best-feasible memory are evaluated separately.

### Amortization path

If the valid-successor controller improves matched QED success, distill its
successor preferences or learn a small `h`/advantage adapter while freezing the
large unconditional backbone. This converts an oracle-expensive proof into a
one-forward-pass conditional sampler. It also keeps the direct, controller,
and combined evidence separable.

## Minimum unconditional handoff

Conditional work does **not** require a publication-frozen FCD optimum. It does
require:

1. executable validity-closed support and a connected-state invariant;
2. canonical molecular-successor aggregation and virtualized molecular
   self-events;
3. calibrated productive family hazards, especially nonzero useful mass for
   insert, delete, retype, Graft, bond revision, and ring revision;
4. a stable time/clock interface and flexible-size source process;
5. canonical state keys, exact transition execution, caches, and pathwise
   accounting;
6. sufficient proposal coverage: guidance cannot select a useful rewrite that
   the base never proposes.

Family top-1 accuracy alone is not the handoff criterion. The relevant gate is
whether productive successor mass, support coverage, and rollouts are credible
enough to expose useful revisions under the target.

## Implementation-delta audit: `FactorizedTraceletRateModel`

### What exists now

The current model already has the right computational skeleton but not the
required scientific separation:

- `FactorizedMarkBatch` contains `property_condition_values [B,P]` and
  `property_condition_mask [B,P]`. These are target values; it has no current
  property, gap, lead similarity, scaffold, or event-budget tensors.
- `property_condition_encoder` maps the concatenated target values and masks,
  `[B,2P] -> [B,H]`, and adds that vector to the time embedding. Its final layer
  is zero-initialized, so a shape-compatible unconditional checkpoint initially
  gives the same generator.
- The resulting context is injected into every message-passing update and the
  global representation. It consequently changes the total-hazard head,
  family head, and all mark heads through one opaque path. There is no tensor
  boundary proving which conditional component owns family mass versus
  within-family allocation.
- The optimizer includes every trainable model parameter. After the first
  conditioned training step, the unconditional generator is no longer frozen.
- `FactorizedMarkPrediction` exposes one total hazard, normalized family
  probabilities, and one selected-mark log probability. It does not expose a
  public, immutable component table on which a sidecar residual can operate.

The zero-initialized target encoder is useful pilot machinery, but it is not the
v1 conditional contract.

### Exact v1 module boundary

Keep `FactorizedTraceletRateModel(property_condition_dim=0)` as the strictly
loaded, frozen base. Add a separate
`FactorizedConditionalResidualAdapter(base, condition_spec)` checkpoint. The
only parameter-free change to the base class is a public component method that
returns a `FactorizedRateComponents` record:

```
node_state                  [B,N,H]
global_state                [B,H]
pair_state                  [B,N,N,H]
base_total_hazard           [B]
base_family_log_probability [B,F]
action_masks                same shapes as the action tables below
base_action_logits          same shapes as the action tables below
base_action_log_partitions  [B,F]
canonical_successor_groups  aligned group IDs / sparse alias map
```

`F=10` is the existing number of rewrite families. The method adds no parameter
and therefore does not change an unconditional state dict or checkpoint hash.
The canonical group map must come from the same executor/support compiler that
qualified the unconditional backbone; a padded mark slot is not a permissible
replacement.

The adapter batch adds optional tensors, all with dataclass defaults of `None`
so old collators remain source-compatible:

```
condition_target_values    [B,P]
condition_target_mask      [B,P]
condition_current_values   [B,P]  # charged state-feedback mode only
condition_current_mask     [B,P]
condition_gap_values       [B,P]  # target-current under one frozen normalizer
events_used_fraction       [B,1]
events_remaining_fraction  [B,1]
lead_similarity            [B,1]  # charged state-feedback mode only
similarity_margin          [B,1]
scaffold_intact            [B,1]
```

The context encoder consumes `[target, current, gap, target_mask,
current_mask, existing time features, progress, constraint features]`. Missing
features are zero with a false mask. An all-missing target gate multiplies every
residual by zero, making the unconditional branch exact by construction rather
than by a learned convention.

The new outputs are deliberately two-level:

```
family_log_hazard_residual  Delta_f [B,F]
within_family_residuals     delta_f(m), aligned to each legal action table
```

The residual mark tables exactly match the current tables:

```
grow_root            [B,T]
grow_connected       [B,N,3,T]
atom_delete          [B,N]
atom_restate         [B,N,T]
bond_reorder         [B,N,N,3]
bond_reroute         [B,N,N]
cycle_insert         [B,C]
cycle_attach         [B,N,A]
ring_system_grow     [B,R]
ring_system_delete   [B,Dmax]
ring_system_restate  [B,Smax]
```

Here `T` is the CNOF atom-type count, and `C`, `A`, `R`, `Dmax`, and `Smax` are
the existing batch/catalog widths. Ring grow also needs residuals at its nested
placement table and semantic electronic-category table (or exact candidate
table in `catalog_exact` mode); conditioning the ring template alone would not
condition the complete within-family mark. All residual heads have zero final
weights and biases.

After executor masking and canonical alias aggregation, compute

```
lambda0_f = H0 * pi0_f
lambdaz_f = lambda0_f * exp(Delta_f)
pz_f(m)   = softmax_m(log p0_f(m) + delta_f(m))
qz(x,y)   = sum over legal (f,m) reaching canonical y of lambdaz_f * pz_f(m).
```

There is no extra global-hazard residual in v1: the ten `Delta_f` values own
all conditional clock/family changes. The within-family softmax centers
`delta_f` automatically and cannot change `lambdaz_f`. This is the mechanical
anti-double-counting guarantee. Classifier-free amplification, if enabled, is
`Delta <- s Delta` and `delta <- s delta` before the within-family softmax; it
is never stacked with family temperatures, hand budgets, or an oracle tilt in
the primary arm.

### Losses

The primary objective remains the existing Poisson Generator-Matching Bregman
loss, but it must use the canonical-successor rate `qz(x,y_teacher)` rather than
one arbitrary alias mark:

```
L_GM = mean w * [Lambda_z - r_teacher * log qz(x,y_teacher)]
Lambda_z = sum_f lambdaz_f.
```

Use a small decomposable Poisson-KL anchor to prevent unsupported drift from the
qualified base:

```
L_anchor = sum_f [
  lambdaz_f log(lambdaz_f/lambda0_f) - lambdaz_f + lambda0_f
  + lambdaz_f KL(pz_f || p0_f)
].
```

No separate family cross-entropy or mark cross-entropy is used in v1; those
would reweight the two decisions a second time. Condition dropout supplies the
unconditional examples, while the structural zero gate makes their equality
exact. Only adapter parameters enter the optimizer.

### Checkpoint compatibility

Adding the optional batch tensors does not alter a checkpoint. Expanding the
existing `property_condition_encoder` from `2P` inputs to current/gap/progress
features would change its first-layer shape and break strict loading. The
sidecar adapter avoids that: the unconditional checkpoint loads strictly and
is hash-identical, then the independently versioned adapter loads separately.
Existing conditional checkpoints remain readable through the legacy path but
are not v1 residual checkpoints. The current compatible-name/shape initializer
can prototype an in-place extension, but it is not sufficient evidence of exact
base recovery and must not be used for the headline run.

### Bounded test plan before any large run

1. Strict-load a real qualified unconditional checkpoint into the frozen base;
   assert no missing/unexpected tensors and an unchanged file SHA.
2. On the 24-state real-chemistry panel, assert zero-initialized and all-target-
   missing adapter outputs match base total hazards, family laws, every legal
   mark probability, and sampled marks under matched RNG.
3. Perturb `Delta` only and prove within-family probabilities are unchanged;
   perturb `delta` only and prove each family hazard is unchanged. Adding a
   constant to every legal `delta` in one family must change nothing.
4. Compare masked/canonical adapter rates and the Poisson loss against exhaustive
   enumeration on tiny fibers, including cross-alias successors, zero-self
   transitions, Graft automorphisms, and ring aliases.
5. Run one optimizer step and prove every frozen-base tensor is bitwise
   unchanged and only adapter tensors receive gradients.
6. Test `subbatch`, device transfer, pinned memory, missing-feature masks,
   frozen normalization, and `gap == target-current` in the same units.
7. Assert CFG scale 0 equals base and scale 1 equals native conditional;
   separately test family and mark scaling.
8. Overfit a 64-example synthetic/real microset with known useful families and
   locations; require target-order monotonicity without family collapse.
9. Run an 8-lead, 2-seed, 20-candidate smoke with all-step validity, similarity
   cliffs, size drift, insert/delete cycling, Graft thrashing, terminal versus
   anytime output, and complete oracle-call accounting.
10. Launch the frozen 800-lead protocol only after the unconditional manifest
    passes and the exact benchmark lead manifest authorizes Protocol A.

## First bounded experiments

1. **Channel audit:** condition family hazards only (`u`, `v=0`).
2. **Location audit:** condition centered within-family successor preferences
   only (`u=0`, `v`).
3. **Joint model:** condition both with exact centering.
4. **Direct versus controller versus combined:** keep the already frozen two-
   protocol fairness boundary.
5. **Target-gap ablation:** target alone versus target + current QED + gap.
6. **Lookahead ablation:** immediate valid-successor score versus learned
   value-to-go, only if the local controller shows greedy failures.
7. **Mechanism diagnostics:** productive family hazard ratios, selected-family
   frequencies, canonical-successor entropy, QED improvement, similarity,
   all-attempt validity, oracle calls, wall time, size drift, backtracking, and
   best feasible anytime candidate.

Failure criteria include channel collapse, excessive delete/insert cycles,
Graft thrashing, property gain obtained by similarity loss, proposal-support
starvation, or a high padding-call fraction.

## Alternative-design audit

- **One global conditional family head:** necessary but insufficient. It learns
  when to grow/delete/restate but cannot identify the useful site, bond,
  template, placement, or electronics. Keep it as the `u`-only ablation.
- **Within-family conditioning only:** preserves the unconditional event clock
  and cleanly tests location control, but cannot allocate mass toward the
  flexible-size families required by GrIDDD-like targeting. Keep it as the
  `v`-only ablation.
- **Unstructured conditional network:** the current pilot path is compact, but
  family and mark effects are not identifiable and the base drifts. Reject for
  v1 evidence.
- **Multiplicative factorized residual:** selected for v1. Positive base rates
  stay positive, absent support stays absent, zero residual exactly recovers the
  qualified generator, and family-versus-mark ownership is testable.
- **Classifier-free direct conditioning:** useful as a validation-selected scale
  ablation. Predictor-free guidance has primary precedent in FreeGress and in
  general discrete guidance, but large scales alter both hazard and entropy;
  native scale 1 remains the primary result.
- **Exact Doob `h` transform:** the non-myopic mathematical target. It requires
  the future success probability/terminal value for the current state and every
  possible successor; those transition quantities are generally intractable.
  Use exact calculations only on tiny fibers and train a value adapter only
  after the local controller demonstrates benefit.
- **Immediate successor-oracle tilt:** cheap to prototype with `K` proposed
  valid successors, but greedy. It can spend similarity for transient QED,
  overvalue dead ends, and double-count control when stacked on a direct model.
  It remains Protocol B with exact proposal, deduplication, selection, and
  padding-call ledgers.
- **Per-channel temperatures or event budgets:** useful stress tests for a
  miscalibrated base, not primary conditioning. They change the learned CTMC,
  can conceal family-rate failures, and duplicate `Delta_f`.
- **Time/progress conditioning:** required. The same edit can be helpful early
  and destructive near termination; remaining events also determine whether a
  temporary similarity detour is recoverable.
- **Hard similarity/scaffold masks:** enforce immutable atom/bond scaffolds only
  when violation is truly irreversible under the executor. Morgan Tanimoto
  `>=0.40` is a terminal, non-hereditary event, so imposing it on every state is
  not an exact constraint and can remove recoverable paths.
- **Anytime feasible memory:** retain the best already visited candidate that
  meets terminal QED/similarity. This is unusually safe in COMPOSE because every
  committed state is a valid molecule. It changes the output policy, not the
  generator, and therefore needs a separate terminal-versus-anytime table.

The amortized residual requires one graph encoding and masked residual tables.
No property oracle or materialized successor fiber is needed in native-direct
mode. Exact Doob control requires the full canonical successor fiber and an
exact `h`, so it is exponentially/combinatorially unattractive outside small
tests. Approximate value guidance draws `K` base proposals, executes and
canonical-deduplicates them, and scores the unique set in a batch; its cost and
bias are explicit functions of `K`. Top-k truncation or proposal resampling is
an approximation and must not be described as the exact conditioned CTMC.

## Evidence boundary

Generator Matching supports learning generators of arbitrary Markov processes
and their superpositions. Discrete Guidance supplies the CTMC rate-tilting
precedent. Generalized `h`-transform fine-tuning supplies a precedent for small
conditional adapters on a frozen unconditional model. COMPOSE's methodological
claim is not that any one of these ingredients is new by itself; it is the
validity-closed stochastic rewrite generator and the ability to apply these
controls over executable, flexible-size molecular successors throughout the
trajectory.

Primary references:

- Holderrieth et al., *Generator Matching: Generative Modeling with Arbitrary
  Markov Processes*, ICLR 2025:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/819aaee144cb40e887a4aa9e781b1547-Paper-Conference.pdf
- Nisonoff et al., *Unlocking Guidance for Discrete State-Space Diffusion and
  Flow Models*, ICLR 2025:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/597254dc45be8c166d3ccf0ba2d56325-Paper-Conference.pdf
- Denker et al., *DEFT: Efficient Fine-tuning of Diffusion Models by Learning
  the Generalised h-transform*, NeurIPS 2024:
  https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf
- Corstanje, van der Meulen, and Schauer, *Conditioning Continuous-Time Markov
  Processes by Guiding* (including jump processes in discrete state spaces):
  https://arxiv.org/abs/2111.11377
- Ho and Salimans, *Classifier-Free Diffusion Guidance*:
  https://arxiv.org/abs/2207.12598
- Schiff et al., *Simple Guidance Mechanisms for Discrete Diffusion Models*:
  https://arxiv.org/abs/2412.10193
- Ninniri, Podda, and Bacciu, *Classifier-Free Graph Diffusion for Molecular
  Property Targeting* (FreeGress): https://arxiv.org/abs/2312.17397
- Ninniri, Podda, and Bacciu, *Graph Diffusion that can Insert and Delete*
  (GrIDDD): https://arxiv.org/abs/2506.15725
- Franke et al., *Generative Molecular Morphing for Flexible-Size Design via
  Unbalanced Optimal Transport* (Morph): https://arxiv.org/abs/2606.07239

# Paper reframe — RGM as a control-closed molecular substrate (anytime Pareto editing)

> ## ⚠️ SUPERSEDED THESIS — 2026-07-28
> **The framing/positioning in this document is SUPERSEDED by
> [`PAPER1_FRAMING_AUTHORITATIVE.md`](PAPER1_FRAMING_AUTHORITATIVE.md)** (RGM-first, trans-dimensional).
> The control-substrate thesis it proposes (control as the paper's starting definition). Control is now the strongest CONSEQUENCE of the learned process, introduced after the framework and molecular kernel.
>
> **STILL VALID and deliberately preserved here — do not delete, this is the granular record:**
> the verbatim advisor memo as a HISTORICAL RECORD of how the framing evolved, and its baseline-map addendum.
>
> When this document and the authoritative framing conflict on thesis, positioning, title, contribution
> order, or novelty claims, **the authoritative framing wins**. Operational detail below remains usable.



- **Date received:** 2026-07-22
- **Status:** Incoming strategic reframe (advisor memo), preserved verbatim below. Not yet
  implemented. Intended to become the paper's thesis — **demoting** the pathwise-safety
  toxicophore spotlight and **narrowing** the "generate-then-filter is the only representable
  baseline" claim.
- **The one new object:** a learned stochastic process whose states are complete molecular
  graphs, whose transitions are executable rewrites, and whose probabilities live on *canonical
  molecular successors* rather than syntactic action descriptions. The paper is about that
  process as a molecular-control substrate.
- **Three-level story:** (1) umbrella = control-closed molecular probability paths; (2) hero =
  anytime Pareto-set editing (every visited molecule is potential output; maintain a
  source-relative nondominated archive); (3) mechanism = Pareto reachability depends on path
  topology (local/monotone control can miss reachable frontier regions that nonmyopic valid-path
  control recovers).
- **How to read the math:** the memo is preserved exactly as received; some equations carry
  OCR/unicode artifacts (e.g. `ⓜ` acting as an argument separator, combining tildes on `Q`,
  `E_C` for the legal edit graph). Reconstruct canonical notation during implementation.
- **Rigor note (ours, not the memo's):** the memo's characterizations of external methods
  (MOG-DFM, AReUReDi, pCoMole, MARS, MIMOSA, junction-tree generation) are strategy-level and
  **must be verified against primary sources** before any of it enters `main.tex` — the same
  discipline we applied to ConStruct / PRODIGY / CoCoGraph.

---

## Verbatim memo

Yes. I think the current framing is clunky, but not because it contains too many capabilities. It is clunky because it presents the consequences of the paradigm—validity, constraints, editing, oracle efficiency, Doob control—as separate contributions rather than identifying the single new object that produces all of them.

Your draft already contains that object: a learned stochastic process whose states are complete molecular graphs, whose transitions are executable rewrites, and whose probabilities live on canonical molecular successors rather than on syntactic action descriptions.

The paper should be about that process as a molecular-control substrate.

### The cleaner thesis

I would not make the central framing merely "a conditional molecular generator." That still sounds like another model for p(G∣c), where c might be a property vector, scaffold, or source molecule.

The stronger framing is:

A generative probability path that remains in molecular space turns generation into an executable, source-conditioned control process. The learned model supplies a prior over chemically legal edits; arbitrary controllers determine what the process should accomplish.

Or even more memorably:

When every state is a molecule, the generative trajectory itself becomes the molecular design object.

That gives you a very clean division of labor:

⏟(G_s )┬"where design begins"   +  ⏟(E_C )┬"what is allowed"   +  ⏟(Q_θ )┬"what edits are plausible"   +  ⏟(Ψ_ω )┬"what the user prefers"   ⟹ "controlled molecular path".

Here:
- G_s is the source molecule;
- E_C is the feasible molecular rewrite graph, possibly restricted by hard requirements;
- Q_θ (Gⓜ,H) is the learned quotient rate over canonical molecular successors;
- Ψ_ω is a controller: MOG-style guidance, an h-transform, SMC, MCTS, MH, or something archive-based.

For a simple rate controller,

Q ̃_t (Gⓜ,H)=1" " {C(H)} Q_(θ,t) (Gⓜ,H) Ψ_t (Gⓜ,Hⓜ,ω), H≠G,

with the diagonal set to the negative outgoing rate.

This single expression contains:
- hard support control through 1{C(H)};
- MOG-style preference tilting through an exponential local score;
- Doob control through h_t (H)/h_t (G);
- source-conditioned editing by setting G_0=G_s;
- dynamic preferences by varying ω_t;
- unguided generation by setting C≡1 and Ψ≡1.

The conceptual property is closure under control: any sampler that only removes or reweights existing legal molecular transitions inherits molecular-state validity. This is the unifying consequence of your construction. Your current draft already gestures toward exactly this division—rewriting defines legal support and guidance reweights executable successors—but it does not yet elevate that interface into the thesis.

### Where the other agent is right—and where I disagree

The other agent is technically right on several points.

The carbon-tree-to-data training versus held-out-lead initialization is a genuine mismatch. The present model is trained on trajectories originating from a carbon-tree prior, while the editing experiments directly initialize the learned dynamics at held-out lead molecules. Re-rooted suffixes are useful because they expose the model to data-like starting states.

But there is an important nuance the other agent missed:

Re-rooting makes the training distribution more editing-like; it does not automatically make the model source-conditioned.

If the rates only receive the current G_t, the result is a generic local molecular transition prior. That may be exactly what you want. But if the claim is genuinely p(G_T∣G_s ), then G_s, source-relative atom mappings, edit cost, or protected structure should be explicit conditioning variables:

Q_(θ,t) (Gⓜ,H∣G_s ).

You should decide deliberately between:
- A universal edit prior, where the source is merely the initial state and controllers preserve source relationships; and
- A source-conditioned edit flow, where rates explicitly know the original lead.

For a headline molecular-editing paper, I lean toward the second.

The weak-generator/MH argument is also conceptually correct: guidance cannot recover absent support, whereas an explicit full-support legal proposal mixed with learned rates can make generator quality primarily an efficiency issue. But it is not as cheap as the feedback makes it sound. Exact MH on molecules requires canonical forward and reverse proposal probabilities, including alias aggregation; your production sampler is time-inhomogeneous; and top-p or candidate pruning can destroy the stated support. The quotient machinery makes this possible, but it is a real method component, not a paragraph.

I also agree that the draft's claim that generate-then-filter is effectively the only relevant learned control is too strong. The current text explicitly says no prior learned constrained generator can represent the studied conditions. That should be removed or narrowed.

But your interruption about MARS is exactly right. MARS and MIMOSA are task-level molecular-optimization baselines, not the closest methodological lineage. MARS is an annealed MCMC method with graph-fragment edits and an adaptive learned proposal; MIMOSA starts from an input molecule and iteratively applies add, replace, and delete operations under multiple property and similarity constraints. They matter because they prevent you from claiming that valid-state multiobjective editing itself is new. They do not compete directly with the contribution "Generator Matching on an executable molecular rewrite process."

Likewise, validity at every construction step has precedents such as junction-tree generation. Your novelty therefore has to be conjunctive:

Generator Matching + flexible non-monotone molecular rewrites + canonical successor semantics + a sampler-agnostic control interface.

Finally, I would not treat the 118-state Pareto barrier analysis as a binary go/no-go test. It is an excellent early diagnostic. But one tiny graph and one choice of objectives cannot invalidate the broader control-substrate thesis. If detours matter strongly, they become a hero result. If they do not on that slice, other path-dependent benefits can still carry the paper.

### What every-step molecular validity can actually buy

The litmus test is:

What can your method do that would become undefined—not merely somewhat worse—if the intermediate states were not molecules?

That is where the framing becomes load-bearing.

#### 1. It gives multiobjective controllers a native molecular interface

MOG-DFM assumes it can evaluate each candidate token replacement with all objective functions, rank its local improvement vector, and test its alignment with a desired Pareto direction. It then reweights the base transition velocity and applies an adaptive hypercone filter. The paper explicitly describes its outputs as near the Pareto front rather than guaranteed Pareto optimal.

COMPOSE can lift that controller from token replacements to molecular successors:

Δs(Gⓜ,H)=s(H)-s(G),
Q ̃_t (Gⓜ,H∣ω)∝Q_(θ,t) (Gⓜ,H)  exp⁡" "  {β" " R" " (Δⓜ,s(Gⓜ,H)ω)},

possibly followed by a cone filter.

The important difference is not simply that the formula works. It is that:
- every H is a complete molecular graph;
- variable-size and topology-changing moves are legitimate candidates;
- black-box molecular objectives can be called directly;
- guidance ranks canonical molecules rather than token changes or rewrite aliases.

That is a very natural collaboration story:

MOG-DFM provides the controller; COMPOSE provides the correct state and transition space for applying that controller to flexible small-molecule graphs.

AReUReDi provides a second controller option. It uses Tchebycheff scalarization, locally balanced single-coordinate proposals, and MH correction. Notably, although its asymptotic argument allows exploration, all reported experiments add a weighted-sum monotonicity constraint to accelerate finite sampling. That creates a direct research question for your rewrite graph: when does finite-budget monotonic guidance prevent access to useful Pareto regions?

pCoMole makes the collaborative lineage even cleaner. It already uses variable-length Edit Flows, terminal feasibility gating, augmented Tchebycheff utility, and an approximate Doob transform for biological sequence editing. Your lane is therefore not "the first Pareto-conditioned editor." It is:

Pareto control over canonical, connected, valence-valid small-molecule graph rewrites, including atom-count, bond, and ring-topology changes, with the option of pathwise rather than only terminal feasibility.

#### 2. It lets the path produce a Pareto set, not merely one endpoint

This is the most interesting direction to me.

Most guidance methods assign a preference vector ω to a trajectory and return its final sample. Even when multiple weights are used, intermediate computation is primarily treated as a means to obtain one endpoint.

Because all COMPOSE states are molecules, you can instead define the output of a trajectory as its nondominated archive:

A_(k+1)=ND⁡(A_k∪{G_(k+1) }).

Then the path-level objective is not merely U_ω (G_T ). It can be

U(τ)=HV⁡(A_T )-λ" " EditCost⁡(τ),

or a reference-vector coverage score when hypervolume is inconvenient.

This changes the generative task:

A trajectory generates an anytime Pareto library of source-related molecules.

Every scored intermediate can:
- enter the returned archive;
- seed a new branch;
- be reused under a different preference;
- be cached across particles and trajectories;
- remain available when the oracle budget is exhausted.

This is where "every step is a molecule" becomes more than a validity statistic. The generative computation itself becomes useful output.

The principled version is to augment the state with the archive:

Z_t=(G_tⓜ,A_t ),

and define

h_t (Gⓜ,A)=E_(Q_θ ) [expⓜ,⁡(βU(A_T ))∣G_t=GA_t=A].

An ideal controlled generator would use

Q_t^h ((Gⓜ,A)ⓜ,(Hⓜ,A^' ) )=Q_(θ,t) (Gⓜ,H)  (h_t (Hⓜ,A^' ))/(h_t (Gⓜ,A) ).

You would approximate this with short rollouts, SMC, or MCTS. Conceptually, this extends terminal-utility Doob control to Pareto-set path utility.

A simpler first implementation is an archive-aware SMC sampler:
- sample legal canonical successors from Q_θ;
- score and deduplicate them by canonical molecular identity;
- add all of them to a shared archive;
- resample particles according to scalarized utility plus archive contribution;
- adapt preference vectors toward under-covered regions of the archive.

The main evaluation becomes feasible hypervolume versus oracle calls, not only endpoint success.

This is not the first optimization algorithm ever to maintain a molecular archive—MARS and evolutionary methods obviously provide relevant precedents. The novelty would be that the archive is built along a learned Generator-Matched molecular probability path, with canonical state merging and arbitrary controller composition.

#### 3. It makes Pareto reachability a path-topology problem

The strongest scientific mechanism remains the one highlighted in the feedback, but I would place it beneath the larger thesis.

A Pareto-optimal endpoint may not be reachable through a sequence of locally improving edits. For a source G_s, legal graph E, and objectives s, distinguish:
- the globally reachable Pareto set;
- the Pareto set reachable through componentwise-monotone moves;
- the set reachable through scalarization-monotone moves;
- the set reachable while staying above objective floors;
- the set reachable with arbitrary valid detours.

The useful quantity is something like an objective barrier:

B_ω (H)=min┬(τ:G_s⇝H)  max┬k [U_ω (G_s )-U_ω (G_k )]_+.

For multiple protected objectives, use a vector floor or maximal floor violation instead.

This lets you ask a sharper question than "does guidance improve properties?":

How much of the molecular Pareto frontier is inaccessible to locally monotone control because of the topology of the legal edit graph?

Then compare:
- local MOG-style guidance;
- MOG with an adaptive viability corridor;
- AReUReDi-style MH without the practical monotonic filter;
- short-horizon rollout/Doob guidance;
- archive-aware SMC.

A favorable result would not merely say "detours help." It would show:
- the endpoint frontier exists in the legal reachable set;
- greedy/local controllers systematically cannot reach part of it;
- nonmyopic control over valid molecular paths recovers that region;
- the recovered states remain usable molecular candidates throughout.

That is a spotlight-grade mechanism if it holds.

But it is an empirical hypothesis, not yet a finding. The exact-state analysis should be run early, across multiple objective pairs and sources, rather than treated as one cherry-picked 118-state example.

#### 4. It enables dynamic preference steering and frontier continuation

A static conditional generator commits to a condition before sampling. A molecular editing process can change the condition during sampling.

For example:
- optimize affinity until a threshold is reached;
- then rotate preference toward solubility;
- branch from the current molecule under two different toxicity preferences;
- return to an archived state and explore another region.

Because all restart points are molecules, preference changes require no re-noising or re-decoding. This suggests a Pareto continuation algorithm: smoothly rotate ω and warm-start the next trade-off search from a nearby Pareto molecule rather than restarting from the lead or noise for every weight vector.

The empirical question is whether continuation covers a frontier using fewer unique oracle evaluations than independent runs. Canonical caching is essential here.

This could also produce a source-rooted Pareto edit atlas: a set of nondominated molecules with valid edit paths back to the lead. That is a qualitatively different artifact from a bag of unrelated generated molecules.

#### 5. Canonical successor semantics become operationally important

This is one of the most technically distinctive pieces of your work.

If several rewrite marks reach the same molecule, a controller must not give that molecule extra probability merely because it has more syntactic derivations. The draft already defines the quotient rate

Q_(θ,t) (Gⓜ,H)=∑_(a:T_a G≃H)▒q_(θ,t)  (a∣G).

For controlled generation, this means:
- rank molecular successors H, not marked actions a;
- calculate hypervolume contribution once per canonical H;
- cache each oracle result once;
- merge particles that reach the same H;
- use quotient forward and reverse probabilities in molecular-state MH.

This gives you an excellent ablation:

Action-level guidance versus quotient-level molecular guidance.

Duplicate some rewrite lowerings or exploit naturally symmetric molecules. A representation-correct controller should produce invariant molecular results; an action-level controller will change merely because the same chemical transition was represented multiple ways.

That would make the quotient machinery visibly load-bearing rather than a formal footnote.

### Are hard structural constraints necessary?

No.

Set C≡1 and the entire control-ready/Pareto-path thesis remains intact. Hard constraints are one especially clean instance of the broader separation:

Hard requirements define support; soft objectives define preference within that support.

I would retain one serious hard condition in the main experiments—perhaps a protected scaffold, pharmacophore, element vocabulary, or maximum edit cost—because real lead optimization nearly always has nonnegotiable requirements. But I would not make "exact structural constraints" the identity of the paper.

Their role should be:
- demonstrate support-level control;
- prevent reward-hacked Pareto solutions;
- define the source-relative feasible design space;
- show that soft Pareto guidance composes with hard invariants.

The pathwise reactive-alert story should probably be demoted or reframed. A rewrite trajectory is not a synthesis route, so "unsafe intermediate" is too physical a phrase. A more rigorous use of path constraints would be:
- predictor-domain or uncertainty thresholds;
- protected pharmacophore retention;
- maximum molecular complexity;
- minimum acceptable floor on a protected property;
- no forbidden motif among any oracle-scored candidates.

Also, your declared domain is only connected, valence-admissible 2D molecular graphs; it does not guarantee synthesis, 3D plausibility, or oracle reliability. The framing should call this molecular-state closure, not comprehensive chemical feasibility.

### The source-conditioned model I would build

If editing becomes primary, I would change the training regime rather than merely reinterpret the current de novo model.

**Source construction.** Use a mixture of:
- suffixes of existing certified programs;
- short valid corruptions of data molecules using inverse rewrite actions;
- source–target pairs sampled from structurally related training molecules;
- optionally carbon-tree-to-data programs to retain de novo generation.

Valid corruption is particularly attractive:

G_"target"  → G_"source" ,

then train the reverse executable program. The corruption depth controls edit scale and ensures that training sources remain molecular and data-adjacent.

**Explicit conditioning.** Use

Q_(θ,t) (Gⓜ,Hⓜ,∣G_s B),

where B may include:
- source atom or substructure correspondence;
- protected atom masks;
- edit budget;
- desired locality or corruption depth.

Keep property objectives out of the base network. The model learns an objective-agnostic molecular edit prior, and MOG/Doob/MH/SMC supplies the preferences at inference.

That is a cleaner modular story than training a separate conditional generator for every objective vector.

### The experimental spine I recommend

The exact analysis is worth doing immediately, but it is one of several decisive probes rather than the only gate.

| Question | Decisive experiment |
|---|---|
| Does the path itself provide value? | All-visited-state Pareto archive versus endpoint-only archive under identical oracle calls |
| Does local monotonicity lose reachable solutions? | Exact Pareto reachability and objective-barrier analysis on several exhaustive rewrite graphs |
| Does the learned generator matter? | Learned quotient rates versus uniform legal rewrites under the same controller |
| Does the molecular substrate matter? | The same MOG/AReUReDi-style controller on COMPOSE and on a SMILES/token or ambient-graph base |
| Is quotient control necessary? | Canonical-successor guidance versus marked-action guidance under alias duplication |
| Do hard requirements compose with Pareto control? | One source-relative multiobjective task with a protected scaffold or pharmacophore |
| Is weak-base robustness real at finite budget? | Checkpoint degradation and learned/uniform proposal mixtures, reporting hypervolume, acceptance, and mixing |

The main real-scale experiment should be genuine source-relative multiobjective editing—not only QED. The current committed result is a single-objective QED improvement under a scaffold constraint, and the draft acknowledges that its SMC controller is oracle-hungry.

I would use at least three conflicting objectives and report:
- feasible hypervolume AUC versus oracle calls;
- unique nondominated molecules;
- frontier coverage;
- source edit distance;
- structural retention;
- all-state molecular validity;
- zero-hazard stalls;
- canonical cache hit rate;
- wall-clock and total candidate evaluations.

MOG-DFM and AReUReDi should appear as controller instantiations or close methodological comparisons, not antagonistic baselines. MIMOSA or MARS can appear as a task-level molecular editor baseline. The same-substrate uniform-legal controller remains the most important ablation for showing that Generator Matching contributes something beyond having a valid edit graph.

### Where the weak-generator story belongs

I would keep it, but as a robustness contribution rather than the top-line thesis.

The clean formulation is

Q_ε (Gⓜ,H)=(1-ε) Q_θ (Gⓜ,H)+εQ_legal (Gⓜ,H).

Then an MH or annealed controller targets an explicit molecular distribution such as

π_(β,ω) (G∣G_s )∝1{G∈R(G_s )}  exp⁡" "  (βU_ω (G)-λd(Gⓜ,G_s )-γE_plaus (G)).

Under appropriate irreducibility and exact proposal-ratio conditions, the target is determined by π; the learned generator determines proposal efficiency. But the finite-budget story still depends heavily on the proposal. Report local quantities such as:

M_θ^+ (Gⓜ,ω)=∑_H▒Q_θ  (Gⓜ,H)1" " {U_ω (H)>U_ω (G)},

top-k improving-successor recall, acceptance rate, and effective unique molecules per oracle call.

The interesting empirical claim would be:

Global unconditional fidelity is not the right predictor of editing performance; local feasible proposal quality is.

That is a hypothesis worth testing, not something to assume in the framing.

### The strongest overall paper shape

I would organize the story at three levels.

**Umbrella contribution.** Control-closed molecular probability paths. RGM learns a prior over executable canonical molecular transitions. Controllers can be attached without retraining and cannot leave the declared molecular state space.

**Hero application and algorithm.** Anytime Pareto-set editing. The sampler treats every visited molecule as potential output, maintains a source-relative nondominated archive, and guides trajectories toward frontier coverage rather than only one terminal scalarized optimum.

**Mechanistic result.** Pareto reachability depends on path topology. Greedy or monotonic local control may miss reachable frontier regions; nonmyopic valid-path control can cross objective barriers while retaining usable molecular states.

Hard structural constraints then become a compositional demonstration, and weak-base MH becomes a robustness result.

### A possible paper pitch

Molecular generators are typically trained to produce valid endpoints, while their intermediate states are internal numerical objects. This makes source-conditioned control dependent on model-specific noisy-state predictors, repair, or terminal filtering. We introduce Rewrite Generator Matching, which learns a stochastic probability path directly on canonical molecular graphs connected by executable grow, shrink, retype, bond, graft, and ring rewrites. Because every state is a molecule, the process is closed under inference-time control: hard requirements restrict legal support, while MOG-, Doob-, SMC-, and MH-style controllers reweight the same molecular successor rates using arbitrary black-box objectives. We exploit this interface for anytime Pareto editing, where every visited state contributes to a source-relative nondominated archive and can seed further search. We study how the topology of the molecular rewrite graph governs Pareto reachability, when locally monotone guidance misses reachable trade-offs, and whether learned edit priors improve frontier coverage over uniform legal search.

### Possible titles

- Every Step Is a Molecule: Anytime Pareto Editing with Rewrite Generator Matching
- From Lead to Pareto Path: Executable Molecular Flows for Multi-Objective Editing
- COMPOSE: Control-Closed Molecular Generation on Valid Rewrite Paths
- Control the Molecule, Not Its Encoding: Generator-Matched Pareto Editing

My strongest recommendation is therefore:

Do not make hard constraints or "conditional generation" the thesis. Make the thesis that RGM turns a molecular generator into a modular, source-conditioned control process. Make anytime Pareto-set generation the hero use of that process, and make Pareto reachability through valid detours the scientific mechanism you test most aggressively.

---

## Baseline literature map (ours, not the memo's — verified 2026-07-26)

**Provenance.** Deep-research sweep (104 agents, 21 primary sources, 24/25 falsifiable
claims confirmed under 3-vote adversarial verification, 1 refuted). Cross-paper numbers
are protocol-dependent — fix ONE evaluation protocol before tabulating ours beside them
(caveat below).

**The differentiation held up under verification:** no published method simultaneously
offers (i) a complete, valid, oracle-scoreable molecule at *every* step, (ii) an anytime
source-relative Pareto archive, (iii) mid-trajectory warm-started steering, and (iv)
constraint-by-construction. Every graph baseline is either latent-decode-at-end or
noisy-diffusion → **valid only at the endpoint**. That is exactly the hero-experiment
wedge below.

### Tier A — must-cite constrained-ZINC editors (all endpoint-only-valid → our contrast)

| Method | Class | Per-step valid? | Objective | Published ZINC (δ≥0.4) |
|---|---|---|---|---|
| VJTNN / +GAN (ICLR'19) | JT-VAE graph→graph translation | No — latent, decode-at-end | baked (paired corpus, model/task) | plogP 3.55±1.67; QED 60.6%; DRD2 78.4% |
| Modof (Nat.Mach.Intel.'21) | JT-VAE single-fragment editor | No — latent diff → AR decode | baked | plogP 5.00 / 5.89 (strongest plogP) |
| GrIDDD (NeurIPS'25) | graph insert/delete DDPM | No — *re-noises* input to edit; admits illegal insert+delete / split mols | baked (classifier-free) | plogP 2.70±0.94; QED 45.1%; DRD2 5.0% |

### Tier B — structural cousin, cite for framing not numbers
**DDSBM** (ICLR'25) — discrete-diffusion Schrödinger bridge / CTMC-over-graphs. Closest in
spirit (a CTMC over graphs) but a **noisy** bridge with no per-step validity; its ZINC "task"
is bespoke distribution-transport (logP≈2→4), so **no citable constrained-QED/plogP number**.
Contrast paragraph: validity-closed executable rewrite + objective-agnostic control vs. noisy
bridge + fixed source/target distributions.

### Tier C — oracle-efficiency context (endpoint-comparable only)
MOGFN, HN-GFN (de-novo multi-objective GFlowNets; no ZINC constrained numbers). **GraphGA /
PMO (Gao'22) is the flagged omission** — the PMO oracle-efficiency top performer a reviewer
expects; cite for context even though de-novo.

### Tier D — steering *philosophy*, related-work only (no ZINC)
MOG-DFM, AReUReDi (objective-agnostic inference-time guidance on frozen bases — closest to our
value-guided SMC idea, but peptide/DNA/SMILES only). The memo already names MOG-DFM / AReUReDi
as controller instantiations; the lit search confirms they have **no** small-molecule/ZINC
numbers → controllers / related work, never ZINC baselines.

### Hero experiment this licenses
**"Carve the Pareto frontier far more efficiently than endpoint-only-valid models."** Two forms:
- **Internal (clean mechanism proof, needs no competitor code):** all-visited-state archive
  vs. endpoint-only archive under **matched oracle calls** — feasible hypervolume AUC vs.
  oracle calls. Every intermediate is already valid + scored + archivable; an endpoint-only
  model harvests one archived molecule per full decode/denoise trajectory. This is the memo's
  decisive-experiment row #1 and isolates "every step is a molecule."
- **External (competitive story):** the JT-VAE / latent foil on our leads for the paradigm
  axes (per-step validity, anytime archive, mid-trajectory steering) the published tables
  don't measure, plus cited Tier-A numbers for endpoint competitiveness. Endpoint-only models
  must re-noise/re-decode per preference ω → cannot warm-start mid-trajectory (claim 1) and
  cannot contribute intermediates to the archive (claim 2).

Metric of record (per the memo): **feasible hypervolume AUC vs. oracle calls**, plus unique
nondominated molecules, frontier coverage, source edit distance, all-state validity, canonical
cache-hit rate.

### Caveat (protocol)
Tier-A numbers span Jin'18 vs Jin'20 suites, best-of-20 candidates, differing similarity
radius/thresholds → not strictly comparable. Fix one protocol (test set, oracle-call budget,
candidate count, similarity radius, success thresholds) before any side-by-side table.
penalized-logP is gameable/saturated → keep endpoint plogP/QED **defensive, competitive-not-
SOTA**, never headline.

### Sources
- VJTNN/+GAN — arXiv:1812.01070 · Modof — PMC8856604 (Nat. Mach. Intel. 2021) ·
  GrIDDD — arXiv:2506.15725 (NeurIPS 2025) · DDSBM — arXiv:2410.01500 (ICLR 2025)
- MOGFN — Jain et al. ICML 2023 (PMLR v202) · HN-GFN — arXiv:2302.04040 ·
  MOG-DFM — arXiv:2505.07086 · AReUReDi — arXiv:2510.00352 · PMO/GraphGA — arXiv:2206.12411
- Unresolved flags: **EDM-SyCo** (Ketata'25, 3D-diffusion optimizer, beats GrIDDD in its own
  appendix under a 400-round budget) not independently verified — decide if in-scope. **HierG2G
  naming split:** arXiv:2002.03230 = HierVAE (unconditional); arXiv:1907.11223 = the HierG2G
  *translation* model — attribute any constrained numbers to 1907.11223.

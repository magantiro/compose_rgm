# Unconditional chemistry failure audit

**Date:** 2026-07-20  
**Scope:** read-only attribution of the calibrated step-6,250 unconditional
rollout; no production-code changes  
**Primary artifacts:**
`diagnostics/pancake6250_calibration_eval600_metrics.json` and
`diagnostics/pancake6250_calibration_eval600_grid100.png`

## Executive decision

The unconditional model is a credible executable base editor and is good
enough to continue into conditional training. It is not yet a faithful final
unconditional molecular distribution. The remaining failures are learned
rate/composition/path-support failures, not evidence that whole-ring growth,
aromatic lowering, or the validity-closed rewrite substrate is broken.

The strongest quantitative failures are:

- triple-bond prevalence of **54.0%** versus **6.91%** in the matched
  reference;
- chemically valid but corpus-atypical N/O-rich ring assignments, including
  rings with at least three heteroatoms at **20.35%** versus **2.99%** of
  rings, ring O-O bonds in **7.0%** versus **0.48%** of molecules, and adjacent
  aromatic `[nH]-[nH]` in **10.5%** versus **0%**;
- small 3/4-member-ring prevalence of **11.33%** versus **6.03%**, after the
  inference-only small-ring thinning already applied;
- cycle rank **2.46** versus **3.35**, with fused/spiro/bridged prevalence
  **28.5/0.67/1.67%** versus **58.0/3.48/4.01%**.

These errors should not block the matched conditional pilot. Conditional
training may update the base generator parameters, including the currently
neutral legacy aromatic-role head. The conditional objective should retain a
base Generator-Matching replay term so property learning does not simply hide
or amplify the measured base-chemistry errors.

## Evidence and comparison population

The audit contains 600 non-null generated molecules. Every endpoint is valid,
connected, neutral C/N/O/F, and within the 40-heavy-atom representation. All
18,646 observed trajectory states are valid and connected; there are no
one-atom/null collapses and no event-budget failures. Therefore none of the
visual examples is an RDKit-invalid molecule under the project's declared
chemistry semantics.

The reference is genuinely matched to the modeled chemistry scope. The
evaluator reads the 5,000 held-out GuacaMol records, calls
`_canonical_cnof_smiles(..., max_atoms=40)`, and deduplicates canonical
SMILES (`modal_apps/evaluate_rollout_shards.py:329-345`). The filter requires a
nonempty connected molecule, neutral formal charges, C/N/O/F only, and at most
40 heavy atoms (`src/compose_v4/data/cnof.py:171-190`). This yields exactly
**2,271** reference molecules, reproducing the count stored in the rollout
artifact. The retained local held-out file has SHA-256
`1f8e92f70018dfb94965977f87b7844dea791e4727ea8f7d251946925d1aa99f`,
matching the run manifest.

The comparison is chemistry-scope matched, not exactly atom-count- or
scaffold-matched. That is acceptable here because generated and reference
means are already close at 27.10 versus 26.35 heavy atoms. All numerical motif
comparisons below use the same deduplicated 2,271-molecule reference.

## Whole-ring growth and aromaticity are already coordinated

Whole-ring growth does not add an untyped ring and then decorate it through
independent sampled actions.

1. The carbon-tree compiler defers all internal ring-system bond orders and
   atom identities into one `RingSystemGrow` transaction
   (`src/compose_v4/rewrite/tree_transport.py:311-401`). The transaction carries
   the complete member set, scaffold, internal bond reorders, atom payloads,
   closure bonds, aromatic edge set, and topology class.
2. The executor lowers that visible transaction through bond reorder, joint
   atom restatement, and closure micro-steps, then requires the exact declared
   cyclic component and aromatic edge set
   (`src/compose_v4/rewrite/tracelets.py:562-678` and `1016-1048`). The learned
   CTMC sees only the atomic transaction.
3. Aromatic electronic categories are decoded jointly under valence,
   matching, and monocycle 4k+2 constraints. A deterministic matching produces
   one integer-order Kekule lowering
   (`src/compose_v4/rewrite/ring_system_fiber.py:211-224` and `591-693`). The
   completed successor is rejected unless RDKit perceives exactly the expected
   aromatic edges (`src/compose_v4/rewrite/ring_system_fiber.py:773-865`).
4. In typed production mode, the micro fiber removes scalar atom restatement
   or deletion on ring atoms and scalar bond reorder or deletion on cycle
   edges. It also removes scalar bond insertion entirely
   (`src/compose_v4/rewrite/tracelet_fiber.py:178-215`). A later independent
   action therefore cannot partially corrupt an installed aromatic ring.
   Coordinated ring-system grow/delete/restate rules remain available.

Targeted regression verification passed 56 tests across semantic aromatic
decoding, ring-system grow, aromaticity, polycyclic parity, and the factorized
mark model:

```text
........................................................  [100%]
```

It would be chemically wrong to mark every installed ring aromatic. The
matched corpus contains saturated and nonaromatic unsaturated rings, and the
whole-ring action intentionally records the target's RDKit-perceived aromatic
edge set rather than assuming aromaticity from topology.

## Failure attribution

### 1. Excess triple bonds are real but mostly rare-valid chemistry

The grid's heavy black parallel strokes can be hard to read at thumbnail
resolution, but the underlying canonical SMILES and RDKit bond objects confirm
the excess. There are **481** generated triple bonds: 261 C#C and 220 C#N.
There are **171** in the reference: 37 C#C and 134 C#N.

| Triple-bond statistic | Generated | Matched reference |
|---|---:|---:|
| molecules with any triple bond | 324/600 = **54.0%** | 157/2,271 = **6.91%** |
| triple bonds per molecule | **0.802** | **0.075** |
| fraction of all bonds | **2.807%** | **0.262%** |
| triple bonds inside rings | **0** | **0** |

Nitriles and alkynes are valid and occur in the reference, so a hard ban would
delete real corpus support. The problem is a roughly ten-fold rate mismatch.
The exact source is the acyclic mark parameterization: connected atom insertion
offers bond orders 1/2/3 whenever local hydrogens permit them, and bond reorder
offers every different order 1/2/3
(`src/compose_v4/rewrite/factorized_fiber.py:141-167` and `194-221`). The
factorized sampler masks by valence and noncycle
status, then relies on the learned insert/reorder logits to supply corpus
frequency (`src/compose_v4/model/factorized_tracelet_rate_model.py:2822-2862`).
The executor is doing what it should; the marked rate law is poorly calibrated
on off-teacher rollout states.

### 2. Heteroatom-heavy structures are a joint electronic-model failure

Element TV of 0.054 looks modest, but it hides a large joint-structure error.

| Composition statistic | Generated | Matched reference |
|---|---:|---:|
| heteroatoms per molecule | **7.655** | **6.131** |
| nitrogen atoms per molecule | **4.148** | **2.923** |
| oxygen atoms per molecule | **3.197** | **2.801** |
| molecules with heteroatom fraction >= 0.4 | 63/600 = **10.5%** | 64/2,271 = **2.82%** |
| heterocyclic rings / all rings | 976/1,479 = **65.99%** | 3,692/7,663 = **48.18%** |
| mean per-ring heteroatom fraction | **24.45%** | **13.43%** |
| rings with at least 3 heteroatoms | 301/1,479 = **20.35%** | 229/7,663 = **2.99%** |
| molecules with a ring O-O bond | 42/600 = **7.0%** | 11/2,271 = **0.48%** |
| molecules with a ring N-N bond | 238/600 = **39.67%** | 383/2,271 = **16.86%** |
| molecules with adjacent aromatic `[nH]-[nH]` | 63/600 = **10.5%** | 0/2,271 = **0%** |
| molecules with a ring containing >=2 aromatic `[nH]` | 91/600 = **15.17%** | 6/2,271 = **0.26%** |

The generated N/O-rich rings remain valence- and aromaticity-valid, and the
reference confirms that N-N, O-O, peroxides, and multi-heteroatom rings are not
universally invalid. They are rare-valid corpus chemistry sampled far too
often. Adjacent aromatic `[nH]` assignments are especially strong evidence of
a distributional failure: RDKit accepts them, but none occurs in the 2,271
matched reference molecules.

The semantic decoder is a correctness/support oracle, not a corpus-frequency
model. At each ring member, the neural score is an additive C/N/O/F type logit
plus an aromatic-role logit, followed by exact completion masking
(`src/compose_v4/model/factorized_tracelet_rate_model.py:2002-2024`). The
restored legacy step-6,250 checkpoint predates the four role-head tensors; the
compatibility loader sets those missing tensors to zero
(`scripts/evaluate_tracelet_rollouts.py:170-203`). Thus legal donor/acceptor
roles receive a neutral role factor, and the exact completion predicate cannot
learn the missing corpus correlations. This is the most concrete source of the
odd heterocycles.

### 3. Nonaromatic ring double bonds are not an excess

The visual concern about double bonds in nonaromatic heterocycles is reasonable
for individual examples, but the aggregate diagnosis points in the opposite
direction.

| Unsaturated-ring statistic | Generated | Matched reference |
|---|---:|---:|
| molecules with an unsaturated nonaromatic ring | **20.83%** | **29.77%** |
| unsaturated nonaromatic rings / all rings | **9.20%** | **10.79%** |
| nonaromatic ring double bonds per molecule | **0.158** | **0.151** |
| ring triple bonds | **0** | **0** |

These motifs are sometimes unusual because of their atom assignment, not
because a scalar bond mutation damaged an aromatic ring. Aromatic rings are
actually underproduced: 63.15% versus 68.86% of rings, 82.83% versus 92.43% of
molecules, and mean aromatic-atom fraction 0.320 versus 0.481.

Some apparent alternating or heavy ring lines in the 100-molecule grid are
representation/rendering effects. Internally, aromatic inputs are stored as
editable integer-order Kekule bonds and aromaticity re-emerges on the RDKit
round trip (`src/compose_v4/chem/molecular_graph.py:494-516`). The serialized
SMILES and evaluator use RDKit-sanitized aromatic perception
(`src/compose_v4/chem/molecular_graph.py:587-656`). A thumbnail is therefore
not evidence of a partially aromatic state.

### 4. Small rings are a support/scheduling and family-rate failure

Small-ring molecule prevalence is **68/600 = 11.33%** versus
**137/2,271 = 6.03%**. The generated SSSR contains 62 three-membered and 12
four-membered rings; the reference contains 108 and 47, respectively, over a
much larger population. These rings are valid and present in the corpus, but
three-membered rings are substantially overrepresented.

The earlier exact state audit already isolates the cause. The catalog has
3,092 semantic templates, including 285 small-ring templates, with only
**3.017%** unconditional empirical small-ring prior mass. Broadly supported
states assign roughly 1.7-4.6% production mass to small rings. Late decorated
states sometimes leave only 2-5 executable templates, all small-ring
topologies. The hierarchical family head sees only that ring grow is enabled
and can assign the whole family probability to this rare residual support
(`src/compose_v4/model/factorized_tracelet_rate_model.py:1874-1878`). The
inference-only `-1.5` small-ring log-rate adjustment improves symptoms but does
not repair this mechanism.

### 5. Fused, spiro, and bridged systems are undercovered by rates, not absent
operators

| Ring-system statistic | Generated | Matched reference |
|---|---:|---:|
| mean cycle rank | **2.458** | **3.353** |
| mean SSSR rings | **2.465** | **3.374** |
| fused prevalence | **28.5%** | **57.99%** |
| spiro prevalence | **0.67%** | **3.48%** |
| bridged prevalence | **1.67%** | **4.01%** |
| polybridged/cage candidate | **0.17%** | **1.45%** |

The independent 200-target full-ring transport audit reports 501 whole-ring
transactions, including 123 bridged-or-fused and 9 spiro systems, with zero
failures and exact valid connected endpoints. The representation can express
these systems. The undercoverage follows from too few/too-late complex
ring-grow events and state-dependent template support. Ring growth is only
7.06% of rollout events; ring deletion occurred once.

The exact template graph distinguishes topologies, but the coarse stored class
merges fused and bridged systems into `bridged_or_fused`
(`src/compose_v4/rewrite/tree_transport.py:454-465`). That is sufficient for
execution, but too coarse for a topology-calibrated group rate. The catalog is
also capped by global frequency
(`src/compose_v4/rewrite/typed_ring_catalog.py:587-610`), which can reduce
rare-class coverage even while every retained
template remains valid.

## Prioritized ground fixes and bounded pilots

All pilots use fixed source trees, seeds, horizons, and matched reference. A
pilot fails immediately on any endpoint/intermediate invalidity, disconnected
state, molecular self-event, one-atom/null collapse, or event-budget failure.
Every chemistry comparison is paired by source and seed and reports a paired
bootstrap confidence interval, not just a point estimate.

### P1. Train corpus-base logits plus learned residuals on the existing cache

**Change.** Add smoothed teacher/corpus base logits for:

- connected atom-insert bond order and bond-reorder target order, conditioned
  at minimum on old order and endpoint elements/valence context;
- acyclic atom type/restatement;
- ring electronic sequences or sufficient joint features such as N/O count,
  donor count, adjacent heteroatom bonds, and aromatic role pattern.

The neural heads learn residual logits above these positive empirical base
measures. Train the four zeroed legacy ring-role tensors and the insert,
restate, reorder, ring-atom, and associated family heads. This is a trained
rate model, not an inference filter: rare alkynes, nitriles, peroxides, and
heterocycles retain positive probability.

**Pilot.** 500 cached updates, evaluate at 250 and 500, then 200 fixed-source
rollouts only from a checkpoint that improves frozen validation. The measured
54.52 examples/s production rate implies about 9.8 raw A100 minutes for 500
updates; authorize 20 minutes including cached validation and at most one hour
for distributed CPU rollouts.

**Cache effect.** Reuse molecular paths, exact support rows, and fixed
evaluation tensors. Invalidate only checkpoint/optimizer and rollout caches.
If paired ring-electronic counts cannot be recovered from cached teacher
actions, use global joint statistics for this pilot and defer a catalog rebuild
to P5.

**Pass gate.** All safety invariants remain 100%; on the 200-sample gate:

- triple-containing molecules fall from 54% to at most **27%** and triple
  bonds per molecule fall to at most **0.40**;
- bond-order TV improves from 0.093 to at most **0.070**;
- mean heteroatoms fall below **7.0**;
- ring O-O prevalence is at most **3.5%**, adjacent aromatic `[nH]-[nH]` at
  most **5%**, and >=3-hetero rings at most **12%**;
- QED does not fall below 0.439, SA does not exceed 4.49, cycle rank does not
  fall below 2.46, and fused/spiro/bridged prevalence does not decrease by
  more than two percentage points.

The eventual full-run target is no more than twice the reference rate for the
rare motifs: <=14% triple-containing molecules, <=1% ring O-O, <=6% rings with
at least three heteroatoms, and <=1% adjacent aromatic `[nH]-[nH]`.

### P2. Replace Boolean ring-family enablement with topology-group rate mass

**Change.** Factor ring grow into corpus-frequency topology groups defined by
cycle-size multiset, fused/bridged/spiro class, aromaticity pattern, and cycle
rank, plus a learned residual. Compute the ring-family hazard as the sum of
absolute supported topology-group rates. A tiny all-small residual support set
then carries tiny absolute mass instead of inheriting all probability assigned
to the Boolean ring family. Derive the finer topology signature from the exact
template graph so the pilot need not change executor semantics.

The existing generic superposed-rate arm did not pass its learning gate; this
narrower pilot is justified only because the exact 12-state audit identified
topology-residual support as the concrete remaining defect.

**Pilot.** 250-500 cached updates and 200 fixed-source rollouts. Expected raw
A100 time is 5-10 minutes; authorize 20 minutes including validation and one
hour CPU rollout time.

**Cache effect.** Exact support and path caches remain valid if topology groups
are derived from existing templates. Model/checkpoint and rollout caches are
new. Do not resize or reorder the catalog for this pilot.

**Pass gate.** Small-ring prevalence is at most **8%** without a hard ban;
states with <=5 supported ring templates all of which are small fall below
**2% of selected ring-grow events**; mean cycle rank reaches at least **2.75**;
fused prevalence reaches at least **38%**, spiro at least **1.5%**, and bridged
at least **2.5%**. Bond-order/element TV and P1 rare-motif rates may not regress
by more than 0.01 TV or two percentage points.

### P3. Add valid recovery supervision for rollout-state errors

**Change.** The one-way corpus path objective rarely trains on states produced
by its own overselected triples or heteroatom edits. Add a 25-50% recovery
mixture: apply one to four valid perturbations to a corpus molecule and train
the exact inverse causal rewrite distribution. The first bounded slice should
target acyclic single/double/triple reorder and C/N/O/F restatement, since these
use cheap local support. Ring recovery should use whole-ring delete/grow
transactions, never scalar edits inside a ring.

**Pilot.** Compile 5,000 acyclic recovery rows, mix them into 500 updates, and
evaluate on both the frozen ordinary validation set and a held-out recovery
set. Budget 15 CPU minutes for local row construction, 10 raw A100 minutes,
and 30 minutes total before rollout. Stop if ordinary validation worsens by
more than 2%.

**Cache effect.** Existing corpus path, ordinary support, and evaluation caches
remain valid. Recovery rows live in a new content-addressed support cache;
checkpoint/optimizer and rollout caches are new.

**Pass gate.** Held-out recovery top-1 successor accuracy is at least **70%**
and ordinary validation loss is no worse than **1.02x** baseline. In rollout,
triple prevalence improves by at least another **25% relative** to the best of
P1/P2, atom/bond TV does not worsen, and event-budget/validity invariants remain
perfect.

### P4. Pilot exact early-ring or causal path supervision before rebuilding scale

**Change.** Use the already implemented exact adjacent-commutation schedule to
move whole-ring transactions to the earliest array-exact valid position, then
pilot randomized/causal successor supervision where the carbon-tree interface
allows it. This attacks both the all-small late-support residue and the lack of
complex-ring exposure. Every transformed path must replay to the exact target;
insert/delete remain phase barriers.

**Pilot.** Separate namespace, 1,000 targets with one tree coupling each,
5,000-20,000 training rows, 500 updates, and 200 paired rollouts. The measured
full-ring audit compiled 200 targets in 21 seconds with eight workers, so raw
path rescheduling should be only minutes; exact support construction is the
real cost. Cap the CPU support stage at one hour and A100 training at 20
minutes. Do not launch a full 1.92-million-row rebuild from this pilot.

**Cache effect.** Path order changes the intermediate states and path
fingerprint, so path, training-support, fixed-evaluation, checkpoint, and
rollout caches all receive a new namespace. Target/source molecules and ring
operator/catalog semantics can be reused.

**Pass gate.** At least **50%** of ring grows occur before normalized path
position 0.8; all-small degenerate support is below **2%** of selected ring
events; small-ring prevalence is <=8%; cycle rank is >=2.9; fused/spiro/bridged
are >=42/2/3%; all exact endpoint and trajectory invariants remain 100%.

### P5. Rebuild the ring representation only if the bounded residual pilots fail

**Change.** Build a v1 paired electronic-alias catalog with observed joint
atom/role counts; preserve exact semantic-decoder fallback; allocate template
capacity by topology-group coverage rather than one global top-frequency cap;
and keep positive support for rare classes. This makes joint heterocycle
frequency and rare fused/spiro/bridged support explicit without adding a new
operator.

**Pilot.** Audit 2,000 training targets plus held-out targets, then train
500-1,000 updates only if coverage passes. Budget 30 CPU minutes for catalog
and path extraction, one hour for exact pilot support, and 20 raw A100 minutes.

**Cache effect.** This is the expensive option. The catalog, ring-specific
model parameters, ring support, training/evaluation batches, checkpoint, and
rollout caches are invalid. Raw corpus molecules, source-tree samples, and
operator code are reusable; molecular paths may be replayed but must be
re-certified against the new catalog signature.

**Pass gate.** Held-out teacher ring-grow support is >=**99% overall** and
>=**95% separately** for fused, spiro, bridged, saturated, aromatic, and
nonaromatic-unsaturated groups. A prior-only sample of executable ring actions
has ring-size/topology total variation <=**0.05** from held-out teacher actions.
After training, ring heterocycle fraction is <=55%, >=3-hetero rings <=8%,
fused/spiro/bridged >=45/2/3%, and all P1/P2 safety/TV gates hold.

## What not to do

- Do not force every ring aromatic. The corpus contains legitimate saturated
  and nonaromatic unsaturated rings, and aggregate nonaromatic ring
  unsaturation is not overproduced.
- Do not hard-ban triples, N-N/O-O bonds, heterocycles, or 3/4-member rings.
  Each has positive matched-corpus support. Use a positive empirical base
  measure plus trained residual rates.
- Do not add another ring operator. Whole-ring grow/delete/restate already
  supplies the required coordination and exact aromaticity semantics.
- Do not treat 100%-RDKit validity as sufficient chemical quality. It proves
  valence, sanitization, and declared execution invariants, not strain,
  stability, synthetic accessibility, or corpus likelihood.
- Do not block the conditional-generator throughput on P4/P5. Run P1/P2 from
  reusable caches in parallel with conditional work; incorporate a successful
  base-rate fix into subsequent conditional fine-tuning with a replay loss.

## Recommended execution order

1. Start conditional training from the qualified executable base, with all
   intended generator parameters trainable and a base-GM replay component.
2. In parallel, run P1 and P2 against the same fixed 200-source panel. They are
   the cheapest principled fixes and do not invalidate chemistry caches.
3. Add P3 only if P1/P2 improve the target marginals but leave exposure-driven
   triples or restatement drift.
4. Run P4 as a bounded separate-namespace mechanism experiment.
5. Rebuild paired electronic/catalog representation under P5 only after the
   cheaper trained residual and scheduling interventions fail their explicit
   gates.

This order preserves conditional throughput while addressing root causes
rather than turning the calibrated unconditional sampler into a collection of
post-hoc filters.

## Implemented bounded P1/P2 slice (2026-07-20)

The cache-compatible pilot slice is implemented behind three default-off
configuration flags:

- `--empirical-mark-prior-mode corpus_residual_v1`
- `--empirical-mark-prior-smoothing 1.0`
- `--ring-family-mass-mode catalog_topology_local_support`

P1 scans the already compiled teacher actions once and stores normalized,
smoothed corpus log bases for root atom type, connected atom type x installed
bond order, atom-restatement type, bond-reorder target order, and the production
semantic ring category (C/N/O/F x electronic role). The existing neural heads
are unchanged and learn residual logits above those bases. Structured ring
actions already persist scaffold/target bond orders, aromatic edges, and final
atom states, so ring electronic roles are read exactly from the saved Kekule
lowering without replaying or re-sanitizing molecular states. The action-only
reader was checked against the production semantic decoder on 632 saved ring
actions with no mismatch. The bases are checkpoint metadata and nonpersistent
model buffers; enabling them does not add, remove, or resize trainable tensors.

The deliberately bounded P1 boundary is global corpus categories, not the
larger proposed contextual table conditioned on both endpoint elements, old
order, and valence. This is the smallest faithful slice that directly attacks
the observed excess C#C/C#N target-order mass and the ring heteroatom/electronic
role drift. It remains a trained residual model with positive support, not a
ban or inference-time calibration. Contextual bases remain P3/P5 work only if
this slice fails its gate.

P2 groups the existing semantic ring templates by
`(topology_class, minimum-cycle-basis size multiset)` and sums their existing
catalog count prior. At a state, the fast local decorated-tree DP determines
which topology groups have structural support; the log of their absolute
catalog mass is added to the ring-grow family logit. The existing cached exact
mask or one-template executor certificate still exclusively decides whether
ring grow is executable, and the ordinary template/placement/electronic
conditional distribution is unchanged. Thus the new scalar is explicitly a
local topology proposal mass, not a claim of exact electronic support.

The bounded P2 key omits an additional aromaticity-pattern subdivision. That
is intentional for this first test: cycle-size/topology is the diagnosed
all-small support-collapse axis, while electronic composition is handled by
P1. Cycle rank is already determined by the minimum cycle basis. A finer
aromatic topology key would be a follow-up only if this slice improves small
rings without restoring complex topology.

### Exact cache contract

- **Reusable unchanged:** raw corpus/split, source trees, typed ring catalog,
  compiled molecular paths and packed checkpoints, training-support stream and
  every existing sparse exact/certificate shard, chemistry feature caches, and
  unconditioned fixed evaluation batches.
- **Conditioned evaluation:** the existing unconditioned evaluation batch can
  still be upgraded by attaching property values. Neither P1 nor P2 changes
  sampled states, teacher actions, support, or property normalization.
- **Old evaluation batches:** they need not contain the new topology-mass
  scalar. The model derives it deterministically from their states with a
  memoized fallback. New training batches precompute the scalar in DataLoader
  workers, outside the GPU update path.
- **New namespace required:** selected/recovery checkpoint, optimizer state,
  training metrics, and rollout cache. These change because the marked rates
  change.
- **No chemistry rebuild:** the empirical bases are a read-only scan of
  compiled teachers; topology groups come from the existing catalog; the P2
  scalar uses the existing local tree DP. Do not rerun path compilation or the
  exact training-support compiler for this pilot.

Checkpoint loading records the prior tables and both modes. Exact resume
requires them to match. Shape-compatible initialization may transfer every
existing neural tensor from the retained unconditional checkpoint while the
new fixed bases take effect; no generator layer is frozen by this feature.

### Conditional alignment and launch boundary

The paired recipes are
`tree_fcd_transfer_unconditional_chemistry_pilot.json` and
`tree_fcd_transfer_qed_conditioned_pilot.json`. Both use the same P1/P2 flags,
seed, path/support chemistry, 500-update horizon, and model architecture; the
conditional arm adds only the QED adapter/objective. The scientific comparison
must include the paired unconditional arm so a chemistry-base improvement is
not misattributed to conditioning. A later conditional run should preferably
warm-start from the P1/P2-qualified unconditional checkpoint, and all shared
generator parameters remain trainable.

### Pilot launch evidence

The first launch (`...-v1`) stopped before any update because the newly explicit
`property_conditioning: null` signature had no 256/512 base evaluation cache.
The paired QED cache was deterministically projected to the unconditional base
by removing only its property-value and property-mask tensors; all states,
teacher actions, and chemistry support tensors remained identical. The result
validated against the exact required signature and now lives at
`/_shared/evaluation_batches/evaluation-batches-v1-57ca0f...pt`.

The second launch exposed a throughput bug before training: the initial P1
implementation replayed every molecular state while the A100 waited. On one
immutable 4,096-record shard the old fitter had not completed after 261.56 s;
the exact action-only implementation completes the same shard in 0.350 s. The
full 98,434-record corpus fit completed remotely in 14.244 s.

The active immutable pilot is Modal app `ap-pvYMJ8MTVm1GVKVDyxG71U`, function
call `fc-01KXZWPHHA906MN5A35KDFAY06`, artifact
`/artifacts/compose-v4-unconditional-chemistry-p1p2-pilot-20260720-v3`.
It loaded the paired 256/512 evaluation cache in 0.080 s, completed compatible
initialization of 109 tensors, and finished all 500 GPU updates in 809.54 s
including corpus/path loading. Initial validation loss was 39.05183 with family
accuracy 0.55497. The selected step-500 validation loss is 11.98764 (69.30%
lower), family accuracy is 0.59686 (+4.19 percentage points), balanced family
accuracy is 0.53050 (+8.02 points), and ring-grow family accuracy is 0.78571
(+30.95 points; top-3 1.0). Held-out test loss is 13.84482 with family accuracy
0.58564, balanced family accuracy 0.52590, and ring-grow accuracy 0.78182.

This is strong optimization evidence but not yet chemical-quality evidence.
The run never met the preregistered 0.65 overall family-accuracy preview gate,
so no promotional chemistry claim is warranted from the training metrics. The
final, best-so-far, and exact-recovery checkpoints plus metrics are durable in
the v3 artifact namespace.

### Posthoc non-promotional rollout diagnostic

A bounded post-training diagnostic was nevertheless launched to locate the
remaining mechanism failures. It is explicitly posthoc and cannot authorize
promotion because the preview gate was missed and fewer than the preregistered
200 samples were requested. The first 100-sample monolithic job was preempted
at 80/100; its immutable retry reached 90/100 but made no further progress for
more than ten minutes and had not persisted a rollout cache. It was stopped and
replaced by 100 independently seeded one-rollout shards under Modal app
`ap-QnzBpdTvBda0pNSwskotLN`, artifact
`/artifacts/compose-v4-unconditional-chemistry-p1p2-posthoc-nonpromotional-eval100-sharded-20260720-v1`.
Each completed shard is durably reusable after preemption.

An interim reducer was run over the first 98 durable shards rather than letting
the two long-tail shards (indices 69 and 86) block diagnosis. The live job was
left running, but this subset may be biased toward faster trajectories. Against
the retained 600-sample calibrated-pancake audit, it gives the following
directional evidence:

- Endpoint and all-step validity/connectivity remain 1.0, with no canonical
  self events or event-budget exhaustion. The transaction/executor is not the
  source of these failures.
- Triple-containing molecules fall from 54.0% to 14.29%, triples per molecule
  from 0.802 to 0.153, and triple-bond share from 2.807% to 0.555% (matched
  heldout reference 0.262%). P1 therefore moves the triple-order marginal in
  the intended direction, though it remains about twice the reference share.
- Rings with at least three heteroatoms fall from 20.35% to 14.39%, and ring
  O-O prevalence from 7.0% to 5.10%. Ring N-N prevalence is essentially
  unchanged (39.67% to 38.78%); adjacent aromatic `[nH]-[nH]` is slightly worse
  (10.5% to 11.22%); and molecules with a ring containing multiple aromatic
  `[nH]` atoms are also unchanged/slightly worse (15.17% to 15.31%). The
  additive atom x role base does not capture the required joint ring-context
  correlations.
- The small-ring failure is substantially worse: molecules containing a
  three- or four-member ring increase from 11.33% to 45.92%, with 39
  three-member and 14 four-member SSSR rings among 98 molecules. Fused-ring
  prevalence falls from 28.5% to 21.43%, and no spiro example appears. The
  bounded topology-mass scalar has increased executable ring activity without
  learning the desired topology mix.
- Immediate backtracking rises from 0.143% to 2.917% per opportunity and from
  3.67% to 45.92% of trajectories, with 51 of 72 observed backtracks being
  `bond_reroute -> bond_reroute`. Mean unique-state fraction falls from 0.9981
  to 0.9691. This is a behavioral-rate/quotient-objective failure, not an
  executor failure.

Accordingly, P1/P2 is not a rollout-level improvement despite the large triple
reduction. Do not promote it or extend this training run. The next principled
lane is a hybrid that preserves the calibrated pancake behavioral family rates
while learning chemistry marks: distill rates over canonical successors or use
quotient-correct inference so syntactically distinct actions leading to the
same state do not induce reroute thrashing. Small-ring/topology mass needs an
explicit corpus-matched scheduling objective or support-aware topology
calibration, and heterocycle marks need joint contextual correlations rather
than independent atom x electronic-role bases. These are model/objective fixes;
no executor filter or aromaticity bandaid is indicated by the diagnostic.

The durable interim evidence is in
`diagnostics/p1p2_posthoc_nonpromotional_eval100_sharded_interim98_metrics.json`,
`diagnostics/p1p2_posthoc_nonpromotional_eval100_sharded_interim98_comparison.json`,
and
`diagnostics/p1p2_posthoc_nonpromotional_eval100_sharded_interim98_grid98.png`.

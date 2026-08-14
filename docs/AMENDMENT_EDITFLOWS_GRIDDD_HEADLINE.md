# Amendment: Edit Flows + GrIDDD are the two headline neighbors; DDSBM leaves the paper

**Canonical. Supersedes the GrIDDD demotion in
`workstreams/EXTERNAL_BASELINE_INDEX.md` and the DDSBM manuscript slot in
`BENCHMARK_RULINGS.md`. Nothing else in the comparator doctrine changes.**

The three-level structure is unchanged and is in fact what this restores:

> **nearest framework neighbor + matched causal controls + bounded task
> competence** — not nine heterogeneous molecule optimizers.

---

## 1. Edit Flows — primary framework neighbor. The "barred" framing was wrong.

Edit Flows is a CTMC over **complete variable-length objects**, learning rates
over **insertion / deletion / substitution** edits, trained by flow matching.
That is strikingly close to the *kind* of generative object COMPOSE proposes.

**Retracted:** listing it under barred/"never ported" in a way that read as
*avoid the comparison*. **It is the closest process-level prior art and should
be embraced as such.**

**Still barred, and this is the only thing that ever was:** *us* building a
molecular-graph Edit Flows ourselves and presenting that reconstruction as their
baseline. That would benchmark our port, not their method.

**No numerical row exists, and none should be manufactured.** Its benchmarks are
sequence tasks — MS-COCO captioning, HumanEval/MBPP, HellaSwag/ARC/PIQA/OBQA/
WinoGrande — compared against autoregressive generation and Mask DFM. There is
no sensible "run COMPOSE on the Edit Flows benchmark."

The comparison is **structural and must be made explicit**: same regime (CTMC,
discrete flow matching, variable-length state, insert/delete/substitute rates),
and the paper must argue why moving from token edits to a **chemically closed
molecular graph** — executable rewrite fiber, canonical-successor quotient,
graph/topology edits, statewise control — is substantive rather than a domain
swap.

## 2. GrIDDD — promoted to primary numerical external comparator

**This reverses my own ruling, and the reason I gave was too aggressive.**

The index recorded GrIDDD as `CONCEPTUAL_LINEAGE_ONLY` because *"its own
ablation shows insert/delete inert on DRD2."* That is true for DRD2 and
penalized logP — and **false for QED**, where removing insertion/deletion drops
GrIDDD's reported success from **45.1 % → 33.8 %**.

Note that `COMPARATOR_ROLES_CANONICAL.md` never demoted it. It said: qualify as
`FRAMEWORK_NEIGHBOR`, *"main row only if native task and conditioning semantics
align without substantial adaptation."* **That condition is now met.** The
amendment restores the canonical intent rather than inventing a new position.

Why it fits, precisely:

| requirement | GrIDDD |
|---|---|
| discrete generative dynamics | ✅ discrete diffusion |
| molecular **graphs**, not SMILES | ✅ |
| **variable size** during the process | ✅ insertion and deletion |
| source-conditioned **editing** of an existing molecule | ✅ its explicit study |
| established benchmark, not self-invented | ✅ **Jin et al. (2020) ZINC-250k** suite |

### The task: QED, not DRD2

**QED is the correct single competence task**, for two independent reasons.

**It interrogates the capability that makes GrIDDD relevant to us.** Insert/
delete carries 45.1 → 33.8 on QED, versus a much smaller effect on logP and
DRD2. The benchmark actually tests variable-size editing.

**Its oracle is deterministic.** QED is computed by RDKit. DRD2 uses a *learned
activity oracle*, so any budget asymmetry between GrIDDD's 20 sampled candidates
and COMPOSE's closed-loop successor inspection becomes a serious fairness
problem there. On QED it is far less offensive.

### The protocol, taken verbatim from GrIDDD

800 ZINC-250k source molecules · 20 generated candidates per source · start
QED 0.7–0.8 · success QED 0.9–1.0 · Tanimoto ≥ 0.4.

### The mandatory caveat — do not overclaim

> **This is an exact TASK and PROTOCOL comparison. It is NOT a matched
> computational budget.**

GrIDDD samples 20 candidates after conditioning; COMPOSE's closed-loop
controller may inspect far more successors and property values internally. Both
resource ledgers are reported separately and **never pooled**, exactly as in
P3/P4. Claiming matched oracle calls here would be false.

### Free contextual rows

GrIDDD's published table on this benchmark already carries **JT-VAE, CG-VAE and
GCPN**. Those come along as *reported* context at zero cost — **one COMPOSE run
on one benchmark carries several prior methods**, which is the comparator
doctrine working as intended. `EDM-SYCO` independently uses the same Jin suite,
confirming it is a continuing benchmark rather than a GrIDDD-specific setup.

**Do not go hunting for newer optimizers merely because JT-VAE and GCPN appear
in that table.** They are inherited benchmark context, not targets.

## 3. DDSBM — removed from the manuscript, retained in the record

> **Classification: developmental transport stress test. Excluded from the
> manuscript because its native scientific question is target-distribution
> transport, not source-conditioned molecular editing.**

**The reason is scope, not the result.** Stating this plainly because the timing
invites the opposite reading, and a later reader deserves the honest version:

**The removal was proposed after the FCD came back at 12.95 against DDSBM's
0.833.** Anyone reviewing this history should see that and judge it. The
defensible grounds are:

- DDSBM's task makes **reproducing the target chemical distribution part of the
  task itself**. COMPOSE's claim is modifying a *realized* molecule under an
  inference-time objective. Different questions.
- With Edit Flows and GrIDDD seated, DDSBM **fills no remaining scientific
  role** — it was the framework neighbor only while GrIDDD was demoted.
- Keeping it imposes a **qualification tax**: several paragraphs explaining why
  a distribution metric we never optimized shouldn't be read as a distribution
  claim. A reviewer sees `FCD 12.95 vs 0.833` and starts litigating whether
  COMPOSE is a bad distribution generator — **a question we would have
  introduced ourselves**, about a paper we are not writing.

**Not in the supplement either.** Supplementary experiments are still scientific
surface area. If we make no distribution-transport claim, we are not obliged to
introduce a distribution-transport benchmark.

### What is preserved, unrescued, in the development record

The finding is real and stays in `CLAIM_LEDGER` / development history:

| | |
|---|---|
| logP `W₁` | **0.0114** vs DDSBM 0.139 — **12.2× better** |
| QED MAD | 0.4161 vs 0.120 — 3.5× worse |
| SA MAD | 1.6911 vs 0.402 — 4.2× worse |
| FCD | **12.95** vs 0.833 — 15.6× worse |
| validity / trajectory-wide validity | 1.0000 / **1.0000** |

And the diagnostic that gives it meaning: `FCD(source→target) = 5.05` against
`FCD(COMPOSE→target) = 12.95`. **COMPOSE ended further from the target
distribution than the untouched sources were.**

**The lesson is load-bearing and must not be lost:** `R_θ` guarantees legality
and plausibility *support*, but does **not** preserve corpus-level distributional
chemistry when a controller aggressively optimizes an unrelated scalar. Mean
edits was exactly 6.0 — the controller never stopped early.

## 4. AReUReDi and pCoMole — lineage, not headline baselines

Neither is a native small-molecule graph editor, so neither is the right
experimental comparator for a variable-size molecular-graph process.

**AReUReDi** — its two benchmark domains are wild-type **peptide sequences** and
chemically modified **peptide SMILES**, with a peptide-specific validity filter.
Relevant lineage for Pareto guidance/control; not ordinary small-molecule design.

**pCoMole** — conceptually very close (pretrained discrete edit flow,
preferences, constraints, Doob-style control) but its work is biomolecular /
peptidomimetic **sequence** editing.

**Cite both conspicuously.** The sanctioned positioning is that same-lab
antecedents are surfaced, never buried — proximity is acknowledged in related
work precisely so no reviewer thinks it was hidden.

## 5. The frozen reviewer story

| method | question it answers |
|---|---|
| **Edit Flows** | *What is the nearest generative-process idea?* — framework comparison, no port |
| **GrIDDD** | *What is the nearest existing molecular graph model we can actually compare against?* — numerical, Jin/QED |
| **matched COMPOSE controls** | *What do executable rewrite support, closed-loop continuation, retargeting and pathwise constraints actually buy?* |

**This makes the novelty claim harder, not easier** — which is the point. We
seat the two methods a knowledgeable reviewer would immediately name, and then
have to show that going from token edits and generic insert/delete graph
diffusion to an executable chemistry-native stochastic process is substantive.

## 6. What this does NOT change

Pareto comparator work continues unchanged: ParetoFlow and A-GPS still need
audits, the exact-alignment-published-rows rule stands, and the Pareto matrix
still freezes before the fresh panel. The non-Pareto matched causal controls are
untouched.

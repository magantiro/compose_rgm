# Same-lab lineage audit

**Artifact status: `DESIGN_ONLY`.** Nothing installed, nothing run, no compute.

**Framing rule, binding, from the charter:** the relationship is complementary.
COMPOSE extends the lab's discrete multiobjective-guidance program to an exact
executable process over canonical molecular graph successor fibers. The result is
**never** framed as "COMPOSE beats Pranam's prior method." No same-lab baseline
is implemented or run without main approval **and** original-author validation.

---

## The shared-lab fact, verified

Three papers share **Tong Chen**, **Yinuo Zhang** and **Pranam Chatterjee**
(corresponding). A fourth, PepTune, shares Yinuo Zhang and Pranam Chatterjee.

| paper | authors as printed | source |
|---|---|---|
| MOG-DFM | Tong Chen, Yinuo Zhang, Sophia Tang, Pranam Chatterjee | arXiv:2505.07086 abstract page, read 2026-08-13 |
| AReUReDi | Tong Chen, Yinuo Zhang, Pranam Chatterjee | arXiv:2510.00352 abstract page |
| pCoMole | Tong Chen, Maximilian Holsman, Lin Zhao, Yinuo Zhang, Pranam Chatterjee | ICLR 2026 ReALM-GEN workshop listing, `iclr.cc/virtual/2026/workshop/10000793` |
| PepTune | Sophia Tang, Yinuo Zhang, Pranam Chatterjee | arXiv:2412.17780 abstract page |

**Do not repeat a search-snippet author list for pCoMole.** Web summaries
attribute it to a different author set; the venue listing above is the primary
record and disagrees with them.

---

## MOG-DFM — `NOT_FAITHFULLY_PORTABLE` (secondary: `CONCEPTUAL_LINEAGE`)

*Multi-Objective-Guided Discrete Flow Matching for Controllable Biological
Sequence Design*, arXiv:2505.07086. ICML 2025 Generative AI for Biology workshop
spotlight. Code `github.com/ynuozhang/MOG-DFM`, CC BY-NC 4.0 plus a custom
licence-agreement requirement; the HuggingFace mirror `ChatterjeeLab/MOG-DFM`
is access-gated. Checkpoints released.

| field | finding |
|---|---|
| native state space | fixed-length token sequence over a finite alphabet — peptide amino acids, enhancer DNA. Abstract: "steer any pretrained discrete flow matching generator". No molecular graphs; the repo has no SMILES tokenizer and no RDKit dependency. |
| transition object | single-position token substitution, realized as a reweighted CTMC factorized velocity, advanced by Euler sampling |
| source conditioning | **NO** — the process is initialized by sampling `x_0` uniformly from the discrete state space |
| base generator fixed | **YES, frozen** — "steer any **pretrained** discrete flow matching generator"; guidance reweights the pretrained velocity field at inference |
| guidance horizon | **LOCAL, one step.** Candidate scoring replaces the `i`-th token and measures the immediate score change. No rollout, no value function. |
| preference mechanism | simplex-lattice weight vector, linear scalarization of the improvement vector, blended with a rank-normalized term, plus an adaptive hypercone angular filter |
| feasibility semantics | no molecular feasibility notion; every fixed-alphabet string is trivially a valid sequence. The guarantee is that the guided velocities preserve valid CTMC dynamics. |
| small-molecule graph support | **NO** — would require building the state space, the successor construction and the source conditioning from scratch |
| reported compute | PepDFM 2×H100 NVL, 200 epochs, batch 512; EnhancerDFM 2×H100, 1500 epochs |

**Reasoning for the status.** Three independent hard blockers: fixed-length token
sequence rather than a graph; uniform initialization rather than a supplied
source; no chemistry in the codebase. Porting it to our task means rebuilding
every one of those, which is building a new method and wearing MOG-DFM's name.

**A caveat that must be checked before the related-work paragraph is written.**
MOG-DFM's causal role — *inference-time preference control over a frozen
reference process, scored one step locally* — is close to the role a COMPOSE
internal arm already occupies. `greedy_preference_run` in `pareto_control.py` is
exactly local preference-scored selection over the frozen process. If that
correspondence holds on inspection, MOG-DFM's status has a secondary reading of
`REDUNDANT_WITH_INTERNAL_ARM`: we already run its causal role as a control, on
our own state space, and running MOG-DFM itself would add a sequence result
rather than a molecular one. Recorded as a reading, not asserted — settling it
is a related-work decision, not a baseline decision.

---

## AReUReDi — `NOT_FAITHFULLY_PORTABLE` (secondary: `CONCEPTUAL_LINEAGE`)

*Annealed Rectified Updates for Refining Discrete Flows*, arXiv:2510.00352. Code
and checkpoints on HuggingFace `ChatterjeeLab/AReUReDi`, ungated, **no licence
declared** (API reports `license: null`).

| field | finding |
|---|---|
| native state space | fixed-length token sequence `S = V^L`. Two instantiations: peptide amino-acid strings, and chemically-modified **peptide SMILES strings** over a 586-token vocabulary. |
| transition object | single-coordinate token substitution proposal with Metropolis–Hastings accept/reject; the pretrained ReDi marginal supplies the proposal prior |
| source conditioning | **NO** — initial sequences are sampled from the pretrained SMILESReDi model from noise. The only length control is a `--gen_len` argument. Note the trap: §4.3's "wild-type" means *natural amino acids* as opposed to chemically modified, **not** a wild-type starting sequence. |
| base generator fixed | **YES, frozen** — guidance assumes access to a pretrained ReDi model and is inference-time only |
| guidance horizon | **LOCAL, one step.** Objectives are evaluated on single-substitution neighbours of the current sequence. No rollout. |
| preference mechanism | **annealed Tchebycheff** scalarization, `min_n ω_n s̃_n(x)`, with a Barker balancing function and MH acceptance. Reported experiments additionally impose a monotonicity gate that accepts only updates increasing the weighted objective sum. |
| feasibility semantics | intermediate filtering exists but is peptide-specific: transitions producing an invalid *peptide* SMILES are rejected, implemented as a weighted term inside the min-scalarization rather than as an exact constructive guarantee |
| small-molecule graph support | **NO** — SMILES is handled as a fixed-length token string; single-token substitution at fixed length cannot express bond or atom graph edits, and there is no source molecule to edit |
| reported compute | SMILESReDi trained on 4× RTX 6000 Ada (48 GB), 5 epochs; sampling budgets of 128 steps × top-200 candidates × 4–5 objectives |

**Reasoning for the status.** This was the expected "sole likely numerical
candidate", and on *availability* it clears the bar the others fail — open
checkpoints, open code, real peptide-SMILES experiments. It fails on **task**,
on every structural axis: fixed-length string state space, no supplied source,
substitution-only edits, peptide-specific soft-gated validity. It is a
sequence-edit method, and porting sequence-edit methods to molecular graphs
ourselves is a hard project stop.

It could become `NUMERIC_BASELINE_IF_AUTHORS_VALIDATE` **only** on a reframed
peptide-SMILES de novo task — which is not the COMPOSE task and would not answer
any of Q1–Q4. My recommendation is that it stays in related work.

---

## pCoMole — `UNVERIFIED`, and this is a gap that must be closed by a human

*pCoMole: Pareto-Constrained Molecule Editing with Discrete Flows*, ICLR 2026
ReALM-GEN workshop.

**Verified:** title, author list, venue — from
`iclr.cc/virtual/2026/workshop/10000793`.

**Not verified: everything else.** The paper is hosted only on OpenReview, which
now serves a browser-verification challenge to non-browser clients; the API
returns `403 ChallengeRequiredError`, and the paper is not on arXiv. **The
primary document was not read.** No code repository was found.

Accordingly, all ten schema fields — `native_state_space`, `transition_object`,
`source_conditioning`, `base_generator_fixed`, `guidance_horizon`,
`preference_mechanism`, `feasibility_semantics`, `official_code`,
`small_molecule_graph_support`, `reported_compute` — are **`UNVERIFIED`**.

Search-engine summaries of the blocked PDF exist and describe specific
mechanisms. **They are not recorded here as findings and must not be cited.**
The project has already paid for this exact mistake once: a committed artifact
stated that a check had refuted a delegated claim from a report that had not
arrived, and the correction notes that *a cited-but-absent source is
indistinguishable, to a later reader, from a verified one*.

**Why this gap is the highest-priority item in this document.** The title alone
— *Pareto-Constrained Molecule Editing* — describes preference-tilted,
constraint-aware editing of molecules under a frozen discrete flow. That is the
same conceptual family as COMPOSE, from the same lab, on molecules rather than
peptides. Two fields decide whether it is a lineage citation or a genuine
prior-art question: `native_state_space` (token string or molecular graph?) and
`guidance_horizon` (local or rollout?).

**Required action, for a human with an OpenReview login or an author contact:**
obtain the PDF. Do not let a submission go out with this cell unread.

---

## PepTune — the lineage neighbour the charter did not name, and the most important one

*PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided
Discrete Diffusion*, Sophia Tang, Yinuo Zhang, Pranam Chatterjee, ICML 2025,
arXiv:2412.17780. Code `github.com/programmablebio/peptune`, Apache-2.0.

This surfaced from the primary-source sweep rather than from the charter's list,
and it changes the positioning more than MOG-DFM or AReUReDi do.

From the abstract, verbatim: *"we introduce Monte Carlo Tree Guidance (MCTG), an
inference-time multi-objective guidance algorithm that balances exploration and
exploitation to iteratively refine Pareto-optimal sequences. MCTG integrates
classifier-based rewards with search-tree expansion."*

| field | finding |
|---|---|
| native state space | therapeutic peptide **SMILES**, under a Masked Discrete Language Model |
| source conditioning | **NO** — de novo generation, per the title |
| base generator fixed | guidance is inference-time; the diffusion model supplies the base process |
| guidance horizon | **ROLLOUT / TREE SEARCH** — the only one of the four with a future-aware guidance mechanism |
| preference mechanism | multi-objective classifier rewards with Pareto-optimal refinement over a search tree |
| official code | Apache-2.0, permissive — the only same-lab code here we could legally vendor |

**Status: `CONCEPTUAL_LINEAGE`.** It is de novo, it is peptide SMILES as token
sequences, and it is not portable to source-conditioned molecular graph editing
by us. It is not a numerical baseline.

**Why it matters anyway.** It establishes, in this lab's own prior work, that
inference-time multiobjective guidance with tree search over a frozen discrete
generative process is effective. So *"future-aware guidance beats greedy
guidance"* is not available to us as a novel Pareto headline — and the charter
had already ruled that out on separate grounds, since exact-target recovery
carries the planning claim. PepTune independently confirms that ruling from the
literature side.

What remains distinctly ours, and what the Pareto block should say:

> COMPOSE reuses one frozen executable molecular graph process across
> target-free preferences, imposing preference-specific purpose through
> inference-time control over **exact canonical successor fibers of a supplied
> source molecule**, rather than through objective-specific updates to the
> learned reference process.

The load-bearing words are *executable*, *exact successor fiber* and *supplied
source*. Tree search is not one of them.

---

## Statuses at a glance

| method | status | numeric baseline? |
|---|---|---|
| MOG-DFM | `NOT_FAITHFULLY_PORTABLE` (secondary `CONCEPTUAL_LINEAGE`; possible `REDUNDANT_WITH_INTERNAL_ARM`) | no |
| AReUReDi | `NOT_FAITHFULLY_PORTABLE` (secondary `CONCEPTUAL_LINEAGE`) | no — recommend related work |
| pCoMole | `UNVERIFIED` — primary document not read | undecidable until the PDF is obtained |
| PepTune | `CONCEPTUAL_LINEAGE` | no |

No same-lab method is recommended for implementation or execution. The one
outstanding obligation is to read pCoMole.

# Paper rewrite brief — control-substrate / anytime-Pareto-editing framing

> ## ⚠️ SUPERSEDED THESIS — 2026-07-28
> **The framing/positioning in this document is SUPERSEDED by
> [`PAPER1_FRAMING_AUTHORITATIVE.md`](PAPER1_FRAMING_AUTHORITATIVE.md)** (RGM-first, trans-dimensional).
> **This brief targets a DIFFERENT manuscript** (`paper_iclr_control_substrate/main.tex`), a 2026-07-22 fork that is framing-only, leaves every number as XXX, and still carries the uncorrected P1 charged-corpus claim. The live manuscript is `paper_iclr_stochastic_rewriting/main.tex`. Do not apply this brief to the live paper.
>
> **STILL VALID and deliberately preserved here — do not delete, this is the granular record:**
> its rewrite mechanics and numbers-macro discipline, if that fork is ever revived.
>
> When this document and the authoritative framing conflict on thesis, positioning, title, contribution
> order, or novelty claims, **the authoritative framing wins**. Operational detail below remains usable.



**Target file:** `paper_iclr_control_substrate/main.tex` (a copy of the old paper; the
original pathwise-framing main is preserved as `main_OLD_pathwise_framing.tex.bak`).
**Framing source (read first):** `docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md` (the advisor memo, verbatim).
**Numbers macros:** `paper_iclr_control_substrate/numbers.tex`.

## The reframe in one paragraph

Stop presenting validity / constraints / editing / Doob / oracle-efficiency as separate
contributions. Name the single new object that produces them all: **a learned stochastic
process whose states are complete molecular graphs, whose transitions are executable
rewrites, and whose probabilities live on canonical molecular successors.** RGM turns a
molecular generator into a **control-closed substrate**: any sampler that only deletes or
reweights legal molecular transitions inherits molecular-state validity. The control law
is one line — `Q̃_t(G,H) = 1{C(H)} · Q_{θ,t}(G,H) · Ψ_t(G,H,ω)` — where `1{C}` is hard
support, `Q_θ` the learned quotient rate over canonical successors, `Ψ` any controller
(MOG-style tilt, Doob `h(H)/h(G)`, SMC, MH). De-novo = `C≡1, Ψ≡1`; editing = `G_0=G_s`.

## Paper shape (three levels, memo §"strongest overall paper shape")

1. **Umbrella contribution:** control-closed molecular probability paths (the substrate + the
   control law + the closure property). Conceptual + `prop:closure`.
2. **Mechanistic spotlight (WE HAVE THIS — make it the empirical spine):** *Pareto reachability
   depends on path topology.* On the exact rewrite graph, greedy/monotone control has a
   structural ceiling below the reachable Pareto frontier; barrier-gated molecules are reachable
   only through valid downhill detours — a statement that is *undefined* if intermediates are
   not molecules.
3. **Supporting (WE HAVE THESE):** structural control (hard constraints in the fiber where
   generate-then-filter collapses); exactness foundation (Doob h-transform).
4. **Framework capability, full hero DEFERRED (XXX):** *anytime Pareto-set editing* — every
   visited molecule is potential output, source-relative nondominated archive. Present as the
   substrate's designed capability + future direction; the learned-editing hero needs the
   source-conditioned model (Stage B) and is NOT yet a result. Do NOT claim it as done.

## HONESTY CONSTRAINTS (do not violate — these were hard-won this session)

- **The anytime-Pareto-EDITING hero is not a result yet.** The current model is a de-novo
  (carbon-tree → refine) generator, OOD for real-lead editing; a fair editing win needs the
  source-conditioned retrain (memo §"source-conditioned model"). Frame editing as designed
  capability + motivated future work, results = **\XXX**.
- **The frontier-recovery *efficiency* claim is UNDECIDED** (an earlier "unguided detours lose
  at matched budget" negative was retracted — it was tuning-dependent). State only what is
  robust: greedy has a structural ceiling; detours can cross it; *efficient* recovery is future.
- **De-novo has a known ring defect.** Do NOT claim SOTA de-novo. Report the partial result
  honestly (below) and the mechanism-proven-but-untrained fix.
- **External-method characterizations (MOG-DFM, AReUReDi, pCoMole, MARS, MIMOSA, ConStruct,
  PRODIGY, junction-tree) must be treated as claims to verify against primary sources.** Keep
  the already-verified ConStruct/PRODIGY/CoCoGraph positioning; mark new comparisons as to-check.
- **"Simple baselines are strong."** Acknowledge GraphGA / PMO (Gao et al. 2022): uniform-legal
  is a strong baseline; our claim is the substrate + mechanism, not beating GA at optimization.

## RESULTS LEDGER — REFERENCE ONLY (this draft is FRAMING-ONLY; leave ALL of these as `\XXX`)

**OVERRIDE (per author): in THIS draft, leave EVERY numerical result as `\XXX` — including the
ones listed below that we already have.** The point of this pass is to get the framing/narrative/
structure right, not to commit numbers (some may still change after retrains). The ledger below is
a reference for a LATER fill-in pass, NOT for the current draft. Write the prose around each result
so a reader knows what will go there, but the value/metric/table-cell = `\XXX`.

### (reference) what we will eventually have
- **Barrier mechanism (the spotlight)** — `diagnostics/reachability/`:
  - cap-4 exact graph: 739 states / 9870 edges. Monotone control provably suboptimal for
    **18.6% (2-obj) / 27.4% (3-obj)** of preference directions (mean over 6 leads); worst single
    lead **33.7% / 53.3%**. When suboptimal it forfeits a **mean 77%** of achievable improvement
    (up to 100% at a strict local max). NOT a size-cap artifact (interior 17% vs boundary 18%).
    Robust to evaluation lookahead: 1-hop provably identical to movement (`lookahead1==lookahead0`);
    erodes only with ≥2-edit lookahead (**16.0% @ d2, 8.9% @ d3** for 2-obj).
  - cap-5 exact graph: 5454 states. Mean **23.9%** (2-obj), **worst lead 91.1%**, forfeit **85%**
    — the phenomenon strengthens with path length.
  - Frontier recovery (exact): greedy structural ceiling **54% frontier coverage / 95.7% HV**
    (2-obj), 43% (3-obj) — barrier-gated remainder greedy-inaccessible at any budget; detour-
    capable control reaches 100%. (Efficiency of recovery = XXX/future, per honesty note.)
- **Structural control** — scaffold **100%** vs generate-then-filter **96%→12%** collapse;
  required/forbidden-SMARTS **100%**; multi-property physchem box under exact scaffold
  **17.8% vs 3.5% per sample (5×)**, 100% scaffold.
- **Oracle efficiency / anytime** — **100%** usable-oracle in-fiber vs **39.8%** generate-then-
  filter; median lead reaches 90% of its gain by **~13** oracle calls.
- **Exactness** — Doob h-transform reproduces the exact conditional to **1.1e-16** on the real
  118-state CTMC; finite-particle guidance provably converges (exact-h ≈ **3×** fewer particles).
- **De-novo (PARTIAL — honest):** validity **1.00**, uniqueness **1.00**, novelty **1.00**,
  internal diversity **0.89**; fused/spiro ~reference. **Flaw:** small-ring **~49%** vs ref 6%,
  aromatic fraction **0.15** vs 0.48 — the "ring taxonomy" defect. Fix (`rate_factorization=
  superposed` + `ring_template_factorization=topology_cycle_hierarchical`, warm-start from B) is
  **mechanism-proven (falsifier test)** but **at-scale training pending (Modal-blocked)**. QED
  ~0.44 raw, **0.881 under QED control** (33% success, 12/12 beat-lead).

### XXX (pending — write the prose, leave the number as \XXX)
- Anytime Pareto-editing hero (feasible-HV vs oracle-calls, unique nondominated, frontier
  coverage, source edit distance, retention) — needs Stage B.
- Learned vs uniform-legal proposal efficiency (needs correct sampling + Stage B).
- Full de-novo sufficiency (V/U/N + descriptor Wassersteins + **fixed** ring marginals) — needs
  the ring-fix at-scale retrain.
- FCD / GuacaMol leaderboard — deferred, not a headline (say so).
- MOG-DFM / AReUReDi / Doob-lifted controller composition demonstrations — future.

## Mechanics
- Keep the method sections (RGM, COMPOSE process, Theoretical Properties, appendices) largely
  intact — the underlying method is unchanged; add the **control law** and the **quotient-
  successor / closure** emphasis, and the **barrier / objective-barrier** definition + prop.
- Rewrite for the new framing: **abstract, introduction, contributions list, experiments intro
  + primary-results framing, discussion/limitations.**
- Use `\XXX` (define `\newcommand{\XXX}{\textcolor{red}{XXX}}` if absent) for **EVERY** number, metric,
  table cell, and result — including ones we already have. FRAMING-ONLY draft; no real numbers.
- Preserve builds: it must `latexmk -pdf` with 0 undefined refs (use \XXX, not broken \ref).
- Titles to consider (memo): "Every Step Is a Molecule: Anytime Pareto Editing with Rewrite
  Generator Matching"; "COMPOSE: Control-Closed Molecular Generation on Valid Rewrite Paths".

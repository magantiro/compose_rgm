> **ARCHIVED — DO NOT USE.**
> Superseded by [`docs/EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), which is the only
> current plan. This document describes an earlier framing of the project and its
> experiment list must not be executed or cited. Kept on disk because other files
> still link to it; read it as history, not as instruction.

# Base element-vocabulary expansion: CNOF → full organic subset

**Status:** scoping doc for discussion (not started). A base-level retrain — decide before launching.

## Problem

Lineage B (our best quotient model) and therefore B-edit can **read** 12 elements but can only
**predict/generate/edit 4** — C, N, O, F. Real drug leads routinely carry S, Cl, Br (and sometimes P,
I, B), so B-edit cannot faithfully edit them: it can preserve such an atom if left untouched, but cannot
add, delete-and-replace, bioisosterically substitute, or ring-place one, and the teacher-scoring path
raises `KeyError` on any non-CNOF atom mark.

This is **by design** (the model matches GrIDDD "at the element level", so the CNOF-benchmark results are
correct as-is) — not a regression. It becomes load-bearing only when the editing target is real,
unfiltered drug leads rather than the CNOF-filtered benchmark.

## Evidence (verified in code, 2026-07-25)

- Input: `atom_embedding = nn.Embedding(M=12, hidden)` — all of `['null','B','C','N','O','F','P','S',
  'Cl','Br','I','SCAR']` have input slots. They exist but are **untrained** (CNOF corpus never fed them).
- Output: every atom-type head is sized `len(CNOF_ATOM_TYPES)=4` — `grow_root_head`, `restate_head`,
  the `atom_insert` head, `root_atom_log_probabilities`, `grow_option` (`3×`), `ring_electronic` (`2×`).
  No `len(ELEMENTS)` anywhere in the heads.
- `CNOF_SYMBOLS = ("C","N","O","F")` has been 4 since the first snapshot (`eaf328b`); never broader.
- Strict warm-start into this 4-wide architecture works → the B checkpoint's heads are 4-wide.
- Crash sites are output-only: `_CNOF_TO_INDEX[int(action.atom_type)]` (`_teacher_action_score`).

## Target

Output atom-type vocabulary = the input's organic subset: **C, N, O, F, S, P, Cl, Br, I** (+ **B**
optional, for boronic-acid warheads). Marginal cost of the rarer elements is a few head rows, so matching
the full input subset is the clean, maximally-comprehensive choice.

## Changes required

1. **Vocabulary (small, drives the heads).** Promote `CNOF_SYMBOLS` → `ORGANIC_SYMBOLS`; the heads and
   `_CNOF_TO_INDEX` resize automatically off `len(...)`. Rename to avoid the misleading "CNOF" label.
2. **Hypervalent H-derivation (the real extension).** S (2/4/6), P (3/5), I (1/3/5) are variable-valence;
   the fiber currently derives H from a single `CNOF_VALENCE[type]`. Switch to `ALLOWED_VALENCES[type]`
   (already defined) — pick the smallest allowed valence ≥ bond-order-sum, `H = valence − bond_sum` —
   in `_factorized_candidates` and the dense `restate`/`insert` masks. Fixed-valence adds (Cl, Br, B) are
   trivial; this step is what sulfonamides/sulfones/phosphates need.
3. **Partial warm-start from B (preserves all CNOF competence).** Expanded heads are `Linear(hidden, N>4)`;
   copy B's 4 CNOF rows into the matching output positions, initialize the new rows. Input embedding needs
   NO shape change (already 12-wide) — its S/halogen rows just start training. This is a **new init mode**
   (not the current strict `load_state_dict`); build + test it explicitly.
4. **Corpus.** Replace the CNOF filter (`data/cnof.py`) with the full drug-like organic subset
   (GuacaMol/ChEMBL/ZINC unfiltered to the 9–10 elements). Both base pretrain and B-edit edit-leads.
5. **Ring catalog.** Add S/P-containing ring templates + electronic aliases (thiophene, thiazole,
   thiadiazole, phosphole, …). `ring_electronic` head width (`2×|vocab|`) grows with the vocabulary.
6. **B-edit corruption.** Once the base handles them, the corruption naturally includes S/halogen leads
   (family-first draw already element-agnostic); re-verify finite loss on S/P/halogen teacher marks.
7. **Fold in #1 (heteroatom-scaffold grow).** The clean-delete inverse (re-cyclize keeping heteroatoms)
   is the same class of change and belongs in this retrain, giving symmetric ring editing.

## Cost / risk

- **Training:** a base pretrain (or long warm-started fine-tune) on the broader corpus — real Modal spend,
  larger than the B-edit fine-tune. New head rows + dormant input embeddings must train to competence.
- **Main code risk:** the hypervalent H-derivation (step 2) — get it right or masks/validity break;
  training-smoke S/P/halogen teacher marks to FINITE loss before any launch (the split-by-direction lesson).
- **Validation:** re-qualify unconditional quality on the wider vocabulary; confirm CNOF behavior is
  unchanged for the CNOF benchmark (the copied rows should keep GrIDDD-comparability intact).

## Open questions for Pranam

1. Element set — full `C,N,O,F,S,P,Cl,Br,I,B` vs a pragmatic `+S,Cl,Br` first? (Recommend full: the
   input already carries them and the marginal head cost is tiny.)
2. Warm-started head-expansion vs a clean from-scratch pretrain on the broader corpus?
3. Sequence — before the CNOF paper (which stands on its own), or after, as the "real-lead editing" step?
4. Corpus source for the broader pretrain (GuacaMol-full vs ChEMBL vs ZINC), and matching edit-lead set.

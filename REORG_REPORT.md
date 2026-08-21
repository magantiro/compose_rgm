# Reorganisation report — 2026-08-19

Scope: make this repository navigable for an ICLR submission and reproducible for
a newcomer, under a limited move authority (archive only, never delete, never
touch `src/`, never move a referenced file, never touch the live paper draft or
the named work-in-progress files).

---

## What changed

**Four files added. Nothing moved, nothing deleted, no existing file modified.**

| File | What it is |
|---|---|
| `README_NAVIGATION.md` | Root entry point. What the project is, which plan governs, where results live, where the papers are, how to run an experiment, and the Modal-volume fact. Created under this name because `README.md` already exists and `pyproject.toml` names it. |
| `docs/INDEX.md` | All 303 files under `docs/` in one table: purpose, status (CURRENT / SUPERSEDED / HISTORICAL / UNKNOWN), last-commit date. Status derived from content and git dates, never from filenames. |
| `archive/ARCHIVE_MANIFEST.md` | Records that zero files were archived, with the reference scans proving why, so nobody re-derives the analysis. |
| `REORG_REPORT.md` | This file. |

The classification is 109 CURRENT, 32 SUPERSEDED, 155 HISTORICAL, 7 UNKNOWN.

Verification after the work (both green):

```
python3 -c "import sys; sys.path.insert(0,'src'); import compose_v4"   # OK
python3 -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('modal_apps/*.py')]"   # 141 apps parse
git status                                                             # no deletion attributable to this work
```

> **Pre-existing working-tree state, not caused by this work.** `git status`
> already showed ` M docs/ARCHIVE_AB_RESULT.json`, ` M docs/RECEDING_HORIZON_AB.json`,
> ` M docs/VALID128_K8_RESULT.json` and ` D modal_apps/hphi_archive_read_app.py`
> **before** anything here was written. The deleted app in particular is an
> uncommitted deletion by someone else and should be resolved by its owner.

---

## Why nothing was moved

The move authority required *clearly superseded* **and** *zero inbound
references*. Those two sets barely intersect here.

- The nine documents carrying an explicit `ARCHIVED — DO NOT USE` banner — the
  only unambiguously superseded material in the tree — are **all referenced**,
  between 2 and 16 times each. `docs/REGISTRY_AUDIT_2026-08-19.md:323` reaches
  the same conclusion independently about the largest of them.
- Extending to all 32 files classified SUPERSEDED: **31 have inbound
  references.** The single exception, `docs/RDKIT_VALIDITY_FINDING.md`, is
  superseded only *in part* (its own banner: "Everything measured here remains
  true of the EAGER path"), already carries a status header, and sits next to the
  CURRENT document that supersedes it.

Full evidence, counts and reproducible commands are in
`archive/ARCHIVE_MANIFEST.md`.

### Status headers were also withheld — and this is the substantive finding

The brief's fallback for a referenced-but-superseded file is a one-line status
header at its top. **That fallback is unsafe in this repository**, because this
project cites documents by line number and the two 2026-08-19 audits are built
almost entirely on such citations. Prepending a line to
`docs/EXPERIMENT_PLAN.md` would invalidate 13 of them, to `docs/PROJECT_BOARD.md`
8, to `docs/CLAIM_LEDGER.md` 6 — including every citation in the audits a
newcomer is now being told to read first.

The extreme case is `CLAUDE_GENERATOR_READ_NOW.md` at the repository root — a
2026-07 note that issues an obsolete instruction ("return the acknowledgement
requested in section 11") to anyone who opens the root directory, and the file
that most looked like a move candidate. It cannot be moved
(`docs/CLAUDE_GENERATOR_RUN_LINEAGE_MANIFEST_V2.json:15` pins its **SHA-256** in
a sealed evidence manifest) and it cannot be edited (the same hash, plus
`docs/GENERATOR_LINEAGE_MAP.md:292,298` quoting it as `:12-13`). It is described
in `docs/INDEX.md` instead. **See the human decisions below — this one needs a
ruling, not a workaround.**

Preregistrations (`docs/AMENDMENT_*.md`, `docs/HPHI_V1_CORPUS_PREREGISTRATION.md`,
`docs/PARETO_DEVELOPMENT_PREREGISTRATION.md`) were excluded from header
consideration on principle: their scientific value is that they were frozen
before their data, and this project does not edit them afterwards.

---

## What was deliberately not touched

| Not touched | Why |
|---|---|
| `src/` | A process-identity SHA over `src/compose_v4/rewrite/` and `src/compose_v4/model/` gates every banked artifact. `docs/ADMISSION_MASK_OPTIMIZATION.md` records a measured, exact 6.28× optimisation that is *blocked* for exactly this reason — the project already pays real cost to keep that hash stable. |
| `paper_iclr2027/` | The live draft. Another agent is writing there now. Not read, not modified, not moved. (The brief named it `paper_iclr2026/`; no such directory exists — the live one is `paper_iclr2027/`, untracked at time of writing.) |
| `paper/`, `paper_arxiv/`, `paper_iclr_control_substrate/`, `paper_iclr_stochastic_rewriting/` | Deliberately preserved; each already carries a `_STATUS.md`. No header added — they have one. |
| `docs/AMENDMENT_*.md`, `docs/MASTER_PLAN_…md`, the named JSON results, `data/`, `third_party/`, `artifacts/`, `local_runtime/`, `tests/` | Named as live for work in progress. |
| `diagnostics/`, `results/`, `configs/`, `scripts/`, `modal_apps/`, `recipes/`, `runs/`, `baselines/` | Outside the archive rationale. `docs/CLAIM_LEDGER.md`'s artifact paths and `docs/SESSION_RUN_MANIFEST_2026-08-18.md`'s invocations resolve into these; moving anything would break a reproduction instruction for a tidiness gain. |
| Root `README.md` | Stale, but named by `pyproject.toml` (`readme = "README.md"`) and not overwritable under this brief. `README_NAVIGATION.md` supersedes it and says so at the top. |

**One caveat on the reference scan.** Generic basenames (`README.md`,
`STATUS.md`, `HANDOFF.md`, `PROTOCOL.md`, `DECISION_LOG.md`, `handoff.json`)
collide across directories, so their inbound counts are upper bounds. That error
only ever made the scan more conservative about moving.

**The repository moved under this work.** `HEAD` advanced four times while this
was being written (`ceb2e23` → `e3b1420`, four commits), and `docs/INVERSIONGNN_FPSI.pt`,
`docs/INVERSIONGNN_HPHI.pt` and `docs/INVERSIONGNN_SUPERVISION_10K.json.gz`
appeared partway through. `docs/INDEX.md` is a snapshot at `e3b1420` and is
generated so it can be regenerated; anything added later will show as UNKNOWN
until reclassified.

---

## What needs a human decision

Ordered by how much damage the ambiguity is doing right now.

**1. `CLAUDE.md` misdirects every session.** It auto-loads, and at lines 17–29 it
names `docs/EXPERIMENT_PLAN.md` as "the ONLY current experiment plan" and repeats
`ESTATE_REGISTRY`'s "84 markdown files". The governing plan is
`docs/MASTER_PLAN_EXPERIMENTS_AND_COMPARATORS.md`, and `docs/` holds 303 files.
Every agent session starts from the wrong plan. This is the single highest-value
fix and it is three lines of edit — but `CLAUDE.md` was not in the move authority
and editing it is a governance change, not a reorganisation.

**2. Which region of the master plan governs.** Lines 1–425 are declared
authoritative and line 426 opens `# ARCHIVED PROVENANCE — NOT GOVERNING`, yet
`# ⭐ CURRENT AMENDMENT` (line 1141) and `# ⭐⭐ CURRENT AMENDMENT II` (line 1740)
sit *below* that divider and both declare themselves governing. Anyone whose
habit is "read the CURRENT AMENDMENT at the end" is reading text the front of the
same file classifies as non-governing. Unresolved.

**3. The submission target.** `docs/PROJECT_BOARD.md:176-182` still records
"which of the four paper directories is the target?" as **unresolved**. A fifth
directory, `paper_iclr2027/`, now exists and is being written. No document
records that decision. Relatedly, all four preserved `paper*/_STATUS.md` notices
point readers at `docs/EXPERIMENT_PLAN.md` as "plan of record" — which is itself
superseded, so the preservation notices now misdirect.

**4. `CLAUDE_GENERATOR_READ_NOW.md`.** Obsolete instruction, root directory,
hash-pinned and line-cited so it can be neither moved nor edited. Options: (a)
leave it and rely on `README_NAVIGATION.md`; (b) add a *new* superseding note
beside it, leaving the original byte-identical; (c) accept breaking the sealed
manifest's hash and say so explicitly in the manifest. Needs an owner's call.

**5. `R_θ`'s frozen step: 12,500 or 8,500?** `docs/PROJECT_BOARD.md:106` and
`docs/DECISION_LOG.md:20` disagree. Independent evidence
(`diagnostics/exactness/editing_v2_experiment_b_exact_control.json`) favours
12,500, but no document records the correction. Two "current" documents, two
answers to what the frozen model is.

**6. Is the scope lock still in force?** `docs/SCOPE_LOCK_AND_KERNEL_COST.md` is
marked UNKNOWN in the index. Its kernel-cost blocker was closed by
`docs/KERNEL_COST_CHARACTERIZATION.md` and its P0c row by `docs/P0C_VERDICT.md`,
but whether the lock itself still binds after the 64-source QED result is
recorded nowhere.

**7. COMPOSE-Lipid / Paper 2: parked or abandoned?** `docs/ARTIFACT_INDEX.md:10`
calls it canonical. "lipid" appears zero times in any current plan and in no
commit in the last 200. Nine files in `docs/research_plans/` and
`docs/audits/2026-07-20_*lung*` depend on the answer.

**8. Provenance on the headline numbers.** 15 of 15 sampled recent `docs/*.json`
results record no git commit and no producing script, and
`docs/EXTENDED_CURVE_64.json`'s producer reads `/tmp/rv5` with no volume
fallback — so the repo's largest development result becomes unreproducible when
`/tmp` is reaped. `docs/HPHI_SMC_REFERENCE_BANKED.json` shows the fix and is
already in the repo. This is an owner task, not a reorganisation one.

**9. Untracked artifacts under `docs/`.** `docs/HPHI_N_SWEEP.json` and
`docs/INVERSIONGNN_BLAMBDA_DIAGNOSTIC.json` are results on disk but not in git.
`docs/INVERSIONGNN_FPSI.pt` and `docs/INVERSIONGNN_HPHI.pt` are ~4.7 MB trained
weights sitting in `docs/`, gitignored by `*.pt`. Decide whether trained weights
belong under `docs/` at all, or on the volume with a hash recorded here.

**10. The remaining UNKNOWN rows** in `docs/INDEX.md` — `GRAPH_ONLY_ENCODE_PLAN.md`
(implemented or not?), `PARETO_DEVELOPMENT_REVISED_PREREGISTRATION.md` and
`REFERENCE_LAW_ABLATION_PREREGISTRATION.md` (did they run?),
`RETARGETING_SAME_PREFIX_DESIGN.md` (is it the banked 3B result?). Each is one
sentence from the person who ran it.

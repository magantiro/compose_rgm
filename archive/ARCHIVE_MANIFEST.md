# Archive manifest

## Nothing was moved.

**Files moved into `archive/`: 0. Files deleted: 0. Existing files modified: 0.**

This directory exists so a future reorganisation has a destination and does not
repeat the analysis below. The reorganisation of 2026-08-19 added four files
(`README_NAVIGATION.md`, `docs/INDEX.md`, this manifest, `REORG_REPORT.md`) and
changed nothing else.

The move authority was: *clearly-superseded material only, and only if a
repo-wide grep for the basename (and, for docs, the stem) finds no hit outside
the file itself.* Those two conditions turned out to have an almost empty
intersection in this repository. The evidence is below so the next agent can
check it rather than re-derive it.

---

## 1 · Every clearly-superseded document is referenced

The nine documents carrying an explicit `ARCHIVED — DO NOT USE` banner are the
only unambiguously superseded material in the tree. **All nine are referenced by
other files**, so all nine are ineligible.

| Candidate (ARCHIVED banner) | Inbound references | Verdict |
|---|---|---|
| `docs/PAPER1_FRAMING_AUTHORITATIVE.md` | 16 | **NOT MOVED** |
| `docs/PAPER_MASTER_PLAN.md` | 11 | **NOT MOVED** |
| `docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md` | 9 | **NOT MOVED** |
| `docs/PAPER_POSITIONING_EXACT_CONTROL.md` | 4 | **NOT MOVED** |
| `docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md` | 4 | **NOT MOVED** |
| `docs/BASE_ELEMENT_EXPANSION_PLAN.md` | 2 | **NOT MOVED** |
| `docs/DEVELOPMENT_PLAN.md` | 2 | **NOT MOVED** |
| `docs/EDITING_V2_CYCLE_OPEN_MIGRATION_PLAN.md` | 2 | **NOT MOVED** |
| `docs/PAPER_REWRITE_BRIEF.md` | 2 | **NOT MOVED** |

`docs/REGISTRY_AUDIT_2026-08-19.md:323` independently reaches the same
conclusion about the largest of them: *"The file must stay where it is — ~13
inbound references depend on it."*

Extended to every file classified `SUPERSEDED` in `docs/INDEX.md` (32 files),
**31 have at least one inbound reference.** Counts, all non-zero:
`EXPERIMENT_PLAN.md` 30, `workstreams/constraints-hard/PROTOCOL.md` 18,
`PAPER1_FRAMING_AUTHORITATIVE.md` 16, `HANDOFF.md` 12, `PAPER_MASTER_PLAN.md` 11,
`CLAIM_LEDGER.md` 10, `EXPERIMENT_INFRASTRUCTURE_PLAN.md` 9, `PROJECT_BOARD.md` 7,
`HPHI_V1_CORPUS_PREREGISTRATION.md` 6, `PROJECT_STATUS.md` 6,
`ESTATE_REGISTRY.md` 5, `LANE2_FIVE_QUESTIONS.md` 5, `MOLLEO_ENV_GATE.json` 4,
`PAPER_POSITIONING_EXACT_CONTROL.md` 4, `PAPER_REFRAME_CONTROL_SUBSTRATE.md` 4,
`PARETO_DEVELOPMENT_PREREGISTRATION.md` 4, `RECEDING_HORIZON_AB.json` 4,
`CURRENT_MODEL.md` 3, `AMENDMENT_SHORTLIST_RETENTION.md` 3, and the rest 1–2.

Method, reproducible:

```bash
git ls-files docs | xargs -n1 basename | sort -u > /tmp/basenames.txt
rg -oIF --no-ignore --hidden -g '!.git/' -f /tmp/basenames.txt \
   --with-filename --no-line-number --no-heading \
   -g '*.{py,md,json,tex,sh,toml,txt,html,yaml,yml,cfg,bib,ipynb}' .
# then aggregate by matched pattern, dropping self-hits
```

> **Known limitation of that scan, recorded so it is not trusted too far.**
> Generic basenames (`README.md`, `STATUS.md`, `HANDOFF.md`, `PROTOCOL.md`,
> `DECISION_LOG.md`, `handoff.json`) collide across directories, so their counts
> are upper bounds. This only ever makes the scan *more* conservative about
> moving, which is the safe direction.

---

## 2 · The one file that passed the reference test, and why it still did not move

`docs/RDKIT_VALIDITY_FINDING.md` — **0 inbound references** (basename and stem;
verified with `rg -F` over the whole tree including untracked files, plus
`git log -S`). It was the only `SUPERSEDED` file to pass.

**NOT MOVED**, on three grounds:

1. It is superseded *in part*, not clearly superseded. Its own banner says so:
   *"Everything measured here remains true of the EAGER path."* The numbers are
   still valid for the path they describe.
2. The document that supersedes it, `docs/LAZY_SAMPLER_RESULT.md`, is `CURRENT`
   and sits beside it, and the banner cross-links the two. Moving one of the pair
   into `archive/` would break that adjacency for no gain.
3. It already carries the status header that the move authority offers as the
   substitute for a move. There is nothing left to add.

---

## 3 · Status headers were also withheld, on evidence

The brief's fallback for a referenced-but-superseded file is *"add a one-line
status header at its top."* **No header was added to any file.** Prepending a
line to these documents would silently invalidate the `file:line` citations that
the 2026-08-19 audits are built on:

| File | `file:line` citations against it | Prepending would break |
|---|---|---|
| `docs/EXPERIMENT_PLAN.md` | 13 | `REGISTRY_AUDIT` §5 C1/C7, `:3-5`, `:14`, `:21`, `:252`, `:665-668`, `:672-673`, `:681` |
| `docs/PROJECT_BOARD.md` | 8 | `REGISTRY_AUDIT` F4.1–F4.5 |
| `docs/CLAIM_LEDGER.md` | 6 | `REGISTRY_AUDIT` F3.1–F3.5, `paper_arxiv/SCIENTIFIC_TRACEABILITY.md` row IDs |
| `docs/ESTATE_REGISTRY.md` | 4 | `REGISTRY_AUDIT` F1.1–F1.6 |
| `docs/HANDOFF.md` | 4 | — |
| `docs/CURRENT_MODEL.md` | 3 | `REGISTRY_AUDIT` F2.3 |
| `docs/ARTIFACT_INDEX.md` | 2 | `REGISTRY_AUDIT` F2.1–F2.4 |
| `docs/PROJECT_STATUS.md` | 1 | `REGISTRY_AUDIT` F2.2 |
| `docs/research_plans/README.md` | 2 | `REGISTRY_AUDIT:219` cites its line 3 |

```bash
rg -o --no-ignore --hidden -g '!.git/' 'EXPERIMENT_PLAN\.md:[0-9]' . | wc -l   # 13
```

The worst case is `CLAUDE_GENERATOR_READ_NOW.md` at the repository root — a
2026-07 note that shouts an obsolete instruction at any newcomer, and the one
file that most looked like a move candidate. It is **doubly pinned**:

- `docs/GENERATOR_LINEAGE_MAP.md:292,298` quote it as `CLAUDE_GENERATOR_READ_NOW.md:12-13`;
- `docs/CLAUDE_GENERATOR_RUN_LINEAGE_MANIFEST_V2.json:15` records a **SHA-256 of
  its contents** (`0f880e47…`) inside a sealed evidence manifest.

So it can be neither moved (breaks the manifest path) nor edited (breaks the
hash *and* the line citations). **NOT MOVED, NOT MODIFIED.** It is described
instead in `docs/INDEX.md` and `REORG_REPORT.md`.

Preregistration documents (`docs/HPHI_V1_CORPUS_PREREGISTRATION.md`,
`docs/PARETO_DEVELOPMENT_PREREGISTRATION.md`, every `docs/AMENDMENT_*.md`) were
excluded from header consideration on principle: this project's scientific
claims rest on those files being frozen before their data, and it does not edit
them after the fact.

---

## 4 · What was deliberately not even considered

- Anything under `src/` — a process-identity SHA over `src/compose_v4/rewrite/`
  and `src/compose_v4/model/` gates every banked artifact.
- `paper/`, `paper_arxiv/`, `paper_iclr_control_substrate/`,
  `paper_iclr_stochastic_rewriting/` — deliberately preserved, each already
  carrying a `_STATUS.md`.
- `paper_iclr2027/` — the live draft, being written concurrently.
- `data/`, `third_party/`, `artifacts/`, `local_runtime/`, `tests/`,
  `diagnostics/`, `results/`, and every file named as live for work in progress.

---

## If you do move something later

1. Re-run the two scans in §1 and §3 first; both are cheap.
2. `git mv` only, so history follows.
3. Mirror the relative path: `docs/FOO.md` → `archive/docs/FOO.md`.
4. Append a row here: original path, new path, why superseded, and the grep
   output proving nothing referenced it.
5. Then verify: `git status` shows no deletion; `python3 -c "import sys;
   sys.path.insert(0,'src'); import compose_v4"` succeeds; and every
   `modal_apps/*.py` still parses.

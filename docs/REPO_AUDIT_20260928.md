# Repository audit — 2026-09-28 (read-only; nothing moved, nothing deleted)

Every number below was measured in this session. Where a first measurement was wrong I say so
and give the corrected one, because two of the instruments I reached for first reported false
values. Scope: the primary checkout `~/compose_rgm_git` (branch `compose-iclr`) plus the
`compose_pmo_chain` and `compose_t4_nitya` worktrees.

---

## 1. Where the 64 GB actually is — not where I assumed

    .worktrees/       38 GB     17 nested worktree checkouts
    diagnostics/      12 GB     main checkout working tree, 533 entries
    .git             583 MB     of which objects 525 MB
    artifacts/       101 MB
    runs/             86 MB
    local_assets/     72 MB
    tmp/              69 MB
    upload/           62 MB
    tests/            41 MB
    src/              40 MB
    docs/             19 MB
    everything else  ~25 MB each or less

**Git history is not the problem.** `.git` is 583 MB of a 64 GB tree. The size is 17 nested
worktree checkouts plus one 12 GB `diagnostics/`.

I had previously written that a full checkout is about 800 MB. That is true for a *fresh* one
and false for these: the largest is 9.1 GB. The difference is untracked output, quantified next.

---

## 2. The critical finding: 15.56 GB of single-copy untracked results

11 of the 17 nested worktrees carry untracked, non-ignored files — overwhelmingly under their
own `diagnostics/`. These are uncommitted experiment outputs. They are **not on origin and
exist only on this disk.**

    worktree                                    untracked files      MB
    fragment-linker-novelty-v1                           10,112   4275.5
    fragment-attachment-library-20260924                 18,520   4148.3
    fragment-decoration-source-coupled-v1                 6,973   4047.4
    fragment-linker-broaden-v1                            6,884   1790.6
    fragment-linker-strict-unseen-v1                      2,261    943.4
    fragment-decoration-breadth-v1                          844    350.9
    t4-objective-dynamic-reset-20260916                     860    313.2
    qed-program-controller-smoke-20260925                   488     39.1
    fragment-interface-ablation-20260923                     31     23.0
    fragment-qed-ablations-20260925                         802      3.6
    denovo-fragment-quality-20260923                         27      0.2
    TOTAL                                                47,802  15.56 GB

**Consequence for any cleanup: a branch-is-on-origin check does NOT protect these.** The branch
can be fully pushed while 4 GB of untracked results in the same checkout are unique. A previous
sweep in this project removed 26 checkouts and recovered 8.7 → 28 GB; whether it destroyed
untracked results of this kind is not established here, but the mechanism is real and the same
check would not have caught it.

### The six checkouts that genuinely are disposable

Zero untracked non-ignored files, zero tracked modifications, HEAD on origin at a matching full
SHA. Recoverable with one `git worktree add <path> <branch>`.

    denovo-legal-mark-prior-pilot-20260924   812 MB
    fragment-chem-prior-pilot-20260924       810 MB
    pmo-existing-region-remodel-20260924     1.4 GB
    pmo-realization-repair-20260920          772 MB
    pmo-seed-replication-20260924            320 MB
    t4-matched-panel-20260924                824 MB
                                            ~4.9 GB total

Also verified clean and fully replicated, left in place per instruction: the three `/private/tmp`
worktrees `armc_frozen` (28 MB), `compose_denovo_prior_split_20260924` (790 MB),
`compose_fragment_linker_old_20260924` (794 MB). Their commits resolve to
`origin/pmo-chain-ablation-20260925`, `origin/denovo-fragment-quality-20260923` and
`origin/fragment-attachment-library-20260924` respectively. **They sit on a reapable filesystem**,
so they are the one group at risk from the OS rather than from us.

---

## 3. The authoritative do-not-move set: 256 hash-pinned paths

Contracts under `configs/` pin files by sha256 and loaders fail closed. The pinned set **differs
by branch**, which is easy to miss:

    worktree                          configs   pinned paths
    compose_rgm_git (compose-iclr)        140             47
    compose_pmo_chain                     253            221
    compose_t4_nitya                      281            256
    UNION                                                256

Union by top-level directory:

    141  src/            52  modal_apps/     43  diagnostics/
      8  tools/           6  (bare files)     3  configs/
      2  docs/            1  data/

### The 11 `diagnostics/` directories that are effectively code

Moving or renaming any of these breaks a contract, and the failure surfaces inside a container
as `input identity mismatch`, far from the edit:

    diagnostics/multi_site_proposal_probe/                   8 pinned files
    diagnostics/t4_second_generation/                       17
    diagnostics/t4_held_target_distillation_quality_v1/      5
    diagnostics/t4_shared_program_controller/                4
    diagnostics/t4_integrated_route_fiber_braf_v1/           2
    diagnostics/t4_program_curriculum/                       2
    diagnostics/t4_integrated_route_fiber_braf_v2/           1
    diagnostics/t4_integrated_route_fiber_parp1_v1/          1
    diagnostics/t4_integrated_route_fiber_v1/                1
    diagnostics/t4_shared_retained_rewrite_v1/               1
    diagnostics/t4_strategy_reset/                           1

Full path list: `/tmp/pinned_union.json` (regenerate with the scan in §6).

**Additionally, and not visible to this scan:** `diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets/`
(71 MB) is **gitignored** yet baked into the PMO Modal image by literal path via `add_local_dir`.
It is not hash-pinned, so no contract scan finds it, and it is not in git, so no clone has it.
Moving it silently breaks image builds.

---

## 4. Reference counts for the consolidation candidates

Files containing a repo-relative literal (`"dir/`, `'dir/` or `(dir/`) across
`src scripts modal_apps tools configs tests`:

    directory        size    referencing files
    diagnostics/      12 G     421
    docs/             19 M     260
    artifacts/       101 M      51
    data/            6.1 M      45
    results/         7.2 M      26
    tmp/              69 M      11
    experiments/     6.3 M       9
    runs/             86 M       6
    local_assets/     72 M       1
    output/          2.0 M       1
    recipes/          60 K       1
    archive/         452 K       0
    upload/           62 M       0
    third_party/     216 K       0
    reviewer/        360 K       0
    baselines/        56 K       0

**A correction worth recording.** My first pass at this counted any line matching `dir/` and
returned inflated numbers — `experiments/` 139, `data/` 123, `artifacts/` 288 — because
`src/compose_v4/experiments/`, `src/compose_v4/data/`, GitHub URLs and absolute `/artifacts/`
container paths all matched. Requiring a quote or paren immediately before the name gives the
table above. `oracle/` is a special case: 24 matches, 0 B on disk, because it is PyTDC's
*runtime relative* path `oracle/<name>.pkl`, not a repository directory.

**Zero references is not a licence to delete.** `upload/` (62 MB), `archive/`, `third_party/`,
`reviewer/` and `baselines/` have no code references, but they may be paper assets, vendored
third-party code, or inputs to manual workflows. Each needs a human look, not a grep.

---

## 5. Safety actions already taken (non-destructive, all verified)

1. **`origin` URL corrected** from `KoshaTx/compose_rgm` to `magantiro/compose_rgm`. GitHub was
   serving the old URL by redirect; redirects serve fetches but **refuse to create new
   branches**, which was the real cause of the earlier push failures. All worktrees share this
   config.
2. **All 15 previously unpushed branches are now on origin**, verified by full-SHA comparison
   rather than by trusting push output.
3. **`remote.origin.fetch` repaired.** It was `+refs/heads/region-resampling:refs/remotes/origin/region-resampling`
   — a *single branch*. That is why only 16 stale remote-tracking refs existed and why
   `git branch -r --contains` cannot answer "is this commit safe on the remote" in this repo. Now
   `+refs/heads/*:refs/remotes/origin/*`, giving **188** usable refs. This fixes an instrument
   that had already produced false readings.
4. **Stale `revX` worktree registration pruned** (its gitdir pointed at a non-existent location).
5. **`~/compose_pmo_ablation` brought under version control** at
   `compose_pmo_chain/diagnostics/pmo_ablation_frozen_v1/`. It was an unversioned plain directory
   holding the only local copy of arm A's per-seed metrics. The home path is left in place because
   the reduction scripts reference it.

---

## 6. How to reproduce this audit

    # pinned-path union across worktrees
    for T in ~/compose_rgm_git ~/compose_pmo_chain ~/compose_t4_nitya; do
      python - <<'EOF'   # walk every configs/*.json, collect keys of any *_sha256 dict
      EOF
    done
    # untracked bytes per nested worktree (use Python; BSD xargs -r is unsupported and
    # `xargs -0 stat -f%z | awk` silently prints 0)
    git -C <worktree> ls-files --others --exclude-standard -z
    # reference counts
    grep -rIl --include='*.py' --include='*.json' -E "[\"'(]DIR/" src scripts modal_apps tools configs tests

---

## 7. Four instrument failures from this session, all of which reported success or absence falsely

Recorded because the repo's own `learnings.md` is full of this class and these are four more.

1. `if timeout … git push -q … | tail -1` — piping makes the `if` test **tail's** exit status,
   always 0. Five failed pushes printed "PUSHED".
2. `for b in $BR` in zsh — **no word splitting on unquoted variables**, so a 15-branch loop ran
   once on the concatenated string and reported one MISSING branch.
3. `"refs/heads/$b:refs/heads/$b"` in zsh — `:r` is parsed as a history modifier and eaten,
   mangling every refspec to `NAMEefs/heads/NAME`. Use `${b}`.
4. `git ls-files -z | xargs -0 -r stat -f%z | awk` — BSD xargs has no `-r`; the pipeline produced
   nothing and the awk printed **0 MB** for 15.56 GB of files.

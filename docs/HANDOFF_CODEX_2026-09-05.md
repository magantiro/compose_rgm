# COMPOSE handoff — region resampling / T4 (2026-09-05)

> Historical handoff. For the current controller state, completed T4 result,
> PMO status, and collaborator workflow, read
> [`START_HERE_ICLR.md`](START_HERE_ICLR.md) first.

## The point of the project

COMPOSE treats molecular design as **control of a frozen learned process**. A
generator `R_theta` is learned once over executable chemical rewrites; a
controller steers trajectories without changing it, as a KL-regularised change
of path measure. Every transformation is realised as primitive executable steps
through complete, chemically valid molecules -- never an endpoint a search must
later justify.

The intended end state, in order:

    executable region rewriting        DONE, qualified
    local <-> global rewrite geometry  DONE, gate passed
    Q(M|x,z)   region/scale selection  mu_exec done; V_z null so far
    Q(o|x,M,z) macro/program option    <- THE CURRENT GAP
    population search                  loop built and running
    T4 benchmark                       in progress, plateaued (see below)
    PMO benchmark                      after T4

`docs/CAMPAIGN_LESSONS.md` holds 41 rules exported from the agent's persistent
memory. Read it before running anything expensive -- most entries exist because
a run was lost or a number was misread.

## Where to start an agent

```
cd /Users/rmaganti/compose_rgm_git      # <- initialize the agent HERE
git checkout region-resampling          # already the current branch
```

Remote `https://github.com/KoshaTx/compose_rgm.git`, branch `region-resampling`,
fully pushed. Nothing needed from any other directory.

### The other directories, and what they are

| path | size | what it is | keep? |
|---|---|---|---|
| `/private/tmp/compose-process-v2-atom-delete` | empty | the shell's cwd only; under `/tmp`, wiped on reboot | no |
| `/Users/rmaganti/compose_rgm_git` | 454M | **the working clone. all campaign work is here and pushed** | YES |
| `/Users/rmaganti/compose_v2_work` | 1.8G | orphaned git worktree; runs were launched from here | see below |
| `/Users/rmaganti/Documents/Codex/2026-07-14/ok-so/compose_rgm` | 6.4G | the original full clone, on the old branch `agent/compose-rgm-full-snapshot-20260720` | optional |

`compose_v2_work` is a worktree whose parent `.git` is at a `~/Documents` path
this process could not read, so it has been OUTSIDE version control all
campaign. Its code was byte-identical to the clone; its unique diagnostics are
now committed. Two files there are still not in git and are load-bearing:

    docs/INVERSIONGNN_FPSI.pt      4.6MB
    docs/INVERSIONGNN_HPHI.pt      4.7MB

`modal_apps/invgnn_hphi_corpus_app.py` mounts those exact paths with
`add_local_file`, so copy them into any tree that must run that app. They match
the `*.pt` gitignore rule, which is why they were never committed.

To work in the 6.4G original instead: `git fetch origin && git checkout
region-resampling`, then copy the two `.pt` files across. There is no other
reason to prefer it.

## What lives outside any folder

Artifacts are on the Modal volume `compose-v4-artifacts`, NOT in the repo:

    /region_committor/committor_bellman_v1.pt   the fitted structural committor
    /region_committor/{screen,collect,arms,race,mass,scale_gate}/
    /t4_population/                             T4 cells
    /artifacts/editing_v2/r_theta_run           frozen R_theta

Read them with `modal volume get compose-v4-artifacts <path> <dest>`.

## Run discipline (learned the hard way)

* **Launch long work into the DEPLOYED app.** `modal deploy
  modal_apps/genmol_t4_opt_app.py`, then `python3 tools/t4_launch.py`. Four T4
  runs died because `modal run --detach` still cancels a `.map()` when the local
  client dies, and spawning from an ephemeral `modal run` is no better -- the
  app is torn down when the entrypoint returns.
* **Persist every iteration**, not on completion. A run at 176/180 iterations
  still lost three whole cells.
* **`python3 tools/preflight.py`** before launching: prints branch, commit and
  any drift under the mounted `src/`+`configs/`, and can abort.
* **Never key a law/successor cache on `canonical_state_key`** -- it collides
  across slot layouts while actions carry slot coordinates. Use
  `(atom_types.tobytes(), bonds.tobytes())`.
* Modal cap is 80 containers; CPU only for this work.

## State of the science

Qualified and frozen:

* structural execution -- guided population completes 7/16 held-out regions
  where base `R_M` completes 0/16;
* the general local->global gate -- every scope band produces coherent rewrites,
  cost rising 7.5x from local to global;
* `mu_exec(M|x)` region prior -- cross-validated AUC 0.884, 2.3x uniform choice.

Null / not supported:

* the QED task tilt `V_z` -- 2 wins, 2 losses, 2 ties on six held-out seeds. The
  estimator had 46 observations and at most a 1.7x preference range. Not a
  verdict on the decomposition.
* "an 80%-scope region replaces 80% of the molecule". Realized coherent change
  sits at 0.05-0.24 regardless of intended scope.

## The open problem, and the next step

T4 on parp1 seed0 (d=0.4) plateaus around -8.0/-8.7 against prior COMPOSE -9.5
at 500 calls and -9.8 at 1000. The winning molecule needed a NEW FUSED RING
(+1 ring for +1 heavy atom); ours add halogens, sulfur and methyls with
heavy-atom bloat, changing a ring system once in forty dockings.

Measured cause: the region controller proposes over RAW primitives. On that
seed's law `atom_insert` carries 0.457 of R_theta's mass and `cycle_close`
0.00007 -- a ~6500:1 deficit against the move that wins -- and a KL trust region
at kappa=1 cannot lift a 7e-5 action (that needs ~7.3 nats).

**COMPOSE already has the layer that solves this, and the new path bypassed it.**
See `docs/MACRO_INVENTORY.md`: fourteen macro options with primitive-support
restrictions and contracts, built from measurements of exactly these biases
(`atom_insert` is halogen-heavy and that is R_theta's own law; ring formation is
NOT `cycle_close` alone -- a six-carbon precursor offered ZERO `cycle_close` and
127 `bond_insert`, two closing a hexagon).

The agreed next step is a three-level controller:

    Q(M|x,z)  ->  Q(o|x,M,z)  ->  q(w|x,M,o)
     WHERE          WHAT             HOW

with bundles becoming `(parent, region, option)`, `o` restricting admissible
primitive support through the existing `MACRO_FAMILIES` /
`macro_action_distribution` machinery, `generic` permanently active, and an
applicability-aware balanced prior over `o` (do NOT train a `Q(o)` model yet).
Do not change `kappa`, retrain `R_theta`, or redesign `Q(M)`.

Open question for the user, unanswered: `BUILD_RING_SYSTEM` is an 11-step
program (`scaffold_extend x8 -> append_system x1 -> restate x2`), not a
single-step option. Treat programs as compound options now, or start with
single-step options and add programs after the audit?

## Key files, and the numbers to beat

New modules from this campaign:

    src/compose_v4/control/region.py            region enumeration, interfaces
    src/compose_v4/control/region_rewrite.py    the frozen inner controller;
                                                kl_tilt lives here, NOT adaptive_tilt
    src/compose_v4/control/region_selector.py   mu_exec and Q = mu_exec*exp(V/tau)
    src/compose_v4/control/task_value.py        V_z, winsorised shrunk means
    src/compose_v4/control/macro_engine.py      PRE-EXISTING option layer, unused
                                                by the region path -- the gap
    modal_apps/committor_bellman_app.py         screen / collect / fit / race / mass
    modal_apps/region_scale_gate_app.py         the local->global gate
    modal_apps/population_search_app.py         the QED population loop + A/B
    modal_apps/genmol_t4_opt_app.py             T4, both the old and new paths
    tools/{preflight,t4_launch,t4_audit,repo_audit,build_dude_pool}.py

Inputs:

    docs/GENMOL_T4_SEEDS.json        15 benchmark seeds (the T4 cells)
    docs/GENMOL_T4_DEV_SEEDS.json    22 dev seeds; [6:22] trained the committor,
                                     [:6] were sentinels -- BOTH are spent
    docs/DUDE_HOLDOUT_POOL.json      743 fresh DUD-E actives, canonical-disjoint
    docs/STRUCTURAL_QUAL_POOL.json   220 MOLLEO/PMO molecules (distribution-shifted)

**The bar, read from artifacts, never transcribed:**

    diagnostics/genmol_t4_official_s2.json   prior COMPOSE @500 calls/cell
        delta=0.4  solved 14/14  mean -10.26  best -11.70
        delta=0.6  solved 11/13  mean  -9.24  best -13.40
    diagnostics/genmol_t4_official_s1.json   prior COMPOSE @200 calls/cell
        delta=0.4  solved 14/15  mean -10.01
    docs/genmol_t4_all_methods.json          published comparators
        GenMol 26/30 (-10.62 / -9.81), GraphGA 19/30, RetMol 11/30

    On the audit cell (parp1 idx=0, delta=0.4): seed -7.3, prior COMPOSE -8.5
    @200 and -9.5 @500, ivg -9.8 @1000, GenMol -10.6. The new path reached -8.7
    at 71 calls before the plateau.

Frozen parameters -- do not tune without a reason: `kappa=1.0` (KL trust region),
`tau=0.05`, `epsilon=0.1` (primitive exploration floor), `epsilon_region=0.2`
(scale floor), 8 lineages, 20 dockings/round.

## Reproducing what has been run

`experiments/INDEX.md` lists 193 apps and 266 entrypoints with exact commands.
`experiments/region_resampling/README.md` is the campaign manifest: commands in
order, the gates as they were fixed BEFORE results, and what the results do not
support. `docs/MACRO_INVENTORY.md` is the option vocabulary.

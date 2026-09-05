# COMPOSE handoff — region resampling / T4 (2026-09-05)

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

## Reproducing what has been run

`experiments/INDEX.md` lists 193 apps and 266 entrypoints with exact commands.
`experiments/region_resampling/README.md` is the campaign manifest: commands in
order, the gates as they were fixed BEFORE results, and what the results do not
support. `docs/MACRO_INVENTORY.md` is the option vocabulary.

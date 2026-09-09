# Resumable frontier, stage 1

Prospective implementation decision, 2026-09-09. Implements the first stage of
`CONTROLLER_RESEARCH_RECOMMENDATION_2026-09-09.md`, not its learned critic or
benchmark campaign. Status before implementation: proposed, no new task results.

## Scope and generated object

The problem is efficient task-directed search over COMPOSE's exact executable
molecular transitions. Outputs are complete molecular graphs with primitive edit
witnesses, not ring templates or target endpoints. The scientific hypothesis is
that retaining unfinished search and exposing completed-option feedback earlier
will use proposal computation better than waiting for whole-horizon returns.
This stage tests engineering behavior, not docking performance or generalization.

Keep WHERE (region), WHAT (option), HOW (primitive), the current region/option
priors and exploration floors, generic support, kappa=1, frozen reference model,
executor, persistent-slot identities and endpoint thresholds. The production
1..40-active-atom, charge-preserving, non-stereochemical support is unchanged.
There is no new local or global operator. Region size remains permission to edit,
not a promise about realized displacement. No winner structures enter this path.

## Explicit first-stage approximation

Use a sparse, exact-transition search graph and a resumable planning cursor.
Each call advances a finite scheduling quantum, then returns its exact cursor;
the quantum does not terminate a molecular program or label it unsuccessful.
An option-complete state can be scored immediately, even with primitive budget
remaining. Endpoint-infeasible states stay in the search frontier.

For the current frozen task snapshot, let g(s) in [0,1] be the existing endpoint
desirability at an option completion. On the *explored* graph, propagate

    B(s) = max({g(t): a witnessed path from s reaches an evaluated completion t},
               default=0).

This is a best-witness search heuristic. It is NOT an expected return, a calibrated
value estimate, a stochastic-policy value lower bound, or an exact Doob transform.
In particular, do not reuse the old importance-weighted return names or claims.
For each full reference row, tilt the core by (1+B(child))/2 with the existing
kappa=1 solver, then mix the unchanged exploration floor. Unknown branches have
neutral value 1/2; zero endpoint desirability is not an infeasibility prune.
Every row is complete, and canonical molecular probability aggregation remains
inside the existing production hierarchy. A larger enumerated frontier does not
constitute additional sampled region bundles.

Planning and committed sampling use separate RNGs. Only committed option
completions enter the initial oracle-candidate interface. Planning witnesses
guide proposals but are not extra oracle slots or extra outer region draws.
The legacy whole-horizon planner remains available as a comparison.

## Persistence and task feedback

Checkpoints contain exact augmented molecular states, all completed cached rows,
the unfinished planning cursor, committed frontier, RNGs, event provenance and
snapshot/process identities. Resume uses exact states, never SMILES reparsing.
Unfinished committed lineages survive preparation slices and oracle-round
boundaries. Completed lineages may restart from the next round's scored archive.
Task snapshot changes invalidate heuristic scores, not executable states or rows.
The archive must be an append-only extension of the previously bound prefix.
No predicted or bootstrapped label is added to the docking archive.

The new T4 entry point is prepare-only and uses a distinct artifact schema. It
must not bypass the existing candidate replay/audit and explicit docking launch
boundary. No Modal jobs, docking calls, model training or whole-suite milestone
claim are authorized by this implementation note.

The prepare-only API is `compose_v4.experiments.t4_frontier_search.prepare_slice`.
Pass its `result["checkpoint"]` into the next call. A completed slice permits a
new scheduling quantum; an interrupted slice retains per-lineage event targets
and requires the original quantum, so already-finished lineages do not receive
extra draws on restart. A zero-event slice can refresh the task model at a round
boundary without executing new chemistry. Progress callbacks receive complete,
self-hashed checkpoints for atomic publication by the caller. Only the final
`proposed_for_audit` field names prospective oracle candidates, not authorization.

Physical HOW witnesses are cached alongside full rows, so resuming a cached
committed decision does not replay that row merely to recover its primitive mark.
This first adapter has no Modal launcher and is deliberately rejected by the
legacy v2 candidate-lock verifier. Its prospective run must add a compatible
candidate-replay/launch contract before docking, rather than relabel this schema.

## Minimal acceptance and comparisons

- Exact split/resume equivalence on a small deterministic search graph, including
  RNG state and a pause inside an unfinished program; no artificial failed return.
- Task feedback before the complete primitive horizon, with unchanged full-row
  support, local/global scale floor, generic option floor and kappa bound.
- Production-state codec round-trip including region, slot lineage, option phase
  and all supported progress payloads; malformed/incompatible checkpoints fail.
- A tiny production-executor T4 fixture exercises prepare, resume, endpoint
  accounting and a new task snapshot without recreating molecules from SMILES.
- Same-snapshot cached rows do not re-enumerate. Snapshot refresh removes old
  scores while retaining the exact frontier. Interrupted work is reported.
- Focused legacy dependency tests, formatting, lint and diff review. No new
  expensive capability panel or target-recovery run.

The next performance comparison is against the same generator with post-hoc
selection, under separately recorded candidate, oracle and compute accounting.
Compare local/global allocation against a declared local-only ablation only after
the full controller is frozen, not by silently disabling larger regions now.
Assess best feasible score, candidate/bundle diversity, intended versus realized
scale, ring-system/cycle-rank deltas and proposal time. No claim that this stage
learns delayed task value or matches InVirtuoGen is licensed by passing its tests.

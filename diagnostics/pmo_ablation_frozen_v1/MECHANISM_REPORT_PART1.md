# What coordinates a COMPOSE program, and what determines its probability
## Part 1 — the program representation, dataflow layer, and the PMO proposal path
Code-grounded. Static code analysis unless a line is marked MEASURED.

---

## §3 (central) — what makes a program *coordinated* rather than a longer legal sequence

### 3.1 Operands are typed symbolic handles, not slot indices

`src/compose_v4/control/edit_program.py` — *"Transferable, attachment-bound primitive
programs with persistent atom handles… These are optional executor-supported optimization
proposals, **not samples from a certified R_theta law**."* (module docstring, lines 1-7)

Every operand inside a program mark is a one-key typed reference:

    {"input": i}     the i-th ROOT atom of whatever molecule the program is bound to
    {"created": j}   the j-th atom BORN by an earlier atom_insert in THIS program

`extract_program` (:171) converts a concrete executed trace into this form: it walks the
trace, assigns `{"created": k}` to each `atom_insert`, and lazily assigns `{"input": rank}`
to any source slot first touched — *erasing the source addresses*. That is what makes a
program transferable to a different molecule rather than a replay of one trace.

### 3.2 The handle discipline is statically enforced

`EditProgram.__post_init__` (:105-142) validates before anything executes:

| invariant | line | failure |
|---|---|---|
| blocks partition the whole trace, strictly increasing | :119 | `program blocks must partition the full primitive trace` |
| birth handles sequential, never reused | :128 | `birth handles must be sequential and never reused` |
| every reference is live (born, not deleted) | :137 | `unbound, deleted or malformed atom reference` |

Liveness is tracked explicitly: `active` starts as every input atom (:120), gains
`("created", j)` on each insert, and `active.remove(...)` on `atom_delete` (:142).

**This is the answer to "how do later edits refer to atoms created earlier":** by persistent
typed handle, with producer/consumer correctness proven statically before execution.

### 3.3 There is a real dataflow graph with dependencies, conflicts and capacity

`src/compose_v4/control/edit_program_graph.py` — *"Joint multi-site proposals, explicit
dataflow and conservative serial scheduling… Every scheduled primitive is revalidated on its
actual predecessor. **No intermediate endpoint utility participates in execution or
scheduling**."* (:1-7)

`compile_program_graph` (:81) builds per-block `BlockFootprint(reads, writes, creates,
deletes, minimum_delta, peak_delta, final_delta)` and two edge sets:

- **dependencies** (true dataflow): if a block references a `created` handle owned by
  another block, add `(owner, consumer)` (:109). A block that consumes an atom born
  elsewhere *must* follow its producer.
- **conflicts** (hazards): blocks i<j conflict when
  `writes_i ∩ (reads_j ∪ writes_j)` or `writes_j ∩ reads_i` (:136) — RAW/WAW/WAR over atom
  handles.
- **serialization_edges** = conflicts not already dependencies (:143). Overlap does **not**
  prune either edit; it *orders* them.

Stated conservatism, in the code's own comment (:110): *"All explicit operands are
conservatively treated as writes, including anchors whose hydrogen counts change. This is
not a full guard model."*

### 3.4 Capacity is reserved across the whole program, not checked at the endpoint

`_topological` (:150) schedules greedily over a priority permutation and admits a block only if

    1 <= atoms + minimum_delta   and   atoms + peak_delta <= 40        (:174-175)

so the 40-heavy-atom ceiling is enforced at every block's **peak**, with a
`capacity_timeline` of (start, peak, end) recorded per block. That is genuine resource
reservation along the program.

### 3.5 Failure semantics — all-or-nothing, no rollback, no retry

- dependency/serialization cycle -> `ValueError: program dependency/serialization graph contains a cycle` (:168)
- nothing schedulable under capacity -> `ValueError: no ready complete block fits the frozen peak-capacity limit` (:179)
- `scheduled_program` docstring (:216-217): *"The bounded greedy resource schedule may abstain
  even when another ordering works. **It never expands the atom ceiling or emits a partially
  completed endpoint.**"*

So a program either completes or is discarded. There is no partial endpoint, no rollback to
an intermediate, and no in-execution retry. Reordering is safe because `scheduled_program`
renames created handles to be globally fresh in the new order (:219-232).

Chemical validity is separate from all of the above and is re-checked per primitive:
`execute_program_graph` receipt records `"scheduling": "deterministic serial; executor
revalidation after every primitive"` (:270), and the graph payload declares
`global_revalidation: [connectivity, valence, aromaticity, capacity, charge]` (:76).

---

## §4 — what determines the probability of a program on the PMO path

**R_theta does not enter it at all.** Three independent pieces of evidence:

1. MEASURED: the scored entry point `compose_v4.experiments.pmo_population_v1` has a
   120-module `compose_v4` import closure taken *by execution*, containing **zero**
   `torch.load` calls and **zero** rate-model instantiations.
2. `edit_program.py` module docstring: *"not samples from a certified R_theta law."*
3. `execute_program_graph` stamps every receipt with
   `"proposal_law": "new optimization proposal; no reference likelihood claimed"` (:271).

The ordering that does determine a proposal is:

    parent selection -> channel dispatch -> program construction -> compile_program_graph
      (dependency + conflict + acyclicity) -> _topological capacity schedule
      -> execute_bound_program with per-primitive executor revalidation
      -> candidate pool -> _credit_allocate selection -> ledger.query(endpoint)

Selection uses `population_features` (pmo_population_controller.py :135-168), a 16-dim
task-free vector: parent reward, incumbent-parent gap, primitive_edits/32, block_count/8,
delta_from_measured_parent/16, realized and target retention, **three one-hot channel
indicators** (shallow / structured / jump), and a five-rule histogram
(atom_insert, atom_delete, cycle_close, cycle_open, ring_system_restate), plus a bias.

Program work limits come from the contract, not the code: `program_max_blocks: 8`,
`program_max_primitives: 32` (`configs/pmo_population_controller_v1.json` :12-13).

---

## §6 — what the PMO structured-vs-chain ablation actually removes

It removes, **together**:

1. **joint structural planning** — the multi-block program with an intended shape;
2. **dependency-aware construction** — typed handles, producer/consumer edges, hazard
   serialization, and peak-capacity reservation (§3.2-3.4);
3. **catalogue / donor information** — the jump lane's teacher-derived plan library
   (`diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json`,
   `fit_scope: shared_all_routes`), and transplant donor material;
4. **program reuse** — `_recombine` / mutation of existing archived programs.

It does **NOT** remove learned scoring, because **there is none in either arm** on this path.

Therefore it must be reported as *structured, dependency-aware program construction versus
uniform legal local editing under a matched planned allowance* — not as an isolated
coordination ablation, and not as a learned-vs-unlearned comparison.

---

## Still to trace (Parts 2-3)
§1 task->implementation map for all eight reported task families; §2 per-constructor
transformation semantics (segment_replace/segment_grow, structured, anchored, transplant,
jump realizer); §5 executed traces per macro family from saved artifacts; the fragment
reference ablation's probability-dependent stages; and the supported scope of the
shared-process claim.

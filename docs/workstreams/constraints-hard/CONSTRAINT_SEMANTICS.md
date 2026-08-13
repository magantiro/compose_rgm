# Constraint semantics — Stage 0 executor audit

**Status: `DESIGN_ONLY`.** No claim-bearing compute. No Modal launch. Every
finding below is a local, CPU-only code read plus a deterministic probe that
needs neither the `R_theta` checkpoint nor the Gate-0 authenticated chain.

**Base commit:** `f6146d7` (branch `codex/compose-constraints-hard`).

---

## 1. The verdict

> **`LABELED_SUBGRAPH_PRESENCE_INVARIANT`.**
>
> **`IDENTITY_INVARIANT` is NOT provable on the production trajectory pipeline
> and must not be claimed.**

The executor supports *exact atom- and bond-labeled subgraph presence* as a
statewise predicate. It does **not** carry a persistent atom-to-atom
correspondence from the source molecule through a multi-step trajectory.

This is the answer to the question the charter called critical, and it is the
answer that constrains every downstream sentence in the paper. The rest of this
document is the evidence and its consequences.

---

## 2. Why the question is subtle: identity *is* preserved by one rewrite

`MolecularGraph` (`src/compose_v4/chem/molecular_graph.py`) is a **fixed-slot
array** representation: `atom_types[i]`, `formal_charges[i]`,
`implicit_h_counts[i]`, `bonds[i, j]`. Slot `i` is a stable address, and unused
slots are `NULL` padding rather than absent.

The micro-operators mutate those arrays **in place at a named slot** and never
renumber:

| operator | file:line | slot behaviour |
|---|---|---|
| `apply_atom_insert` | `src/compose_v4/rewrite/operators.py:443` | writes at `op.slot`, a previously `NULL` address |
| `apply_atom_delete` | `src/compose_v4/rewrite/operators.py:490` | sets `op.v` to `NULL_IDX`; all other slots untouched |
| `apply_atom_restate` | `src/compose_v4/rewrite/operators.py:517` | mutates element/charge/H at `op.v` only |
| `contract_scars` | `src/compose_v4/chem/molecular_graph.py:519` | drops scar slots to `NULL_IDX` **in place**; no compaction |

So a **single** `RewriteSystem.apply` is slot-preserving, and
`contract_scars` — the one read-out post-process that removes sites — is
slot-preserving too. Probe 1 confirms it: deleting slot 3 of `CCCO` leaves
surviving atoms at slots `(0, 1, 2)`, exactly their original addresses.

**This is why the answer is not obvious, and it is also the trap.** Identity
survives the *operator*. It does not survive the *pipeline*.

---

## 3. Three independent places where atom identity is destroyed

Any one of these is sufficient to bar `IDENTITY_INVARIANT`. All three are
present in the production path.

### 3.1 The trajectory is carried as a canonical SMILES string, not a graph

`modal_apps/retarget_intervention_app.py:243-256` is the production
state-advance idiom, and it is representative of every controller app that
consumes the kernel:

```python
def successors(key: str, slots: int):
    ...
    state = pad_molecular_graph(smiles_to_molecular_graph(key), slots)
    ...
    enumeration[key] = [(s.key, float(s.probability))
                        for s in result.batch.successors]
```

The trajectory is `trajectory: list[str]` of canonical SMILES
(`retarget_intervention_app.py:276-282`). At every step the `MolecularGraph`
object — and with it the slot labelling — is **discarded and rebuilt by parsing
the canonical SMILES**. RDKit assigns parse-order indices to the canonical
string, which are unrelated to the previous step's addresses.

Probe 3 shows the renumbering is not hypothetical and not rare:

```
built from 'OCCN' : atom_types [4, 2, 2, 3, ...]  key NCCO
re-parsed from key: atom_types [3, 2, 2, 4, ...]  key NCCO
identical atom_types array? False
```

A single round-trip through the molecule's own canonical key **reverses the slot
labelling**. `O` was slot 0 and is now slot 3.

`src/compose_v4/experiments/editing_v2_bridge_control.py:99-101` does the same
thing explicitly: `states[successor.key] = pad_molecular_graph(
smiles_to_molecular_graph(successor.key), int(slots))`.

### 3.2 Alias collapse substitutes an arbitrary representative labelling

`src/compose_v4/experiments/production_successor_kernel.py:573`:

```python
successor_states.setdefault(key, successor)
```

Distinct marks that reach the same canonical SMILES are merged into one
`CanonicalSuccessor`, and the **retained `state` is whichever mark the
enumerator happened to reach first**. `alias_count` records how many were
merged; which one survived is not recorded.

This is not a symmetry technicality. Probe 2:

```
source          : CCC   occupied (0, 1, 2)
delete slot 0   : key=CC  retained slots=(1, 2)
delete slot 2   : key=CC  retained slots=(0, 1)
SAME canonical key?      True
SAME retained slot set?  False
```

Two actions that **retain disjointly different atoms** produce the same
canonical successor and are merged. A controller protecting "the atom at slot 0"
cannot distinguish them, because the object the controller sees is the key.

Alias collapse is also *load-bearing and frozen*: `validate_successor_batch`
(`src/compose_v4/experiments/successor_kernel.py:271-276`) **raises** if two
successors share a key — "aliases were not merged". The canonical successor
fiber is defined modulo this collapse. Undoing it to recover identity would not
be a constraint experiment; it would be a different process.

### 3.3 Canonicalization is lossy in ways the representation documents

`molecular_graph_to_smiles` ends in `Chem.MolToSmiles(mol)`. The module
docstring (`molecular_graph.py:43-47`) states the known limitations plainly:
**stereochemistry is not encoded** (cis/trans and R/S are lost on round-trip)
and **isotopes are not encoded**. Whatever "the same atom" could mean, it cannot
mean more than the constitutional 2D labelled graph carries.

---

## 4. What this licenses, and the exact wording to use

**Provable, and the claim we may make:**

> At every committed state of a COMPOSE trajectory, at least one exact atom- and
> bond-labeled subgraph embedding of the protected core exists.

**Not provable, and barred:**

> The atoms of the source scaffold are the same atoms throughout the trajectory.

The difference is real and a reviewer will find it. In principle a trajectory
could destroy the source's scaffold atoms and rebuild an isomorphic copy
elsewhere in the molecule, and the predicate would be satisfied throughout. We
should say so rather than be asked. The honest framing is that the constraint is
on the **molecular state space**, not on an atom-tracking bookkeeping layer —
which is in fact the stronger architectural statement, because it is a property
of `F_C(x)` rather than of a wrapper.

### Barred phrasings

- "atom-mapped", "atom-tracked", "persistent atom identity", "the same atoms"
- "the scaffold atoms are never touched"
- anything implying a source→state correspondence survives canonicalization

### Permitted phrasings

- "labeled-subgraph presence is invariant along the trajectory"
- "no committed state lacks an embedding of the protected core"
- "the protected core is present in every realized molecule"

---

## 5. Where a hard constraint may and may not be injected

### The `constraints=` hook does NOT work for masking

`RewriteSystem.__init__` accepts `constraints: Iterable[Constraint]` with
`Constraint = Callable[[MolecularGraph, Any, MolecularGraph], bool]`, and
`RewriteSystem.apply` (`src/compose_v4/rewrite/kernel.py:71-75`) enforces them.
It looks like the natural injection point. **It is not usable here.**

`apply` **raises** `InvalidRewrite` on a failing constraint (Probe 4), and
`production_successor_kernel.canonical_successor_result:557-567` wraps `apply`
in a `try/except` that converts **any** exception into a fatal
`ProductionSuccessorKernelError` whose message is *"the model's legal mask
admitted an action rejected by the production executor"*. A protected-core
constraint installed there would not filter the fiber; it would crash the kernel
on the first core-violating mark — i.e. on nearly every state.

That guard is correct and should not be weakened: it exists to catch genuine
model/executor mask disagreement, which is a real defect class. Installing a
*semantic* constraint into a hook reserved for *legality* would blind it.

### The correct injection point

Filter the enumerated canonical successor list and renormalize:

```
F(x)   = result.batch.successors                       # exact legal fiber
F_C(x) = [s for s in F(x) if C(s.key)]                 # hard admissibility
         then renormalize probability over F_C(x)
```

Renormalization is **required**, not optional: `validate_successor_batch`
(`successor_kernel.py:297-302`) rejects a batch whose successor probabilities do
not sum to 1. `SuccessorBatch.is_terminal` already covers the empty case, so a
mask-empty state is representable as terminal without special-casing.

This placement is also the architecturally honest one. It leaves `R_theta`
completely untouched — same weights, same marked law, same enumeration — and
expresses the constraint as exactly what the paper claims it is: a restriction
of the successor support, applied at inference time, composed after legality and
before plausibility is renormalized.

The project already has vocabulary for this: `SUPPORT_ABLATION` in
`successor_kernel.py:308` — *"support differs ONLY in preregistered fields"* —
versus `LAW_ONLY`. **The hard-mask arm is a `SUPPORT_ABLATION` and must be
declared as one.**

---

## 6. The protected-object definition (frozen BEFORE any outcome)

**Primary protected object:** the **atom- and bond-labeled Bemis–Murcko scaffold
of the source molecule**, computed once from `x_0` and then held fixed for the
whole trajectory.

Frozen match semantics — what counts as "the same labeled subgraph":

| feature | in the label? | reason |
|---|---|---|
| element symbol | **yes** | the core's atoms are the core |
| formal charge | **yes** | already conserved by `editing_charge_policy_constraint` |
| bond order / aromatic class | **yes** | the representation stores class 4 explicitly |
| ring membership of query bonds | **implied** | a query cycle forces a cycle in the target |
| implicit H count | **no** | substituents attach by consuming H; requiring it would forbid all decoration and make the constraint vacuous-by-strangulation |
| stereochemistry | **no** | not representable (`molecular_graph.py:43-47`) |
| isotope | **no** | not representable |

**Two consequences that must be measured rather than assumed:**

1. **Fusion is permitted.** A core ring atom may end up inside a larger fused
   system and the embedding still exists. This is defensible ("the core is still
   present") but it is a *choice*. Do not silently rely on it — report
   `source_murcko == successor_murcko` as a **secondary descriptive statistic**,
   never as the constraint.
2. **Aromaticity relabelling can cause false violations.** An edit outside the
   core can change RDKit's aromaticity perception of a fused core ring, failing a
   strict aromatic-class match even though every core atom survives. The census
   **must** report the rate at which the mask rejects a successor solely because
   of a core aromaticity relabel. If that rate is material, it is a finding about
   the predicate, not about COMPOSE.

**Eligibility, frozen from starting-state applicability only** (charter Stage 0
item 5): nonempty protected scaffold; editable structure outside the core;
native method applicability. **No eligibility criterion may reference any
controller outcome**, and no source may be dropped because its scaffold result is
unfavourable. This mirrors `diagnostics/retarget_heldout_panel.json`'s
`not_an_eligibility_criterion` field, which is the house pattern.

---

## 7. The guarantee that must never be reported as a result

The `hard_scaffold_mask_verified` arm retains the protected core in **100%** of
committed states. This is **true by construction** — the mask removes every
core-violating successor before the controller ever sees it — and under the
project's sign-guarantee meta-rule it is a **construction check, recorded
without a denominator**, never an empirical success.

Per the meta-rule, the declaration must carry both halves:

**(1) Mathematical reason.** For the estimand *"fraction of committed states
satisfying C"* on the arm `hard_scaffold_mask_verified`: the commit set is
`F_C(x) = {y in F(x) : C(y) = 1}` by construction, so every committed `y`
satisfies `C(y) = 1`. The estimand is identically 1 on this arm and has no
falsifying range.

**(2) Adversarial fixture — a neighbouring metric that CAN go the other way.**
On the *same arm*, **support retention** `|F_C(x)| / |F(x)|` and **mask-empty
frequency** are not sign-fixed: they range over `[0, 1]`, and a mask that
strangles the fiber shows up there immediately. Likewise **terminal objective
improvement** on the hard arm can be worse than, equal to, or better than the
unconstrained arm. `tests/test_hard_scaffold_constraint.py` must contain a
fixture in which the hard arm's retention collapses and mask-empty is high, and
assert that the reported metrics say so.

**The actual measurement in this experiment is what the *unconstrained* arm
does** — how often it leaves the admissible region — **and what the constraint
costs.** Not that the constrained arm obeys its constraint.

---

## 8. Reproducing this audit

```bash
python3 /path/to/identity_probe.py     # Probes 1-4, no checkpoint, no network
```

The probe source is reproduced at
`docs/workstreams/constraints-hard/probes/identity_probe.py`. It imports only
`compose_v4.chem.molecular_graph`, `compose_v4.chem.state`,
`compose_v4.rewrite.operators` and `compose_v4.rewrite.kernel`. It constructs no
model, opens no Gate-0 decision, and touches no network. Runtime is under two
seconds on CPU.

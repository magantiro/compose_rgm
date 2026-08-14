# Same-lab lineage and positioning

**Status: `DESIGN_ONLY`.** No same-lab numerical baseline was run, designed, or
costed. This document exists to fix the framing before any figure or sentence is
written.

---

## 1. The frozen framing sentence

> Prior work develops guidance and constrained control for discrete generative
> processes; COMPOSE supplies an exact executable molecular graph substrate over
> canonical legal successor fibers.

Use this. It is accurate, it is generous, and it locates the contribution in the
**substrate**, which is what is actually new.

## 2. Hard rules

- ❌ **Never frame any result as COMPOSE defeating a previous same-lab method.**
  Not in a table, not in a caption, not in a sentence of the form "unlike X, we…".
- ❌ **Do not run a same-lab numerical baseline without original-author
  validation.** If a number is wanted, the author of the original method
  validates the configuration first.
- ❌ **Do not port sequence methods to molecular graphs.** Building "method X for
  molecular graphs" to put a name in a table is inventing a new method and
  attributing it to someone else.

## 3. Lineage, and what each item is allowed to establish

| work | relationship | allowed role |
|---|---|---|
| **pCoMole and related same-lab sequence-control methods** | conceptual lineage — guidance and constrained control over discrete generative processes | **cited as lineage.** No homemade graph port. Per `BASELINE_IMPLEMENTATION_POLICY.md` this is the "published idea, no usable implementation for *our* object" case, and option 3 — related work, not an engineering branch — is the correct one |
| **Edit Flows** | conceptually adjacent; published object is variable-length **sequences** | related work only. The policy names this case explicitly: we do **not** build "Edit Flows for molecular graphs" |
| **ConStruct** | hard structural constraints throughout graph-generation trajectories | **cite, do not port.** General graph method; released projectors are `planar`/`tree`/`lobster`; no source conditioning; no molecular config shipped |

## 4. Why the substrate framing is the honest one

The constraint mechanism this lane audited is **not** a new algorithm. It is a
filter on an enumerated successor list plus a renormalization
(`PROTOCOL.md` §4). That is deliberately unglamorous, and it is the point:

- the *legality* is the executor's, and it is exact;
- the *plausibility* is `R_theta`'s, and it is untouched;
- the *admissibility* is a predicate, injected at inference time on the realized
  state;
- the *purpose* is the controller's.

The claim is that these compose over a shared executable substrate — not that any
one of them is individually novel. Guidance under constraints is prior art;
having a canonical legal successor fiber to apply it to is the contribution.

## 5. What would make a same-lab comparison legitimate

All three, together:

1. the original author validates the configuration and agrees the task is native
   to the method;
2. the comparison is on an object both methods natively address — **not** a graph
   port of a sequence method;
3. it is reported as a **capability/scope** comparison, not a leaderboard.

Absent all three, the correct output is a citation.

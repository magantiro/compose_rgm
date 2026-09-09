# Target-informed whole-ring planning diagnostic

## Scope and acceptance

User-authorized on 2026-09-09: compile the first PARP1 seed0, delta=0.4 IVG
winner as coherent structural stages before attempting autonomous discovery.
The platform remains a frozen executable molecular process. This diagnostic's
output is an exact primitive trace plus named program boundaries, not a new
generator, learned proposal law, or docking result.

Hypothesis: selecting a complete ring descriptor before execution avoids
unnecessary search over unrelated insertions and closure endpoints. A compiled
ring remains a sequence of existing Active8 edits; it is not a new whole-ring
executor transition. All states remain connected supported molecules in the
40-active-atom, 48-persistent-slot, charge-preserving, achiral 2D support.

Use pair `5705946f25f35dd88189b1204f01f294e96a43e78fd1c5030967c48cecceccc8`
from the existing input-hashed winner-path audit. Preserve its exact source
state, not a SMILES reconstruction. The winner and its path are explicitly
development/answer-known inputs. No motifs, templates, learned parameters or
proposal weights are extracted for blind search.

The structural plan is:

1. Remove the terminal dimethylamino group and extend the surviving linker.
2. Construct one pendant aromatic C6 ring, choosing the complete descriptor.
3. Fuse one nonaromatic C6 ring onto it and add its carbonyl oxygen.
4. Replace one original core ring bond with a carbonyl-bearing carbon path.

The two ring stages must satisfy the existing `RingSpec`, `RingProgress`,
descriptor support and completed-construction contracts. Local stages are
explicit primitive programs and must not be counted as pre-existing compound
options. Every action must replay through the production semantic executor.
Final canonical 2D identity must equal the pinned winner. Preserve failures.
These are upper-bound witnesses, never a shortest-path proof.

For the core stage, try the four saved open/insert/insert/close actions first.
If the executor rejects that sequence or the complete endpoint differs, record
the outcome and try the saved six-action variant including its two electronic
preparation edits. These are the only declared variants; no molecular search,
new law enumeration, docking, model fitting, or remote launch is authorized by
this diagnostic. Compare counts against the saved 23-edit witness, not against
unmeasured blind optimizer performance. Record compilation, validation and
replay separately, including software, exact inputs, code closure, all attempts,
program boundaries and primitive work.

Keep the production WHERE/WHAT/HOW controller, generic support, R_theta and
kappa unchanged. After this check, report which program descriptors fit the
existing menu and which need a separately authorized general compiler before
they can become blind proposal channels. Endpoint descriptors alone do not
prove learned-law support or useful controlled probability.

## Runnable check and test asset

`tools/t4_whole_ring_plan.py --audit <existing ivg_winner_paths/audit.json>
--output <new result.json>` runs from clean committed source under RDKit
2024.03.5, with `PYTHONPATH=src:.` plus the chemistry environment if needed.
It refuses output overwrite and binds the pinned audit, physical pair receipt,
source closure, executor attempts, exact states and software. The small
`tests/test_whole_ring_plan.py` suite includes the real first-winner regression;
only that test skips when its documented vendored compressed receipt is absent.

## How this can inform the controller

The existing `MolecularHierarchy` selects regions and options, then the
`OptionContinuationKernel` samples primitive HOW decisions. The two ring
requests used here belong to the existing default `RingSpec` menu. This check
additionally fixes the attachment, atom order and bond pattern before executing,
so there is no full learned-law enumeration in this diagnostic. It is an
executor-level compilation witness, not an exact sampler of that hierarchy.

A prospective controller integration should expose a complete ring request as
the planning object and evaluate its completed result. It must specify a
normalized, target-independent descriptor proposal and its relation to the
frozen reference process. The forced winner recipe must never become the blind
proposal law. Generic atom/bond editing and the current region/scale selection
remain available. A different proposal law is an explicit controller change,
not a claim that the original Doob or KL law was preserved by compilation.

The current one-step shrink/grow options can execute the linker edits, but
`remodel_linker` is not an existing compound option. Likewise the core program
is not the current `expand_ring` option: it adds carbonyl oxygen in addition
to inserting a ring atom, and the current expansion contract excludes junction
bonds and requires only one newly added atom. These need generic program
interfaces, not a lookup table of winner fragments. Their independent primitive
routes remain valid even when an option contract cannot express them.

The useful subsequent question is whether a target-independent proposal can
offer such complete programs and whether task guidance ranks their outcomes
well enough. That requires a separate matched comparison. This diagnostic
cannot establish that either property holds, that more rings improve docking,
or that the resulting path is shortest under any declared cost metric.

# Contextual edit replay: new molecule, negative primary policy comparison

Durable raw receipts and executed source: volume `compose-v4-artifacts`, prefix
`pmo_edit_replay/5f576529f386971a4d4eda78dde602607e843559fc60f71b77e662385263fdae/`.
`receipts.tgz` has the prefix digest; `source.tgz` SHA-256 is
`391880bb74b3d39551394ab5d5a1c18d66bd2e5a2f687b50a90dceaa2503cb03`.
Both uploads completed. The source archive predates this location-only note.

The locked batch improved the prescreened perindopril development best from
0.6030226892 to **0.6495190528**, through uniform donor proposals. Contextual
successful-edit replay reached 0.6365315497. Replay did not pass the prespecified
criterion of exceeding both the incumbent and concurrent control.

| Outcome | Uniform | Contextual replay |
|---|---:|---:|
| Attempted programs | 64 | 64 |
| Compiled programs | 40 | 46 |
| Distinct completed molecules | 40 | 29 |
| Unique improving offspring | 3 | 9 |
| Best score | 0.649519 | 0.636532 |
| Archive top-ten mean | 0.597691 | 0.605168 |
| Proposal seconds | 20.358 | 26.104 |

Both arms share 16 exact starting states and the original 100-donor bank.
Replay uses 33 positive prior edits from 29 parents, with a 20% uniform donor
exploration component. All unique compiled endpoints were queried; there was
no learned endpoint filter or public winner input. The 17 duplicate completed
replay draws are included in its compute and attempt counts, not replaced.
Top-ten means include each arm's common initial archive and its own proposals.

The uniform best replaces 14 atoms with 22 through 39 executor-verified edits:
intended release 0.4516, largest changed region/original size 0.7097,
heavy-atom delta +8, cycle-rank delta +1, ring-system delta -1. Its final graph
has 39 heavy atoms, five independent cycles and two ring systems. The replay
best is a 52-edit replacement with intended release 0.7353, largest changed
region/original size 0.7059, heavy-atom delta -2, unchanged cycle rank and one
additional ring system. These fields distinguish restructuring from decoration;
ring-system count is not cycle rank. Exact SMILES and operands are in report.json.

Total: 63 new physical oracle calls in 50.413 seconds, including 0.988 seconds
of context features, 0.090 seconds fitting and 0.024 seconds of oracle execution.
Historical accounting retains 249455 prescreen calls and 669 earlier development
calls (732 development calls after this batch). No Modal worker, GPU, or neural
reference enumeration was used. This is not official PMO AUC, a matched external
comparison, or evidence that replay is the best complete controller.

Reproduce the read-only audit with the qualified local chemistry path:

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. \
  /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_edit_replay_report.py \
  --input /private/tmp/compose-pmo-edit-replay-20260911a \
  --output diagnostics/pmo_edit_replay/report.json
```

The audit verifies sealed receipts, configuration/training/candidate locks,
unchanged candidate membership, exact graph identities, the charged query union,
and both reported best and top-ten values. The original three focused replay
tests passed in 2.71 seconds; touched-code lint passed. No full suite was run,
and no milestone-completion claim is made. Raw execution records are kept
separately from the compact report. Source hashes and exact environment are
bound in the run configuration, including the uncommitted local source status.

Decision: retain the new exact states and paid trajectories. Stop this unchanged
replay recipe rather than promote its secondary top-ten gain into a primary
success. Next test whether updating the proposal donor population from actual
discoveries enables successive useful edits, against a frozen-memory control,
while retaining the full generic/reference proposal channel in both controllers.
This next policy is proposed, not yet implemented or running.

# Method map

COMPOSE has three distinct parts. The executor defines legal edits on complete
molecular graphs. The frozen model assigns probabilities to those edits. A task
controller constructs or selects edits and multi-edit programs under its own
constraints and feedback. Every step of a multi-edit program is executed as a
primitive edit, so its intermediate states obey the same molecular rules.

For a state `x`, the executor exposes the legal marks `A(x)`. The checkpoint
defines `p_theta(a | x, t)` over that set. The time or progress coordinate `t`
is explicit in code and suppressed in some manuscript notation. Several marks
can produce the same molecule. The canonical molecular kernel `R_theta(y | x, t)`
sums their mass, removes canonical self-events, and renormalizes over productive
successors. The [alias test](../tests/test_program_reference.py) checks this
against the complete production successor kernel.

The four task configurations pin the same NLL-trained checkpoint by SHA-256.
Its parameters are not retrained or fine-tuned for fragment generation, QED
editing, PMO or T4. Each task uses the same primitive rewrite semantics. The
checkpoint scores states within its declared molecular support. The controllers
differ in how they use its mark probabilities and induced canonical successor
kernel.

| Task | Where the fixed reference affects decisions | Task-specific control |
| --- | --- | --- |
| Fragment generation | Learned primitive edit preferences in superstructure sampling. The finite-panel selector uses native mark scores from the same checkpoint. | Protected structure, attachment rules, executable program construction, endpoint selection. |
| QED editing | Canonical successor probabilities at the configured time. | A separately fitted finite-horizon value head and sequential Monte Carlo. |
| PMO | Canonical successor scores of exact executed programs in one exploration allocation. | Structured proposal construction, online archive, value and query allocation, optional legal-chain and binding interventions. |
| T4 | Canonical successor scores of exact executed programs in non-floor selection slots. | Lead constraints, program template prior, score-blind exhaustion-tempered parent sampling, archive updates and docking evaluation. No program-value model is fitted. |

PMO and T4 construct programs from those same primitive edits. The executor
checks every intermediate state. Their controllers use the fixed reference to
weight supported programs in the selection allocations shown above. The
configured coverage policy handles programs that cannot be scored by the
checkpoint. T4's program template prior and QED's value head are separate
controller inputs. Neither changes or retrains `R_theta`.

The PMO and T4 example strengths are implementation settings, not operating
points selected by a completed benchmark comparison. Offline tests verify model
identity, executable traces, probability normalization, selection behavior,
resume guards and scoring boundaries. They do not reproduce the manuscript's
reported optimization or docking measurements. See the task guides in
[`experiments/`](../experiments/README.md) for assets and commands.

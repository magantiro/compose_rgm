# Development source snapshots for the support repair

Status: explicitly approved on 2026-09-11. After the launch safeguard rejected
the initial interpretation of the instruction to defer Git operations, the user
answered the scoped exception request: "approve pls u don't ahve to wait ofr me
in the future". Use content-hashed, explicitly uncommitted source snapshots for
bounded controller development without requesting the same operational exception
again. This does not authorize changes to scientific gates or new training lanes.
The exception changes no molecular support, model, numerical tolerances, oracle
allowance, search bounds or acceptance criteria. Concurrency is limited to the
user's separately authorized ceiling of 30 total Modal containers.

The initial `db85525914ba` deployment stopped in the driver import before any
molecular worker or oracle call: the analysis-only slot-permutation import
transitively required PyYAML. The replacement uses an exact array permutation
without importing that registry and is checked against the prior helper.

The next attempt records the base commit and a self-hashed inventory of every
serialized source/configuration byte, explicitly marked as an uncommitted
development snapshot. The deployed worker recomputes those hashes and rejects
any mismatch. No clean-commit assertion is forged. The P50/training launch gates
remain unchanged. Snapshot and output identities, not the base commit alone,
identify this run. Git operations can be batched after useful work is complete.

The failed invocation remains recorded at
`/private/tmp/compose-pmo-route-support-runs/b208c5be694e44630c935f2b4c5e883d4cea440ba6a9c8f8dc2505aff179bcff/spawn.json`,
call `fc-01M27GMWC9KJCKQVDNGD88T339`. It must not be reported as a chemistry failure.

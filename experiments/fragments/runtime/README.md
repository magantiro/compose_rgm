# Preserved fragment source runtimes

Each task directory contains a deterministic `source.zip` and `manifest.json`.
The archive holds unchanged Python source, not weights or an executable binary.
The manifest lists every file hash and the inspected source Git revision. Inspect
with `unzip -l source.zip` or extract into a new directory.

The fragment branches have a model sampling API and executor revision different
from the primary checkout. Overwriting shared core files would invalidate other
frozen contracts. The normal task interface is in
`src/compose_v4/experiments/fragments/`; its worker imports only the selected
snapshot's domain implementation. This preserves existing behavior while keeping
new orchestration separate from domain logic.

The capture tool follows project imports, including function-local imports and
package initializers. Explicit roots include constructor inputs passed by callers
rather than imported by the sampler. Capture reads tracked source and verifies
it against the selected Git revision. Execution needs no historical worktree.
Extra transitive source does not authorize other controllers or cloud jobs.

Maintainer example with an explicitly selected historical checkout:

```bash
python3 tools/capture_fragment_runtime.py --source /path/to/historical-checkout --task motif_extension --output /path/to/new-snapshot
```

Do not edit archive members to make parity pass or substitute a later development
branch. A future domain-code migration needs its own compatibility tests. Bounded
parity does not prove equality on every molecular state or full reproducibility.

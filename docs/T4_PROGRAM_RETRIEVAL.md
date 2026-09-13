# T4 context-ranked direct program retrieval

## Identity and evidence boundary

- **Scientific problem:** the fast controller can exhaust a useful neighborhood
  because every cold-start program draw is mutated before execution, even when a
  retrieved complete transformation is already strongly matched to the source.
- **Primary output:** eligible exact endpoints from a bounded direct-retrieval
  slice, with the same typed programs, contextual attachment bindings and exact
  executor as the existing cold start.
- **Claim under test:** allowing some high-probability context-matched programs to
  execute unchanged increases useful cold-start support without a material
  proposal-time penalty.
- **Setting:** all fifteen exact T4 source states at delta=0.4, using the shared
  146-program development library and no new task-oracle calls.
- **Control:** the frozen benchmark cold start, which mutates or recombines every
  retrieved construction.
- **Intervention:** target eight of sixteen candidate slots with direct retrieval;
  the remaining attempts use the unchanged mutation/recombination recipe.
- **Support:** unchanged 40-heavy-atom molecular support, 32 primitives, eight
  blocks, strict similarity greater than 0.4, QED greater than 0.6 and SA less
  than 4, with every intermediate checked by the exact executor.

Public IVG routes and winners are explicit development inputs to the shared
library. Exact recovery is therefore an answer-known diagnostic, not autonomous
discovery or a fresh docking result. The comparison reports public-winner overlap
separately from general eligible endpoint yield. External recorded scores are not
used as new COMPOSE oracle labels.

The live full-suite benchmark remains fixed at commit `c272b88`; this optional
channel cannot affect it. The default is zero direct-retrieval candidates, so old
recipes retain their behavior.

## Mechanism

`propose_programs` already constructs the finite context-conditioned
program-binding pool. The new optional output ranks that same pool by proposal
probability, breaks ties deterministically and initially exposes at most one
binding per program. This prevents many symmetric bindings for a small program
from consuming the whole bounded slice.

Cold start executes ranked programs without mutation until it admits the requested
number of eligible, canonically distinct endpoints or exhausts at most four ranked
attempts per requested candidate. All rejected, ineligible and duplicate attempts
remain explicit. The rest of the batch uses the existing random mutation and
branch-recombination proposal. No endpoint score, target fingerprint or docking
value enters the ranking.

## Frozen local comparison

For every source, generate one control batch and one retrieval batch using the
same library, source tensor, controller seed, endpoint gate, candidate cap and
attempt cap. Regenerate the retrieval batch and require an identical batch ID.
Persist complete batches and a summary with input hashes, source-code hashes,
software identity, candidate/channel counts, proposal seconds, canonical overlap
and public-winner reconstruction.

The structural gate is positive only if:

1. every retrieval batch remains nonempty;
2. retrieval recovers a public winner absent from its control in at least five of
   the fifteen cells; and
3. summed retrieval proposal time is at most 1.5 times the control.

The result is negative if there is no new public-winner recovery and no increase
in eligible support. Other outcomes are inconclusive. A positive result supports
including bounded direct retrieval in a separately frozen controller version. It
does not authorize docking or changing the active benchmark.

Run locally with:

```sh
PYTHONPATH=src:. .venv/bin/python tools/t4_program_retrieval_probe.py \
  --output diagnostics/t4_program_retrieval/attempt_1
```

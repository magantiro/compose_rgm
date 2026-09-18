# Anchored-replacement cross-source support

The frozen generic anchored-replacement proposal transferred to JAK2 seed 0 but
not JAK2 seed 2 under the preregistered zero-oracle gate.

At an equal 512-program budget per lane, seed 0 produced 218 unique eligible
anchored-replacement endpoints and 165 shallow endpoints. Every anchored worker
returned at least one eligible endpoint. Seed 2 produced 0 unique eligible
anchored-replacement endpoints and 17 shallow endpoints. No worker failed, so the
seed-2 result is a proposal-support failure rather than an execution failure.

The engineering gate passed. The cross-source transfer-support and full-panel-
yield gates failed because the frozen proposal expert had zero eligible support
on seed 2. No oracle or docking call occurred. The result does not measure
docking utility, FiberControl, or route-distilled proposal quality.

The negative result limits the claim from the seed-1 pilot: anchored replacement
is a productive generic structural option with demonstrated support on a second
JAK2 source, but it is not a universal JAK2 proposal law. A mixed controller must
retain an independent local/exploratory lane and treat empty expert output as an
abstention rather than spending or reserving oracle budget for that expert.

Authoritative artifacts:

- `result.json`: sealed aggregate, configuration, hashes, and candidate sets.
- `result.jsonl.gz`: all 256 independently seeded worker outputs.
- Frozen code revision: `1a7a4efbe007c5ed04ae6c5c4ef6a94e3fa77885`.
- New oracle calls: `0`.

# Experiments

**[`paper/`](paper/) is the front door: what the current ICLR submission
contains, which module implements each part, and what you need to run it.**

Three other kinds of file live here, and they answer different questions.

- **`<task>/README.md` + `<task>/manifest.json`** — *how do I verify this
  benchmark, and how do I run it?* One per benchmark task
  ([`t4`](t4/), [`pmo`](pmo/)): the objective, a reproduction path that needs no
  cloud credentials and makes no oracle calls, the exact input manifest with
  hashes, the pinned environment, documented missing assets, and the
  limitations. Verify a manifest with
  `python3 tools/verify_experiment_inputs.py`.

- **[`INDEX.md`](INDEX.md)** — *what can I run?* Every app and entrypoint in the
  tree with its exact command, generated from source by
  `tools/gen_experiment_index.py`. Regenerate it after adding an entrypoint;
  never hand-edit it, or it will start describing experiments that do not exist.
- **`<campaign>/README.md`** — *what did we ask, and what came back?* Written by
  hand, one per campaign: the question, the commands in order, the gates fixed
  **before** results were seen, the inputs, where artifacts land, what was
  established, and what the result does **not** support.

## Writing a campaign manifest

Record the gate before the run, not after. A gate written afterwards is a
description of the result, not a test of it. State limitations in the same
document as the findings — the two that mattered most in the region-resampling
campaign were both limitations of the *question*, not the code:

- "reachable" means *completable within B primitive edits*, not "rewriteable";
- a probe that varies depth at fixed beam says nothing about breadth.

If a run selects its cases conditional on something (for instance, regions
chosen *because* a rewrite is known to exist), say so in the manifest and in the
entrypoint's own output, so a number can never be read as coverage when it is
conditional efficiency.

## Campaigns

| Campaign | Status | What it settles |
|---|---|---|
| [`region_resampling`](region_resampling/) | splitting phase closed | Directed proposals reach structural handoff where the base kernel does not; budget is not the limiter at fixed breadth; the open item is the general local→global gate. |

## Benchmark tasks

| Task | Result status | Reproduce with |
|---|---|---|
| [`paper`](paper/) | **the current ICLR submission** | paper section/table -> module, plus required external assets |
| [`t4`](t4/) | T4 task guide | protocol, pinned environment, reading hazards |
| [`pmo`](pmo/) | PMO task guide | protocol, information boundary, reading hazards |

A task guide is not a campaign manifest: it describes a standing benchmark and
its reproduction path, not one question asked once. Campaigns above may target
either task.

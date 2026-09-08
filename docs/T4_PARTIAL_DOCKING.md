# Saved-prefix T4 docking diagnostic

User authorization, 2026-09-08: dock the saved eligible, previously undocked
batch of at most 13 molecules, and parallelize across containers if faster.
This explicitly amends the interrupted warm-round protocol for one diagnostic.
It does not declare the eight-parent optimizer round complete or authorize
another proposal, retry, training run, or automatic next round.

COMPOSE's platform identity and bounded molecular support remain those in
AGENTS.md. This inspected PARP1 seed0 d=0.4 development diagnostic asks whether
the saved feasible proposals, including two exact-witness fused C6 products,
improve the docking score. Compare with the original 20-call guided archive.
No matched reference is run, and unseeded docking limits attribution. This is
not held-out evaluation, synthesis validation, or evidence of guidance benefit.

## Frozen source, selection, and accounting

`configs/t4_partial_docking.json` binds seven completed parent units and the
full exact-state warm archive from source revision
`34aa672eee4cb5d60bfa92d4c5386bd455617c73`. Its self-hash is
`3a56fe201c5498ea44961d8580c019af30fe87da65006a1908194fcabbd32166`.
The source volume namespace was inventoried before launch; no prior partial
docking namespace or result existed. No missing parent is reconstructed and
no completed molecular work is repeated. Failed source preparation remains a
failure at 20,000 executor calls.

The approved 13 canonical identities are frozen in the contract before any
new docking outcomes. They come from all 13 eligible unevaluated bundles in
the saved prefix, not a ring-only or winner-similarity selection. One additional
locally feasible molecule was already evaluated and is excluded. Source
states are recovered from saved sampled transitions, not rebuilt from SMILES.

Recompute QED, SA, and Morgan radius-2/2048-bit similarity against the original
seed in the pinned RDKit 2024.03.5 runtime. Use the same shared production
property calculation and thresholds: QED >= 0.6, SA <= 4, similarity >= 0.4.
Newly ineligible approved molecules are omitted with reasons; no backfill or
expansion to other source candidates. Keep all 25 emitted source records in
the lock. Freeze the full batch and exact states before invoking any worker.

Each candidate is attempted once, including failures. Every attempted call
counts against the prior 40-new-call allowance. Thirteen attempts would bring
the guided total to 33 and leave 27 unused calls, not authorize their use.
Do not resample parents or update a full optimizer archive from this partial
batch. Complete results and interrupted attempts are reusable receipts, not
permission to repeat docking.

## Execution and cost boundary

One deployed driver: 1 CPU, 2 GiB, one-hour timeout. At most four independent
worker containers: 1 CPU, 2 GiB each, 600-second timeout, zero retries. Each
uses the unchanged `_dock` evaluator with QuickVina cpu=1, exhaustiveness=1,
10 modes, the same PARP1 receptor/box, and the inherited unseeded protocol.
Input receptor and executable hashes and serialized source revision are checked.
This path does not initialize R_theta or execute molecular rewrites.

The previous guided batch took 47.439 seconds for 20 serial dockings. That is
about 2.37 seconds per candidate, not a guaranteed duration for these larger
products. Expected docking work is under a minute with four workers; allow a
few minutes for deployment, image startup and validation. The driver and
worker timeouts are hard limits, not runtime forecasts.

At published Modal base rates checked 2026-09-08, 1 CPU plus 2 GiB costs
0.00001754 USD per container-second. A planning allowance of 0.01-0.03 USD
for task compute covers several minutes of startup and evaluation; it is not
an invoice. The configured driver plus all 13 worker task-time limits total
11,400 container-seconds, about 0.200 USD at those requested resources.
This is not a billing cap and excludes builds, storage, overhead, usage above
requests, account charges and any applicable pricing multipliers.
[Modal pricing](https://modal.com/pricing).

The driver emits 30-second heartbeats and durable per-result progress. Worker
started/result receipts bind indices and the candidate-lock hash. Out-of-order
completion never changes candidate selection. A started receipt without a
complete result blocks automatic retry. No other project jobs are modified.

## Launch and verification

Use a clean committed worktree, excluding unrelated in-progress scaffold work:

```sh
python3 tools/preflight.py --strict
modal deploy modal_apps/genmol_t4_opt_app.py
python3 tools/t4_launch.py --partial-docking
```

Focused checks cover exact-state reuse, source hashes, approved membership,
eligibility, canonical duplicates, before-oracle locking, out-of-order identity,
all attempted-call accounting, completed-result reuse, interrupted-row/batch
retry prevention, and the four-container resource boundary. They use saved
development receipts and synthetic oracle responses, never network calls.
The existing broad-suite non-green result is not re-run for this bounded task.

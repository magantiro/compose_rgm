# T4 — constrained lead optimization

**Status: table FROZEN (`diagnostics/T4_FROZEN_RESULT_v1.json`, locked 2026-09-22).**

The question: given a lead molecule, a similarity ball `δ`, and QED/SA
constraints, can a validity-closed rewrite process find a better-docking
eligible molecule than the comparator within a fixed oracle budget?

    objective   QuickVina2 docking of the best eligible endpoint, lower is better
    subject to  Tanimoto ≥ δ  AND  QED ≥ 0.6  AND  SA ≤ 4  AND  heavy ≤ 40
    budget      COMPOSE 250 charged docking calls per cell vs InVirtuoGen 1000
    panel       5 targets × 3 seeds × δ ∈ {0.4, 0.6}

---

## 1. Reproduce the reported table — no credentials, no docking

```
python3 tools/reproduce_t4_table.py
```

Bare `python3`; no RDKit, no torch, no Modal, no oracle. It rebuilds the table
from the sealed artifact and runs three checks, each able to fail on its own:

| Check | What it catches |
|---|---|
| seal | `sha256(canonical(payload)) == payload_sha256` — any edit after freezing |
| arithmetic | every stored aggregate recomputed **from the rows** |
| row integrity | `gap == compose - ivg`, no duplicate cells |

Exit codes: `0` clean, `1` internally inconsistent, `2` seal broken or artifact
missing.

**It currently exits 1, and that is correct.** See §4.

## 2. What the table says

    δ = 0.4   COMPOSE -169.4  IVG -160.4   sum gap -9.0   wins 10/15   coverage 15/15
    δ = 0.6   COMPOSE -153.9  IVG -145.4   sum gap -8.5   wins 13/14   coverage 14/15

Negative gap favours COMPOSE. One blank cell (FA7 seed 1, δ 0.6) is a
**deliberate scoped negative**, not a missing measurement: the cell is reachable
— all five stored witnesses pass the production `Fiber.check` — but was reached
only once across 3 seeds × 480 draws, so docking calls were declined.

**Rows are not interchangeable.** Each carries `source`:

- `panel` — the historical per-cell procedure that produced the submitted table
- `support_expansion` — a later campaign

The reproducer counts and labels them separately (δ 0.4: 13 panel / 2 expansion;
δ 0.6: 11 panel / 4 expansion). Do not merge them in prose.

## 3. Verify the inputs

```
python3 tools/verify_experiment_inputs.py --task t4
```

`strict` pins must match; `informational` pins are live code and drift is
reported, not failed. Assets the manifest declares missing — the docking binary,
the per-cell Modal volumes — are listed as documented gaps and are **not** needed
for §1.

## 4. Known discrepancy, deliberately not fixed

At δ 0.6 the artifact stores `sum_gap = -7.6`, but the rows sum to **-8.5**, and
two independent routes agree with the rows:

    compose_sum - ivg_sum   = -153.9 - (-145.4) = -8.5
    mean_gap * paired_cells = -0.6071 * 14      = -8.4994

So the stored field is wrong and it **understates** COMPOSE's margin by 0.9
kcal/mol. The artifact is LOCKED and was **not** edited: the reproducer prints
the row sum, flags the stored field, and exits 1. Quote -8.5, with this note.

## 5. Running new cells

Not covered by §1 and not runnable from a clone: it needs Modal, the pinned
image, and a signed contract. The per-cell launchers are 9-line env-var wrappers
over one shared implementation:

```python
os.environ.setdefault("COMPOSE_HELD_CONTRACT", "configs/t4_held_target_distilled_5ht1b_d06_250.json")
...
from modal_apps.t4_integrated_route_fiber_parp1_app import app, main
```

Discipline that is not optional, each of which has already cost a run here:

- **Read `delta` from the contract, never from the app or volume name.** One
  contract shipped `delta: 0.4` under a name and `claim_boundary` saying 0.6.
  Diff the executable field against the prose every time.
- **`--detach` is mandatory** (`main()` uses `.spawn()`).
- Never resume an arm while its original driver is alive — two `run_cell`
  containers writing one cell folder is the checkpoint-overwrite race.
- A round lock is authority; a checkpoint is overwritable.

## 6. Environment

Pinned production kernel — **python 3.11, rdkit 2024.3.5, numpy 1.26.4, scipy
1.13.1, networkx 3.3, torch 2.4.0**.

```
uv venv --python 3.11 <dir>
uv pip install --python <dir>/bin/python rdkit==2024.3.5 numpy==1.26.4 \
    scipy==1.13.1 networkx==3.3 torch==2.4.0 pytest
```

torch is needed only because `t4_fiber_campaign` imports it transitively — but it
**is** needed, because without it you must transcribe the eligibility gate rather
than import it, and a transcribed gate cannot fail usefully.

**This is not the PMO kernel.** PMO runs rdkit 2023.9.6
(`experiments/pmo/README.md`). Two production kernels exist in this repo, and a
number computed under the wrong one needs a parity statement.

## 7. Limitations

- **Docking is not reproducible across runs.** `qvina02` is seeded and the box is
  fixed, but `obabel --gen3D` takes no seed. One molecule scored **-7.5, -8.30,
  -8.8** across three runs — a 1.3 kcal/mol spread, wider than many per-row
  margins here. Quote the aggregate and the win count; do not defend a single
  cell's 0.2-0.7 gap. Within-run search is unaffected, since candidates and
  incumbent share one pipeline.
- **Best-of-N is a biased statistic and its bias grows with N.** On the
  comparator, the same fixed cell set moved 10.4 kcal/mol purely by changing the
  summary statistic (best-of-3 → best-of-8 → mean-of-8). Any cross-system verdict
  must fix the statistic and N on both sides.
- Benchmark eligibility is a floor, not a quality score: endpoints can pass all
  four constraints and still be exocyclic quinoids or strained azirines. Report
  eligibility and chemical plausibility as two numbers.
- The frozen table is a *historical method replication*. The unified adaptive
  controller is a **separate** experiment and must not be substituted into these
  rows — the artifact says so itself under `two_experiments_do_not_conflate`.

# Learnings

Durable, dated gotchas + design calls. Append; don't rewrite history.

## 2026-07-21
- **macOS fork deadlock:** the multiprocessing rollout pool (`tracelet_sampling_worker`) deadlocks
  at 0% CPU on this Mac at `workers>=2`. Use `workers=1` (serial) locally; Modal for scale. Killing
  the parent orphans the workers — `pkill -9 -f tracelet_sampling_worker` too, or they leak CPU/RAM.
- **Base corpus = GuacaMol, not ZINC.** `train_tracelet_gm.py` loads `guacamol_subset_500000_…smiles`;
  the earlier handoff's "ZINC-250k" claim was wrong. Constrained-design *leads* are ZINC-derived
  (Jin QED set). "Matches GrIDDD" holds at the element level (CNOF), not the corpus.
- **Base = Lineage B is a step-1000 preview** and overproduces small rings (aziridine/epoxide) — this
  inflates the pathwise-safety *magnitude* (not the guarantee/free-cost). See lineage-correction V2.
- **Conditional results are mechanism-driven** (fiber + SMC), so base-independent; they carry to any
  backbone. Only E1 (unconditional quality) and pathwise-safety magnitude depend on the base.
- **Framing (Paper 1):** pathwise-constrained generation is the spine/spotlight; structural control +
  oracle efficiency + anytime are supporting; exactness (E0 + guidance-ground-truth) is the rigor
  foundation. QED optimization is NOT a GrIDDD beat (oracle-hungry); FCD is deferred, not a headline.
- **Fairness:** never compare our *selected* top-k to a baseline's *all-sample* rate — use per-sample.

## 2026-07-22
- **Full-suite segfault = dual OpenMP, not a code bug.** `pytest tests/` segfaults (exit 139) at
  ~40% on this Mac: torch 2.11 (`~/Library/Python/3.14`, bundles its own libomp) + rdkit/numpy/
  scipy/sklearn (`/opt/homebrew`, brew's libomp) load two OpenMP runtimes in one process on
  Python 3.14.2; the `.venv` pip-manages none of them (only `_virtualenv.pth`). `KMP_DUPLICATE_LIB_OK=TRUE`
  silences the abort but not the corruption — after enough thread-pool churn it faults (tips at the
  sklearn-based `test_pan_lung_filtering` lipid test, whichever test crosses the threshold). Every
  test passes in isolation. **Fix (env, not code): `OMP_NUM_THREADS=1`** → segfault gone. The one
  remaining fail (`test_parallel_tracelet_sampling`) is a subprocess `PYTHONPATH` gap — the worker
  spawns `python -m compose_v4...` which needs `src` on the path; **`PYTHONPATH=src`** fixes it.
  Green invocation: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src pytest tests/` → 427/427.
  Real root cause is that `.venv` isn't a clean single-source install (`uv sync`/`pip install -e .`
  would put one OpenMP + compose_v4 in the venv); don't reinstall reactively — document + move on.

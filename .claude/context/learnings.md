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

"""WE must reproduce the exact weighted terminal distribution. Toy chain.

WE is statistically exact for a broad class of processes and binning rules
provided resampling preserves per-stratum weight. The theorem is not in doubt;
the IMPLEMENTATION is. So this runs a small enumerable Markov chain where the
exact terminal distribution is computable in closed form and compares three
estimators on it:

    exact          matrix power
    plain          weighted particles, no resampling at all
    WE             the real we_resample, with adaptive rank strata

If the WE resampler leaked or renormalised weight -- for instance by resetting
descendants to 1/N, which is invisible when there is only one stratum -- the WE
column would drift from exact while `plain` stayed correct. That is precisely
the bug this is built to catch, and it cannot be caught on molecules where the
truth is unknown.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from compose_v4.experiments.hphi_we_resample import we_resample  # noqa: E402

S, H, N = 6, 8, 64          # states, steps, particles
rng0 = np.random.default_rng(5)
P = rng0.random((S, S)) + 0.05
P /= P.sum(axis=1, keepdims=True)

exact = np.zeros(S); exact[0] = 1.0
for _ in range(H):
    exact = exact @ P

def run(use_we: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(N, dtype=int)
    w = np.full(N, 1.0 / N)
    for t in range(H):
        for i in range(N):
            x[i] = rng.choice(S, p=P[x[i]])
        if use_we and t % 2 == 1:          # resample on a fixed schedule
            # score is an ARBITRARY progress coordinate: WE must stay exact for
            # any binning, so a deliberately uninformative one is the right test
            parents, nw, _ = we_resample(
                [int(v) for v in x], w, lambda k: (k * 7) % S, rng, n_slots=N)
            x = np.array([x[p] for p in parents], dtype=int)
            w = nw
    out = np.zeros(S)
    for i in range(N):
        out[x[i]] += w[i]
    return out / out.sum()

R = 400
plain = np.mean([run(False, 1000 + r) for r in range(R)], axis=0)
we = np.mean([run(True, 5000 + r) for r in range(R)], axis=0)
print("state:      " + "".join(f"{i:>9}" for i in range(S)))
print("exact:      " + "".join(f"{v:9.4f}" for v in exact))
print("plain:      " + "".join(f"{v:9.4f}" for v in plain))
print("WE:         " + "".join(f"{v:9.4f}" for v in we))
tv_p = 0.5 * np.abs(plain - exact).sum()
tv_w = 0.5 * np.abs(we - exact).sum()
print(f"\ntotal variation vs exact   plain {tv_p:.5f}   WE {tv_w:.5f}")
ok = tv_w < 3 * max(tv_p, 0.002)
print(f"WE EXACTNESS: {'PASSED' if ok else 'FAILED'}  "
      f"(WE must not be materially worse than no-resampling)")
w_ok = True
for r in range(30):
    rng = np.random.default_rng(90000 + r)
    w = rng.random(N) + 1e-3
    parents, nw, _ = we_resample(list(rng.integers(0, S, N)), w,
                                 lambda k: float(k), rng, n_slots=N)
    if abs(nw.sum() - w.sum()) > 1e-9 * max(1.0, w.sum()):
        w_ok = False
print(f"WEIGHT PRESERVATION: {'PASSED' if w_ok else 'FAILED'} "
      f"(sum of descendant weights == sum of incoming, 30 random draws)")
raise SystemExit(0 if (ok and w_ok) else 1)

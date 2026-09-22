"""Power for the predeclared paired McNemar test on macro-option reach rates.

Run BEFORE any measurement; its output is the power table sealed into
diagnostics/pmo_macro_option_v1/falsifier_predeclaration_v1.json.
"""

import numpy as np
from scipy.stats import binom

def mcnemar_one_sided(b, c):
    n = b + c
    return 1.0 if n == 0 else float(binom.sf(b - 1, n, 0.5))

def power(n, p0, true_effect, noise, trials=40000, seed=11, alpha=0.05, min_diff=0.25):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(trials):
        both = rng.binomial(n, p0)
        prot_only = rng.binomial(n - both, true_effect / max(1e-9, 1 - p0))
        lost = rng.binomial(both, noise)           # protection loses some greedy wins
        b, c = prot_only, lost
        r1 = (both - lost + prot_only) / n
        r0 = both / n
        if (r1 - r0) >= min_diff and mcnemar_one_sided(b, c) < alpha:
            ok += 1
    return ok / trials

print("threshold min_diff=0.25, alpha=0.05 one-sided exact McNemar, noise=0.05")
for n in (40, 60, 80, 100, 120):
    row = [f"n={n:4d}"]
    for p0, eff in ((0.40, 0.35), (0.40, 0.40), (0.30, 0.45), (0.40, 0.30)):
        row.append(f"p0={p0},eff={eff}: {power(n, p0, eff, 0.05):.3f}")
    print("  ".join(row))

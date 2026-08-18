"""Does threading `budget_max` change ANYTHING at budget_max = 24?

The H40 adaptation touches the feature builder, the budget clamp, the head
width and the two build_features call sites. Every one of those is on the path
that produced the FROZEN H24 head, so the adaptation is only safe if the
default reproduces the old behaviour exactly -- not approximately.

This replays the assemble() inner loop twice on identical inputs: once through
the OLD path (module constants BUDGET_MAX / INPUT_DIM, build_features called
without a budget_max argument) and once through the NEW path (explicit
budget_max=24). It compares the resulting matrices BITWISE, not by tolerance,
because a plumbing change has no licence to move a single bit.

No Modal, no volume, no cost. Embeddings are synthetic: the check is over the
feature/label assembly, which is what the edit touched, and synthetic vectors
exercise it exactly as real ones do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_v4.experiments.hphi_region_features import (  # noqa: E402
    BUDGET_MAX, EMBED_DIM, INPUT_DIM, build_features, in_region, input_dim,
)
from compose_v4.experiments.hphi_rollout import registered_regions  # noqa: E402


def assemble_loop(paths, emb, e_src, regions, budget_max, *, new_path):
    """The assemble() inner loop. `new_path` selects threaded vs module-constant."""
    X, Y, B, M = [], [], [], []
    clamp = budget_max if new_path else BUDGET_MAX
    for path, qed, sim in paths:
        H = len(path) - 1
        for i, s in enumerate(path):
            b = H - i
            if b > clamp:
                continue
            for reg in regions:
                if in_region(qed[i], sim[i], reg):
                    continue
                hit = 0
                for k in range(i, len(path)):
                    if (k - i) <= b and qed[k] >= reg[0] and sim[k] >= reg[1]:
                        hit = 1
                        break
                if new_path:
                    f = build_features(np.asarray(emb[s]), e_src,
                                       qed[i], sim[i], reg, b, budget_max)
                else:
                    f = build_features(np.asarray(emb[s]), e_src,
                                       qed[i], sim[i], reg, b)
                X.append(f); Y.append(hit); M.append((reg, b))
                if i + 1 < len(path) and b >= 1 and not in_region(
                        qed[i + 1], sim[i + 1], reg):
                    if new_path:
                        B.append(build_features(
                            np.asarray(emb[path[i + 1]]), e_src,
                            qed[i + 1], sim[i + 1], reg, b - 1, budget_max))
                    else:
                        B.append(build_features(
                            np.asarray(emb[path[i + 1]]), e_src,
                            qed[i + 1], sim[i + 1], reg, b - 1))
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32),
            np.asarray(B, dtype=np.float32), M)


def main() -> int:
    rng = np.random.default_rng(0)
    regions = registered_regions()
    print(f"{len(regions)} registered regions")

    # Fixed synthetic trajectories, lengths spanning the H24 clamp on BOTH
    # sides so the `b > budget_max` branch is genuinely exercised.
    paths = []
    emb: dict[str, list[float]] = {}
    for t, L in enumerate((25, 41, 12)):
        p = [f"S{t}_{i}" for i in range(L)]
        for s in p:
            emb[s] = rng.normal(size=EMBED_DIM).tolist()
        paths.append((p, list(rng.uniform(0.2, 0.95, L)),
                      list(rng.uniform(0.1, 0.9, L))))
    e_src = rng.normal(size=EMBED_DIM)

    fails = []

    # ---- 1. the constants themselves -----------------------------------
    if input_dim(24) != INPUT_DIM or INPUT_DIM != 4 * EMBED_DIM + 6 + 25:
        fails.append(f"input_dim(24)={input_dim(24)} != INPUT_DIM={INPUT_DIM}")
    print(f"input_dim(24) = {input_dim(24)}   INPUT_DIM = {INPUT_DIM}   "
          f"input_dim(40) = {input_dim(40)}")

    # ---- 2. the assemble loop, old vs new -------------------------------
    Xo, Yo, Bo, Mo = assemble_loop(paths, emb, e_src, regions, 24, new_path=False)
    Xn, Yn, Bn, Mn = assemble_loop(paths, emb, e_src, regions, 24, new_path=True)
    print(f"old: X{Xo.shape} B{Bo.shape}   new: X{Xn.shape} B{Bn.shape}")
    for name, a, b in (("Xtr", Xo, Xn), ("Ytr", Yo, Yn), ("Btr", Bo, Bn)):
        if a.shape != b.shape:
            fails.append(f"{name} shape {a.shape} != {b.shape}")
        elif not np.array_equal(a, b):
            n = int((a != b).sum())
            fails.append(f"{name} differs in {n} of {a.size} entries")
        else:
            print(f"  {name}: BITWISE IDENTICAL  {a.shape}")
    if Mo != Mn:
        fails.append("Mva differs")
    else:
        print(f"  Mva: identical  ({len(Mo)} rows)")

    # ---- 3. the H40 path is genuinely wider, and clamps at 40 -----------
    X40, _y, _b, _m = assemble_loop(paths, emb, e_src, regions, 40, new_path=True)
    if X40.shape[1] != input_dim(40):
        fails.append(f"H40 width {X40.shape[1]} != {input_dim(40)}")
    if X40.shape[0] <= Xn.shape[0]:
        fails.append("H40 assembled no extra rows; the clamp did not move")
    print(f"  H40: width {X40.shape[1]}, rows {X40.shape[0]:,} "
          f"vs H24 {Xn.shape[0]:,}")

    # ---- 4. a budget the H24 head cannot express still RAISES -----------
    try:
        build_features(np.asarray(emb[paths[0][0][0]]), e_src, 0.5, 0.5,
                       regions[0], 25)
        fails.append("build_features accepted budget 25 at budget_max=24")
    except ValueError:
        print("  budget 25 at budget_max=24: correctly REJECTED")

    print()
    if fails:
        print("REGRESSION FAILED")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS: budget_max=24 reproduces the frozen H24 assembly bitwise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

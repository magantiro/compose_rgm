"""Online surrogate-guided oracle allocation for PMO.

COMPOSE proposes candidate endpoints cheaply; the PMO oracle charges per call and
the benchmark budget is the scarce resource.  This module fits a lightweight
predictor ``f_hat`` ONLINE -- trained only on molecules the current run has
already purchased -- ranks a pool of free candidates, and spends the next batch
of oracle calls on the ranked head under an acquisition that includes calibrated
uncertainty.

Data structure
--------------
``ArchiveRow``      one purchased evaluation: (smiles, score, provenance).
``OnlineArchive``   the run's purchased set; the ONLY training source permitted.
``Surrogate``       fit(X, y) -> predict(X) -> (mean, sigma).  Closed form; no
                    pretrained weights, no external corpus, no oracle internals.
``SurrogateAllocator``  archive + surrogate + acquisition -> ranked batch.

Invariants maintained here and asserted in tests
------------------------------------------------
1. NO-PRESCREEN.  ``OnlineArchive.observe`` is the only way a (molecule, score)
   pair enters training, and every row carries a provenance tag naming the run
   that paid for it.  ``training_provenance()`` reports that set verbatim so an
   artifact can state where every training row came from.  A surrogate is never
   constructed with data and cannot be warm-started.
2. EXPLORATION FLOOR.  ``select_batch`` reserves ``explore_fraction`` of every
   batch for acquisition-independent picks, so a confident-but-wrong surrogate
   cannot starve exploration.  The floor is >= 1 call whenever k >= 2 and
   explore_fraction > 0.
3. COLD START IS EXPLICIT.  Below ``min_fit_rows`` the allocator reports
   ``surrogate_active=False`` and falls back to the exploration policy rather
   than ranking on an uninformed predictor.
4. SELECTION IS PURE.  ``select_batch`` never calls an oracle.  It returns
   indices into the candidate pool; charging is the caller's responsibility.

Features are Morgan (ECFP) bits ONLY, by default.  Physicochemical descriptors
are available but OFF by default because a descriptor correlated with the
objective (QED, logP) would leak the objective into the features for the PMO
tasks built on that same descriptor.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import cho_factor, cho_solve

RDLogger.DisableLog("rdApp.*")

# ---- Featurization ----

DEFAULT_FP_BITS = 2048
DEFAULT_FP_RADIUS = 2

_GENERATOR_CACHE: dict[tuple[int, int], object] = {}


def _morgan_generator(radius: int, n_bits: int):
    key = (radius, n_bits)
    generator = _GENERATOR_CACHE.get(key)
    if generator is None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        _GENERATOR_CACHE[key] = generator
    return generator


class MorganFeaturizer:
    """Binary ECFP bits.  Structural only: cannot encode an objective value."""

    def __init__(self, *, radius: int = DEFAULT_FP_RADIUS, n_bits: int = DEFAULT_FP_BITS) -> None:
        if radius < 0 or n_bits < 8:
            raise ValueError("radius >= 0 and n_bits >= 8 required")
        self.radius = radius
        self.n_bits = n_bits
        self._cache: dict[str, np.ndarray | None] = {}

    @property
    def identity(self) -> str:
        return f"morgan_r{self.radius}_b{self.n_bits}"

    def featurize_one(self, smiles: str) -> np.ndarray | None:
        cached = self._cache.get(smiles)
        if cached is not None or smiles in self._cache:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self._cache[smiles] = None
            return None
        fp = _morgan_generator(self.radius, self.n_bits).GetFingerprintAsNumPy(mol)
        vec = np.asarray(fp, dtype=np.float64)
        self._cache[smiles] = vec
        return vec

    def featurize(self, smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
        """Return (X, kept_indices).  Unparseable SMILES are dropped, not faked."""
        rows: list[np.ndarray] = []
        kept: list[int] = []
        for i, smiles in enumerate(smiles_list):
            vec = self.featurize_one(smiles)
            if vec is not None:
                rows.append(vec)
                kept.append(i)
        if not rows:
            return np.zeros((0, self.n_bits), dtype=np.float64), []
        return np.vstack(rows), kept


# ---- Kernels ----


def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between binary rows of A and B.

    T(a, b) = <a, b> / (|a| + |b| - <a, b>).  Defined as 1.0 for two all-zero
    vectors (identical objects), which keeps the kernel PSD-consistent.
    """
    inner = A @ B.T
    a_sum = A.sum(axis=1)[:, None]
    b_sum = B.sum(axis=1)[None, :]
    denom = a_sum + b_sum - inner
    out = np.divide(inner, denom, out=np.ones_like(inner, dtype=np.float64), where=denom > 0)
    return out


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine (row-normalised dot product), so the kernel diagonal is 1.0.

    Dividing the raw dot product by the bit count instead puts the signal far
    below the noise floor -- a fingerprint sets ~30 of 2048 bits, so
    <x, x> / 2048 ~ 0.015 against a noise term of 0.05 -- and the arm returns
    the prior mean for every candidate.  That was MEASURED here, not assumed,
    and would have made this a strawman comparator.
    """
    an = np.linalg.norm(A, axis=1, keepdims=True)
    bn = np.linalg.norm(B, axis=1, keepdims=True)
    an = np.where(an > 0, an, 1.0)
    bn = np.where(bn > 0, bn, 1.0)
    return (A / an) @ (B / bn).T


_KERNELS = {"tanimoto": tanimoto_kernel, "linear": linear_kernel}


# ---- Surrogates ----


@dataclass
class Prediction:
    mean: np.ndarray
    sigma: np.ndarray


class Surrogate:
    """Interface: fit on purchased rows only, predict mean and sigma."""

    name = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Prediction:
        raise NotImplementedError


class MeanSurrogate(Surrogate):
    """Null control: archive mean and archive spread, ignoring structure.

    Any structural model that cannot beat this is not reading the molecule.
    """

    name = "mean"

    def __init__(self) -> None:
        self._mu = 0.0
        self._sd = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanSurrogate:
        self._mu = float(np.mean(y)) if len(y) else 0.0
        self._sd = float(np.std(y)) if len(y) > 1 else 1.0
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        n = X.shape[0]
        return Prediction(np.full(n, self._mu), np.full(n, max(self._sd, 1e-9)))


class GaussianProcessSurrogate(Surrogate):
    """Exact GP regression.  ``kernel='tanimoto'`` or ``'linear'``.

    The linear-kernel case IS Bayesian ridge regression on the fingerprint bits,
    computed in the n-dimensional dual space, which is the cheap direction when
    n (purchased rows) is far below the 2048 features -- exactly the PMO regime.

    Amplitude is set to the archive variance and the noise floor is a fixed
    fraction of it, so the model is scale-free and needs no held-out tuning at
    n = 16.  This is a deliberate choice for cold start: hyperparameter search on
    16 points overfits harder than the fixed prior does.
    """

    def __init__(self, *, kernel: str = "tanimoto", noise: float = 0.05, jitter: float = 1e-8):
        if kernel not in _KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        if not (0.0 < noise < 1.0):
            raise ValueError("noise must be a fraction of signal variance in (0, 1)")
        self.kernel = kernel
        self.noise = noise
        self.jitter = jitter
        self.name = f"gp_{kernel}"
        self._X: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._chol = None
        self._mu = 0.0
        self._amp = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessSurrogate:
        if X.shape[0] != len(y):
            raise ValueError("X and y disagree on row count")
        self._X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._mu = float(np.mean(y))
        centered = y - self._mu
        self._amp = float(np.var(y))
        if self._amp <= 0.0:
            # Degenerate archive (all scores identical): fall back to a pure
            # prior so sigma stays meaningful instead of collapsing to zero.
            self._amp = 1e-6
        K = _KERNELS[self.kernel](self._X, self._X) * self._amp
        K[np.diag_indices_from(K)] += self._amp * self.noise + self.jitter
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, centered)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        if self._X is None or self._alpha is None:
            raise RuntimeError("predict before fit")
        Ks = _KERNELS[self.kernel](np.asarray(X, dtype=np.float64), self._X) * self._amp
        mean = Ks @ self._alpha + self._mu
        v = cho_solve(self._chol, Ks.T)
        prior = self._amp * np.ones(X.shape[0])
        var = prior - np.einsum("ij,ji->i", Ks, v)
        var = np.clip(var, 1e-12, None)
        return Prediction(mean, np.sqrt(var))


class KNNTanimotoSurrogate(Surrogate):
    """Similarity-weighted k-nearest-neighbour prediction.

    The classical chemical-series baseline.  sigma is the similarity-weighted
    spread of the neighbours, inflated when the nearest neighbour is far, so an
    off-manifold candidate is reported as uncertain rather than as the mean.
    """

    name = "knn_tanimoto"

    def __init__(self, *, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k >= 1 required")
        self.k = k
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNTanimotoSurrogate:
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        if self._X is None or self._y is None:
            raise RuntimeError("predict before fit")
        S = tanimoto_kernel(np.asarray(X, dtype=np.float64), self._X)
        k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(S.shape[0])[:, None]
        sims = S[rows, idx]
        vals = self._y[idx]
        w = sims + 1e-9
        w = w / w.sum(axis=1, keepdims=True)
        mean = (w * vals).sum(axis=1)
        var = (w * (vals - mean[:, None]) ** 2).sum(axis=1)
        nearest = sims.max(axis=1)
        global_sd = float(np.std(self._y)) if len(self._y) > 1 else 1.0
        # Blend toward the global spread as the nearest neighbour recedes.
        sigma = np.sqrt(var) * nearest + global_sd * (1.0 - nearest)
        return Prediction(mean, np.clip(sigma, 1e-9, None))


class BootstrapEnsembleSurrogate(Surrogate):
    """Bagged ensemble; sigma is the disagreement across members.

    Stands in for the random-forest arm the brief names.  scikit-learn is not
    installed in this environment, so the ensemble is built from the closed-form
    members above over bootstrap resamples, which gives the same
    variance-from-disagreement signal without a new dependency.
    """

    def __init__(self, *, base: str = "linear", n_estimators: int = 16, seed: int = 0) -> None:
        if n_estimators < 2:
            raise ValueError("n_estimators >= 2 required")
        self.base = base
        self.n_estimators = n_estimators
        self.seed = seed
        self.name = f"bagged_{base}_{n_estimators}"
        self._members: list[GaussianProcessSurrogate] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> BootstrapEnsembleSurrogate:
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self._members = []
        for _ in range(self.n_estimators):
            take = rng.integers(0, n, size=n)
            member = GaussianProcessSurrogate(kernel=self.base)
            member.fit(X[take], y[take])
            self._members.append(member)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        if not self._members:
            raise RuntimeError("predict before fit")
        means = np.vstack([m.predict(X).mean for m in self._members])
        return Prediction(means.mean(axis=0), np.clip(means.std(axis=0), 1e-9, None))


SURROGATE_FACTORIES = {
    "mean": lambda: MeanSurrogate(),
    "gp_tanimoto": lambda: GaussianProcessSurrogate(kernel="tanimoto"),
    "gp_linear": lambda: GaussianProcessSurrogate(kernel="linear"),
    "knn_tanimoto": lambda: KNNTanimotoSurrogate(k=5),
    "bagged_linear": lambda: BootstrapEnsembleSurrogate(base="linear", n_estimators=16),
}


def build_surrogate(name: str) -> Surrogate:
    if name not in SURROGATE_FACTORIES:
        raise ValueError(f"unknown surrogate {name!r}; have {sorted(SURROGATE_FACTORIES)}")
    return SURROGATE_FACTORIES[name]()


# ---- Acquisition ----


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def acquisition_scores(
    pred: Prediction, *, kind: str = "ucb", beta: float = 1.0, best: float = 0.0
) -> np.ndarray:
    """Acquisition over predicted mean and sigma.  Higher is better."""
    if kind == "greedy":
        return pred.mean
    if kind == "ucb":
        return pred.mean + beta * pred.sigma
    if kind == "ei":
        sigma = np.clip(pred.sigma, 1e-12, None)
        z = (pred.mean - best) / sigma
        return (pred.mean - best) * _standard_normal_cdf(z) + sigma * _standard_normal_pdf(z)
    if kind == "sigma":
        return pred.sigma
    raise ValueError(f"unknown acquisition {kind!r}")


# ---- Online archive ----


@dataclass(frozen=True)
class ArchiveRow:
    """One PURCHASED evaluation.  ``provenance`` names who paid for it."""

    smiles: str
    score: float
    provenance: str
    step: int


@dataclass
class OnlineArchive:
    """The run's purchased evaluations -- the only legal surrogate training set.

    There is deliberately no constructor argument carrying data and no bulk
    loader: every row must arrive through ``observe`` with a provenance tag, so
    a prescreen against an external scored database cannot be introduced by
    accident.
    """

    rows: list[ArchiveRow] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def observe(self, smiles: str, score: float, *, provenance: str, step: int) -> bool:
        """Record a purchased evaluation.  Returns False if already present."""
        if not provenance:
            raise ValueError("every purchased row needs a provenance tag")
        if not math.isfinite(score):
            raise ValueError("non-finite oracle score")
        if smiles in self._seen:
            return False
        self._seen.add(smiles)
        self.rows.append(ArchiveRow(smiles, float(score), provenance, int(step)))
        return True

    def __len__(self) -> int:
        return len(self.rows)

    def contains(self, smiles: str) -> bool:
        return smiles in self._seen

    @property
    def best(self) -> float:
        return max((r.score for r in self.rows), default=0.0)

    def scores(self) -> np.ndarray:
        return np.asarray([r.score for r in self.rows], dtype=np.float64)

    def smiles(self) -> list[str]:
        return [r.smiles for r in self.rows]

    def training_provenance(self) -> dict[str, int]:
        """Row counts per provenance tag, for the artifact's disclosure block."""
        out: dict[str, int] = {}
        for row in self.rows:
            out[row.provenance] = out.get(row.provenance, 0) + 1
        return out


# ---- Allocator ----


@dataclass
class BatchDecision:
    selected: list[int]
    surrogate_active: bool
    explore_slots: int
    exploit_slots: int
    rank_seconds: float
    predicted_mean: list[float]
    predicted_sigma: list[float]


class SurrogateAllocator:
    """Rank free candidates, spend the scarce oracle calls on the ranked head.

    ``min_fit_rows`` is the cold-start guard: with fewer purchased rows the
    allocator does not pretend to rank and returns an exploration batch.
    ``explore_fraction`` is the standing floor that survives even once the
    surrogate is active.
    """

    def __init__(
        self,
        *,
        surrogate_name: str = "gp_tanimoto",
        acquisition: str = "ucb",
        beta: float = 1.0,
        explore_fraction: float = 0.25,
        min_fit_rows: int = 8,
        diversity_penalty: float = 0.0,
        featurizer: MorganFeaturizer | None = None,
        seed: int = 0,
    ) -> None:
        if not (0.0 <= explore_fraction <= 1.0):
            raise ValueError("explore_fraction must be in [0, 1]")
        if min_fit_rows < 2:
            raise ValueError("min_fit_rows >= 2 required for a variance estimate")
        if not (0.0 <= diversity_penalty <= 1.0):
            raise ValueError("diversity_penalty must be in [0, 1]")
        self.surrogate_name = surrogate_name
        self.acquisition = acquisition
        self.beta = beta
        self.explore_fraction = explore_fraction
        self.min_fit_rows = min_fit_rows
        self.diversity_penalty = diversity_penalty
        self.featurizer = featurizer or MorganFeaturizer()
        self.rng = np.random.default_rng(seed)

    def _explore_slots(self, k: int) -> int:
        if self.explore_fraction <= 0.0:
            return 0
        # Guarantee at least one exploratory call whenever more than one is
        # being spent, so the floor cannot round away on small batches.
        return max(1, round(self.explore_fraction * k)) if k >= 2 else 0

    def select_batch(
        self, archive: OnlineArchive, candidates: list[str], k: int
    ) -> BatchDecision:
        """Choose ``k`` candidate indices to charge.  Never calls an oracle."""
        if k < 1:
            raise ValueError("k >= 1 required")
        started = time.perf_counter()
        pool = [i for i, s in enumerate(candidates) if not archive.contains(s)]
        if not pool:
            return BatchDecision([], False, 0, 0, time.perf_counter() - started, [], [])
        k = min(k, len(pool))

        if len(archive) < self.min_fit_rows:
            picked = [int(i) for i in self.rng.choice(pool, size=k, replace=False)]
            return BatchDecision(
                picked, False, k, 0, time.perf_counter() - started, [], []
            )

        X_train, kept = self.featurizer.featurize(archive.smiles())
        y_train = archive.scores()[kept]
        if len(kept) < self.min_fit_rows:
            picked = [int(i) for i in self.rng.choice(pool, size=k, replace=False)]
            return BatchDecision(
                picked, False, k, 0, time.perf_counter() - started, [], []
            )

        pool_smiles = [candidates[i] for i in pool]
        X_pool, pool_kept = self.featurizer.featurize(pool_smiles)
        if not pool_kept:
            picked = [int(i) for i in self.rng.choice(pool, size=k, replace=False)]
            return BatchDecision(
                picked, False, k, 0, time.perf_counter() - started, [], []
            )
        usable = [pool[j] for j in pool_kept]

        surrogate = build_surrogate(self.surrogate_name).fit(X_train, y_train)
        pred = surrogate.predict(X_pool)
        scores = acquisition_scores(
            pred, kind=self.acquisition, beta=self.beta, best=archive.best
        )

        explore_slots = min(self._explore_slots(k), k)
        exploit_slots = k - explore_slots

        chosen: list[int] = []
        taken: set[int] = set()
        if exploit_slots > 0:
            order = np.argsort(-scores)
            if self.diversity_penalty > 0.0:
                chosen_local = self._diverse_head(X_pool, scores, exploit_slots)
            else:
                chosen_local = [int(j) for j in order[:exploit_slots]]
            for j in chosen_local:
                chosen.append(usable[j])
                taken.add(usable[j])

        remaining = [i for i in pool if i not in taken]
        if explore_slots > 0 and remaining:
            n_take = min(explore_slots, len(remaining))
            for i in self.rng.choice(remaining, size=n_take, replace=False):
                chosen.append(int(i))

        elapsed = time.perf_counter() - started
        local_of = {usable[j]: j for j in range(len(usable))}
        means = [float(pred.mean[local_of[i]]) if i in local_of else float("nan") for i in chosen]
        sigmas = [float(pred.sigma[local_of[i]]) if i in local_of else float("nan") for i in chosen]
        return BatchDecision(
            chosen, True, explore_slots, exploit_slots, elapsed, means, sigmas
        )

    def _diverse_head(self, X_pool: np.ndarray, scores: np.ndarray, n: int) -> list[int]:
        """Greedy top-n with a similarity penalty against what is already picked.

        Plain top-k on a fingerprint acquisition returns near-duplicates, which
        wastes a scarce batch on one chemical neighbourhood.
        """
        picked: list[int] = []
        working = scores.astype(np.float64).copy()
        for _ in range(min(n, X_pool.shape[0])):
            j = int(np.argmax(working))
            picked.append(j)
            working[j] = -np.inf
            sims = tanimoto_kernel(X_pool, X_pool[j : j + 1]).ravel()
            live = np.isfinite(working)
            working[live] -= self.diversity_penalty * sims[live] * np.abs(scores[live])
        return picked


# ---- Acquisition engagement ----
#
# diagnostics/task3_surrogate_acquisition_kind_mismatch.json records a general
# failure this allocator must not repeat: any convex-combination estimator
# (k-NN mean, kernel regression, bagged average) paired with an acquisition that
# is zero unless the prediction exceeds the incumbent (EI, PI, marginal
# hypervolume) CANNOT FIRE, because a weighted average of observed labels cannot
# exceed the largest of them.  Nothing errors -- the acquisition still returns
# numbers and still ranks -- it just degrades silently into its tie-break.
#
# The one-line detector from that record is ``engagement_fraction``: the share of
# candidates the acquisition scores above the incumbent.  Near zero means the
# mechanism is not being tested at all.


def engagement_fraction(
    pred: Prediction, incumbent: float, *, kind: str = "ucb", beta: float = 1.0
) -> float:
    """Share of candidates whose acquisition clears the incumbent score.

    For ``kind='greedy'`` this is the share whose predicted MEAN exceeds the
    incumbent, which is exactly the quantity that collapses to ~0 for an
    averaging surrogate once the archive holds a good molecule.
    """
    if kind == "greedy":
        return float(np.mean(pred.mean > incumbent))
    if kind == "ucb":
        return float(np.mean(pred.mean + beta * pred.sigma > incumbent))
    if kind in ("ei", "sigma"):
        scores = acquisition_scores(pred, kind=kind, beta=beta, best=incumbent)
        return float(np.mean(scores > 1e-9))
    raise ValueError(f"unknown acquisition {kind!r}")


def select_beta_by_engagement(
    pred: Prediction,
    incumbent: float,
    *,
    candidates: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0),
    threshold: float = 0.5,
    kind: str = "ucb",
) -> dict:
    """Smallest beta whose engagement clears ``threshold``.

    Selection is on ENGAGEMENT -- whether the acquisition can discriminate at
    all -- which is independent of any benchmark outcome, so beta is never tuned
    on the score it will later be judged by.  Pick quality falls monotonically as
    beta rises (the ranking drifts toward pure uncertainty), hence "smallest
    that clears" rather than "largest available".
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    sweep = {
        float(b): engagement_fraction(pred, incumbent, kind=kind, beta=float(b))
        for b in candidates
    }
    chosen = next((b for b in sorted(sweep) if sweep[b] >= threshold), None)
    return {
        "sweep": sweep,
        "threshold": threshold,
        "selected_beta": chosen if chosen is not None else max(sweep),
        "cleared_threshold": chosen is not None,
        "selection_rule": "smallest beta clearing the engagement threshold",
    }

"""Goal-conditioned search over the macro vocabulary. No fixed program.

WHY THIS REPLACES THE FIXED RECIPE. `local_extend:2 -> close_disjoint ->
restate2` was discovered on parp1 s0 at delta=0.4 and hard-coded: 2 anchors,
length 8, disjoint closure, 2 restates. Applied unchanged elsewhere it produced
0 T4-feasible endpoints from 300 rollouts on parp1 s0 d0.6, 5ht1b s7 d0.4 and
5ht1b s7 d0.6 -- because the similarity budget is a property of the SEED, not a
constant. parp1's seed carries 41 fingerprint bits, 5ht1b's only 29, and delta
0.6 permits far less perturbation than 0.4.

So the controller must choose its own program given (x0, delta, constraints,
budget) rather than execute a script. Macros are a VOCABULARY of temporally
extended actions, not a recipe.

Three of the four T4 constraints -- QED, SA, similarity -- are cheap. Only
docking is expensive. So the search optimises the cheap margins first and
spends the oracle only on molecules already known to be feasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: (macro, length) choices the controller may take. Lengths are per-macro
#: because a cyclisation is one structural event while growth is many.
ACTION_SPACE: tuple[tuple[str, int], ...] = tuple(
    [("local_extend", k) for k in (1, 2, 3, 4, 6, 8)]
    + [("grow", k) for k in (1, 2, 4)]
    + [("cyclize", 1)]
    + [("close_disjoint", 1)]
    + [("annulate", 1)]
    + [("restate", k) for k in (1, 2, 3)]
    + [("shrink", k) for k in (1, 2)]
    + [("decorate", 1)]
    # BUILD_RING_SYSTEM is one temporally extended action, judged only at its
    # endpoint. It is parameterised by ring size because a 5- and a 6-ring are
    # different chemical decisions, not different lengths of the same one.
    # Frozen 2026-08-26 at 8/8 qualification on 5ht1b seed 2; see
    # docs/BUILD_RING_SYSTEM_FROZEN.md. Exposed here so the controller can
    # select it, rather than having to rediscover the eight primitive edits
    # that compose it -- none of which improves any local measure until all of
    # them are done, which is why decomposed search never found this.
    # SEMANTIC ring actions. The controller decides what KIND of ring system to
    # build -- topology, size, composition, electronic state -- because those
    # choices change the reachable molecular class. The attachment atom / fusion
    # edge is a CONDITIONAL REALIZATION chosen underneath, not a top-level
    # action.
    #
    # Measured basis for that split. Ring construction is strongly useful:
    # ring-containing trajectories averaged -8.96 against -7.75 without, over
    # 33 docked molecules. But per-site global credit earned nothing: with
    # total initial ring mass matched at 0.24, six site-visible actions and one
    # opaque action performed identically (best -11.3 vs -11.2, mean -8.46 vs
    # -8.57, two 28-heavy molecules each) -- every gap far under the 0.70
    # kcal/mol noise floor. An 8-replicate isomer test agreed: meta -10.62 vs
    # para -10.36 at sd ~1.2.
    #
    # The earlier -9.6 -> -11.4 "win" was a confound: six actions at 0.04 gave
    # 0.24 initial ring mass against 0.05 for one, so rings were simply sampled
    # 5x more often. Hence the explicit prior below rather than an accidental one.
    #
    # Sites are NOT discarded -- the compiler still enumerates every admissible
    # realization and its predicted endpoint, so a contextual value model can
    # score the candidate molecules directly once one exists. What is dropped
    # is only splitting 30 docking observations across six unrelated global
    # parameters.
    + [(f"ring:{t}/{k}/{c}/{st}", k)
       for (t, k, c, st) in (("linked", 6, "C", "aromatic"),
                             ("fused", 6, "C", "aromatic"),
                             ("fused", 6, "C", "saturated"))]
    + [("STOP", 0)]
)


@dataclass
class Margins:
    """How far an endpoint is from each cheap constraint. >= 0 means satisfied."""
    sim: float
    qed: float
    sa: float
    valid: bool

    @property
    def feasible(self) -> bool:
        return self.valid and self.sim >= 0 and self.qed >= 0 and self.sa >= 0

    @property
    def shortfall(self) -> float:
        """Total violation, normalised. 0 exactly when feasible."""
        if not self.valid:
            return 10.0
        return (max(0.0, -self.sim) / 0.4 + max(0.0, -self.qed) / 0.6
                + max(0.0, -self.sa) / 4.0)


def margins(smiles: str, seed_smiles: str, delta: float,
            qed_min: float = 0.6, sa_max: float = 4.0) -> Margins:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator
    import os, sys
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.gates.med_chem_gate import is_valid
    m = Chem.MolFromSmiles(smiles)
    s = Chem.MolFromSmiles(seed_smiles)
    if m is None or s is None:
        return Margins(-9, -9, -9, False)
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sim = float(DataStructs.TanimotoSimilarity(gm.GetFingerprint(s),
                                               gm.GetFingerprint(m)))
    try:
        sa = float(sascorer.calculateScore(m))
    except Exception:
        return Margins(-9, -9, -9, False)
    # PRIMARY benchmark feasibility is exactly the published GenMol criterion:
    # QED >= 0.6, SA <= 4, Tanimoto >= delta, and a parseable/sanitizable
    # molecule. Our stricter medicinal-chemistry screen is NOT part of it --
    # the comparator has no such rule, so applying it here would run the
    # benchmark harder than GenMol and understate COMPOSE. It is reported
    # separately as a secondary validity analysis.
    #
    # `is_executable` still constrains the PATH (real molecules at every step);
    # this is only about what may be RETURNED.
    return Margins(sim=sim - delta, qed=float(QED.qed(m)) - qed_min,
                   sa=sa_max - sa, valid=True)


def med_chem_ok(smiles: str) -> bool:
    """Secondary validity screen -- reported alongside, never gating the
    primary benchmark number."""
    from compose_v4.gates.med_chem_gate import is_valid
    return bool(is_valid(smiles))


@dataclass
class Particle:
    smiles: str
    program: list = field(default_factory=list)
    stopped: bool = False


class CEMController:
    """Cross-entropy over ACTION_SPACE, reweighted by endpoint shortfall.

    Deliberately simple: no learned value function yet. The point is to test
    whether ANY adaptive search over the existing vocabulary can satisfy the
    cheap constraints on cells where the fixed recipe scored 0/300. If a plain
    CEM can, the fixed recipe was the problem. If it cannot, the controller is.
    """

    def __init__(self, n_actions: int, rng, elite_frac: float = 0.25,
                 smoothing: float = 0.3, init_probs=None):
        # init_probs allows a NON-UNIFORM prior. Needed to control for a
        # confound: replacing one ring action with six site actions silently
        # raised total initial ring mass from 1/20=0.05 to 6/25=0.24, so the
        # site-visible arm sampled ring macros ~5x more often before learning
        # anything. Any comparison of the two must match that mass.
        if init_probs is None:
            self.logits = np.zeros(n_actions)
        else:
            q = np.asarray(init_probs, dtype=float)
            q = q / q.sum()
            self.logits = np.log(np.maximum(q, 1e-9))
        self._prior = self.probs()
        self.rng = rng
        self.elite_frac = elite_frac
        self.smoothing = smoothing

    def probs(self) -> np.ndarray:
        e = np.exp(self.logits - self.logits.max())
        return e / e.sum()

    def sample(self, n: int) -> np.ndarray:
        return self.rng.choice(len(self.logits), size=n, p=self.probs())

    def update_by_rank(self, action_idx, scores, explore: float = 0.15,
                       groups=None, site_shrink: float = 0.25) -> None:
        """Update toward actions used by the best-RANKED trajectories.

        Rank-based because docking noise is ~0.70 kcal/mol -- absolute score
        magnitudes are not trustworthy at this budget, orderings are more so.
        `explore` mixes the initial uniform distribution back in so one lucky
        early action cannot collapse the controller.

        `groups` enables PARTIAL POOLING over realizations of the same macro
        (e.g. the attachment sites of one ring specification). Evidence is
        pooled to the parent and each member keeps only `site_shrink` of its
        own residual:

            Q(m, xi) = Q_macro(m) + shrink * delta_site(xi)

        Measured reason: with sites as six independent actions the controller
        put 0.254 mass on @0 against 0.096 on @1, while replicated docking
        showed @1 marginally BETTER (-10.13 vs -9.88 mean) and an 8-replicate
        isomer test put the two products 0.26 kcal/mol apart at sd ~1.2. The
        site-level estimator was pure noise-chasing. The macro-level effect was
        real over the same data -- ring-containing trajectories averaged -9.04
        against -7.76 without -- so the pooled parent should carry that
        evidence and the per-site residual should stay near zero until it
        earns otherwise.
        """
        action_idx = np.asarray(action_idx)
        scores = np.asarray(scores, dtype=float)
        if action_idx.size == 0:
            return
        k = max(1, int(len(scores) * self.elite_frac))
        elite = action_idx[np.argsort(scores)[:k]]
        counts = np.bincount(elite, minlength=len(self.logits)).astype(float)
        target = counts / max(counts.sum(), 1.0)
        for grp in (groups or []):
            g = [i for i in grp if 0 <= i < len(target)]
            if len(g) < 2:
                continue
            shared = float(np.mean([target[i] for i in g]))
            for i in g:
                # inherit the pooled macro evidence, keep a shrunken residual
                target[i] = shared + float(site_shrink) * (target[i] - shared)
        cur = self.probs()
        new = (1 - self.smoothing) * cur + self.smoothing * target
        # explore mixes back the INITIAL prior, not a flat uniform, so a
        # deliberately non-uniform arm is not silently pulled toward uniform
        new = (1 - explore) * new + explore * self._prior
        self.logits = np.log(np.maximum(new, 1e-9))

    def update(self, action_idx, scores) -> None:
        """Raise the probability of actions used by low-shortfall trajectories."""
        action_idx = np.asarray(action_idx)
        scores = np.asarray(scores, dtype=float)
        if action_idx.size == 0:
            return
        k = max(1, int(len(scores) * self.elite_frac))
        elite = action_idx[np.argsort(scores)[:k]]      # lower shortfall is better
        counts = np.bincount(elite, minlength=len(self.logits)).astype(float)
        target = counts / max(counts.sum(), 1.0)
        cur = self.probs()
        new = (1 - self.smoothing) * cur + self.smoothing * target
        self.logits = np.log(np.maximum(new, 1e-9))


class PrefixArchive:
    """Every realized molecular state is eligible for endpoint bookkeeping.

    COMPOSE's states are explicit molecules, so an episode
    x0 -> x1 -> x2 -> x3 does not have one endpoint; it has four candidate
    answers, and T4 asks for a single returned molecule. If x2 is feasible and
    a later destructive edit ruins x3, x2 is still a molecule COMPOSE actually
    generated and is a legitimate candidate.

    Scoring only the final state cost a whole sentinel round: a program built a
    feasible terphenyl (sim 0.610, QED 0.710) with two ring macros, a following
    restate de-aromatised it, and the episode was recorded as a plain failure
    with the good molecule discarded. That produced 0 feasible -> 0 docked ->
    0 controller updates -> uniform forever.

    Credit is attributed to the PREFIX that produced each state, never to the
    actions that came after it -- several of those are actively destructive and
    crediting them teaches the controller the opposite of the truth.
    """

    def __init__(self, margins_fn, seed_smiles: str | None = None):
        self._margins = margins_fn
        self._seen: dict[str, tuple[int, object]] = {}
        # x0 is a realized state and may be noted, but it must NEVER become a
        # candidate: the seed trivially satisfies sim=1 and, on a QED-rich seed,
        # every other constraint too. Letting it through reintroduces the
        # trivial-solution failure where optimising shortfall made x = x0 the
        # optimum and STOP0 dominated the archive.
        self._seed_key = self._canon(seed_smiles) if seed_smiles else None

    @staticmethod
    def _canon(smiles: str):
        try:
            from rdkit import Chem
            m = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(m) if m is not None else None
        except Exception:
            return None

    def note(self, smiles: str, step: int) -> bool:
        """Record a realized state. `step` is its index in the program, so the
        producing prefix has length step + 1. Returns True if it was feasible.

        Must be called from EVERY control path, including branches that
        `continue` and loops that `break`. The bug this replaces was exactly a
        `continue` in the build_ring_system branch jumping past a tail-only
        check -- so the single macro whose products matter was the one action
        whose feasible states were never recorded.
        """
        if not smiles:
            return False
        if self._seed_key is not None and self._canon(smiles) == self._seed_key:
            return False                    # noted, never a candidate
        try:
            m = self._margins(smiles)
        except Exception:
            return False
        if not m.feasible:
            return False
        prior = self._seen.get(smiles)
        # dedup by molecule, keeping the SHORTEST producing prefix: the extra
        # actions in a longer route did not contribute to this molecule
        if prior is None or (step + 1) < prior[0]:
            self._seen[smiles] = (step + 1, m)
        return True

    def entries(self):
        """[(smiles, prefix_len, margins)], most-constructed last."""
        return [(k, v[0], v[1]) for k, v in
                sorted(self._seen.items(), key=lambda kv: kv[1][0])]

    def best(self):
        """The furthest-along feasible state, or None."""
        e = self.entries()
        return e[-1] if e else None

    def __len__(self) -> int:
        return len(self._seen)


def realization_groups(action_space):
    """Indices of actions that are realizations of the SAME ring specification.

    `build_ring_system@0..@5` at a given size are six attachment realizations
    of one semantic macro, not six unrelated actions -- and @k means nothing
    across molecules, since the k-th admissible site on one state is unrelated
    to the k-th on another. Grouping them lets evidence pool to the macro while
    per-site residuals stay shrunk.
    """
    from collections import defaultdict
    g = defaultdict(list)
    for i, (name, k) in enumerate(action_space):
        if name.startswith("build_ring_system@"):
            g[(name.split("@", 1)[0], k)].append(i)
    return [v for v in g.values() if len(v) > 1]


# Mass-matched control arm: ONE opaque ring action instead of six site-visible
# realizations. Used with init_probs so total initial ring probability equals
# the site-visible arm's, isolating "does naming the site help?" from "does
# sampling rings more often help?".
ACTION_SPACE_OPAQUE: tuple[tuple[str, int], ...] = tuple(
    [a for a in ACTION_SPACE if not a[0].startswith("build_ring_system@")]
    + [("build_ring_system", 6)]
)


def is_ring_action(name: str) -> bool:
    """True for any BUILD_RING_SYSTEM action, under either naming scheme."""
    return name.startswith("ring:") or name.startswith("build_ring_system")


def matched_prior(action_space, ring_mass: float = 0.24):
    """Uniform over non-ring actions, `ring_mass` shared across ALL ring actions.

    PARENT-LEVEL invariance: the total probability of building a ring must be a
    deliberate choice, never a side effect of how many semantic variants happen
    to exist. Adding fused took the ring family from 1 variant to 3; without
    this, exploration mass would have shifted silently -- the same confound that
    made six site-actions look like an architectural win when they had merely
    raised ring sampling from 0.05 to 0.24.

    An earlier version of this function matched on "build_ring_system" only and
    so found ZERO of the "ring:" actions, quietly yielding 3/22 = 0.136 instead
    of the requested 0.24.
    """
    ring = [i for i, (n, _k) in enumerate(action_space) if is_ring_action(n)]
    q = np.zeros(len(action_space))
    other = [i for i in range(len(action_space)) if i not in ring]
    if ring:
        for i in ring:
            q[i] = float(ring_mass) / len(ring)
    for i in other:
        q[i] = (1.0 - float(ring_mass)) / max(len(other), 1)
    return q / q.sum()


def parse_ring_spec(name: str):
    """`ring:topology/size/composition/state` -> dict, or None."""
    if not name.startswith("ring:"):
        return None
    try:
        t, k, c, st = name[5:].split("/")
    except ValueError:
        return None
    return dict(topology=t, size=int(k),
                composition={"C": "carbon_rich"}.get(c, c), state=st)


def select_realization(candidates, prior=None, value_fn=None, rng=None,
                       temperature: float = 1.0):
    """Choose one realization xi of a ring macro from its admissible set.

        P(xi | x, m, z)  proportional to  R_theta(xi | x, m) * H(xi; x, m, z)

    `candidates` is what the cheap compiler produced: each entry carries the
    site/edge AND its predicted endpoint molecule, so `value_fn` -- a future
    contextual h_phi(y_xi, z, b) -- can score the actual candidate futures
    rather than arbitrary rank tokens. With `value_fn=None` this reduces to the
    goal-independent prior alone, which is the honest default while no
    trustworthy site-level objective evidence exists.

    Returns (index, probabilities).
    """
    import numpy as np
    n = len(candidates)
    if n == 0:
        return None, np.zeros(0)
    p = np.ones(n, dtype=float) if prior is None else np.asarray(prior, float).copy()
    p = np.maximum(p, 1e-12)
    if value_fn is not None:
        h = np.array([max(float(value_fn(c)), 1e-12) for c in candidates])
        p = p * h
    if temperature != 1.0:
        p = p ** (1.0 / max(temperature, 1e-6))
    p = p / p.sum()
    if rng is None:
        return int(np.argmax(p)), p
    return int(rng.choice(n, p=p)), p

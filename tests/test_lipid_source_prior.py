"""LipidCarbonTreePrior: valid lipid-shaped carbon skeletons with long linear tails."""
import numpy as np

from compose_v4.chem.lipid_source_prior import LipidCarbonTreePrior
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state


def _terminal_chains(state):
    n = int(state.n_real_atoms)
    b = np.asarray(state.bonds)[:n, :n]
    adj = {i: list(np.flatnonzero(b[i] != 0)) for i in range(n)}
    deg = {i: len(adj[i]) for i in range(n)}
    out = []
    for leaf in [i for i in range(n) if deg[i] == 1]:
        length, prev, cur = 1, None, leaf
        while True:
            nxt = [x for x in adj[cur] if x != prev]
            if len(nxt) != 1 or deg[nxt[0]] >= 3:
                break
            prev, cur = cur, nxt[0]
            length += 1
        out.append(length)
    return out


def test_lipid_prior_is_degree_bounded_subclass():
    lp = LipidCarbonTreePrior(sizes=(44,), probabilities=None, max_degree=4)
    assert isinstance(lp, DegreeBoundedCarbonTreePrior)  # passes flexible_size_graft type check


def test_lipid_prior_samples_valid_long_tailed_skeletons():
    rng = np.random.default_rng(1)
    lp = LipidCarbonTreePrior(sizes=(44,), probabilities=None, max_degree=4)
    longest = []
    for _ in range(30):
        st = lp.sample(rng, n_slots=64)
        assert is_valid_state(st) and is_connected_or_null(st)
        assert int(st.n_real_atoms) == 44
        n = int(st.n_real_atoms)
        b = np.asarray(st.bonds)[:n, :n]
        assert int(b.max()) <= 1  # single bonds only
        assert (np.asarray(st.bonds)[:n, :n].sum(axis=1) <= 4).all()  # degree <= 4
        longest.append(max(_terminal_chains(st)))
    assert sorted(longest)[len(longest) // 2] >= 10  # lipid-scale tails


def test_lipid_prior_beats_generic_on_tail_length():
    rng = np.random.default_rng(2)
    lp = LipidCarbonTreePrior(sizes=(44,), probabilities=None, max_degree=4)
    gp = DegreeBoundedCarbonTreePrior(sizes=(44,), probabilities=None, max_degree=4)
    lp_max = np.median([max(_terminal_chains(lp.sample(rng, n_slots=64))) for _ in range(20)])
    gp_max = np.median([max(_terminal_chains(gp.sample(rng, n_slots=64))) for _ in range(20)])
    assert lp_max >= 2 * gp_max  # lipid tails are much longer

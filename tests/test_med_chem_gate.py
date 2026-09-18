"""The gate must not reject legitimate chemistry. Calibrated on molecules we did
not author: the 15 T4 seeds and the 25 published InVirtuoGen winners."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from compose_v4.gates.med_chem_gate import validity_reasons, is_valid

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEEDS = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
WINS  = json.loads((ROOT / "diagnostics/ivg_winners.json").read_text())

def test_accepts_every_t4_seed():
    bad = [(s["target"], validity_reasons(s["smiles"])) for s in SEEDS
           if not is_valid(s["smiles"])]
    assert not bad, f"gate rejects real T4 seeds: {bad}"

def test_accepts_every_published_ivg_winner():
    bad = []
    for cell, rec in WINS.items():
        for w in rec["winners"]:
            if not is_valid(w["smiles"]):
                bad.append((cell, w["smiles"], validity_reasons(w["smiles"])))
    assert not bad, f"gate rejects published IVG winners: {bad}"

def test_rejects_the_known_bad_compose_molecules():
    known_bad = {
        "O=C1NCCn2c3c(c4cccc1c42)C=[IH2]O3":            "hypervalent_halogen",
        "O=C1NCCn2cc([IH4])c3cccc1c32":                 "hypervalent_halogen",
        "O=C1NCCc2c(OC(O)=C[PH2](F)F)[nH]c3cccc1c23":   "non_phosphate_P",
        "CC1=C2c3c(ccnc3[PH3]1)-c1ccccc1C(=O)NC2c1ccc(O)cc1F": "non_phosphate_P",
        "CCC(CCCC(C)=C1N=N1)NC(=O)C1CCCN1C(=O)C(N)Cc1ccccc1":  "strained_NN_ring",
        "C=CC=c1cc(F)cc(C(=O)N(CCC(C)C)Cc2cccccc(C(N)=O)c(F)cc2)c1=C": "isolated_ring",
    }
    for smi, kind in known_bad.items():
        r = validity_reasons(smi)
        assert r, f"gate accepted a known-bad molecule: {smi}"
        assert any(x.startswith(kind) for x in r), f"{smi}: expected {kind}, got {r}"

def test_unparseable_is_rejected():
    assert validity_reasons("not_a_smiles") == ["unparseable"]


def test_rejects_hypervalent_sulfur():
    """Found in macro endpoints AFTER the gate shipped: the gate checked
    halogens and phosphorus but not sulfur."""
    for smi in ("C1#[SH2]CNC=N1", "C1C[SH4]CN1"):
        assert not is_valid(smi), f"gate accepted hypervalent sulfur: {smi}"
        assert any(r.startswith("hypervalent_S") or r.startswith("element")
                   for r in validity_reasons(smi)), validity_reasons(smi)


def test_ordinary_sulfur_still_accepted():
    for smi in ("c1ccsc1", "CSC", "CS(=O)(=O)C", "CCS"):
        assert is_valid(smi), f"gate wrongly rejected normal sulfur: {smi}"


def test_pathwise_gate_is_strictly_weaker_than_the_full_gate():
    from compose_v4.gates.med_chem_gate import is_executable, is_valid
    # an isolated macrocycle is a real molecule -- a search may pass through it
    macro = "C1CCCCCCCCCCCC1"
    assert is_executable(macro), "pathwise gate must not block a real molecule"
    assert not is_valid(macro), "full gate should still flag it as an endpoint"


def test_pathwise_gate_still_blocks_impossible_chemistry():
    from compose_v4.gates.med_chem_gate import is_executable
    for bad in ("C1#[SH2]CNC=N1", "O=C1NCCn2cc([IH4])c3cccc1c32", "not_a_smiles"):
        assert not is_executable(bad), f"pathwise gate must block {bad}"


def test_endpoint_gate_adds_qed_and_sa():
    from compose_v4.gates.med_chem_gate import is_acceptable_endpoint, is_executable
    ugly = "O=C1NCc2cc(C3CCCC3Cc3ccc4c(c3)CNC(=O)c3cccn3C4=O)ccc21"   # IVG winner
    assert is_acceptable_endpoint(ugly), "a real winner must pass the endpoint gate"
    # something executable but not a good answer
    junk = "C1CCCCCCCCCCCC1"
    assert is_executable(junk) and not is_acceptable_endpoint(junk)


def test_all_seeds_and_winners_pass_both_gates():
    import json, pathlib
    from compose_v4.gates.med_chem_gate import is_executable, is_acceptable_endpoint
    root = pathlib.Path(__file__).resolve().parents[1]
    for s in json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text()):
        assert is_executable(s["smiles"])
    W = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    for cell, rec in W.items():
        for w in rec["winners"]:
            assert is_executable(w["smiles"]), cell
            assert is_acceptable_endpoint(w["smiles"]), cell


def test_t4_feasible_requires_similarity_not_just_qed_and_sa():
    """The exact bug: a molecule with fine QED/SA but sim 0.14 is NOT feasible."""
    import json, pathlib
    from compose_v4.gates.med_chem_gate import t4_feasible, is_acceptable_endpoint
    root = pathlib.Path(__file__).resolve().parents[1]
    seed = [s for s in json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
            if s["target"] == "parp1" and s["idx"] == 0][0]["smiles"]
    # a real carbon-rich endpoint: passes QED/SA, fails similarity
    far = "CC1CCC2C(=O)NCc3c(ccc(C(CCN)N4CCN(C)C4)c3F)N12"
    assert not t4_feasible(far, seed, 0.4), "similarity must be enforced"
    # the seed itself trivially satisfies all four
    assert t4_feasible(seed, seed, 0.4)


def test_t4_feasible_matches_the_published_winners():
    import json, pathlib
    from compose_v4.gates.med_chem_gate import t4_feasible
    root = pathlib.Path(__file__).resolve().parents[1]
    seeds = {f"{s['target']}_s{s['idx']}": s for s in
             json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    W = json.loads((root / "diagnostics/ivg_winners.json").read_text())
    for cell, rec in W.items():
        tgt, si, dl = cell.rsplit("_", 2)
        seed = seeds[f"{tgt}_{si}"]["smiles"]; delta = float(dl[1:])
        for w in rec["winners"]:
            assert t4_feasible(w["smiles"], seed, delta), \
                f"a published winner must be T4-feasible: {cell}"


def test_saturated_polyaza_rings_are_rejected():
    """Hexazine and relatives are not synthesizable. The pre-existing strained
    N-N rule only fired on rings of size <= 4, so BUILD_RING_SYSTEM with
    composition='hetero_rich' built an N1NNNNN1 pendant and both gates passed
    it."""
    from compose_v4.gates.med_chem_gate import is_valid, validity_reasons
    assert not is_valid("N1NNNNN1")
    assert any("polyaza" in r for r in validity_reasons("N1NNNNN1"))
    assert not is_valid("FC(F)(F)c1cc(N2CC[NH2+]CC2)ccc1N1NNNNN1")


def test_aromatic_polyaza_rings_are_still_accepted():
    """Counted on non-aromatic N-N bonds only. Tetrazole has four contiguous
    nitrogens and appears in marketed drugs; an aromatic-blind rule would
    reject real chemistry to catch hexazine."""
    from compose_v4.gates.med_chem_gate import is_valid
    for smiles in ("CCn1nnnc1",            # 1-ethyltetrazole
                   "c1ccc2[nH]nnc2c1",     # benzotriazole
                   "Cc1nnc(C)o1"):         # 1,3,4-oxadiazole
        assert is_valid(smiles), smiles


def test_one_saturated_nn_bond_is_still_allowed():
    """Pyrazolidine is rare but real; the rule triggers at two N-N bonds."""
    from compose_v4.gates.med_chem_gate import is_valid
    assert is_valid("C1CNNC1")


def test_terminal_hypervalent_sulfur_is_flagged():
    """The rule has been wrong twice, in opposite directions; both are pinned here.

    Absent, it let `C1#[SH2]CNC=N1` -- a triple bond to hypervalent sulfur -- into macro
    endpoints. Written as `H and degree >= 2`, it missed TERMINAL hypervalent sulfur:
    `CC([SH3])...` has one carbon neighbour and three hydrogens, so degree is 1 and the
    rule never fired. That molecule was docked in an autonomous delta=0.6 round.
    """
    for smiles in ("CC([SH3])CC(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23", "C1#[SH2]CNC=N1"):
        reasons = validity_reasons(smiles)
        assert any(r.startswith("hypervalent_S") for r in reasons), (smiles, reasons)


def test_ordinary_sulfur_chemistry_still_passes():
    """Counting bonds to non-oxygen is what separates hypervalent from ordinary.

    Thiol, thioether and thiophene sit at two; sulfoxide and sulfone spend their extra
    bonds on oxygen. A rule that rejected any of these would be worse than the bug.
    """
    for smiles in ("CSC", "CCS", "CS(=O)C", "CS(=O)(=O)C", "c1ccsc1",
                   "CS(=O)(=O)N", "O=S(=O)(N)c1ccccc1", "CSSC"):
        assert validity_reasons(smiles) == [], smiles

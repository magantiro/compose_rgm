# PMO 3x250 scored run — zero-oracle chemical and trajectory autopsy

Run `79fec4984821e97de6decefd213e45a47b3d2b901a1a9c021ee0dfd3ed1561f4`,
volume `compose-v4-artifacts`, prefix `pmo_population_controller_v1/<run_id>/<task>/`.

**No oracle call was made by this analysis and nothing was launched.** Every score quoted
was charged by the run under audit. The two objective decompositions are offline
re-derivations validated against all 250 charged scores per task before use.

Machine-readable companion: `diagnostics/pmo_3x250_autopsy_v1.json`.
Drivers (all read-only): `scripts/pmo_3x250_autopsy.py`,
`scripts/pmo_3x250_celecoxib_distance.py`, `scripts/pmo_3x250_binder_probe.py`,
`scripts/pmo_3x250_jump_parent_counterfactual.py`, `scripts/pmo_3x250_autopsy_report.py`.

`gsk3b` returned exactly 0.0 on all 250 calls and is **descriptive only**. Its task-blind
binder numbers are retained, because the jump plan bank and the binder never see the
objective.

**Join caveat, found and corrected during this analysis.** A candidate record carries two
different "parents". The **measured parent** is `entries[provenance["entry_id"]]`, the
archive entry whose reward the controller credits — its ledger score equals the recorded
`parent_measured_score` on **202/202** rows in every task. `trace["states"][0]` is the
**program source**, the state the (possibly mutated) program was replayed from, and it is
a *different molecule* on **53/202 (perindopril)** and **59/202 (celecoxib)** rows, one
generation further back. Every parent-relative quantity below is computed against the
measured parent. The one exception is `retained fraction`, which is reported against the
program source because that is what the controller itself computes.

---

## 0. Verdict

| task | class | meaning | more v1 budget? |
|---|---|---|---|
| perindopril_mpo | **(A)** | still climbing in the correct chemical direction | yes — the curve is not flat at the boundary |
| celecoxib_rediscovery | **(B)** | plateaued in the wrong basin | no — fix realization / structural jumps |

One mechanism is common to both and is classified **(D)** on its own axis: the
`joint_dependency_region_jump` lane is structurally unable to realize its programs
(measured below), and for the *mpo* family it has nothing relevant to offer in the first
place.

---

## 1. Objective re-derivation (MEASURED, exact)

Both scoring functions were reproduced offline and agreed with the charged ledger on
**250/250 rows each**, to 1e-9. Every component number below is therefore measured, not
assumed.

* `celecoxib_rediscovery` = ECFP4 **count**-fingerprint Tanimoto to celecoxib.
  (Note: the bit-vector Morgan r=2/2048 Tanimoto — the controller's own
  `fiber_fingerprint` convention — is a *different* number and is reported separately.)
* `perindopril_mpo` = geometric mean of
  [ECFP4 count Tanimoto to perindopril, `Gaussian(mu=2, sigma=0.5)` on aromatic ring count].

Data-format correction: `oracle/query_NNNNNN` are **directories** containing
`started.json` + `result.json`, not JSONL files.

**Reproducibility note (worth recording, cause not proven).** One of four full runs of
the driver disagreed with the ledger on exactly **one** perindopril row —
`CCNCNCNn1cccc(NC(=O)COc2ccccc2)ccsc1=NC(=O)C(C)C1CNNC(N)C1`, a fused **9-membered**
N/S-containing ring, where RDKit reported 1 or 3 aromatic rings instead of 2 and the
Gaussian term therefore collapsed from 1.0 to 0.1353. That run was executed concurrently
with two other RDKit-heavy probe processes. Three subsequent runs on an unloaded machine
reproduce **250/250 twice over and are byte-identical to each other**, and the molecule
scores correctly in isolation. MEASURED: the anomaly occurred once and does not reproduce.
INFERRED: concurrency is the likely cause. The committed JSON is from a clean run. Anyone
recomputing a molecular property offline against a remotely charged oracle should check
the reproduction row-by-row rather than in aggregate, which is what caught this.

---

## 2. Best-so-far lineage

Lane ↔ mode ↔ scale vocabulary (from `pmo_population_controller.MODE_BY_CHANNEL` and
`pmo_credit.SCALE_BY_CHANNEL`):

| lane | mode | scale |
|---|---|---|
| `shallow_program_channel` | `refine_elite` | `refine` |
| `structured_program_channel` | `global_explore` | `medium` |
| `joint_dependency_region_jump` | `jump_from_elite` | `jump` |

### 2.1 perindopril_mpo — 6 improvements, 4 of them initialization molecules

| call | score | role / lane | parent (score) | family | prims | retained | ΔHA | ΔRings | scaffold changed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0076 | initialization | — | — | — | — | — | — | — |
| 3 | 0.0104 | initialization | — | — | — | — | — | — | — |
| 4 | 0.1516 | initialization | — | — | — | — | — | — | — |
| 12 | **0.4655** | initialization | — | — | — | — | — | — | — |
| 205 | 0.4714 | shallow / refine_elite | `CC(C)C(=O)N=c1sc2cc(NC(=O)COCC3CCCC3)ccc2n1F` (0.4635) | atom_insert+cycle_close | 7 | 1.00 | +6 (27→33) | +1 | yes |
| 240 | **0.4865** | structured / global_explore | same parent (0.4635) | atom_delete+atom_insert+cycle_close | 8 | 1.00 | +5 (27→32) | +1 | yes |

Molecules asked for:

* **call 1** `CC1CCC(C)(C(=O)N2CCCC(C)(C(=O)[O-])C2)O1` — 0.0076
* **first improvement** is call 1 itself; the first *controller* improvement is **call 205**
  `CC(C)C(=O)N=c1sc2cc(NC(=O)COCC3CCC(C4CCCCO4)C3)ccc2n1F` — 0.4714
* **midpoint (call 125)** queried `Cc1ncccc1CCC(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1` — 0.0051;
  best-so-far at 125 was still the call-12 initialization molecule at 0.4655
* **best (call 240)** `CCC(=O)N=c1sc2cc(NC(=O)COCC3CCC(C4CCCN4O)C3)ccc2n1F` — 0.4865

Bemis-Murcko scaffolds along that line — note they are all the same benzothiazolylidene
amide family, differing only in the saturated tail:

| molecule | scaffold |
|---|---|
| init best (call 12) | `N=c1[nH]c2ccc(NC(=O)COC3CCCCC3)cc2s1` |
| parent (call 172) | `N=c1[nH]c2ccc(NC(=O)COCC3CCCC3)cc2s1` |
| call 205 | `N=c1[nH]c2ccc(NC(=O)COCC3CCC(C4CCCCO4)C3)cc2s1` |
| **best (call 240)** | `N=c1[nH]c2ccc(NC(=O)COCC3CCC(C4CCCN4)C3)cc2s1` |
| *perindopril itself* | `C1CCC2NCCC2C1` |

Realized ancestry of the best molecule — a genuine 4-generation chain with **monotone**
improvement:

```
call   9  0.1154  initialization  CC(C)C(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1      HA 26  arom 3
call 117  0.4307  structured  12p CC1CCC(COCC(=O)Nc2ccc3[nH]c(=NC(=O)C(C)C)sc3c2)C1   HA 27  arom 2
call 172  0.4635  shallow      2p CC(C)C(=O)N=c1sc2cc(NC(=O)COCC3CCCC3)ccc2n1F        HA 27  arom 2
call 240  0.4865  structured   8p CCC(=O)N=c1sc2cc(NC(=O)COCC3CCC(C4CCCN4O)C3)ccc2n1F HA 32  arom 2
```

The call-117 step is the chemically meaningful one: it replaced a phenoxy with a
cyclopentylmethyl ether, taking the molecule from **3 aromatic rings to exactly 2** and
flipping the ring term from 0.1353 to 1.0 (MEASURED: aromatic ring counts 3, 2, 2, 2 along
the chain). Every later step then works purely on the similarity component.

### 2.2 celecoxib_rediscovery — 7 improvements, 6 of them initialization molecules

| call | score | role / lane | parent (score) | family | prims | retained | ΔHA | ΔRings | Tc(2048-bit) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0339 | initialization | — | — | — | — | — | — | 0.0548 |
| 2 | 0.0351 | initialization | — | — | — | — | — | — | 0.0556 |
| 3 | 0.0496 | initialization | — | — | — | — | — | — | 0.0556 |
| 4 | 0.0783 | initialization | — | — | — | — | — | — | 0.1081 |
| 5 | 0.1339 | initialization | — | — | — | — | — | — | 0.1143 |
| 6 | **0.1897** | initialization | — | — | — | — | — | — | **0.1486** |
| 216 | **0.1958** | structured / global_explore | `CC1=NC=CCC=C1CC(C)C(=O)N=c1[nH]c2ccc(NC(=O)COc3cccc4c3N4N)cc2s1` (0.1837) | atom_delete+atom_insert+cycle_close | 16 | 1.00 | **−1 (36→35)** | 0 | **0.1089** |

Molecules asked for:

* **call 1** `CC1CCC(C)(C(=O)N2CCCC(C)(C(=O)[O-])C2)O1` — 0.0339
* **best initialization (call 6)** `CC(C)n1cc(S(=O)(=O)[N-]c2ccccc2-c2ncnn2C)cn1` — 0.1897
  (this one *does* carry a pyrazole and an aryl sulfonamide)
* **midpoint (call 125)** `Cc1ncccc1CCC(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1` — 0.1884
* **best (call 216)** `Cc1ncccc1CC(C)C(=O)N=c1[nH]c2ccc(NC(=O)COc3cccc4c3N4N)cc2s1` — 0.1958.
  Against its measured parent (call 172) this is a **one-heavy-atom** change:
  `CC1=NC=CCC=C1…` → `Cc1ncccc1…`, i.e. aromatizing the pendant dihydropyridine. The run's
  single controller improvement is a micro-move, not a structural jump.

Bemis-Murcko scaffolds:

| molecule | scaffold |
|---|---|
| best initialization (call 6) | `O=S(=O)([N-]c1ccccc1-c1ncn[nH]1)c1cn[nH]c1` |
| parent (call 172) | `O=C(CCC1=CCC=CN=C1)N=c1[nH]c2ccc(NC(=O)COc3cccc4c3N4)cc2s1` |
| **best (call 216)** | `O=C(CCc1cccnc1)N=c1[nH]c2ccc(NC(=O)COc3cccc4c3N4)cc2s1` |
| *celecoxib itself* | `c1ccc(-c2ccnn2-c2ccccc2)cc1` |

The initialization molecule at call 6 is the only one in the whole run whose scaffold is
even the right *kind* of object (a sulfonyl-linked biaryl carrying two azoles). The
controller then left that basin entirely and never returned.

Ancestry (5 generations, **not** monotone — 0.1884 → 0.1761 → 0.1837 → 0.1958):

```
call   9  0.1694  initialization  CC(C)C(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1           HA 26
call 125  0.1884  structured   9p Cc1ncccc1CCC(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1     HA 32
call 145  0.1761  shallow     10p CC1=NC=CCC=C1CCC(=O)N=c1[nH]c2ccc(NC(=O)COc3ccccc3)cc2s1 HA 33
call 172  0.1837  shallow     17p CC1=NC=CCC=C1CC(C)C(=O)N=c1[nH]c2ccc(...N4N)cc2s1        HA 36
call 216  0.1958  structured  16p Cc1ncccc1CC(C)C(=O)N=c1[nH]c2ccc(NC(=O)COc3cccc4c3N4N)cc2s1 HA 35
```

### 2.3 The headline fact both tables share (MEASURED)

`best_after_16 == best_after_32 == best_after_64 == best_after_125` for **both** tasks.
The best molecule of each 250-call run was found inside the **16 charged initialization
molecules**; 234 controller proposals then bought **+0.0210 (perindopril, at call 240)**
and **+0.0061 (celecoxib, at call 216)**.

---

## 3. Celecoxib: is it closing on the target, or sitting in unrelated chemistry?

**Answer: sitting in unrelated chemistry. It never makes the required scaffold jump.**

Celecoxib is `Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1` — 26 heavy atoms,
3 aromatic rings, Bemis-Murcko scaffold `c1ccc(-c2ccnn2-c2ccccc2)cc1`.

Structural piece census over **all 250 charged molecules** (MEASURED):

| piece | count / 250 |
|---|---|
| 1,5-diarylpyrazole core (the target scaffold, as SMARTS) | **0** |
| N-aryl pyrazole | **0** |
| pyrazole ring (any) | 2 (both initialization-era) |
| primary sulfonamide `NH2-SO2-` | **0** |
| benzenesulfonamide | **0** |
| any sulfonamide | 1 |
| trifluoromethyl | **0** |
| tolyl | 9 |
| molecules carrying celecoxib's Murcko scaffold | **0** |

Distance trajectory per 50 calls (MEASURED):

| calls | max charged score | mean Tc(count) | **mean Tc(2048-bit)** | **max Tc(2048-bit)** | mean HA | mols with pyrazole |
|---|---|---|---|---|---|---|
| 1–50 | 0.1897 | 0.0721 | 0.0610 | **0.1486** | 17.4 | 2 |
| 51–100 | 0.1552 | 0.0647 | 0.0553 | 0.1231 | 13.2 | 0 |
| 101–150 | 0.1884 | 0.0797 | 0.0492 | 0.1158 | 17.8 | 0 |
| 151–200 | 0.1837 | 0.0738 | 0.0587 | 0.1273 | 16.8 | 0 |
| 201–250 | **0.1958** | 0.0860 | 0.0643 | 0.1273 | 15.5 | 0 |

Sharpest form of the same fact (MEASURED): the only two molecules in the entire run
carrying a pyrazole are **initialization** molecules at calls 4 and 6, and the only one
carrying a sulfonamide is the initialization molecule at call 6. **In 234 proposals for a
target that is a sulfonamido-diarylpyrazole, the controller generated neither a pyrazole
nor a sulfonamide, not once.**

The closest structural approach to celecoxib happened in the **first 50 calls** and was
never beaten. The single controller improvement (call 216) *raised* the count-fingerprint
score by +0.0061 while *lowering* the bit-vector similarity from 0.1486 to 0.1089 — it
gained by accumulating fragment multiplicity, not by acquiring the missing pharmacophore.

Mechanism (MEASURED): across 886 `atom_insert` actions in the scored programs, fluorine
is 1.0% and sulfur 0.2%. The three missing pieces are each *coordinated* multi-primitive
motifs (a CF3 = three F on one carbon; a primary sulfonamide = S + 2 O + N; a pyrazole =
a cycle_close joining two adjacent ring nitrogens), while the autonomous programs run a
median of 7 primitives dominated by carbon insert/delete.

---

## 4. Perindopril MPO component decomposition

`perindopril_mpo` is *not* "make perindopril". Perindopril itself has **0 aromatic rings**
and the benchmark demands **2**, so the objective is intrinsically in tension.

The aromatic-ring term is a Gaussian on an **integer**: it takes only
**1.0** (exactly 2 rings), **0.1353** (1 or 3), **0.00034** (0 or 4). It is a cliff, not a
gradient.

Top molecules (MEASURED, all components exact):

| call | score | **ECFP4 similarity** | ring term | #arom | HA | lane |
|---|---|---|---|---|---|---|
| 240 | 0.4865 | **0.2366** | 1.0000 | 2 | 32 | structured |
| 242 | 0.4793 | 0.2297 | 1.0000 | 2 | 39 | structured |
| 205 | 0.4714 | 0.2222 | 1.0000 | 2 | 33 | shallow |
| 244 | 0.4704 | 0.2213 | 1.0000 | 2 | 28 | structured |
| 249 | 0.4695 | 0.2205 | 1.0000 | 2 | 30 | structured |
| 12 (init) | 0.4655 | 0.2167 | 1.0000 | 2 | 26 | initialization |

**Every top molecule already sits at ring term 1.0. The bottleneck is the similarity
component, which moved 0.2167 → 0.2366 (+0.0199) over 238 calls.**

Where the budget went (MEASURED): aromatic-ring distribution over all 250 charged
molecules is `{0: 134, 1: 48, 2: 51, 3: 16, 4: 1}` — **199 of 250 calls (79.6%) were spent
on molecules outside the feasible band**, where the score is capped at ~0.37 of its
similarity potential.

Children split by whether they kept the ring term:

| parent band | children | kept 2 rings | mean Δ when kept | P(improve\|kept) | lost term | mean Δ when lost | P(improve\|lost) |
|---|---|---|---|---|---|---|---|
| ≥ 0.35 (elite) | 46 | 37 (80%) | **−0.0266** | 0.162 | 9 | **−0.3231** | 0.000 |
| < 0.35 | 156 | 7 (4.5%) | **+0.1063** | **1.000** | 149 | −0.0046 | 0.503 |

Reaching 2 aromatic rings from a non-elite parent improved the score **7 times out of 7**,
and was proposed on only 4.5% of those children. Among elite children that *kept* the ring
term, the mean similarity delta is **−0.0215** and the best is **+0.0294** — so even the
feasible edits are, on average, similarity-destructive at the top of the archive.

Control (MEASURED): the best molecule carries an aromatic **N–F**, which is medicinally
implausible. It is **not load-bearing** — the des-fluoro analogue
`CCC(=O)N=c1[nH]c2ccc(NC(=O)COCC3CCC(C4CCCN4O)C3)cc2s1` scores **0.4902**, slightly
higher. The headline survives the control.

Direction of travel on the bottleneck, per 50 calls (MEASURED):

| calls | max score | **max similarity among feasible (2-ring) molecules** | #feasible |
|---|---|---|---|
| 1–50 | 0.4655 | 0.2167 | 3 |
| 51–100 | 0.3743 | 0.1401 | 10 |
| 101–150 | 0.4342 | 0.1885 | 9 |
| 151–200 | 0.4635 | 0.2149 | 11 |
| 201–250 | **0.4865** | **0.2366** | **18** |

Monotone after the first block, exceeding the initialization ceiling in the final block,
with the feasible population growing. **This is the (A) evidence.**

---

## 5. Is recursive optimization happening?

**Yes, structurally — and that is the surprise.** The failure is not an absence of
descendants; it is that the proposal distribution is not conditioned on parent quality,
and that the joint credit cell cannot accumulate evidence.

| statistic | perindopril | celecoxib | gsk3b |
|---|---|---|---|
| generation histogram (gen: n) | 1:32 2:47 3:56 4:49 5:18 6:25 7:4 8:3 | 1:32 2:53 3:61 4:36 5:18 6:14 7:10 8:7 9:3 | 1:32 … 8:13 |
| children whose parent is an initialization molecule | 32 / 234 | 32 / 234 | 32 / 234 |
| distinct parents used | 75 | 71 | 81 |
| mean / max children per parent | 2.69 / 10 | 2.85 / 10 | 2.49 / 10 |
| productive parents (≥1 child beating them) | 48 | 45 | 0 |
| **budget on children of productive parents** | **156 (66.7%)** | **142 (60.7%)** | 0 |
| children that beat their parent | 88 | 76 | 0 |
| …later used as parents themselves | 30 → 97 grandchildren | 23 → 81 grandchildren | 0 |

`P(child improves | parent score bin)` — **monotonically decreasing**, MEASURED:

perindopril:

| parent bin | n | improved | P | mean Δ | best Δ |
|---|---|---|---|---|---|
| [0.00, 0.05) | 122 | 70 | 0.574 | +0.0050 | +0.2170 |
| [0.10, 0.15) | 17 | 7 | 0.412 | +0.0160 | +0.2852 |
| [0.20, 0.35) | 8 | 4 | 0.500 | −0.0497 | +0.1008 |
| [0.35, 0.45) | 28 | 4 | **0.143** | **−0.0926** | +0.0329 |
| [0.45, 1.01) | 18 | 2 | **0.111** | **−0.0723** | +0.0229 |

celecoxib:

| parent bin | n | improved | P | mean Δ | best Δ |
|---|---|---|---|---|---|
| [0.00, 0.05) | 102 | 46 | 0.451 | +0.0076 | +0.1026 |
| [0.05, 0.10) | 46 | 21 | 0.457 | −0.0050 | +0.0754 |
| [0.10, 0.15) | 18 | 5 | 0.278 | −0.0257 | +0.0206 |
| [0.15, 0.20) | 36 | 4 | **0.111** | −0.0202 | +0.0121 |

### 5.1 The hierarchical-credit defect, measured

`credit_key_from_candidate` keys a cell on `(basin_of_the_CHILD, parent_entry_id,
program_family, scale)`. Because the basin is the **child's** Bemis-Murcko scaffold, and
66–67% of scored children change the scaffold, a new child almost always opens a **new
cell**:

| | perindopril | celecoxib | gsk3b |
|---|---|---|---|
| active cells at the end | 198 | 195 | 197 |
| total trials recorded | 202 | 202 | 202 |
| **mean trials per cell** | **1.020** | **1.036** | **1.025** |
| cells with more than one trial | 4 | 6 | 4 |
| credit draws over the run | 152 | 160 | 134 |
| **draws into a cell that had ANY prior evidence** | **2** | **5** | **1** |
| draws into an untried cell | 150 (98.7%) | 155 (96.9%) | 133 (99.3%) |

An untried cell has `value = 0 + prior_weight/sqrt(1) = 0.25`, identically for every
untried cell, so `q_credit` is uniform and `allocate` degenerates to uniform sampling over
the available cells. **In 98% of draws the joint credit steered nothing.** This is the
defect named in the brief, now measured end to end.

### 5.2 Disconfirming evidence for a "no recursion" story

* 60–67% of the charged budget went to children of productive parents.
* The best perindopril molecule sits at the end of a 4-generation monotone chain.
* Cross-task proposal overlap is confined to the two bootstrap rounds: after candidate 117
  the three tasks (same seed, same pool) share **0** proposals at the same position. The
  reward signal *does* reach the proposal distribution.

So the correct statement is **not** "descendants are never exploited". It is: budget
reaches descendants, but the *edit distribution offered to a good parent is the same one
offered to a bad parent*, so it is destructive at the top of the archive, and the credit
mechanism intended to fix that is degenerate.

---

## 6. Lane contribution and attrition

Proposal-side counts come from `campaign/round_*/pending.json → batch.attempts`.

### perindopril_mpo

| lane | proposed | eligible | duplicate | exec-rejected | scored | parent-improvements | global bests | best score |
|---|---|---|---|---|---|---|---|---|
| shallow_program_channel | 370 | 240 | 18 | 112 | 116 | 40 | 1 | 0.4714 |
| structured_program_channel | 412 | 240 | 34 | 138 | 113 | 45 | 1 | **0.4865** |
| joint_dependency_region_jump | 232 | **5** | 0 | **227** | 5 | 3 | **0** | 0.4229 |
| initialization bank | — | — | — | — | 16 | — | **4** | 0.4655 |

### celecoxib_rediscovery

| lane | proposed | eligible | duplicate | exec-rejected | scored | parent-improvements | global bests | best score |
|---|---|---|---|---|---|---|---|---|
| shallow_program_channel | 380 | 240 | 35 | 105 | 117 | 37 | 0 | 0.1931 |
| structured_program_channel | 378 | 240 | 33 | 105 | 114 | 39 | 1 | **0.1958** |
| joint_dependency_region_jump | 259 | **3** | 0 | **256** | 3 | 0 | **0** | 0.1357 |
| initialization bank | — | — | — | — | 16 | — | 6 | 0.1897 |

### gsk3b (descriptive)

| lane | proposed | eligible | exec-rejected | scored |
|---|---|---|---|---|
| shallow | 363 | 240 | 92 | 114 |
| structured | 409 | 240 | 106 | 118 |
| joint jump | 387 | **2** | **385** | 2 |

### 6.1 Decomposing the joint-jump failures (MEASURED by offline replay)

All 227 / 256 / 385 jump failures carry the identical reason string
`joint plan has no legal binding on this parent`, which the controller raises whenever
`bind_joint_plan` returns an empty list — conflating two different events. Replaying 120
sampled failures per task through the production primitives (`enumerate_role_successors`,
`action_role_supervision`), with both directions of the cross-check against the real
`bind_joint_plan` asserted and **0 probe defects**:

| | perindopril | celecoxib | gsk3b |
|---|---|---|---|
| failures with a *different* reason string | 0 | 0 | 0 |
| **A. no compatible realization** (beam emptied) | **120/120** | **120/120** | **120/120** |
| **B. binding discarded** by the dependency-region post-filter | **0** | **0** | **0** |
| **C. execution rejected** after a successful bind | **0** | **0** | **0** |
| died at role **step 0** | 98 (81.7%) | 86 (71.7%) | 93 (77.5%) |
| median fraction of the plan bound before death | **0.0** | **0.0** | **0.0** |
| **rescued by beam width 8 / 16 / 32** | **0 / 0 / 0** | **0 / 0 / 0** | **0 / 0 / 0** |
| blocking rule (atom_delete / atom_insert / restate / cycle_open) | 73/16/10/21 | 77/25/5/13 | 83/19/5/13 |

**The `beam_width=4` versus the function default of 8 is not the cause.** Nothing is
rescued at 8, 16 or 32, because in ~75% of cases the *first* role already has no legal
match. The binder requires an **exact** role identity at every step, where a role carries
element, degree, formal charge, implicit-hydrogen count, bond-class histogram and
neighbour-element histogram.

### 6.2 Parent selection is not the cause either (MEASURED counterfactual)

The lane's median parent carried **8 heavy atoms** (archives are 41–44% ≤10 heavy atoms),
which invited the hypothesis that a 28-primitive plan simply had too few atoms to land on.
Replaying `bind_joint_plan` over stratified parents from the run's own archive, 20 plans
each, 600 binding attempts in total:

| stratum | median HA | median score | attempts | bound @ beam 4 | bound @ beam 8 |
|---|---|---|---|---|---|
| perindopril top-scoring | 32 | 0.4714 | 100 | **3 (3.0%)** | 3 |
| perindopril largest | 39 | 0.3743 | 100 | 0 | 0 |
| perindopril small (as sampled) | 7 | 0.0031 | 100 | 0 | 0 |
| celecoxib top-scoring | 35 | 0.1931 | 100 | 0 | 0 |
| celecoxib largest | 37 | 0.1503 | 100 | 0 | 0 |
| celecoxib small (as sampled) | 7 | 0.0444 | 100 | 0 | 0 |

Aiming the lane at elite, drug-like parents lifts the bind rate from ~0% to **at most 3%**.
**This is a binder/plan-representation problem, not a parent-selection or beam problem.**

---

## 7. Teacher routes versus autonomous proposals

Corpus: `diagnostics/pmo_dependency_region_program_v2/attempt_1/training_dependency_region_corpus.json.gz`,
184 routes over 7 task families. Used as a **counterfactual capability diagnostic**; no
prior was fit from it.

### 7.1 Move scale

| axis | teacher (n=184) | autonomous scored (n=234/task) |
|---|---|---|
| program primitives: median (p10–p90) | **31 (23.6–43)** | **7 (1–13/14)** |
| heavy-atom delta vs the transition source: median (p10–p90) | −1 (−5 … +12) | +1 / +0.5 (−2 … +7/+6) |
| retained fraction (vs program source): median | 1.00 | 1.00 |
| ring delta: median | +1 | 0 |

Autonomous programs reaching the teacher **10th percentile** length (23.6 primitives):
**0/234 (perindopril)**, **1/234 (celecoxib)**. Longest scored programs: 23 and 24.

CAVEAT (INFERRED): teacher route *length* is partly a property of the route compiler's
decomposition, so the trustworthy axis is move-class diversity, not primitive count alone.

### 7.2 Move type

| class | teacher share | autonomous share |
|---|---|---|
| grow (atom_insert) | 0.249 | **0.573** |
| prune (atom_delete) | 0.214 | 0.251 |
| **remodel (bond_reorder / bond_reroute)** | **0.194** | **0.012** |
| **replace (atom_restate)** | **0.183** | **0.023** |
| ring_create (cycle_close) | 0.104 | 0.107 |
| ring_destroy (cycle_open) | 0.044 | 0.025 |
| ring_remodel (ring_system_restate) | 0.011 | 0.007 |

The autonomous vocabulary is **82.4% atom insert/delete**. Bond remodelling is
under-represented **16x** and atom replacement **8x**.

### 7.3 Topology

| | teacher | autonomous scored (perindopril / celecoxib) |
|---|---|---|
| changes ring count | 132/184 (72%) | 106 / 107 of 234 (45%) |
| changes Bemis-Murcko scaffold | 182/184 (99%) | 155 / 157 of 234 (66–67%) |

Topology *is* being touched autonomously. The gap is coordination, not blindness.

### 7.4 Sequence

| | teacher | autonomous |
|---|---|---|
| mean distinct move classes per program | **5.66** | **2.10** |
| multi-component routes | 80/184 | mean 1.56 scheduled blocks |

By family, the two families matching this run's tasks:

| family | routes | median prims | median ΔHA | ring change | scaffold change | move classes |
|---|---|---|---|---|---|---|
| **mpo** | 16 | **40** | +11 | 0.88 | 1.00 | 5.69 |
| **rediscovery** | 45 | **35** | +3 | 0.67 | 1.00 | 5.67 |

**Yes — success in the teacher corpus requires several coordinated changes before reward
improves.**

### 7.5 The four questions, per task

**Can COMPOSE represent the transition?**
Partly, and the exception is decisive. All **184/184** teacher routes replay exactly
(execution-level: yes). But only **106/184** are `complete_representation_supported`; the
other 78 abstain with `primitive_budget_exceeded` against `MAXIMUM_PRIMITIVES = 32`, and
the plan bank is fit **only** on the 106. Broken out:

| family | representable | abstained | abstained route lengths |
|---|---|---|---|
| **mpo** | **0 / 16** | 16 | 38–67 |
| **rediscovery** | 15 / 45 | 30 | 33–43 |
| similarity | 30 / 30 | 0 | — |
| bioactivity | 23 / 48 | 25 | — |
| druglikeness | 18 / 25 | 7 | — |
| multi_property, formula | 20 / 20 | 0 | — |

So for `perindopril_mpo` the jump bank contains **zero** plans from its own task family,
and for `celecoxib_rediscovery` it holds only the 15 shortest rediscovery routes while the
30 longest — the real scaffold jumps — are excluded by the primitive budget.

**Does autonomous COMPOSE ever propose anything comparable?**
Yes, by design: the bank holds 95 plans, median 28 primitives, 87/95 in the "large"
(>15) band, and the lane attempted 232 / 259 / 387 bindings per task. So teacher-scale
programs *are* proposed. Outside that lane, essentially never: 0–1 of 234 scored programs
reach the teacher p10 length.

**Can the binder realize it on the current parent?**
Almost never — 5 / 3 / 2 realizations out of 232 / 259 / 387 (2.2% / 1.2% / 0.5%), and the
replay above shows 100% of sampled failures are "no compatible realization", 72–82% dying
at the very first role, with no rescue at any beam width and at most a 3% rate even on
elite parents. Note also that the 5 and 3 programs that *did* bind carried 14–23
primitives, i.e. the short tail of the bank, not its median of 28.

**Does FiberControl ever evaluate/exploit it?**
It evaluates all of them — the lane's quota is 1 slot per round and it usually cannot fill
it. Of the 10 jump candidates ever scored across three tasks, **0 produced a global best**,
and the best jump molecule scored below the best initialization molecule in both scientific
tasks (0.4229 vs 0.4655; 0.1357 vs 0.1897).

---

## 8. Chemical realism (descriptive)

RDKit-valid but medicinally unusual motifs among the charged molecules:

| motif | perindopril all-250 / top-20 | celecoxib all-250 / top-20 |
|---|---|---|
| aromatic N–F | 18 / **14** | 0 / 0 |
| hypervalent iodine | 19 / 0 | 21 / **6** |
| aromatic phosphorus | 4 / 0 | 1 / 1 |
| three-membered ring | 50 / 1 | 70 / **8** |

For perindopril this is benign — the des-fluoro control in §4 shows the N–F is not
load-bearing. For celecoxib, 6 of the top 20 contain hypervalent iodine and 8 contain a
strained three-membered ring (typically a benzo-fused 1-aminoaziridine), so part of the
celecoxib top-of-list is the fingerprint metric rewarding exotic but unmakeable structures.

---

## 9. What follows

MEASURED implications, in priority order:

1. **Do not buy 750 more v1 calls for celecoxib.** Five independent measurements
   (zero target pieces, zero target scaffold, falling max bit-similarity, a best
   improvement that moved structurally away, and a 1.2% element rate for the missing
   pharmacophore atoms) all say the search is in unrelated chemistry.
2. **Perindopril budget is genuinely unfinished** — 2 of 5 global bests landed in the last
   50 calls, the bottleneck component is still rising, and the feasible population is
   still growing. (INFERRED: a linear extrapolation of that rate is not a prediction.)
3. **The credit cell must stop keying on the child's basin.** Keying on the *parent's*
   basin, or dropping the basin axis, is what would let a cell accumulate more than 1.02
   trials and let allocation mean anything.
4. **The jump lane needs plan/binder work, not tuning.** Beam width and parent selection
   are both measured non-causes. Two concrete targets: the `MAXIMUM_PRIMITIVES = 32`
   budget excludes 100% of the mpo family and the 30 longest rediscovery routes; and the
   exact role-identity match kills 72–82% of bindings at step 0.
5. **The proposal distribution is not conditioned on parent quality.** `P(improve)` falls
   from 0.57 to 0.11 as the parent improves, and the mean delta goes negative. For
   perindopril there is a cheap, measured, task-specific version of this: reaching 2
   aromatic rings improved the score 7/7 times and was proposed on 4.5% of eligible
   children.

## 2026-09-20 (T4 fa7_0: `candidate_exhaustion` was an OBJECTIVE miss, and one legal rewrite clears it)

- **The comparator rescues NOTHING, measured on all 15 held-target cells in the pinned
  production env.** The live gate already uses the contract form (`sim < delta or qed <
  QED_MIN or sa > SA_MAX` -> refuse, `t4_fiber_campaign.py:153`), so the documented legacy
  strictness is not what is running. Re-scoring every stored endpoint under both forms gives
  `comparator_only_gain = 0` in **every** cell, and **0** endpoints anywhere sit exactly on the
  QED or SA bound -- those are continuous and never land on a rational bound. **45 fa7_0
  endpoints (6-129 across the panel) sit at similarity EXACTLY 0.6**, because Tanimoto is a
  ratio of small integers, so the comparator is live and load-bearing for similarity alone.
  Reverting similarity to `>` would silently delete those. Check the comparator first, then
  stop -- it is cheap, and here it is a clean negative.
- **fa7_0's real failure: the similarity ball and the QED bound are nearly disjoint.** Of 7,286
  distinct scored endpoints, 1,535 pass similarity and their best QED is **0.598337 (margin
  -0.001663)**; 160 pass QED and their best similarity is 0.5373. SA never binds (best endpoint
  margin +1.72, `first_failing_stage sa = 0`). Report the pairwise intersection, never per-gate
  pass rates -- each bound individually admits over a thousand endpoints.
- **The whole QED deficit is ONE functional group, and BOTH alerts are it.** `QED.properties`
  gives ALERTS=2 on the fa7_0 source AND on its best endpoint; the matching SMARTS are
  `[C&!R]=[N&!R]` and `N=[C&R0][N,n,O,S]`, both firing on the same acyclic **amidine** -- the FA7
  S1-pocket pharmacophore. ALERTS carries QED's largest weight (0.95 of 3.92), so relieving it is
  worth **x1.41**: measured, the alert-free Pareto points reach QED 0.84-0.90 against the
  leader's 0.598. **Every one of them leaves the delta=0.6 ball** (cyclise the amidine -> sim
  0.319/0.409; amidine -> primary amide -> sim 0.557). Inside the ball the ALERTS lever is
  unreachable and only the small descriptors remain.
- **Inside the ball one `atom_delete` of a peripheral CH2 clears the bound**: propanamide ->
  acetamide, QED 0.598337 -> **0.632753 (+0.034416)**, similarity 0.6724 -> **0.6964 (UP)**, SA
  2.280 -> 2.266, ALERTS unchanged at 2 so the amidine survives. It moves three descriptors at
  once (MW 325.5->311.4, ALOGP 3.91->3.52, ROTB 7->6), two of them QED's next-heaviest weights.
  The similarity RISE is not a general law -- the deleted carbon belonged to a group absent from
  the seed, so its bits were pure mismatch.
- **The trade surface is 5 Pareto points and only ONE holds similarity.** Among SA-passing
  depth-1 successors of the leader: 733 successors, 97 hold similarity AND SA, **7** improve QED
  at all, and exactly **1** improves QED without spending similarity. Unconstrained front max QED
  0.901; constrained to sim >= 0.6 it is 0.633. Quote both or the headroom is fiction.
- **None of the rescues were ever generated.** All four eligible successors of the leader are
  absent from the 9,527 distinct endpoints the cell produced, and `round_one_decision.selected =
  0` means that set IS the campaign's entire output -- it never reached round two. So this is a
  missing TERMINAL LOCAL POLISH step, not a gate rejection and not a support failure: the lanes
  synthesise programs from the docked root and never take one exact edit from a near-miss
  endpoint.
- **SELECT THE PANEL BY VIOLATION, NOT BY WHICH BOUND FAILS -- I got this wrong from fa7_0 and
  braf_1 refuted it.** On fa7_0 the QED-only label is perfectly predictive (**5 fired -> 5 lifted;
  20 did not fire -> 0 lifted**), which is exactly the kind of clean 2x2 that invites a false
  generalisation. On braf_1 the violation-ranked panel lifts **5** endpoints while the QED-only
  panel, handed the top 20 of a **1,404**-endpoint trigger set, lifts **1**. Four of braf_1's five
  rescues are SIMILARITY failures the label never sees.
- **There are TWO repairs under one selector, and they are mechanically different.** fa7_0 is
  objective POLISH: trim a peripheral aliphatic carbon, moving MW/ALOGP/ROTB together. braf_1 is
  proposal-ARTIFACT repair: the lane emitted `c1c2cc(...)cc1-2`, a spurious ring fused across a
  benzene, and ONE `cycle_open` restores the plain ring -- lifting QED 0.6481 -> 0.7574 **and**
  similarity 0.5658 -> 0.6857 simultaneously. So the generator's own defects are locally
  repairable and a bound-specific trigger is blind to them. Worth knowing independently: the
  proposal lanes are manufacturing strained-ring artifacts at measurable rates.
- **CORRECTION, made against my own first rationale.** I justified refining only QED by claiming
  similarity does not move under local editing. False: one rewrite moves fa7_0's best QED-passing
  endpoint from similarity 0.5373 to **0.6129**, clearing delta. The real reason is DISJOINTNESS
  -- of its 953 successors the 331 holding QED >= 0.6 and SA <= 4.0 top out at similarity 0.5606,
  and none are eligible. A bound moving is not a bound being repairable jointly with the others.
- **Count distinct, and classify before quoting a yield.** The fa7_0 panel produced "15 eligible"
  but **14 distinct**, and of those only **4 are conventional drug-like**: the rest are exocyclic
  quinoids (`C=c1ccc(...)cc1=C`) and strained azirines (`C1C=N1`) that pass the benchmark
  thresholds, `med_chem_gate` AND the stricter `LEGACY_SCREENED` instability bundle (15/15
  survive). The structural classification is MY reading, not a repo gate -- report benchmark
  eligibility and chemical plausibility as two numbers, never one.
- **Eligibility is established; docking is NOT.** The rescue converts "returns nothing" into
  "returns something the benchmark accepts". Whether any of it docks well is unmeasured and costs
  one oracle call each. Any number produced this way must be versioned as a distinct rescue phase
  and must never splice into an unchanged-v1 table.
- **Environment parity is exact here, so it is not a caveat.** rdkit **2024.03.5** (production
  pin) vs **2026.03.6** on the five decisive molecules: `max_abs_delta = 0.0` on QED, similarity
  and SA, with identical gate verdicts, and the full test file passes under both. Build the
  pinned env (`uv venv --python 3.11` + rdkit/numpy/scipy/networkx pins) -- add `torch==2.4.0`
  too, because `t4_fiber_campaign` and `generic_legal_action_policy` import it transitively even
  though this path never runs a tensor, and without it you are forced to transcribe the gate
  instead of importing it.
- **Pad to 40, not 48.** `Fiber` refuses `heavy > REPRESENTABLE_HEAVY_ATOMS = 40`, so the T4
  proposal lanes' 48 slots can only manufacture successors the gate always rejects. Measured on
  the leader: tight 24 slots -> 428 marks with **0** `atom_insert`; padded 40 -> 733 marks with
  **305**. A tight graph understates the local support by 42% and deletes atom birth outright.
- **A T4 endpoint can parse in RDKit and be unbuildable in COMPOSE.** `[CH]c1ccc(...)` -- a real
  fa7_0 lane product -- raises `MolecularGraphError` from `smiles_to_molecular_graph` and killed
  the first sweep mid-panel. Radicals are exactly what the `Fiber` structural gate exists to
  remove, so any batch refinement must report them rather than die (`SourceNotRepresentable`);
  3 of 25 fa7_0 panel endpoints and 4 of 25 fa7_2 were unrepresentable.

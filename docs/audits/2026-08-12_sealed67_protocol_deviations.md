# Sealed 67-pair panel — protocol deviations and bookkeeping defects

Full disclosure for the reproducibility appendix. Neither item changed the
controller or any reported outcome, and both are mechanically checkable from the
repository. They belong in an appendix, not the main text.

Result: `diagnostics/editing_v2_sealed67_result.json`
Protocol: `diagnostics/editing_v2_sealed67_preregistration.json`
Amendment: `diagnostics/editing_v2_sealed67_amendment.json`

---

## D1 — The primary denominator was corrected before any outcome was evaluated

The dev/sealed split made at commit `571ec9d` was **pair-disjoint only**.
Endpoint-disjoint connected-component splitting was applied to `h_phi`'s
train/validation carve but not to the evaluation panel — the stricter standard
went to the model's data and the looser one to the evaluation, which is
backwards. A pre-outcome integrity audit found 2 of the 67 sealed pairs reuse a
source molecule appearing in the 24-pair development panel used to select the
controller configuration.

**Disposition.** The primary denominator was fixed at the 65 endpoint-clean
pairs and the amendment committed **before the seal was opened**. All 67 were
run and all 67 are reported as a sensitivity analysis. No pair was replaced and
none was dropped after an outcome was seen. The exclusion criterion is
mechanical and outcome-independent: *exclude any sealed pair whose source
endpoint appears in the 24 development pairs.*

**Effect on conclusions: none qualitatively.** 65 vs 67: verified rollout 40 vs
41, `h_phi` top-1 40 vs 40, similarity top-2 37 vs 38, `R_theta` top-1 33 vs 33.
The two excluded pairs are both ones full rollout rescues and `h_phi` top-1 does
not, which is the whole reason `h_phi` gain retention reads 100% on the clean 65
and 93% on the 67. Both numbers are reported.

## D2 — The executed app hash differs from the preregistered app hash

The preregistration records `app_sha256 = 232d66ce…` (commit `3e1172b`). The
app actually executed hashes to `e9f48cbb…` (commit `8d5c987`), because D1's
amendment had to be wired into the runner after the protocol was hashed.

**The entire difference** is `git diff 3e1172b 8d5c987 --
modal_apps/h_phi_verified_hybrid_app.py`: it mounts the amendment JSON into the
image and stamps `excluded_from_primary` on each row so the runner can print how
many pairs are primary. `run_pair`, the six arms, the candidate universe and
every shortlist are byte-identical. The controller as executed **is** the
preregistered controller. Both hashes and this reasoning are recorded in the
`provenance` block of the result JSON.

## D3 — The exclusion flag was dropped from the returned payload

The D2 wiring set `excluded_from_primary` on the task dict, but `run_pair`
rebuilt its result payload from an explicit key list and never copied the field
through. Nothing behavioural read it — which is exactly why the loss was silent,
and also why it cannot have affected any arm. The first analysis pass therefore
printed identical "primary" and "sensitivity" tables, both over 67.

**Disposition.** The split is reconstructed in
`scripts/editing_v2_analyse_sealed67.py` by joining each shard's `index` against
the sealed panel order and then against the committed amendment's
`excluded_pairs`. The join is **asserted endpoint-for-endpoint** before use, so
a silent index shift fails loudly rather than mislabelling a pair. Because the
criterion was committed before any outcome existed, applying it after the fact
is arithmetic, not a choice. The app now emits the field.

---

## What was *not* deviated from

- No arm was added, removed or retuned after the preregistration. Six arms, as
  declared: greedy, full verified rollout, similarity top-1/top-2, `h_phi`
  top-1, `R_theta` top-1.
- No `K` was changed, no threshold reselected, no model retrained, no pair
  rerun. The stopping rule — "whatever happens is the answer" — held.
- `h_phi` weights are frozen at `12e66467…` and the panel was sealed at
  `571ec9d`, before any `h_phi` work existed.

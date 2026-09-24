# Linker perceived-ring-cell repair

The frozen v1 linker metric pilot stopped after 127 complete attempts and six
locked prompt rows. The started, uncommitted attempt is Liothyronine index 7.
The original v1 manifest, its 127 complete attempts, six row locks and rows,
and the started-attempt RNG record remain immutable.

Exact replay of the saved pre-attempt RNG state found the failure at offered
draw 4. The rooted connector `[1*]N1C2CNCC1C2[2*]` has graph cycle rank 2 but
RDKit symmetrized perceived-ring count 3. The two supplied cores contribute
one graph cycle and one perceived ring. The assembler produced a valid 21-atom
endpoint with graph cycle rank and perceived-ring count 3. The v1 sampler
expected perceived-ring count 4 by adding the connector's pre-assembly
perceived count. That additivity assertion is false for this bridged structure.

The narrow repair converts this one exact-cell mismatch into a typed, recorded
proposal refusal. It consumes the offered draw and records planned and actual
cells. It does not accept the valid molecule under an unplanned cell, resample
the connector, change the learned prior or change chemical execution. Other
proposal behavior remains unchanged.

The v2 measurement is an explicit new revision under
`configs/fragment_training_linker_metric_pilot_v2.json`. It imports 127
byte-verified v1 attempts and six locked metric rows. It replays the one
interrupted started attempt from its saved pre-attempt RNG state under the v2
rule, then performs only the remaining 72 attempts. The old failure and all
v1 artifacts remain visible. A new v2 interrupted start fails closed and is
not automatically redrawn. All ten prompt rows retain 20 attempted outputs
and exactly eight offered draws per attempt. The same official evaluator and
original published-comparator arithmetic apply. This is an operational
correctness repair, not an independent replicate or a new fragment model.

Acceptance before launch: frozen v1 manifest and all 127 attempts physically
verify; old sampler bytes can be recovered from the v1 source commit; the
new mismatch regression and existing sampler tests pass; the v2 manifest is
sealed from a clean source revision. Acceptance after run: all 200 attempts
accounted for, ten prompt locks, old imported attempts byte-identical, exact
core/path fidelity and output coverage gates unchanged, and every mismatch
represented as a consumed refusal rather than a crash or silent relabel.

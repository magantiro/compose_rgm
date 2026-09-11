# Evolving donor memory: first comparison positive, replication required

Six synchronous rounds completed from identical starts (best 0.6495190528).
The evolving arm reached **0.6720215050**, fixed memory **0.6580812003**.
Archive top-ten means were 0.6502614449 and 0.6482515168 respectively. This
passes the declared first-run best-score criterion, not a reproducible or
external-superiority claim.

Cost: 109 new physical PMO queries, 619.577 seconds wall, 558.892 proposal
seconds, 1981.904 summed worker-seconds over 176 distinct workers, 0.107 oracle
seconds. Historical accounting retains 249455 prescreen and 732 prior development
calls (841 after this run). Deployment took 89.413 seconds separately. The run
used no GPU and respected the maximum of 29 proposal containers plus one driver.
The $10 cap is a reserved authorization, not an actual billing measurement.

The initial 116-donor memory stayed fixed in the control; treatment memory sizes
before successive rounds were 116, 128, 141, 156, 170 and 182. Ten draws chose
new donor molecules, six compiled, and **all six decreased the parent's score**.
Do not claim that newly evolved donor components directly caused the improvement.
The treatment's altered population may have helped indirectly or the advantage
may be seed-dependent. Its direct winning path used the unchanged broad reference:
0.649519 scaffold-replacement descendant, then one-step hydroxyl addition to
0.652960, then three generic edits opening a ring to 0.672022. The last edit
sequence has intended release 0.025, heavy-atom delta -1 and cycle-rank delta -1.
This supports retaining broad chemistry, including ring opening and refinement,
not pursuing ring additions alone.

Fixed memory returned 76 distinct completed products from 96 attempts; evolving
memory returned 77 distinct products from 78 completed draws and 96 attempts.
Mean pairwise Morgan distances over their full queried archives are 0.731636 and
0.737900. Cycle-rank changes include both additions and removals. Full option
counts, intended/realized changes and donor-draw audits are in report.json.
The largest-changed-region/original-size metric can exceed one after growth; it
is not a bounded fraction of the final molecule and must not be described as one.

Source: clean commit `eda6e8963d7f`, run
`e111b2aac585acaf2c5c8d6338599f6a53856db9e3c5a7132b9cd77e53292189`, call
`fc-01M291A3D6D26JKTAH17S4CCPB`. Raw receipts remain on
`compose-v4-artifacts/pmo_evolving_memory/<run>/`. Local raw result, source
snapshot and launch receipt are in `/private/tmp/compose-pmo-evolving-memory-runs/<run>/`.
The report binds their SHA-256 hashes and verifies source bytes, oracle accounting,
arm-only archive membership, dynamic donor memories and selections, exact
ancestry and parent selection. Numeric replay error was zero. No new oracle
or neural calls were needed for the audit.

Focused checks: ten memory/worker/driver tests passed in 4.10 seconds; the
interruption-extended memory file passed three tests in 3.24 seconds. Nine
related diagnostic/archive tests passed in 3.30 seconds. Touched-code lint and
strict source preflight passed. The modified report reproduced the prior archive
comparison unchanged. No full-suite or milestone-completion claim.

Decision: run the one earned fresh-seed replication from the same 0.649519 starts
and original 116-donor memory, not from this run's winner. Keep the mechanism
caveat above even if the score advantage repeats. Do not compare these warm
best scores with a published no-prescreen PMO AUC.

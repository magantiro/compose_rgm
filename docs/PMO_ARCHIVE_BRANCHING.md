# Archive branching versus forward-only particles

Question: can revisiting successful states improve beyond the current PMO
champion, rather than losing them after one successor draw? Both completed donor
comparisons discovered their best at boundary two, gave it one follow-up draw,
and then lost it from the active population. The reporting archive was not an
active search resource. This identifies an allocation hypothesis, not proof
that an archive-based optimizer will improve scores.

Output and scope: two populations of 16 exact supported molecular states, four
whole-option rounds on perindopril_mpo. Same top-16 distinct exact molecules from
both completed comparisons, starting best 0.6030226891555273. No public winner
endpoints. These are exposed warm development starts, not benchmark initialization.
Both arms retain the same 50/50 donor/reference law, original 100 donors,
WHERE/WHAT/HOW broad options, 64 primitive cap, and unchanged valid-state executor.
No reference-model or controller training. Failures are recorded without retry.

Baseline is current actual-score SMC, beta=10, ESS<N/2 systematic resampling at
nonterminal option boundaries. The new arm instead retains exact states for
every canonically distinct scored candidate it has independently produced.
After each locked score batch, one next-round parent slot is the best archive
molecule. The other 15 are sampled independently from the full archive using
the existing reciprocal-score-rank distribution with 20% uniform exploration.
Canonical ties are deterministic; all exact-state variants are not extra votes.
An unsuccessful proposal does not erase its ancestor from the archive.

Mathematically the optimization state is the scored archive A_t, with
A_{t+1}=A_t union scored offspring. Parent selection is a mixture of a single
elite slot and draws proportional to 0.8*(1/rank)/sum(1/rank)+0.2/|A_t|.
Complete option programs then execute under the same proposal law. This is
archive-based population optimization, not a Doob transform or SMC terminal-law
sampler. It does not guarantee improvement or solve delayed credit. Retaining
a best state is not a positive performance result by itself.

Controls: same initial exact states, same frozen donors, proposal mixture,
primitive support, RNG derivation, four rounds and 16 attempt slots per arm.
Identical parent/slot/proposal tasks share one computation. Each complete batch
is locked before scoring, and every independently requested candidate and failed
program is recorded. Canonical score queries are deduplicated across arms and
compatible history, but one arm cannot use the other arm's unrequested candidates
as parents. No new labels influence proposals still running in the same round.

Accounting: preserve 249455 reported legacy prescreen calls, 194 physical
probe/parity calls, and 305 physical calls from the two completed comparisons.
Shared historical molecules may have fewer unique labels; report both rather
than subtracting actual calls. At most 128 new physical calls and 128 proposal
tasks, 29 one-CPU/8 GiB workers plus one driver (30 total), no GPU or retries,
180-second worker and 900-second driver caps. Maximum CPU reservation 6.65 hours,
$10 reserved cap, expected 3–8 minutes. Reuse compatible exact neural-law caches.
No T4 job runs concurrently with this 30-container allocation.

Primary decision: a new best exceeding 0.6030226891555273 and the concurrent
SMC arm earns a fresh-seed replication before scaling. Top-ten-only improvement
is narrower refinement evidence. Preserving the old best without new gains is
a null result. A null stops this unchanged recipe; report whether it revisited
the incumbent and what those branches produced. Timeout is inconclusive, not
permission to rerun completed work or weaken the task. Report all oracle calls,
proposal time, topology, canonical diversity and parent-selection decisions.

# COMPOSE 48-hour results tracker

**Window opened:** 2026-07-19 18:08 EDT  
**Rule:** no experiment starts without a named failure, a bounded runtime,
reused artifacts, a stop rule, and a result that changes the next decision.

This is a computational-results sprint.  It can produce qualified generators,
benchmark measurements, conditional proof points, lipid data audits, and lipid
smoke samples.  It cannot produce prospective synthesis, formulation, or
in-vivo results within 48 hours; those remain explicit Paper-2 future work.

## Live tracker

| Block | Deliverable | Runtime authorization | Pass / stop rule | Status |
|---|---|---:|---|---|
| U1 | Quotient-correct unconditional continuation from selected step 2,500 | 750 updates; evaluate every 250; no new chemistry compilation | Continue only while validation and 100-sample chemistry improve; stop on any invalid/disconnected state, canonical self-event, non-finite loss, event exhaustion, or two deteriorating evaluations | **Running** — app `ap-Vl2HEZcpNLhyGlTi2Ftv2M`; initial loss 11.6977, family accuracy 58.08% |
| U2 | 100-sample checkpoint audit | One deterministic sample set per materially improved checkpoint | 100% final and intermediate validity/connectivity; zero molecular self-events; report operator, atom, bond, size, QED, ring, and trajectory diagnostics | Queued at updates 250/500/750 when warranted |
| U3 | Unconditional MVP decision | Use U1/U2 evidence; no new long run | Select the first chemically credible checkpoint; FCD at n=100 is diagnostic, not the gate | Queued |
| C1 | Matched QED protocol and valid-rewrite conditional smoke | Engineering/smoke only before U3; bounded small oracle budget after U3 | Same starts, proposals, oracle calls, validity denominator, seeds, and similarity rule for selection and guidance arms | Protocol build may proceed now; benchmark waits for U3 |
| C2 | QED authorization result | Small pilot first; expand only on positive paired signal | Legal-rewrite guidance/recovery must beat unconditional selection under the frozen matched budget without sacrificing validity | Queued after U3 |
| L1 | Lipid dataset registry and audit | Starts immediately when files/links and endpoint schema arrive | Provenance, licenses, deduplication, chemistry coverage, graph size, charge, rings, fragments, assay/formulation separation, and leakage-safe split all recorded | Awaiting user dataset package |
| L2 | Lipid kernel/throughput smoke | Small audited slice only; no corpus-scale cache build without a measured pilot | Exact reachability on eligible molecules and tractable measured examples/s at lipid sizes | Queued after L1; independent of final Paper-1 tables |
| L3 | COMPOSE-Lipid smoke samples | Restricted, non-claim-bearing run after U3 and C2 | 100% validity/connectivity, corpus-matched size/charge/head-linker-tail diagnostics, unique samples, complete provenance | Queued |
| P1 | Paper-1 results package | Populate only measured results | Unconditional MVP + QED proof + retained formal/executor evidence; all unfinished cells remain `XXX`, never invented | In progress |
| P2 | Paper-2 computational package | Populate only measured retrospective/smoke results | Dataset card + chemistry/kernel qualification + lipid generator smoke; prospective panels remain preregistered future work | In progress through L1 planning |

## Current experiment: U1

- **Question:** did the previous run stop learning because its 3,000-step cosine
  schedule decayed the learning rate too early?
- **Only intervention:** selected step-2,500 weights, fresh optimizer, 50-update
  warmup, then a fixed `1e-4` learning rate for 750 updates.
- **Reused exactly:** 50,000-target compiled paths, fixed validation/test batch,
  exact training-support cache, model architecture, rewrite semantics, catalog,
  split, and seed.
- **Decision points:** update 250, 500, and 750.
- **Fallback if it fails:** one matched warm start from the better-trained
  legacy checkpoint with changed ring heads reinitialized.  Do not add a new
  operator, objective, or exhaustive cache build first.

## What is deliberately not blocking progress

- A publication-quality 10,000-sample unconditional table is not required to
  begin the QED proof.
- Perfect GuacaMol fused-ring fidelity is not required to audit lipid data,
  whose ring support will be derived from the actual corpus.
- The sparse unique-state/path-witness compiler is required before another
  large from-scratch or lipid-scale stream, but not for the cached U1 run.
- Paper 2 does not require Paper 1 submission or acceptance.  Our operational
  gate is a credible base editor plus one matched QED proof before expensive
  lipid generation and prospective work.

## Change-authorization card

Before any further scientific or infrastructure change, record:

1. the measured failure and artifact that demonstrates it;
2. whether the change alters semantics, learning, or only representation;
3. pilot size, expected wall time, compute type, and maximum cost;
4. which caches remain reusable and which signature changes invalidate them;
5. the numerical pass criterion and automatic stop criterion; and
6. the downstream decision enabled by either a positive or negative result.


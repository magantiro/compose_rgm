# Four-call PARP utility preparation

This directory seals, but does not authorize or launch, four single-attempt PARP1 docking queries
selected by the preregistered zero-oracle topology-support rule.

- Contract payload: `4f973c171dd9392f23819b362c6b1946d846e712103d8f58b8a2ce29f70a8da6`
- Candidate proposal payload: `41d67dd0026e1e120bf130446845f229efb0ef13b19699d7efb4fef2aa043d4e`
- Calls: exactly four unique candidates, one attempt each
- Retries, replacements, backfill, and replicates: zero
- Calls spent while preparing this directory: zero

Any future launcher must refuse to run unless
`scored_authorization.json` is a valid self-hashed receipt that explicitly names the contract path,
the exact contract payload above, and a four-call ceiling. Evaluator assets must be rehashed before
launch. A failed or timed-out query is terminal and may not be replaced.

The exact required user sentence is:

> I authorize exactly 4 scored docking calls under contract payload 4f973c171dd9392f23819b362c6b1946d846e712103d8f58b8a2ce29f70a8da6, with zero retries, replacements, or backfill.

The resulting evidence, if authorized and collected, is only a prospective candidate-utility
diagnostic. It is not a controller benchmark or a Full-versus-NoDistill causal comparison.

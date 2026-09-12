# PMO complete-plan policy comparison

The updating complete-plan policy did not improve over its frozen-policy control.
Both arms finished at best score `0.6835298930947339`, top-ten mean
`0.6787222096517102`, and mean round-end top-ten mean `0.6781711013900468`.
The current champion is unchanged.

This is a genuine null comparison rather than a failure to update. The learned
actor changed its snapshot after every round and reached mean plan-pool total
variation near 0.13. The final archives contain 12 arm-specific molecules each,
but the elite sets remain tied. The learning arm produced 50 distinct proposal
SMILES and four parent improvements; the frozen arm produced 49 and three. The
learning arm's largest parent gain was only 0.00555, versus 0.24266 for a frozen
proposal from a much weaker parent, and neither arm exceeded the retained
champion.

The run consumed 58 new physical PMO calls and completed in 276.953 seconds.
Proposal work dominated: 243.536 summed round proposal seconds versus 0.066
seconds in the oracle. All 58 oracle receipts completed. The run used 112 CPU
worker tasks, one thread per worker, and no public winning molecule.

This rejects the unchanged endpoint-plan reranking recipe. It does not reject the
executor, broad edit support, local-to-global proposal mixture, or future-aware
control, because the experiment did not use a committor, a learned future value,
twisted sequential Monte Carlo, or an exact Doob transform. The next experiment
must test joint multi-step reachability and delayed credit, not another scale-up
of this actor.

## Zero-oracle failure separation

The post-run audit in `failure_separation.json` examines all 46 recorded
complete-plan pools without adding labels. The pools contain 1,719 product slots
and 890 distinct products, but only 297 slots (17.3%) and 76 distinct products
(8.5%) had scores before this run. Four pool instances contain a pre-run known
parent improvement. They represent three distinct parent structures, and the
behavior policies assign 0.89% mean probability mass to known improving products.
The learning policy selects the known improvement from the meaningful high-score
parent pool in round four, but that 0.674748 product is below the retained elite
set and cannot improve the primary metric.

Counterfactual label coverage is too low to decide whether unselected plans are
mostly bad or merely unmeasured. The lineage evidence is less ambiguous: frozen
and learning continue six and eight scored temporary-loss intermediates,
respectively, and recover zero above the corresponding pre-loss score. Allowing
low intermediates plus endpoint policy updates is therefore insufficient under
the tested continuation allocation. This does not test whether wider,
rollout-valued allocation could recover them.

The next discriminating experiment is a locked pool-prevalence assay on strong
parents. Randomly selected, previously unscored endpoints must be locked before
labels are obtained. If useful endpoints are absent, change the proposal process;
if present but missed, compare endpoint ranking with rollout-derived continuation
value on the same fixed pools. This is a proposed new paid-label milestone, not
authorized by the completed comparison.

Authoritative evidence is `report.json`. It binds deployed commit
`8150257d7d5f2e6d022cd648177b4803cd4c1948`, contract
`f26d61756bf0eb577d33ee21edb7959bfd182cfc0a1a948339de8a4d60fe03d5`, and
raw result SHA-256
`4dca72933d8268408c32d4da9d79acfb76deb6964c63aa48c6bd043154ab2cd1`.
The full result and receipts are stored at
`compose-v4-artifacts/pmo_plan_policy/170ca5944c646076d6adf1be75edd5ba781119eddc23d12fa6d2d11b45510e4b`.

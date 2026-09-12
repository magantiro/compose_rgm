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

Authoritative evidence is `report.json`. It binds deployed commit
`8150257d7d5f2e6d022cd648177b4803cd4c1948`, contract
`f26d61756bf0eb577d33ee21edb7959bfd182cfc0a1a948339de8a4d60fe03d5`, and
raw result SHA-256
`4dca72933d8268408c32d4da9d79acfb76deb6964c63aa48c6bd043154ab2cd1`.
The full result and receipts are stored at
`compose-v4-artifacts/pmo_plan_policy/170ca5944c646076d6adf1be75edd5ba781119eddc23d12fa6d2d11b45510e4b`.

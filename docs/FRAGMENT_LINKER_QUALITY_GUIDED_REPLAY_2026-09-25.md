# Conditional quality-guided linker panel replay

## Frozen question and policy

The quality-headroom diagnostic found selected failures when a saved panel
already contained a distinct endpoint meeting the public benchmark quality
predicate. Measure the effect of an explicit quality-aware selection policy
on exactly those locked development panels. This is a counterfactual selector
replay, not a fresh benchmark evaluation.

At each attempt, inspect only the eight already constructed, model-supported,
canonical-distinct offers. Compute each offer's QED and SA with the pinned
upstream property implementation. Partition offers into three tiers in this
fixed priority order: previously unreturned offers with QED at least 0.6 and
SA at most 4.0; all other previously unreturned offers; and the full panel
when no new offer exists. Sample within the first nonempty tier using the
unchanged softmax of mean native log-mark scores. Count a forced repeat as an
output. Do not redraw offers, increase the panel size, alter the executor,
change the learned reference, or consult another property or task result.

Replay 16 SHA-derived selector streams per prompt on the exact panels used by
the strict-unseen replay, verify input hashes and the pinned evaluator, and
score each completed 100-output prompt with the official metrics. The
development screen is mean quality at least 35.0%, uniqueness at least 87.1%,
diversity at least 0.542, and 100% validity and prompt fidelity. The 35.0%
threshold is the historical small development-row quality, not a published
large-sample result. Preserve all per-prompt and per-stream outcomes.

This policy directly consults the benchmark's QED and SA criterion before
returning a candidate. It is a materially different, property-guided
controller. Passing the conditional replay cannot authorize replacing the
running novelty-4 benchmark row or comparing it as an unguided fragment
sampler. A fresh, separately contracted evaluation and explicit disclosure
of the in-loop property computation are required before any manuscript use.

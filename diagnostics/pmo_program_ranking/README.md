# Historical complete-edit ranking check

The unchanged preference model fit 256 complete edits from 156 parents in the
isolated donor probe and first matched comparison. Later runs supplied 247
distinct validation edges; 54 were excluded because a parent or product appeared
in training. Split/exclusions were published before feature construction. The
193 remaining edges are endpoint-disjoint, not scaffold-disjoint. These inspected
development runs are not a sealed test.

On 51 donor edges from 39 new parents, 11/17 predicted improvements were positive
(64.7% precision), recovering 11/15 actual improvements (73.3% recall). In eight
multi-candidate donor parent groups, 10/16 informative pairs were ordered
correctly. Highest-preference choice increased mean selected score by 0.0474
versus uniform choice. That denominator is eight groups, not 51 choices.

The model ranked the earlier 47-step ring-system improvement above its two
alternatives, but below its parent. It did not correctly identify that edge as
an improvement. This is a preference model, not calibrated score or future value.

Fit took 5.062 seconds and batched prediction 11.715 seconds, with zero new
oracle/reference calls. `split.json`, `model.json`, and `report.json` retain input
identities, exclusions, coefficients and predictions. Four focused split/metric/
audit tests passed in 1.93 seconds. The earned fresh comparison's negative result
in `../pmo_program_choice/README.md` prevents treating this check as deployment
success.

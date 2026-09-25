# Oversubscribed QED program-controller diagnostic

The completed v1 two-source run is retained as an immutable negative result.
It charged 25 local QED evaluations per source and arm, but every feedback
round returned exactly eight candidates for eight query slots. The recorded
selection mode was `direct`, so the downstream allocation comparison was
vacuous. The PMO joint-dependency channel reached one scored candidate across
the two sources. These facts motivate a mechanism test, not an outcome-based
change to the QED score or source list.

V2 repeats the same exact Jin source indices 0 and 1, seeds, local QED and
Morgan-Tanimoto definitions, 25-call per-cell ceiling, three rounds and eight
queries per round. The only operational change is a 16-candidate proposal pool
for non-bootstrap rounds. The shared bootstrap still requests eight
candidates. This permits the existing score-blind selector to choose eight
from a larger legal pool and preserves all unselected proposals in the round
receipt. The Dynamic-v2.1 and PMO population arms use the same change.

As in v1, this is a two-source development diagnostic. The score is QED(y) for
similarity at least 0.4 and zero otherwise. Off-floor scored endpoints remain
charged. Neither source result is an independent sample-based GrIDDD
comparison, and no claim about the 800-source benchmark follows from v1 or v2.
Report whether oversubscription actually occurs, whether the PMO jump channel
is proposed and selected, and the resulting QED/similarity outcomes. If the
eligible pool remains at or below eight, report that failure rather than
describing the controller as exercised.

# Qualitative figure — "one molecular history, many preference-dependent futures"

**Artifact status: `DESIGN_ONLY`. Not rendered. No example has been selected.**

## The statement

One realized molecular prefix `x_0 -> x_1 -> x_2 -> x_3`, then five branches
from the **identical** intermediate `x_3` under five preferences, ending at five
distinct endpoints with their property coordinates shown.

This complements the retargeting figure rather than duplicating it. Retargeting
shows *one* alternative future after an unanticipated goal change. This shows
that the set of reachable futures from a realized state is **indexed by the
user's tradeoff**, with no answer molecule anywhere in the picture.

## Why it is honest as a figure

The two arms of the branch differ in **exactly one thing**: the preference
weight. Same frozen `R_theta`, same kernel, same start state `x_3`, same
remaining budget `H - 3 = 3`. That is contrast **P6** in the parity table, and
the figure is the qualitative rendering of a contrast that already isolates one
dimension. A figure whose branches differed in controller *and* preference would
illustrate nothing in particular.

## Panels

**Panel A — the shared prefix.** `x_0 .. x_3` drawn left to right, each a
complete molecule. Annotate each edge with the operator family applied. The
point of drawing all four is that every intermediate **is** a molecule; that is
what makes branching from `x_3` meaningful rather than a decoder trick.

**Panel B — the fan.** Five branches from `x_3`, one per `w in {0.1, 0.3, 0.5,
0.7, 0.9}`, each showing its endpoint structure. Order top to bottom by `w`, so
the visual gradient runs from one objective to the other.

**Panel C — the coordinates.** A 2-D scatter in normalized objective space with:

- the five endpoints, coloured to match their branches in Panel B;
- `x_3` marked, so the movement each preference achieved is visible;
- the frozen utopia `z*` and reference `r` points;
- the achieved nondominated set outlined.

Panel C is what stops Panel B being decorative: a reader can check that the
endpoints really do trade off rather than merely differ.

**Panel D — the honest caveat strip.** A one-line annotation of the source's
endpoint diversity and preference coverage, plus the same two numbers averaged
over the whole held-in panel, so the reader can see whether the chosen example
is typical or flattering.

## Selection rule — binding, and written before any example exists

1. The example is chosen **only after** the quantitative results are frozen.
   Examples illustrate a claim; they do not establish one.
2. The source must be drawn from the **held-in** panel. A held-out source may
   not be used for illustration — it would spend confirmatory data on a picture.
3. The source is selected as the one whose endpoint diversity is **closest to
   the panel median**, not the maximum. A figure built from the best case
   misrepresents the effect, and the median case is the one the caption can
   honestly describe.
4. If the panel median endpoint diversity is low, the figure is **not drawn**.
   A fan that visually separates while the panel does not is a misleading
   picture, and "only qualitative examples work" is already a recorded stop rule
   in the retargeting design.
5. The five branches must come from a single arm (`greedy_pref`), not the best
   arm per preference. Mixing arms across branches would vary controller and
   objective together inside one figure.

## What the caption may and may not say

May say: one realized history; five preferences; five distinct endpoints; the
coordinates achieved; that the process and its parameters were identical across
branches.

May **not** say: that these endpoints are optimal, that they are synthesizable,
that they are medicinally plausible, or that any of them would have been found
by a baseline — no external baseline is qualified in this lane.

## Data needed

Only the held-in smoke output: `states`, `endpoints`, `endpoint_z`,
`action_sequences` per preference, all already recorded per source by
`modal_apps/pareto_control_app.py`. The prefix `x_0..x_3` is the first three
committed states of any one branch — the branches share them by construction
only if the preference is applied from `x_3` onward, so the figure run must use
a **common prefix policy**, exactly as the retargeting protocol did: generate
and commit the prefix first, hash it, and only then inject the preference.

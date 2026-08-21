# Figure 1 image-generation prompts

Send to ChatGPT ONE per message, re-uploading the newest image each time.
Never batch them: each pass can silently drop a constraint fixed by an
earlier one.

| file | what it does | status |
|---|---|---|
| `00_fig1_generate_from_scratch.txt` | full spec, plus an alternate route asking for SVG source code instead of an image | used |
| `01_fig1_edits_1to4.txt` | 1 molecule invariance across panels, 2 the A(x)->S(x) marks column with the merge, 3 the +1/-1/0/ring badges, 4 unify panel tints and R_theta hue | applied |
| `02_fig1_edit5_molecules.txt` | clean up the molecule cartoons | applied, only partly took |
| `03_fig1_edit5b_connectivity.txt` | connect the warm-coloured atoms Edit 5 left floating | **SEND NEXT** |
| `04_fig1_edit6_arrows.txt` | panel (c) arrow semantics, and the 7-marks-to-5-successors arithmetic in (a) | send after 5b |

Send 5b before 6: 6 rewrites every arrow in panel (c), which makes any later
molecule pass riskier.

Three constraints generation keeps dropping, so re-check them after every
pass: the molecules must be identical across all three panels, every circle
must have at least one bond, and two marks must converge on y3.

If those keep breaking, the alternative is `figures/fig1_overview.py`, where
the molecules come from one shared array drawn three times and connectivity
and invariance are properties of the code rather than things to re-verify.

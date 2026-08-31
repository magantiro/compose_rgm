# COMPOSE figure palette

SAMPLED from the installed `figures/fig1_overview.png`, not eyeballed. These are
the values actually in the paper. Match them exactly.

## Grounds and cards

    page ground             #E1E6DB   sage-tinted, NOT cream
    panel (a) card          #F4F4EC   warm off-white
    panel (b) card          #EDF1EF   cool off-white
    panel (c) card          #F8F4ED   warmest cream
    molecule tile           #FAF7F2

The three cards are deliberately DIFFERENT off-whites. Do not flatten them to
one colour; the variation is what stops the figure looking like a template.

## Ink

    teal, primary           #43848B   R_theta, reference, baseline, local
    terracotta, accent      #CE785E   control, future-aware, COMPOSE
    deep terracotta         #C3543B   stars, the single strongest highlight
    sand                    #FAEDD7   h_phi box fill, support/pathwise
    heading navy            #1B3A4B
    body ink                #2B3F4E
    hairline / axis         #D8D4C8

## The rule that makes it work

Teal is the ordinary condition. Terracotta is the future-aware condition and
appears ONLY where that is the subject. In Figure 1 terracotta is confined to
panel (c). Keep it scarce; one strong accent per panel at most.

Bonds are one colour and one width everywhere. Arrow WIDTH carries magnitude;
arrow COLOUR carries which law is acting.

## Figure system

    Figure 1   conceptual framework          fully schematic      DONE
    Figure 2   mechanism and core claim      3 data + 1 schematic
    Figure 3   task results and breadth      data-forward, minimal schematic

Figure 1 is the polished conceptual anchor. Figures 2 and 3 should feel more
empirical, deliberately. If all three read as infographics, a reviewer
subconsciously asks where the hard evidence is.

Do NOT ship default plotting-package styling: no heavy black axes, no saturated
primaries, no rainbow, no glossy markers, no gridlines beyond a faint axis.
Plot cleanly, then restyle to the values above.

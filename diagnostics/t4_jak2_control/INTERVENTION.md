# Can a COMPOSE program be held fixed while one decision is resampled?

Zero oracle calls. 147 structural subgoals from the 77 teacher routes, 0 refused.

The contrastive controller rests entirely on this. A matched bundle is only matched if
`P` and `P^(j<-d'_j)` differ in one structural decision and agree elsewhere.

The substrate already existed: `encode_patch_stream` gives a canonical typed token
sequence, `token_domain` gives each token's frozen legal value fiber, and
`decode_patch_stream` reads a stream back. An intervention is encode, substitute one
token from its own domain, decode. Every token already carries a `factor` naming its
semantic coordinate, so the coordinate system is inherited, not invented.

## Per-coordinate result

| coordinate | tokens | decode alone | independent rate | alternatives/decision |
| --- | ---: | ---: | ---: | ---: |
| `atom_attributes` -- element, charge, hydrogens, degree | 6,971 | 5,644 | **0.810** | 4.8 |
| `bond_attributes` -- bond order | 1,207 | 1,207 | **1.000** | 3.0 |
| `attachments` -- edge presence | 3,118 | 178 | 0.057 | 1.1 |
| `target_topology` -- edge presence | 5,540 | 0 | **0.000** | 1.0 |
| `dependencies` -- dependency parent | 601 | 0 | **0.000** | 18.5 |
| `control` -- `output_count` | 294 | 0 | **0.000** | 31.0 |

FREE: heteroatom/element pattern and bond order can be resampled alone.

COUPLED: ring size and scale (`output_count`), topology, attachment presence, and
dependency structure cannot. The grammar is why -- `output_count` determines how many
later tokens the stream contains, and `edge_presence` determines whether a `bond_order`
token follows it. These need INTERVENTION BLOCKS, a constrained coherent re-decode, not a
token flip. Coupling is reported rather than forced into a fake factorial design.

## Binding is a separate coordinate, and JAK2 has none of it

`attachment_bindings` enumerates where a patch may attach without changing what it is.

| target | subgoals | median legal bindings | share with more than one |
| --- | ---: | ---: | ---: |
| braf | 23 | 2 | **74%** |
| fa7 | 38 | 1 | 32% |
| 5ht1b | 26 | 1 | 31% |
| parp1 | 28 | 1 | 21% |
| **jak2** | 32 | 1 | **0%** |
| all | 147 | 1 | 29.3% |

On JAK2 every teacher patch binds in exactly one place, so the attachment coordinate is
not varyable there at all.

## What this means for a matched bundle

The motivating design was `5-ring vs 6-ring`, `1N vs 2N`, `attachment x vs y`. Measured
against this machinery, on JAK2 only the middle contrast is free. Ring size needs a block
intervention; attachment has no alternatives to offer.

So the contrastive controller is viable but narrower than assumed: a bundle can vary
element identity and bond order freely, and must construct coherent blocks for scale,
topology and dependency. Attachment contrasts are available on BRAF (74%) and should not
be assumed on JAK2.

CAUTION on a near miss: the first run of this reported 0 varyable bindings for all 147
patches, because the census field is `assignments` and the probe read a non-existent
`bindings` attribute through `getattr(..., default)`. A defaulted attribute read returns a
confident, wrong, and entirely plausible number.

---

# Minimal intervention closures: the block generator

Zero oracle calls. Same 147 teacher subgoals.

To intervene on a semantic decision, find the smallest set of decisions whose existence,
domain or value must also change for the program to stay legal, resample exactly those,
and preserve the maximal compatible complement. A free coordinate is a singleton closure
and a coupled coordinate is a larger one, from one rule rather than two mechanisms.

| coordinate | kind | tried | legal | round-trips | complement preserved | binds to source | mean closure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| element | free | 702 | 702 | 702 | 702 | 702 | 1.0 |
| bond_order | free | 294 | 294 | 294 | 294 | 294 | 1.0 |
| scale (`output_count`) | BLOCK | 365 | 281 | 281 | 281 | 281 | **3.6** |
| edge_presence | BLOCK | 147 | 147 | **72** | 147 | 147 | 2.0 |

Closure size 3.6 against 1.0 is the grammar-induced coupling made concrete: a scale
intervention moves the created-atom count plus the roles and bonds it drags, and nothing
else. `complement_preserved` verifies the source context never moved, which is what the
parent-cancellation argument requires.

## The JAK2 gate is met

Scale-block siblings per target (legal and round-trip coincide exactly for scale, so these
are usable counts):

| target | scale siblings | of attempts | edge siblings |
| --- | ---: | ---: | ---: |
| **jak2** | **73** | 81 (90.1%) | 32 |
| parp1 | 71 | 79 (89.9%) | 28 |
| braf | 42 | 54 (77.8%) | 23 |
| fa7 | 58 | 90 (64.4%) | 38 |
| 5ht1b | 37 | 61 (60.7%) | 26 |

JAK2 has the HIGHEST block-intervention rate of all five targets, on the target where the
binding coordinate offers no alternatives at all (0% of patches). The coupled-block
mechanism therefore supplies exactly the contrast JAK2 was missing, and a JAK2 matched
bundle no longer has to rely on element identity alone.

## Defect: half of edge interventions do not round-trip

`edge_presence` is legal on 147 of 147 and round-trips on only 72. Those 75 encode to a
stream that decodes back to a DIFFERENT patch, so the runtime would execute something
other than what was constructed. They are not safe siblings.

ADMISSION RULE: a candidate is usable only if `legal AND round_trips AND
complement_preserved AND binds`. Legality alone is not sufficient and was not sufficient
here. `scale` is unaffected (281 of 281 round-trip), so the JAK2 result stands.

---

# Reference-process support audit

Zero oracle calls. Can any reference process actually PROPOSE the admitted siblings?

| law | median log q | expected draws |
| --- | ---: | ---: |
| `q_legal` (uniform over the grammar fiber), teacher patch | -105.2 | 10^46 |
| `q_legal`, admitted scale sibling | -196.8 | **10^85** |
| `q_block` (closure-conditional on the current patch) | **-3.18** | **24** |

Teacher patches carry a median 79 real decisions at mean fiber width 4.9.

Every admitted sibling has strictly positive mass under `q_legal`: absolute continuity
holds, and there are no literal zeros anywhere. It is still useless. Reaching one specific
patch needs 10^85 draws against a budget of 10^3.

| target | admitted scale siblings | median log q_legal | expected draws |
| --- | ---: | ---: | ---: |
| 5ht1b | 37 | -231.5 | 10^101 |
| braf | 42 | -138.3 | 10^60 |
| fa7 | 58 | -113.0 | 10^49 |
| jak2 | 73 | -184.1 | 10^80 |
| parp1 | 71 | -249.5 | 10^108 |

`q_block` reaches the same sibling in 24 expected draws, probability 0.744 within 32.

CONCLUSION: the exploration floor must be LOCAL IN PROGRAM SPACE. A uniform-over-grammar
component satisfies absolute continuity on paper and cannot be sampled in practice, so it
is not a usable epsilon-component. Conditioning on the current patch and resampling one
minimal closure is both a valid process over the legal fiber and reachable inside a
budget.

The closure was derived to make contrasts identifiable by cancelling parent terms. It is
independently the only tractable broad-support reference. Same object, two reasons.

STRUCTURAL CONSEQUENCE: `q_block` is a kernel over patches rather than a marginal law over
options given G, so the reference process is patch-local and the inner path measure is
over (current patch, intervention) pairs.

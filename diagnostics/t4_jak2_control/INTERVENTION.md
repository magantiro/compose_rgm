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

# T4 actual-sampler proposal-access probe

Decision: **PASS**. Contract `20aa83709c96aec4`.
Zero oracle calls, no model fit, no sampler or compiler change.

## Per-context yield

| Cell | Bootstrap attempts | Distinct proposals | Unique endpoints | Eligible | New proposal each round | Archive attempts | Archive pool |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| jak2_1 | 256 | 249 | 225 | 55 | yes | 320 | 30 |
| braf_1 | 256 | 227 | 179 | 5 | yes | 320 | 43 |
| 5ht1b_0 | 256 | 198 | 159 | 16 | yes | 320 | 14 |
| parp1_0 | 256 | 245 | 217 | 84 | yes | 320 | 32 |
| fa7_0 | 256 | 255 | 240 | 0 | yes | 320 | 10 |

A distinct proposal is a distinct (lane, module families and parameters, endpoint,
status) draw. Two different proposals can legitimately reach the same molecule, so
unique endpoints are reported but the repetition repair is judged on proposals.

Only the bootstrap lane is teacher-free. The archive lane injects one historical
champion for diagnostics, so its numbers are conditional support around a known
good endpoint, never autonomous recovery.

## Declared family support, pooled over the five contexts

| Family | Realized | Attempted |
| --- | ---: | ---: |
| segment_grow | 145 | 154 |
| segment_shrink | 233 | 236 |
| segment_replace | 218 | 221 |
| substituent_delete | 846 | 848 |
| append_ring | 147 | 171 |
| fuse_ring | 162 | 185 |
| functionalize | 251 | 254 |
| carbonyl_insert | 201 | 206 |
| heteroatom_substitute | 231 | 231 |
| bond_reroute | 153 | 189 |
| cycle_open | 175 | 175 |
| cycle_close | 163 | 163 |
| ring_system_restate | 193 | 193 |
| ring_path_remodel | 263 | 263 |
| construct_substituted_ring | 359 | 374 |

## Gate

Exact realization, continued exploration and family support all held.

## Nearest historically scored molecule

| Cell | Lane | Generated | Median nearest | Exact hits | Best score among exact hits | Best score reachable in cell |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| jak2_1 | archive | 30 | 0.800 | 3 | -10.6 | -11.7 |
| jak2_1 | bootstrap | 225 | 0.623 | 26 | -9.8 | -11.7 |
| braf_1 | archive | 43 | 0.831 | 2 | -10.2 | -11.2 |
| braf_1 | bootstrap | 179 | 0.564 | 3 | -8.8 | -11.2 |
| 5ht1b_0 | archive | 14 | 1.000 | 9 | -11.2 | -14.0 |
| 5ht1b_0 | bootstrap | 159 | 0.686 | 14 | -12.1 | -14.0 |
| parp1_0 | archive | 32 | 0.728 | 2 | -9.1 | -14.3 |
| parp1_0 | bootstrap | 217 | 0.588 | 26 | -9.3 | -14.3 |
| fa7_0 | archive | 10 | 0.873 | 2 | -8.0 | -9.7 |
| fa7_0 | bootstrap | 240 | 0.497 | 0 | none | -9.7 |

Similarity is to historically scored endpoints of the same cell. Their docking
scores stay attached to those historical molecules; no generated molecule is given
a value here, and section 7 of the strategy report measured that structural
proximity is not a dependable utility label.

## The starting molecule each cell must stay similar to

| Cell | Heavy atoms | Root QED | Root passes the endpoint gate | Bootstrap eligible |
| --- | ---: | ---: | --- | ---: |
| parp1_0 | 19 | 0.888 | yes | 84 |
| jak2_1 | 22 | 0.712 | yes | 55 |
| 5ht1b_0 | 39 | 0.438 | no | 16 |
| braf_1 | 39 | 0.346 | no | 5 |
| fa7_0 | 32 | 0.284 | no | 0 |

Eligible yield is ordered by the root's own QED across all five cells. The gate
requires QED above 0.6 at the completed endpoint while similarity to this same
root stays above the threshold, so a root far below the gate constrains how much
a bounded edit can move. This is a measured association over five cells, not a
proof that no supported program clears the gate.

## Limits

- Failing to sample an exact historical endpoint inside this budget is not a
  failure criterion and is never a proof of zero support.
- Local RDKit is newer than the pinned benchmark image. This is a support probe,
  not a substitute for a pinned-image result.
- The wall-clock stopping budget is lifted so the attempt count binds; realized
  attempts and elapsed seconds are recorded.

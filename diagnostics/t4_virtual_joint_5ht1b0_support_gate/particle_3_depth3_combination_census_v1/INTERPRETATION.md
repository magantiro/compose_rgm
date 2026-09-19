# Particle-3 depth-three combination census

## Outcome

The sealed attempt_2 miss is a planning-beam miss, not an absence of valid
complete targets in particle 3. Under the actual key-monotone prefix semantics
of commit `5f9d0820`, exhaustive validation of all `C(48,3) = 17,296`
depth-three combinations finds 254 valid STOP occurrences and 189 unique
canonical targets. The incremental proposer validated zero depth-three STOPs.
No valid exhaustive combination appears in the global marginal top 64; the
first appears at rank 4,938.

All three strong-route teacher triples form valid complete STOP targets and
reconstruct their exact endpoints. Their ranks are:

| Route | All-combination marginal rank | Valid-combination rank | Unique-valid-target rank |
|---|---:|---:|---:|
| `2b0aff16...` | 16,329 | 252 | 161 |
| `88e87055...` | 15,421 | 205 | 160 |
| `bbd7db0d...` | 15,923 | 240 | 154 |

The joint marginal log score is `-18.054426015057185` for each teacher triple.
Ranks sort by descending joint marginal score and then by constituent keys.
Thus, valid targets and the known targets are well below the incremental
planning frontier under the current marginal. This diagnostic does not test a
new budget or allocation policy.

## Reconciliation with the prior ad hoc count

The reported ad hoc result of 257 valid STOPs and teacher ranks near 241 to 253
is exactly reproducible only when each combination retains particle selection
order internally. That nonproduction ordering yields 257 valid occurrences,
194 unique targets and teacher valid-combination ranks 253, 247 and 241.

The production incremental planner instead permits a next constituent only
when its key exceeds the preceding key. Its combinations therefore use
monotonically increasing constituent-key order. Because complete-target
instantiation assigns created slots in constituent order, the two orderings are
not equivalent: production order yields 254 valid occurrences, 189 unique
targets and teacher valid-combination ranks 252, 205 and 240. The artifact
records the selection-order result as a separate nonproduction comparison and
uses production prefix semantics for its primary result.

## Scope and reproducibility

This is a separate post-hoc zero-oracle census over the 48 constituents already
selected by binding particle 3. It calls the unchanged private binder and
`_valid_stop_target` from implementation commit `5f9d0820`. It does not call the
proposer planner, realize or compile a target, widen a budget, inject a teacher
target, dock, launch Modal or access a scored artifact. Teacher identities are
joined only after exhaustive validation.

Another collaborator modified the worktree proposer after attempt_2 was
sealed. To prevent that concurrent change from contaminating this diagnostic,
the measured runs imported `src` from an immutable `git archive` of
`5f9d0820`. The runner verifies the loaded proposer SHA-256 is
`fc6ecde5c375abc3a94df6fd4bc9d9096f69789f38e52748a5cb3fe05fbf05c8`
before computing the census.

The commands were:

```bash
mktemp -d /tmp/t4-5ht1b-5f9d0820.XXXXXX
git archive 5f9d08203016b7d978fe604f0aa30db09d5361b3 src | \
  tar -x -C /tmp/t4-5ht1b-5f9d0820.dHcxpV
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
PYTHONPATH=/tmp/t4-5ht1b-5f9d0820.dHcxpV/src:src:. \
/Users/rmaganti/compose_rgm_git/.venv/bin/python \
tools/t4_virtual_joint_5ht1b0_particle3_combination_census.py \
--output diagnostics/t4_virtual_joint_5ht1b0_support_gate/particle_3_depth3_combination_census_v1/result.json
```

An independent second execution reproduced `result.json` byte-for-byte. The
payload SHA-256 is
`8f29bd27cf3339a531be394bb9e455e2d324a7abf8e256e78a5d5de2571ce6b8`;
the physical-file SHA-256 is
`5548cbc3f44ab97110eaa996554a3f40ac665e61ec2737515533d1684fbc2071`.

# Frozen held-prompt test of single-interface release

## Decision frozen before held outcomes

The development-only interface-release pilot
`diagnostics/fragment_interface_release_pilot_v1/summary.json` (SHA-256
`bb2c9fcb1954afa36dc0b92383239fa4d61e4c496cbd4bb737e1cc5fec4145db`)
found a quality/diversity tradeoff. On four development prompts per family,
releasing permanent core-interface restriction after declared-site coverage
raised within-prompt diversity, raised motif uniqueness, but lowered decoration
quality. These development observations motivate exactly one **generic,
constraint-derived** candidate:

- zero declared interfaces: controller remains a no-op;
- exactly one declared interface: retain attachment-first staging and
  redirection, but release the ban on growth from undeclared locked-core atoms
  only after required coverage;
- two or more declared interfaces: retain the old permanent interface
  restriction before and after coverage.

This policy may read only `AttachmentSpec.interfaces` and current coverage.
It may not read task names, drug names, prompt identity, hidden targets, or
outcomes. Strict (`restrict_interfaces=True`) and global release
(`restrict_interfaces=False`) remain frozen controls. Legacy Boolean/default
configuration identities must remain unchanged; the candidate gets its own
distinct serialized controller identity. No other sampler, checkpoint,
executor, region-lock, evaluator, or seed setting changes.

## Held qualification sample

The same pinned fragment environment, checkpoint, manifest, upstream IVG
evaluator, 20 attempts per prompt, and seed 0 from
`docs/FRAGMENT_INTERFACE_RELEASE_PILOT_2026-09-23.md` apply. These prompt/task
pairs were not used in that development pilot:

| Family | Held drugs |
| --- | --- |
| Motif extension | CYCLOTHIAZIDE, ELIGLUSTAT, FUTIBATINIB, LESINURAD, LOVASTATIN, SPIRAPRIL |
| Scaffold decoration | CYCLOTHIAZIDE, ELIGLUSTAT, FUTIBATINIB, LESINURAD, LOVASTATIN, SPIRAPRIL |
| Superstructure no-interface control | CYCLOTHIAZIDE, ELIGLUSTAT, ERLOTINIB, FUTIBATINIB, LESINURAD, LIOTHYRONINE |

Run each of these 18 prompt/task pairs once in each of three arms: strict,
global release, and the frozen single-interface policy. This is 360 attempts
per arm, 1,080 attempted trajectories total. Save per-prompt atomic units,
all attempt-aligned committed/emitted molecules, exact seed, events, family
and refusal counters, model and code hashes, official evaluator hashes, and
software identity. Report attempted, committed, chemically valid, exact lock
preservation by path/test, independent fragment containment, covered
interfaces, full task success, uniqueness, quality, and within-prompt
diversity on explicit denominators. Failed prompt builds and no-output attempts
are failures, not exclusions.

## Gates and interpretation

Every committed endpoint must be connected, sanitizable, and fragment-
preserving. Focused tests must show exact locked atom/bond preservation and
allowed-site first growth. The candidate must be attempt-by-attempt identical
to global release on all held single-interface prompts, identical to strict
on all held multi-interface prompts, and all three arms identical on all held
zero-interface prompts. A mismatch is an implementation failure, not a
statistical result.

There is no post-held tuning. Report the numeric effects and any regressions
even if they contradict the development pilot. In particular, distinguish
official quality (unique QED+SA-passing molecules over submitted molecules)
from mean QED/SA and from chemical validity. A held result can qualify the
candidate for a separately frozen full 100-attempt/three-seed official suite,
but cannot itself establish IVG/GenMol superiority. The historical fragment
document's GenMol V1 comparator is not the stronger GenMol V2 comparator; both
must be pinned and labelled separately before a paper comparison. No paid
oracle, PMO/T4 run, training, or live-job modification is authorized here.

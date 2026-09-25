# Scaffold-decoration content-breadth decision

The exact-contract matched pilot finished all 400 planned attempts on the ten
pinned scaffold-decoration prompts. The frozen square-root training-content law
returned 200/200 valid, prompt-faithful molecules, with quality 39.5%,
uniqueness 98.0%, and diversity 0.559613. Uniform training-content sampling
within the same structural cells also returned 200/200 valid, prompt-faithful
molecules, with quality 28.0%, uniqueness 99.5%, and diversity 0.602241. The
uniform arm gained 0.042628 diversity and lost 11.5 quality points. It failed
the prespecified maximum two-point quality-loss criterion, so this intervention
is not promoted to the full benchmark. Both arms used the same frozen model,
executor, structural restrictions, joint mass prior, eight-offer panel size,
and official evaluator. Neither arm used QED or SA to generate or select.

The post-lock offer audit identifies a proposal-content loss. The frozen arm
had at least one QED/SA-passing model-supported offer in 126/200 panels, with
534 passing offers among 1,443 supported offers. The uniform arm had a passing
offer in 95/200 panels, with 308 passing offers among 1,301 supported offers.
The frozen selector chose a passing offer 83 times, and the uniform selector
did so 57 times. Those raw passing counts exceed the official unique-passing
quality counts, which are 79 and 56, respectively. The selector improved
passing yield over each arm's offer-level prevalence, but more diverse content
proposals left it fewer quality-passing choices. This is a post-lock
diagnostic, not a proposal-time quality policy.

The quality decline was not uniform across prompts. Uniform content maintained
or improved quality on Baricitinib and Lesinurad, but lost 25 points on
Erlotinib, 30 on Futibatinib, 40 on Liothyronine, and 20 on Maribavir. The
three zero-quality prompts Cyclothiazide, Lovastatin, and Spirapril remained
zero in both arms. The completed result does not establish that a higher
selector temperature or uniform proposal mixture would improve the official
quality-diversity tradeoff. A future intervention should preserve the
training-content coherence that generated passing offers and add structural
breadth without replacing it. Source-coupled content is one separate
train-only hypothesis; its earlier support census also identified a diversity
risk, and it has not passed a scored gate.

The matched pilot is a development comparison at 20 attempts per prompt and
one fresh seed. It does not supply a three-seed, 100-attempt-per-prompt
decoration benchmark row or seed variance. The manuscript row must not be
updated from this failed pilot.

Authoritative pilot summary:
`diagnostics/fragment_content_breadth_pilot_v1/summary.json` (SHA-256
`ac0bd37c289306f1a9cf75a729df598add42a47d5e55cc1b4294c4c1c5882354`).
Post-lock audit:
`diagnostics/fragment_content_breadth_offer_audit_v1/result.json` (SHA-256
`dc850eabb7f2c83eb5582cc19bc70d7c67db5a6fb72129057441e35809841d85`).
The audit hashes the summary, manifest, all 20 prompt rows, locks, 400 attempts,
and its own analysis code. The larger raw pilot directory remains an untracked
local result and must not be deleted or overwritten.

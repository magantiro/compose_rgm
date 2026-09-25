# Matched source-coupled scaffold-decoration pilot

The 60-attempt zero-quality execution smoke found complete output, exact prompt
fidelity, and nearly matched learned-reference support under the frozen and
source-coupled content proposals. It did not measure benchmark quality or
diversity. The scored comparison below tests whether train-source coupling
improves whole-completion quality without collapsing output diversity.

The self-hashed contract is
`configs/fragment_source_coupled_decoration_pilot_v1.json`, payload SHA-256
`a6ecc715ca9fc70900e18d292ab8160f358649ad0e5cf06b7cd4a210cb0c4716`.
Both arms use the ten pinned scaffold-decoration prompts, fresh seed seven,
20 attempted outputs per prompt, eight complete offers per attempt, one CPU
worker, the same RingCore checkpoint, split-first training pendant catalog,
train-only joint mass prior, exact executor, learned panel selector, prompt
checker, and official evaluator. The intervention replaces only pendant content
draws after each attachment-context/size/ring category is chosen. The new law
selects a compatible shared training-source row with probability one half and
otherwise retains independent content draws. If no shared source exists, it
falls back to the independent draw. Every original content combination keeps
positive proposal probability. No QED/SA value is used during proposal or
selection.

The complete comparison contains 200 attempts and 1,600 offers per arm. Every
attempt is recorded, including refusals. All prompt-arm molecule sets are
locked before the official QED/SA evaluator runs. The frozen promotion gate
requires the source-coupled arm to improve matched quality by at least three
percentage points, reach diversity at least 0.56 and uniqueness at least
90%, return at least 190/200 outputs in each arm, and produce only chemically
valid, exact prompt-faithful committed molecules. Failure is retained as a
negative result. Passing this one-seed development test would warrant a
separate fresh-seed full benchmark; it would not itself replace the current
paper row.

`tools/run_fragment_source_coupled_decoration_pilot_v1.py prepare` verifies
input hashes, pinned software, source ancestry, prompt identity, and evaluator
identity, then writes the manifest. `run` additionally requires an exact
authorization receipt with
`{"approved": true, "payload_sha256": "a6ecc715ca9fc70900e18d292ab8160f358649ad0e5cf06b7cd4a210cb0c4716"}`.
An interrupted started attempt is never redrawn. The result is separate from
the completed 71 MB source-coupled execution smoke and the running linker
novelty pilot.

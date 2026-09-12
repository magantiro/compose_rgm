# Shared-program T4 development curriculum

The expanded common library fixes the initial cross-target eligibility failures
without changing the 40-heavy-atom support. No candidate in these preparation
artifacts has a new docking score yet.

| Preparation | JAK2 seed1 | FA7 seed0 | BRAF seed1 | 5HT1B seed0 |
| --- | ---: | ---: | ---: | ---: |
| PARP1-only shared library, attempt_1 | 8 | 0 | 0 | 0 |
| Expanded shared library, attempt_2 | 8 | 8 | 8 | 8 |

Entries count unique eligible locked endpoints, not discoveries of strong binders.
The second preparation used 146 typed programs, 25.06 seconds, 2,304 executor
calls and zero oracle calls. All tasks use the same proposal recipe and shared
library. Public winner routes are development inputs; their scores are not
transferred to other targets. Unavailable local broad-runtime draws are retained
as failures rather than reallocated.

Each attempt stores input hashes, software, configuration, exact traces, complete
attempt ledgers and candidate locks. The frozen scoring manifest is
`configs/t4_program_curriculum_lock.json`: all 32 candidates plus four seed
controls, maximum 36 new docking calls. Scores belong in the separate
`diagnostics/t4_program_curriculum/` result directory when returned.

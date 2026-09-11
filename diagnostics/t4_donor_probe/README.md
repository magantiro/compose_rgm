# T4 transfer: completed negative development result

One PARP1 seed0, delta=0.4 batch from the same eight saved T4 parents. Both arms
retain the broad reference options and unchanged executor. Hybrid additionally
uses the unchanged 50/50 donor proposal recipe tested on PMO. No public winner
injection, predictor or future-value model. The archive has 134 historical
docking attempts; this is an exposed warm-start experiment.

| Measure | Broad reference | Donor/reference hybrid |
| --- | ---: | ---: |
| Attempted options | 16 | 16 |
| Completed options | 16 | 15 |
| Eligible endpoints, including old molecules | 4 | 5 |
| Novel eligible endpoints docked | 0 | 2 |
| New scores | none | -7.5, -7.6 |
| Recorded worker seconds | 336.183 | 185.491 |

Endpoint exclusions overlap: QED rejected 9/16 and 8/15 completed proposals,
similarity 9/16 and 7/15, SA 3/16 and 2/15. No additional med-chem-only rejection.
Seven eligible proposals were already observed. The two new docked molecules
removed three and eleven heavy atoms; neither added a ring. The latter removed
two ring systems. Whole candidate pools did include ring construction, but it
did not survive endpoint eligibility and novelty allocation in this batch.

Fixed controls: seed -7.3/-7.3, incumbent -11.8/-12.3. The incumbent was recorded
at -11.0 in its source run and -9.8 in another diagnostic. Two new repeats do
not precisely characterize this pipeline's noise. Neither new candidate is
competitive with the incumbent in this batch, and the absent baseline docking
pool prevents a meaningful head-to-head docking comparison.

Decision: stop the unchanged one-option T4 transfer recipe, not the broad chemistry.
31/32 completed programs rule out pervasive compiler failure in this batch.
The next hypothesis is coordinated structural editing plus endpoint-property
repair using saved exact states. Endpoint gates were never applied to internal
primitive states; do not describe this result as a pathwise-similarity restriction.

Run `f01772ffc5e8b46e8038de69ae7e09b47e1fde3c6be0ebc976c7e8bedf608166`,
source `85c948083f30`, call `fc-01M28THS9HPKKCW7924H6E6STJ`.
Durable raw result, full primitive traces and candidate lock:
`compose-v4-artifacts/t4_donor_probe/<run>/`.
Six new physical docking attempts, no failures or replacement queries. Deployment
took 114.424 seconds, driver execution 114.772 seconds. Up to 14 one-CPU workers
plus one driver, no GPU. `report.json` records exact inputs, software, locked
configuration, chemistry, controls and per-arm work. It binds the source archive
and rederives the same score-blind candidate allocation. Property arithmetic
has a 1e-12 audit tolerance for ARM/x86 roundoff; molecular identity, eligibility
and allocation remain exact. No full repository suite or milestone completion.

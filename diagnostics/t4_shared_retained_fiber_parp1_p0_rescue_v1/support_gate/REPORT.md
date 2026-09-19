# PARP1-0 route-timeout support gate

**Decision: PASS**

The actual production route expert returned after independently bounding each candidate realization at 10 seconds. This was a local zero-oracle run. It made no docking call, launched no Modal job, and accessed no live campaign.

## Measured result

- Elapsed wall time: 107.690234 seconds
- Returned records: 87
- Realization statuses: `{"committed": 87, "realizer_candidate_timeout": 9}`
- Realized primitive bands: `{"large": 28, "medium": 34, "small": 25}`
- Exact realization precision: 87/87
- Expected endpoint recovery: 3/3

## Gate checks

- PASS: `material_inputs_match_code_revision`
- PASS: `exact_rescue_configuration_used`
- PASS: `full_realization_prefix_accounted`
- PASS: `candidate_timeout_path_exercised`
- PASS: `returned_after_candidate_timeouts`
- PASS: `exact_realization_precision_one`
- PASS: `small_medium_large_programs_committed`
- PASS: `all_three_expected_endpoint_keys_recovered`
- PASS: `zero_oracle_execution`

## Claim boundary

This gate establishes production-path runtime support and exact recovery of three previously identified structural endpoints. It does not measure docking utility, autonomous novelty, or prospective optimization performance.

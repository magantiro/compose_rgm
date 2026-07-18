from compose_v4.rewrite.fuzz import run_compiled_trace_fuzz


def test_randomized_compiled_program_fuzz() -> None:
    report = run_compiled_trace_fuzz(target_commits=250, seed=91)
    assert report.committed_rewrites >= 250
    assert report.exact_forward == report.programs
    assert report.exact_reverse == report.programs
    assert report.canonical_permutation_matches == report.programs

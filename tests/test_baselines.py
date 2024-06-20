"""
Tests for the ipwgml.baselines module.
"""
from ipwgml.baselines import load_baseline_results


def test_load_baseline_results():
    """
    Test that loading of baseline results works as expected.
    """
    results = load_baseline_results("gmi")
    assert "imerg_final_v07b_gmi" in results.algorithm
    assert "gprof_v07a_gmi" in results.algorithm

    results = load_baseline_results("gmi", ["imerg_final_v07b_gmi"])
    assert "imerg_final_v07b_gmi" in results.algorithm
    assert "gprof_v07a_gmi" not in results.algorithm

    results = load_baseline_results("gmi", ["gprof_v07a_gmi"])
    assert "imerg_final_v07b_gmi" not in results.algorithm
    assert "gprof_v07a_gmi" in results.algorithm

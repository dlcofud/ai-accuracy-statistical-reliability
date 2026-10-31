import math
from analyze import analyze_case, monte_carlo


def test_small_case():
    r = analyze_case(18, 20)
    assert r['accuracy'] == .9
    assert math.isclose(r['pvalue'], .206084718948474, rel_tol=1e-10)
    assert math.isclose(r['ci_low'], .698966, abs_tol=.0001)
    assert math.isclose(r['ci_high'], .972134, abs_tol=.0001)
    assert not r['reject']


def test_large_case():
    r = analyze_case(900, 1000)
    assert math.isclose(r['pvalue'], 8.731206511212449e-18, rel_tol=1e-9)
    assert r['reject']
    assert r['ci_high']-r['ci_low'] < .04


def test_simulation_reproducible():
    assert monte_carlo(20, .9, 1000, 29) == monte_carlo(20, .9, 1000, 29)

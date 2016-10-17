import pymc3_wrap as pw
import pymc3 as pm
import numpy as np

def test_AB():
    m = pw.test_AB()
    assert set(pw.m_type_inference(m).values()) == set({((), np.dtype('float64')), ((750,), 'int64'), ((1500,), 'int64')})
    model = pw.m_pymc3(m)
    assert len(model.deterministics) == 3
    assert len(model.free_RVs) == 2
    assert len(model.observed_RVs) == 2
    assert len(model.potentials) == 0
    assert len(model.named_vars) == 5

def test_dark_skies():
    m, trace, t = pw.test_dark_skies()
    model = pw.m_pymc3(m)
    assert len(model.deterministics) == 3
    assert len(model.free_RVs) == 2
    assert len(model.observed_RVs) == 1
    assert len(model.potentials) == 0
    assert len(model.named_vars) == 4
    assert set(pw.m_type_inference(m).values()) == set({((), np.dtype('float64')), ((1, 2), np.dtype('float64')), ((578, 2), np.dtype('float64')), ((578, 2), 'float64')})
    assert len(trace[0]) == 5
    np.testing.assert_allclose(np.mean(t, axis=0), [2312, 1127], rtol=0.01)

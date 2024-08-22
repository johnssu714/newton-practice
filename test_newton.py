import pytest
import numpy as np
from newton import newtons_method

def test_input_validation():
    f = lambda x: x**2 - 4

    with pytest.raises(TypeError):
        newtons_method("not a function", 1.0)

    with pytest.raises(TypeError):
        newtons_method(f, "not a number")

    with pytest.raises(ValueError):
        newtons_method(f, 1.0, epsilon=-1e-6)

    with pytest.raises(ValueError):
        newtons_method(f, 1.0, max_iter=0)

def test_successful_optimization():
    f = lambda x: x**2 - 4
    root, success = newtons_method(f, 1.0)
    assert pytest.approx(root, 0.001) == 2.0
    assert success is True

def test_unsuccessful_optimization():
    f = lambda x: x**4 - x**3 - x
    root, success = newtons_method(f, 0.5)
    assert success is False

def test_warning_large_step():
    f = lambda x: np.exp(x) - 1  # Exponential function

    with pytest.warns(UserWarning, match="Large step detected"):
        root, success = newtons_method(f, x0=10.0)

def test_warning_zero_derivative():
    f = lambda x: x**3

    with pytest.warns(UserWarning, match="Second derivative near zero"):
        root, success = newtons_method(f, x0=0.0)
        assert success is False

def test_convergence_failure():
    f = lambda x: x**3

    with pytest.warns(UserWarning, match="Newton's method did not converge"):
        root, success = newtons_method(f, x0=1.0, max_iter=5)
        assert success is False

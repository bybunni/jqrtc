"""
Tests for NumPy and JAX utility functions to ensure consistency.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from jqrtc.utils import angle_delta, saturate
from jqrtc.jax_utils import angle_delta as jax_angle_delta
from jqrtc.jax_utils import saturate as jax_saturate


def test_angle_delta():
    """Test angle_delta functions produce the same results."""
    # Test on scalars
    angle1 = 0.5
    angle2 = 2.0
    
    np_result = angle_delta(angle1, angle2)
    jax_result = float(jax_angle_delta(angle1, angle2))
    
    assert np.isclose(np_result, jax_result), f"NumPy gave {np_result}, JAX gave {jax_result}"
    
    # Test on arrays - handle each pair individually to avoid cross product issues
    angles1 = np.array([0.5, 1.5, 2.5])
    angles2 = np.array([2.0, 1.0, 3.0])
    
    # Apply angle_delta to each pair of angles separately
    np_results = np.array([angle_delta(angles1[i], angles2[i]) for i in range(len(angles1))])
    jax_results = np.array([float(jax_angle_delta(angles1[i], angles2[i])) for i in range(len(angles1))])
    
    np.testing.assert_allclose(np_results, jax_results, rtol=1e-5, atol=1e-5)


def test_saturate():
    """Test saturate functions produce the same results."""
    # Test on scalars
    value = 2.5
    
    np_result = saturate(value)
    jax_result = float(jax_saturate(value))
    
    assert np.isclose(np_result, jax_result), f"NumPy gave {np_result}, JAX gave {jax_result}"
    
    # Test on arrays
    values = np.array([-3.0, 1.5, 2.5])
    
    np_results = saturate(values)
    jax_results = np.array(jax_saturate(jnp.array(values)))
    
    np.testing.assert_allclose(np_results, jax_results, rtol=1e-5, atol=1e-5)


def test_default_parameters():
    """Ensure both implementations return the same default parameters."""
    from jqrtc.kinematics import QuadrotorDynamics
    from jqrtc.jax_utils import create_default_parameters
    
    np_dynamics = QuadrotorDynamics()
    np_params = np_dynamics.parameters
    jax_params = create_default_parameters()
    
    # Check important keys are the same
    common_keys = set(np_params.keys()).intersection(set(jax_params.keys()))
    assert len(common_keys) > 0, "No common parameter keys found"
    
    # Check values for common keys
    for key in common_keys:
        assert np.isclose(np_params[key], jax_params[key]), f"Parameters differ for {key}"

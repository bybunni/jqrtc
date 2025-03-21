"""
Tests for NumPy and JAX kinematics functions to ensure consistency.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import vmap

from jqrtc.kinematics import quadrotor_kinematics, QuadrotorDynamics
from jqrtc.jax_kinematics import quadrotor_kinematics as jax_quadrotor_kinematics
from jqrtc.jax_kinematics import QuadrotorDynamics as JaxQuadrotorDynamics


def test_quadrotor_kinematics_single(random_state, jax_random_state, random_control_input, 
                                     jax_random_control_input, parameters, jax_parameters):
    """Test that NumPy and JAX kinematics produce the same results for a single quadrotor."""
    # Default dt value to use
    dt = 0.01
    
    # Run NumPy kinematics
    np_result = quadrotor_kinematics(random_state, random_control_input, parameters, dt)
    
    # Run JAX kinematics
    jax_result = jax_quadrotor_kinematics(jax_random_state, jax_random_control_input, jax_parameters, dt)
    jax_result_np = np.array(jax_result)
    
    # Compare results
    np.testing.assert_allclose(np_result, jax_result_np, rtol=1e-5, atol=1e-5,
                              err_msg="NumPy and JAX kinematics produced different results for single quadrotor")


def test_quadrotor_kinematics_batch(random_batch_state, jax_random_batch_state, 
                                   random_batch_control_input, jax_random_batch_control_input,
                                   parameters, jax_parameters):
    """Test that JAX kinematics works correctly in batch mode."""
    # Default dt value to use
    dt = 0.01
    
    # Process each state individually with NumPy
    np_results = np.zeros_like(random_batch_state)
    for i in range(len(random_batch_state)):
        np_results[i] = quadrotor_kinematics(
            random_batch_state[i], random_batch_control_input[i], parameters, dt
        )
    
    # Process in batch with JAX using vmap
    batched_kinematics = vmap(jax_quadrotor_kinematics, in_axes=(0, 0, None, None))
    jax_results = batched_kinematics(
        jax_random_batch_state, jax_random_batch_control_input, jax_parameters, dt
    )
    jax_results_np = np.array(jax_results)
    
    # Compare results
    np.testing.assert_allclose(np_results, jax_results_np, rtol=1e-5, atol=1e-5,
                              err_msg="NumPy and JAX kinematics produced different results in batch mode")


def test_quadrotor_dynamics_class(random_state, jax_random_state, random_control_input, 
                                 jax_random_control_input, parameters):
    """Test that NumPy and JAX dynamics classes produce the same results."""
    # Default dt value to use
    dt = 0.01
    
    # Initialize classes
    np_dynamics = QuadrotorDynamics(parameters)
    jax_dynamics = JaxQuadrotorDynamics(parameters)
    
    # Run dynamics for one step
    np_result = np_dynamics.step(random_state, random_control_input, dt)
    jax_result = jax_dynamics.step(jax_random_state, jax_random_control_input, dt)
    jax_result_np = np.array(jax_result)
    
    # Compare results
    np.testing.assert_allclose(np_result, jax_result_np, rtol=1e-5, atol=1e-5,
                              err_msg="NumPy and JAX dynamics classes produced different results")


def test_quadrotor_dynamics_getstate(parameters):
    """Test that both dynamics class state dimensions are as expected."""
    # Initialize classes
    np_dynamics = QuadrotorDynamics(parameters)
    jax_dynamics = JaxQuadrotorDynamics(parameters)
    
    # Let's define the expected state dimension (typically 12 for quadrotors)
    expected_state_dim = 12
    
    # Create a sample state and check its dimension
    sample_state = np.zeros(expected_state_dim)
    updated_state = np_dynamics.step(sample_state, np.zeros(4), 0.01)
    
    assert len(updated_state) == expected_state_dim, f"NumPy state dimension is {len(updated_state)}, expected {expected_state_dim}"
    
    # Do the same for JAX
    jax_sample_state = jnp.zeros(expected_state_dim)
    jax_updated_state = jax_dynamics.step(jax_sample_state, jnp.zeros(4), 0.01)
    
    assert len(jax_updated_state) == expected_state_dim, f"JAX state dimension is {len(jax_updated_state)}, expected {expected_state_dim}"

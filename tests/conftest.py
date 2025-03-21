"""
Pytest configuration for testing JAX and NumPy implementations.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from jqrtc.kinematics import QuadrotorDynamics
from jqrtc.jax_utils import create_default_parameters

@pytest.fixture
def parameters():
    """Return default parameters for quadrotor simulation."""
    dynamics = QuadrotorDynamics()
    return dynamics.parameters

@pytest.fixture
def jax_parameters():
    """Return default parameters for JAX quadrotor simulation."""
    return create_default_parameters()

@pytest.fixture
def random_state():
    """Generate a random state vector for testing."""
    # Set seed for reproducibility
    np.random.seed(42)
    return np.random.randn(12)

@pytest.fixture
def jax_random_state():
    """Generate the same random state vector as random_state but as a JAX array."""
    # Using the same seed for consistency
    np.random.seed(42)
    state = np.random.randn(12)
    return jnp.array(state)

@pytest.fixture
def random_batch_state():
    """Generate a batch of random state vectors for testing."""
    np.random.seed(43)
    return np.random.randn(10, 12)  # 10 quadrotors, 12 state dimensions

@pytest.fixture
def jax_random_batch_state():
    """Generate the same batch of random states as random_batch_state but as a JAX array."""
    np.random.seed(43)
    state = np.random.randn(10, 12)
    return jnp.array(state)

@pytest.fixture
def random_key():
    """Generate a random JAX key for testing."""
    return random.PRNGKey(42)

@pytest.fixture
def random_control_input():
    """Generate random control inputs for testing."""
    np.random.seed(44)
    return np.random.randn(4)  # 4 motor speeds

@pytest.fixture
def jax_random_control_input():
    """Generate the same random control inputs as random_control_input but as a JAX array."""
    np.random.seed(44)
    control = np.random.randn(4)
    return jnp.array(control)

@pytest.fixture
def random_batch_control_input():
    """Generate a batch of random control inputs for testing."""
    np.random.seed(45)
    return np.random.randn(10, 4)  # 10 quadrotors, 4 motor speeds

@pytest.fixture
def jax_random_batch_control_input():
    """Generate the same batch of random control inputs as random_batch_control_input but as a JAX array."""
    np.random.seed(45)
    control = np.random.randn(10, 4)
    return jnp.array(control)

@pytest.fixture
def reference_position():
    """Generate a reference position for testing."""
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def jax_reference_position():
    """Generate the same reference position as reference_position but as a JAX array."""
    return jnp.array([1.0, 2.0, 3.0])

@pytest.fixture
def reference_velocity():
    """Generate a reference velocity for testing."""
    return np.array([0.1, 0.2, 0.3])

@pytest.fixture
def jax_reference_velocity():
    """Generate the same reference velocity as reference_velocity but as a JAX array."""
    return jnp.array([0.1, 0.2, 0.3])

@pytest.fixture
def reference_acceleration():
    """Generate a reference acceleration for testing."""
    return np.array([0.01, 0.02, 0.03])

@pytest.fixture
def jax_reference_acceleration():
    """Generate the same reference acceleration as reference_acceleration but as a JAX array."""
    return jnp.array([0.01, 0.02, 0.03])

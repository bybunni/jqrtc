"""
Tests for the factory module that creates NumPy or JAX implementations.
"""

import pytest
import numpy as np
import jax

from jqrtc.factory import create_dynamics, create_controller, create_simulator
from jqrtc.kinematics import QuadrotorDynamics as NumpyQuadrotorDynamics
from jqrtc.controller import TrackingController as NumpyTrackingController
from jqrtc.simulator import Simulator as NumpySimulator
from jqrtc.jax_kinematics import QuadrotorDynamics as JaxQuadrotorDynamics
from jqrtc.jax_controller import TrackingController as JaxTrackingController
from jqrtc.jax_simulator import JaxSimulator


def test_create_dynamics():
    """Test that factory creates the correct dynamics implementation."""
    # Test NumPy implementation
    numpy_dynamics = create_dynamics(implementation="numpy")
    assert isinstance(numpy_dynamics, NumpyQuadrotorDynamics)
    
    # Test JAX implementation
    jax_dynamics = create_dynamics(implementation="jax")
    assert isinstance(jax_dynamics, JaxQuadrotorDynamics)
    
    # Test with custom parameters
    params = {"m": 1.0, "g": 9.8, "k": 1.0, "L": 0.25}
    custom_numpy_dynamics = create_dynamics(implementation="numpy", parameters=params)
    assert isinstance(custom_numpy_dynamics, NumpyQuadrotorDynamics)
    assert custom_numpy_dynamics.parameters["m"] == params["m"]
    
    # Test invalid implementation
    with pytest.raises(ValueError):
        create_dynamics(implementation="invalid")


def test_create_controller():
    """Test that factory creates the correct controller implementation."""
    # Test NumPy implementation
    numpy_controller = create_controller(implementation="numpy")
    assert isinstance(numpy_controller, NumpyTrackingController)
    
    # Test JAX implementation
    jax_controller = create_controller(implementation="jax")
    assert isinstance(jax_controller, JaxTrackingController)
    
    # Test with custom parameters
    params = {"m": 1.0, "g": 9.8, "k": 1.0, "L": 0.25}
    pos_gain = 2.0
    vel_gain = 15.0
    custom_numpy_controller = create_controller(
        implementation="numpy", 
        parameters=params,
        position_gain=pos_gain,
        velocity_gain=vel_gain
    )
    assert isinstance(custom_numpy_controller, NumpyTrackingController)
    assert custom_numpy_controller.position_gain == pos_gain
    assert custom_numpy_controller.velocity_gain == vel_gain
    
    # Test invalid implementation
    with pytest.raises(ValueError):
        create_controller(implementation="invalid")


def test_create_simulator():
    """Test that factory creates the correct simulator implementation."""
    # Test NumPy implementation
    numpy_simulator = create_simulator(implementation="numpy")
    assert isinstance(numpy_simulator, NumpySimulator)
    
    # Test JAX implementation
    jax_simulator = create_simulator(implementation="jax")
    assert isinstance(jax_simulator, JaxSimulator)
    
    # Test with custom parameters
    dt = 0.02
    total_time = 10.0
    num_quadrotors = 5
    custom_jax_simulator = create_simulator(
        implementation="jax", 
        dt=dt,
        total_time=total_time,
        num_quadrotors=num_quadrotors
    )
    assert isinstance(custom_jax_simulator, JaxSimulator)
    assert custom_jax_simulator.dt == dt
    assert custom_jax_simulator.total_time == total_time
    assert custom_jax_simulator.num_quadrotors == num_quadrotors
    
    # Test invalid implementation
    with pytest.raises(ValueError):
        create_simulator(implementation="invalid")

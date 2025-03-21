"""
Factory module for creating simulator instances.

This module provides a unified interface to create either NumPy or JAX-based
simulator components, allowing for easy switching between implementations.
"""

from typing import Dict, Optional, Union, Literal

# Import both implementations
from .kinematics import QuadrotorDynamics as NumpyQuadrotorDynamics
from .controller import TrackingController as NumpyTrackingController
from .simulator import Simulator as NumpySimulator

from .jax_kinematics import QuadrotorDynamics as JaxQuadrotorDynamics
from .jax_controller import TrackingController as JaxTrackingController
from .jax_simulator import JaxSimulator

# Type aliases for better documentation
ImplementationType = Literal["numpy", "jax"]


def create_dynamics(
    implementation: ImplementationType = "numpy",
    parameters: Optional[Dict[str, float]] = None
) -> Union[NumpyQuadrotorDynamics, JaxQuadrotorDynamics]:
    """
    Create a QuadrotorDynamics instance of the specified implementation type.

    Parameters
    ----------
    implementation : str
        The implementation to use ('numpy' or 'jax')
    parameters : dict, optional
        Dictionary of quadrotor parameters

    Returns
    -------
    QuadrotorDynamics
        A dynamics instance of the specified implementation
    """
    if implementation == "numpy":
        return NumpyQuadrotorDynamics(parameters)
    elif implementation == "jax":
        return JaxQuadrotorDynamics(parameters)
    else:
        raise ValueError(f"Unknown implementation: {implementation}. Use 'numpy' or 'jax'.")


def create_controller(
    implementation: ImplementationType = "numpy",
    parameters: Optional[Dict[str, float]] = None,
    position_gain: float = 1,
    velocity_gain: float = 10
) -> Union[NumpyTrackingController, JaxTrackingController]:
    """
    Create a TrackingController instance of the specified implementation type.

    Parameters
    ----------
    implementation : str
        The implementation to use ('numpy' or 'jax')
    parameters : dict, optional
        Dictionary of quadrotor parameters
    position_gain : float, optional
        Gain for position error (k1)
    velocity_gain : float, optional
        Gain for velocity error (k2)

    Returns
    -------
    TrackingController
        A controller instance of the specified implementation
    """
    if implementation == "numpy":
        return NumpyTrackingController(parameters, position_gain, velocity_gain)
    elif implementation == "jax":
        return JaxTrackingController(parameters, position_gain, velocity_gain)
    else:
        raise ValueError(f"Unknown implementation: {implementation}. Use 'numpy' or 'jax'.")


def create_simulator(
    implementation: ImplementationType = "numpy",
    parameters: Optional[Dict[str, float]] = None,
    dt: float = 0.01,
    total_time: float = 50,
    num_quadrotors: int = 1
) -> Union[NumpySimulator, JaxSimulator]:
    """
    Create a Simulator instance of the specified implementation type.

    Parameters
    ----------
    implementation : str
        The implementation to use ('numpy' or 'jax')
    parameters : dict, optional
        Dictionary of quadrotor parameters
    dt : float, optional
        Time step in seconds
    total_time : float, optional
        Total simulation time in seconds
    num_quadrotors : int, optional
        Number of quadrotors to simulate (JAX implementation can efficiently handle many)

    Returns
    -------
    Simulator
        A simulator instance of the specified implementation
    """
    if implementation == "numpy":
        return NumpySimulator(parameters, dt, total_time, num_quadrotors)
    elif implementation == "jax":
        return JaxSimulator(parameters, dt, total_time, num_quadrotors)
    else:
        raise ValueError(f"Unknown implementation: {implementation}. Use 'numpy' or 'jax'.")

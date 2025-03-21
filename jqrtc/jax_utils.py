"""
JAX utility functions for quadrotor simulation.

This module contains JAX-based implementations of helper functions used 
across the quadrotor simulation.
"""

from typing import Union, Any
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def angle_delta(p2: float, p1: float) -> float:
    """
    Calculate the angle between two angles on a plane using JAX.

    Parameters
    ----------
    p2 : float
        Target angle in radians
    p1 : float
        Reference angle in radians

    Returns
    -------
    float
        The signed angle difference in radians
    """
    # Convert angles to unit vectors
    v1 = jnp.array([jnp.cos(p1), jnp.sin(p1)])
    v2 = jnp.array([jnp.cos(p2), jnp.sin(p2)])

    # Calculate the determinant to get the sign
    det_sign = jnp.sign(jnp.cross(jnp.append(v1, 0), jnp.append(v2, 0))[2])

    # Calculate the dot product for the angle
    dot_prod = jnp.dot(v1, v2)

    # Ensure dot product is in valid range for arccos
    dot_prod = jnp.clip(dot_prod, -1.0, 1.0)

    # Return the signed angle
    return det_sign * jnp.arccos(dot_prod)


def saturate(input_value: ArrayLike) -> jax.Array:
    """
    Saturate an input value or array using hyperbolic tangent with JAX.

    Parameters
    ----------
    input_value : ArrayLike
        Input value or array to saturate

    Returns
    -------
    jax.Array
        Saturated value or array
    """
    return jnp.tanh(input_value)


def create_default_parameters() -> dict[str, float]:
    """
    Create a dictionary of default quadrotor parameters.

    Returns
    -------
    dict
        Dictionary of default quadrotor parameters
    """
    return {
        "g": 9.8,       # Gravity acceleration (m/s^2)
        "m": 1.2,       # Mass (kg)
        "Ix": 0.05,     # Moment of inertia around x-axis (kg·m^2)
        "Iy": 0.05,     # Moment of inertia around y-axis (kg·m^2)
        "Iz": 0.1,      # Moment of inertia around z-axis (kg·m^2)
        "b": 1e-4,      # Thrust coefficient
        "l": 0.5,       # Arm length (m)
        "d": 1e-6,      # Drag coefficient
        "Jr": 0.01,     # Rotor inertia (kg·m^2)
        "k1": 0.02,     # Damping coefficient for x
        "k2": 0.02,     # Damping coefficient for y
        "k3": 0.02,     # Damping coefficient for z
        "k4": 0.1,      # Damping coefficient for phi
        "k5": 0.1,      # Damping coefficient for theta
        "k6": 0.1,      # Damping coefficient for psi
        "omegaMax": 330,  # Maximum motor speed (rad/s)
    }

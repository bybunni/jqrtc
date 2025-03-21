"""
JAX-based quadrotor kinematics and dynamics simulation.

This module implements the equations of motion for a quadrotor using JAX for
accelerated computation and vectorization capabilities.
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def quadrotor_kinematics(
    s: ArrayLike, omega: ArrayLike, para: Dict[str, float], dt: float
) -> jax.Array:
    """
    Update the state of the quadrotor based on its current state and motor speeds.
    
    JAX-based implementation that can be vectorized with vmap.

    Parameters
    ----------
    s : jax.Array
        Current state vector of shape (12,) containing:
        [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
        where:
        - (x, y, z) is the position
        - (vx, vy, vz) is the velocity
        - (phi, theta, psi) are the Euler angles (roll, pitch, yaw)
        - (vphi, vtheta, vpsi) are the angular velocities

    omega : jax.Array
        Motor speeds vector of shape (4,)

    para : dict
        Dictionary of quadrotor parameters

    dt : float
        Time step in seconds

    Returns
    -------
    jax.Array
        Updated state vector
    """
    # Extract state variables
    x, y, z = s[0], s[1], s[2]
    vx, vy, vz = s[3], s[4], s[5]
    phi, theta, psi = s[6], s[7], s[8]
    vphi, vtheta, vpsi = s[9], s[10], s[11]

    # Thrust and torque transformation matrix
    tr = jnp.array(
        [
            [para["b"], para["b"], para["b"], para["b"]],
            [0, -para["b"], 0, para["b"]],
            [-para["b"], 0, para["b"], 0],
            [-para["d"], para["d"], -para["d"], para["d"]],
        ]
    )

    # Square of motor speeds
    omega2 = omega**2

    # Overall propeller speed
    Omega = -omega[0] + omega[1] - omega[2] + omega[3]

    # Control inputs (thrust and torques)
    u = tr @ omega2

    # Update linear velocities and positions
    vx_new = vx + dt * (
        u[0]
        * (jnp.cos(psi) * jnp.sin(theta) * jnp.cos(phi) + jnp.sin(psi) * jnp.sin(phi))
        / para["m"]
        - para["k1"] * vx / para["m"]
    )
    x_new = x + dt * vx

    vy_new = vy + dt * (
        u[0]
        * (jnp.sin(psi) * jnp.sin(theta) * jnp.cos(phi) - jnp.cos(psi) * jnp.sin(phi))
        / para["m"]
        - para["k2"] * vy / para["m"]
    )
    y_new = y + dt * vy

    vz_new = vz + dt * (
        u[0] * (jnp.cos(theta) * jnp.cos(phi)) / para["m"]
        - para["g"]
        - para["k3"] * vz / para["m"]
    )
    z_new = z + dt * vz

    # Update angular velocities and angles
    vphi_new = vphi + dt * (
        (para["Iy"] - para["Iz"]) / para["Ix"] * vtheta * vpsi
        - para["Jr"] * vtheta * Omega
        + u[1] * para["l"] / para["Ix"]
        - para["k4"] * vphi * para["l"] / para["Ix"]
    )
    phi_new = phi + dt * vphi

    vtheta_new = vtheta + dt * (
        (para["Iz"] - para["Ix"]) / para["Iy"] * vphi * vpsi
        + para["Jr"] * vphi * Omega
        + u[2] * para["l"] / para["Iy"]
        - para["k5"] * vtheta * para["l"] / para["Iy"]
    )
    theta_new = theta + dt * vtheta

    vpsi_new = vpsi + dt * (
        (para["Ix"] - para["Iy"]) / para["Iz"] * vphi * vtheta
        + u[3] * 1 / para["Iz"]
        - para["k6"] * vpsi / para["Iz"]
    )
    psi_new = psi + dt * vpsi

    # Assemble and return the new state
    return jnp.array(
        [
            x_new,
            y_new,
            z_new,
            vx_new,
            vy_new,
            vz_new,
            phi_new,
            theta_new,
            psi_new,
            vphi_new,
            vtheta_new,
            vpsi_new,
        ]
    )


# Vectorized version of quadrotor_kinematics that can handle multiple quadrotors
vmap_quadrotor_kinematics = jax.vmap(
    quadrotor_kinematics, in_axes=(0, 0, None, None), out_axes=0
)


class QuadrotorDynamics:
    """
    A class for simulating quadrotor dynamics with JAX.

    This provides an object-oriented interface to the JAX-based quadrotor kinematics.
    """

    def __init__(self, parameters: Dict[str, float] = None):
        """
        Initialize the quadrotor dynamics with physical parameters.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of quadrotor parameters. If None, default parameters are used.
        """
        # Default parameters if none provided
        if parameters is None:
            from .jax_utils import create_default_parameters
            self.parameters = create_default_parameters()
        else:
            self.parameters = parameters

    def step(self, state: ArrayLike, omega: ArrayLike, dt: float) -> jax.Array:
        """
        Update the quadrotor state for one time step.

        Parameters
        ----------
        state : jax.Array
            Current state vector of shape (12,) or (B, 12) for batch processing
        omega : jax.Array
            Motor speeds vector of shape (4,) or (B, 4) for batch processing
        dt : float
            Time step in seconds

        Returns
        -------
        jax.Array
            Updated state vector of shape (12,) or (B, 12)
        """
        # Check if we're dealing with a batch of states or a single state
        if len(jnp.shape(state)) > 1:
            # Batch mode - use vectorized function
            return vmap_quadrotor_kinematics(state, omega, self.parameters, dt)
        else:
            # Single quadrotor mode
            return quadrotor_kinematics(state, omega, self.parameters, dt)

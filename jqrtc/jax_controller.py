"""
JAX-based quadrotor tracking controller implementation.

This module implements controllers for tracking a reference trajectory
with a quadrotor using JAX for accelerated computation and vectorization.
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .jax_utils import angle_delta, saturate


def quadrotor_controller(
    s: ArrayLike,
    xl: ArrayLike,
    vl: ArrayLike,
    psil: float,
    para: Dict[str, float],
    k1: float,
    k2: float,
) -> jax.Array:
    """
    Compute control inputs to track a reference trajectory.

    JAX-based implementation that can be vectorized with vmap.

    Parameters
    ----------
    s : jax.Array
        Current state vector of shape (12,) containing:
        [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
    xl : jax.Array
        Leader/reference position vector of shape (3,)
    vl : jax.Array
        Leader/reference velocity vector of shape (3,)
    psil : float
        Desired yaw angle
    para : dict
        Dictionary of quadrotor parameters
    k1 : float
        Position gain
    k2 : float
        Velocity gain

    Returns
    -------
    jax.Array
        Motor speeds vector of shape (4,)
    """
    # Extract state variables
    x, y, z = s[0], s[1], s[2]
    vx, vy, vz = s[3], s[4], s[5]
    phi, theta, psi = s[6], s[7], s[8]
    vphi, vtheta, vpsi = s[9], s[10], s[11]

    # Initialize control vector
    u = jnp.zeros(4)

    # Thrust and torque transformation matrix
    tr = jnp.array(
        [
            [para["b"], para["b"], para["b"], para["b"]],
            [0, -para["b"], 0, para["b"]],
            [-para["b"], 0, para["b"], 0],
            [-para["d"], para["d"], -para["d"], para["d"]],
        ]
    )

    # Compute control input based on position and velocity errors
    pos_error = jnp.array([x, y, z])
    vel_error = jnp.array([vx, vy, vz])
    input_delta = k1 * (xl - pos_error) + k2 * (vl - vel_error)

    # Apply saturation to control input
    attr = saturate(input_delta) * 2 + jnp.array([0, 0, para["g"]])

    # Rotation matrix for yaw
    rot = jnp.array(
        [[jnp.cos(psi), jnp.sin(psi), 0], 
         [-jnp.sin(psi), jnp.cos(psi), 0], 
         [0, 0, 1]]
    )

    # Apply rotation
    attr = rot @ attr

    # Total thrust
    u = u.at[0].set(para["m"] * jnp.linalg.norm(attr))

    # Desired roll and pitch angles
    phi_p = jnp.arcsin(-attr[1] / jnp.linalg.norm(attr))
    theta_p = jnp.arctan(attr[0] / attr[2])

    # Adjust theta_p if attr[2] < 0
    theta_p = jnp.where(attr[2] < 0, theta_p - jnp.pi, theta_p)

    # Desired yaw angle
    psi_p = psil

    # Compute torque control inputs
    u = u.at[3].set(para["Iz"] * saturate(angle_delta(psi_p, psi) + 0 - vpsi))
    
    u = u.at[1].set(
        20
        * para["Ix"]
        * (u[0] + u[3])
        / 2
        * saturate(angle_delta(phi_p, phi) + 0 - 1 * vphi)
        / para["l"]
    )
    
    u = u.at[2].set(
        20
        * para["Iy"]
        * (u[0] - u[3])
        / 2
        * saturate(angle_delta(theta_p, theta) + 0 - 1 * vtheta)
        / para["l"]
    )

    # Compute motor speeds from control inputs
    omega2 = jnp.linalg.solve(tr, u)

    # Ensure non-negative values
    omega2 = jnp.maximum(omega2, 0)

    # Convert to motor speeds
    omega = jnp.sqrt(omega2)

    return omega


# Vectorized version of quadrotor_controller that can handle multiple quadrotors
vmap_quadrotor_controller = jax.vmap(
    quadrotor_controller, 
    in_axes=(0, None, None, None, None, None, None), 
    out_axes=0
)


class TrackingController:
    """
    A class for implementing quadrotor tracking control with JAX.

    This provides an object-oriented interface to the JAX-based tracking controller.
    """

    def __init__(
        self,
        parameters: Dict[str, float] = None,
        position_gain: float = 1,
        velocity_gain: float = 10,
    ):
        """
        Initialize the tracking controller with parameters and gains.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of quadrotor parameters. If None, default parameters are used.
        position_gain : float, optional
            Gain for position error (k1)
        velocity_gain : float, optional
            Gain for velocity error (k2)
        """
        # Default parameters if none provided
        if parameters is None:
            from .jax_utils import create_default_parameters
            self.parameters = create_default_parameters()
        else:
            self.parameters = parameters

        self.position_gain = position_gain
        self.velocity_gain = velocity_gain

    def compute_control(
        self,
        state: ArrayLike,
        leader_pos: ArrayLike,
        leader_vel: ArrayLike,
        desired_yaw: float = 0,
    ) -> jax.Array:
        """
        Compute control inputs to track a reference trajectory.

        Parameters
        ----------
        state : jax.Array
            Current state vector of shape (12,) or (B, 12) for batch processing
        leader_pos : jax.Array
            Leader/reference position vector of shape (3,)
        leader_vel : jax.Array
            Leader/reference velocity vector of shape (3,)
        desired_yaw : float, optional
            Desired yaw angle in radians

        Returns
        -------
        jax.Array
            Motor speeds vector of shape (4,) or (B, 4)
        """
        # Check if we're dealing with a batch of states or a single state
        if len(jnp.shape(state)) > 1:
            # Batch mode - use vectorized function
            return vmap_quadrotor_controller(
                state, 
                leader_pos, 
                leader_vel, 
                desired_yaw, 
                self.parameters, 
                self.position_gain, 
                self.velocity_gain
            )
        else:
            # Single quadrotor mode
            return quadrotor_controller(
                state, 
                leader_pos, 
                leader_vel, 
                desired_yaw, 
                self.parameters, 
                self.position_gain, 
                self.velocity_gain
            )

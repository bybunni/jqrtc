"""
JAX-based quadrotor simulation engine.

This module provides a simulator for running quadrotor tracking simulations
with JAX acceleration and parallelization via vmap.
"""

from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import random
from functools import partial

from .jax_kinematics import quadrotor_kinematics, vmap_quadrotor_kinematics
from .jax_controller import quadrotor_controller, vmap_quadrotor_controller


class JaxSimulator:
    """
    A JAX-accelerated class for simulating quadrotor dynamics and control.
    
    This implementation leverages JAX's vectorization capabilities to
    efficiently simulate multiple quadrotors in parallel.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, float]] = None,
        dt: float = 0.01,
        total_time: float = 50,
        num_quadrotors: int = 1,
    ):
        """
        Initialize the JAX simulator with parameters.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of quadrotor parameters. If None, default parameters are used.
        dt : float, optional
            Time step in seconds
        total_time : float, optional
            Total simulation time in seconds
        num_quadrotors : int, optional
            Number of quadrotors to simulate in parallel
        """
        # Default parameters if none provided
        if parameters is None:
            from .jax_utils import create_default_parameters
            self.parameters = create_default_parameters()
        else:
            self.parameters = parameters

        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.num_quadrotors = num_quadrotors

        # Dimension of state space
        self.state_dim = 3  # [x, y, z]

    def initialize_state_random(
        self, key: jax.random.PRNGKey, init_range: float = 0.0
    ) -> jax.Array:
        """
        Initialize the state of the quadrotor(s) with random values.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key
        init_range : float, optional
            Range for random initialization

        Returns
        -------
        jax.Array
            Initial state vector of shape (num_quadrotors, 12)
        """
        # Initialize state vector: [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
        s = jnp.zeros((self.num_quadrotors, 12))

        # Generate 3 more keys for the different random components
        key1, key2, key3, _ = random.split(key, 4)
        
        # Random positions
        s = s.at[:, 0:3].set(
            random.uniform(
                key1, 
                shape=(self.num_quadrotors, 3), 
                minval=-init_range, 
                maxval=init_range
            )
        )
        # Random velocities
        s = s.at[:, 3:6].set(
            random.uniform(
                key2, 
                shape=(self.num_quadrotors, 3), 
                minval=-init_range, 
                maxval=init_range
            )
        )
        # Random orientations
        s = s.at[:, 6:9].set(
            random.uniform(
                key3, 
                shape=(self.num_quadrotors, 3),
                minval=-init_range * jnp.pi, 
                maxval=init_range * jnp.pi
            )
        )

        return s
        
    def initialize_state_zeros(self) -> jax.Array:
        """
        Initialize the state of the quadrotor(s) with zeros.

        Returns
        -------
        jax.Array
            Initial state vector of shape (num_quadrotors, 12)
        """
        # Initialize state vector with all zeros
        return jnp.zeros((self.num_quadrotors, 12))

    def generate_leader_trajectory(
        self, speed: float = 1.0
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Generate a trajectory for the virtual leader.

        Parameters
        ----------
        speed : float, optional
            Speed parameter for the trajectory

        Returns
        -------
        tuple
            (positions, velocities, accelerations) of the leader for each time step
        """
        # Ensure calculations are done with JAX arrays
        speed = jnp.array(speed, dtype=jnp.float32)
        
        # Define accelerations for different time segments
        def get_acceleration(normalized_time, velocity):
            conditions = [
                normalized_time < 0.1,
                (normalized_time >= 0.1) & (normalized_time < 0.2),
                (normalized_time >= 0.2) & (normalized_time < 0.4),
                (normalized_time >= 0.4) & (normalized_time < 0.6),
                (normalized_time >= 0.6) & (normalized_time < 0.8),
                (normalized_time >= 0.8) & (normalized_time < 0.9),
                normalized_time >= 0.9
            ]
            
            choices = [
                jnp.array([0, 0, speed]) - velocity,
                jnp.array([speed, 0, 0]) - velocity,
                jnp.array([0, speed, 0]) - velocity,
                jnp.array([-speed, 0, 0]) - velocity,
                jnp.array([0, -speed, 0]) - velocity,
                jnp.array([speed, 0, speed]) - velocity,
                jnp.array([speed, 0, 0]) - velocity
            ]
            
            return jnp.select(conditions, choices)
        
        # Initialize arrays to store trajectories
        leader_positions = jnp.zeros((self.num_steps + 1, self.state_dim))
        leader_velocities = jnp.zeros((self.num_steps + 1, self.state_dim))
        leader_accelerations = jnp.zeros((self.num_steps, self.state_dim))
        
        # Initial state
        xl = jnp.zeros(3)
        vl = jnp.zeros(3)
        leader_positions = leader_positions.at[0].set(xl)
        leader_velocities = leader_velocities.at[0].set(vl)
        
        # Define the scan function that will generate the trajectory
        def trajectory_step(carry, t):
            pos, vel = carry
            n_time = t / self.num_steps
            acc = get_acceleration(n_time, vel)
            
            # Update velocity and position
            new_vel = vel + self.dt * acc
            new_pos = pos + self.dt * vel
            
            return (new_pos, new_vel), (new_pos, new_vel, acc)
        
        # Generate trajectory using scan
        init_state = (xl, vl)
        _, scanned_outputs = jax.lax.scan(
            trajectory_step,
            init_state,
            jnp.arange(self.num_steps)
        )
        
        # Extract positions, velocities, and accelerations
        positions_scan, velocities_scan, accelerations_scan = scanned_outputs
        
        # Update trajectory arrays
        leader_positions = leader_positions.at[1:].set(positions_scan)
        leader_velocities = leader_velocities.at[1:].set(velocities_scan)
        leader_accelerations = leader_accelerations.at[:].set(accelerations_scan)
        
        return leader_positions, leader_velocities, leader_accelerations
    
    def step_simulation_batch(
        self,
        state: jax.Array,
        leader_pos: jax.Array,
        leader_vel: jax.Array,
        position_gain: float,
        velocity_gain: float,
        desired_yaw: float
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Perform a single simulation step for all quadrotors in parallel using vmap.
        
        Parameters
        ----------
        state : jax.Array
            Current state vectors of shape (num_quadrotors, 12)
        leader_pos : jax.Array
            Leader/reference position vector
        leader_vel : jax.Array
            Leader/reference velocity vector
        position_gain : float
            Position error gain
        velocity_gain : float
            Velocity error gain
        desired_yaw : float
            Desired yaw angle
            
        Returns
        -------
        tuple
            (new_state, omegas) for all quadrotors
        """
        # Compute control inputs for all quadrotors in parallel
        omegas = vmap_quadrotor_controller(
            state, 
            leader_pos, 
            leader_vel, 
            desired_yaw, 
            self.parameters, 
            position_gain, 
            velocity_gain
        )
        
        # Update states for all quadrotors in parallel
        new_state = vmap_quadrotor_kinematics(
            state, 
            omegas, 
            self.parameters, 
            self.dt
        )
        
        return new_state, omegas

    @partial(jax.jit, static_argnums=(0,))
    def run_simulation_with_state(
        self,
        initial_state: jax.Array,
        leader_speed: float = 1.0,
        position_gain: float = 1,
        velocity_gain: float = 10,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Run the full simulation with JAX acceleration using a provided initial state.

        Parameters
        ----------
        initial_state : jax.Array
            Initial state vector of shape (num_quadrotors, 12).
        leader_speed : float, optional
            Speed parameter for the leader trajectory
        position_gain : float, optional
            Gain for position error (k1)
        velocity_gain : float, optional
            Gain for velocity error (k2)

        Returns
        -------
        tuple
            (leader_positions, quadrotor_positions, leader_velocities, omega_histories)
        """
        # Use the provided state
        s = initial_state

        # Generate leader trajectory
        leader_positions, leader_velocities, _ = self.generate_leader_trajectory(leader_speed)
        
        # Initialize history arrays
        quadrotor_positions = jnp.zeros((self.num_steps + 1, self.num_quadrotors, 3))
        omega_histories = jnp.zeros((self.num_steps, self.num_quadrotors, 4))
        
        # Store initial positions
        quadrotor_positions = quadrotor_positions.at[0, :, :].set(s[:, 0:3])
        
        # Define the scan function for the simulation steps
        def simulation_step(carry, t):
            state, _ = carry
            
            # Current leader position and velocity
            xl = leader_positions[t]
            vl = leader_velocities[t]
            
            # Update state and get control inputs
            new_state, omegas = self.step_simulation_batch(
                state, 
                xl, 
                vl, 
                position_gain, 
                velocity_gain, 
                0.0
            )
            
            # Store positions and omegas
            positions_t = new_state[:, 0:3]
            
            return (new_state, omegas), (positions_t, omegas)
        
        # Run simulation using scan
        init_carry = (s, jnp.zeros((self.num_quadrotors, 4)))
        _, scanned_outputs = jax.lax.scan(
            simulation_step,
            init_carry,
            jnp.arange(self.num_steps)
        )
        
        # Extract positions and omegas
        positions_scan, omegas_scan = scanned_outputs
        
        # Update trajectory arrays
        quadrotor_positions = quadrotor_positions.at[1:, :, :].set(positions_scan)
        omega_histories = omega_histories.at[:, :, :].set(omegas_scan)
        
        return leader_positions, quadrotor_positions, leader_velocities, omega_histories
        
    def run_simulation(
        self,
        initial_state: Optional[jax.Array] = None,
        key: Optional[jax.random.PRNGKey] = None,
        leader_speed: float = 1.0,
        position_gain: float = 1,
        velocity_gain: float = 10,
        random_init: bool = False,
        init_range: float = 0.0,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Run the full simulation with JAX acceleration.
        
        This is a non-JIT wrapper function that handles initialization logic
        and then calls the JIT-compiled simulation function.

        Parameters
        ----------
        initial_state : jax.Array, optional
            Initial state vector of shape (num_quadrotors, 12). If None, will be initialized.
        key : jax.random.PRNGKey, optional
            JAX random key for state initialization if random_init is True
        leader_speed : float, optional
            Speed parameter for the leader trajectory
        position_gain : float, optional
            Gain for position error (k1)
        velocity_gain : float, optional
            Gain for velocity error (k2)
        random_init : bool, optional
            Whether to initialize with random positions and orientations
        init_range : float, optional
            Range for random initialization

        Returns
        -------
        tuple
            (leader_positions, quadrotor_positions, leader_velocities, omega_histories)
        """
        # Handle initialization logic outside of JIT compilation
        if initial_state is None:
            if key is None:
                key = random.PRNGKey(0)
                
            if random_init:
                initial_state = self.initialize_state_random(key, init_range)
            else:
                initial_state = self.initialize_state_zeros()
        
        # Call the JIT-compiled simulation function with the initialized state
        return self.run_simulation_with_state(
            initial_state,
            leader_speed,
            position_gain,
            velocity_gain
        )

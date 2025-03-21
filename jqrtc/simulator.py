"""
Quadrotor simulation engine.

This module provides a simulator for running quadrotor tracking simulations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from .kinematics import quadrotor_kinematics
from .controller import quadrotor_controller


class Simulator:
    """
    A class for simulating quadrotor dynamics and control.
    """

    parameters: Dict[str, float]
    dt: float
    total_time: float
    num_steps: int
    num_quadrotors: int
    state_dim: int
    position_history: Optional[NDArray[np.float64]]
    omega_history: Optional[NDArray[np.float64]]

    def __init__(
        self,
        parameters: Optional[Dict[str, float]] = None,
        dt: float = 0.01,
        total_time: float = 50,
        num_quadrotors: int = 1,
    ):
        """
        Initialize the simulator with parameters.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of quadrotor parameters. If None, default parameters are used.
        dt : float, optional
            Time step in seconds
        total_time : float, optional
            Total simulation time in seconds
        num_quadrotors : int, optional
            Number of quadrotors to simulate
        """
        # Default parameters if none provided
        if parameters is None:
            self.parameters = {
                "g": 9.8,  # Gravity acceleration (m/s^2)
                "m": 1.2,  # Mass (kg)
                "Ix": 0.05,  # Moment of inertia around x-axis (kg·m^2)
                "Iy": 0.05,  # Moment of inertia around y-axis (kg·m^2)
                "Iz": 0.1,  # Moment of inertia around z-axis (kg·m^2)
                "b": 1e-4,  # Thrust coefficient
                "l": 0.5,  # Arm length (m)
                "d": 1e-6,  # Drag coefficient
                "Jr": 0.01,  # Rotor inertia (kg·m^2)
                "k1": 0.02,  # Damping coefficient for x
                "k2": 0.02,  # Damping coefficient for y
                "k3": 0.02,  # Damping coefficient for z
                "k4": 0.1,  # Damping coefficient for phi
                "k5": 0.1,  # Damping coefficient for theta
                "k6": 0.1,  # Damping coefficient for psi
                "omegaMax": 330,  # Maximum motor speed (rad/s)
            }
        else:
            self.parameters = parameters

        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.num_quadrotors = num_quadrotors

        # Initialize storage for simulation history
        self.position_history = None
        self.omega_history = None

        # Dimension of state space
        self.state_dim = 3  # [x, y, z]

    def initialize_state(
        self, random_init: bool = False, init_range: float = 0.0
    ) -> NDArray[np.float64]:
        """
        Initialize the state of the quadrotor(s).

        Parameters
        ----------
        random_init : bool, optional
            Whether to initialize with random positions and orientations
        init_range : float, optional
            Range for random initialization

        Returns
        -------
        numpy.ndarray
            Initial state vector of shape (12, num_quadrotors)
        """
        # Initialize state vector: [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
        s = np.zeros((12, self.num_quadrotors))

        if random_init:
            # Random positions
            s[0:3, :] = np.random.uniform(
                -init_range, init_range, (3, self.num_quadrotors)
            )
            # Random velocities
            s[3:6, :] = np.random.uniform(
                -init_range, init_range, (3, self.num_quadrotors)
            )
            # Random orientations
            s[6:9, :] = np.random.uniform(
                -init_range * np.pi, init_range * np.pi, (3, self.num_quadrotors)
            )

        return s

    def generate_leader_trajectory(
        self, speed: float = 1.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
        # Initialize arrays to store the leader trajectory
        leader_positions = np.zeros((self.state_dim, self.num_steps + 1))
        leader_velocities = np.zeros((self.state_dim, self.num_steps + 1))
        leader_accelerations = np.zeros((self.state_dim, self.num_steps))

        # Initial position and velocity
        xl = np.zeros(3)
        vl = np.zeros(3)

        # Store initial state
        leader_positions[:, 0] = xl
        leader_velocities[:, 0] = vl

        # Generate trajectory for each time step
        for t in range(self.num_steps):
            # Normalized time
            normalized_time = t / self.num_steps

            # Generate acceleration based on time (same logic as in MATLAB)
            if normalized_time < 0.1:
                al = np.array([0, 0, speed]) - vl
            elif normalized_time < 0.2:
                al = np.array([speed, 0, 0]) - vl
            elif normalized_time < 0.4:
                al = np.array([0, speed, 0]) - vl
            elif normalized_time < 0.6:
                al = np.array([-speed, 0, 0]) - vl
            elif normalized_time < 0.8:
                al = np.array([0, -speed, 0]) - vl
            elif normalized_time < 0.9:
                al = np.array([speed, 0, speed]) - vl
            else:
                al = np.array([speed, 0, 0]) - vl

            # Update velocity and position
            vl = vl + self.dt * al
            xl = xl + self.dt * vl

            # Store current state
            leader_accelerations[:, t] = al
            leader_velocities[:, t + 1] = vl
            leader_positions[:, t + 1] = xl

        return leader_positions, leader_velocities, leader_accelerations

    def run_simulation(
        self,
        initial_state: Optional[NDArray[np.float64]] = None,
        leader_speed: float = 1.0,
        position_gain: float = 1,
        velocity_gain: float = 10,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Run the full simulation.

        Parameters
        ----------
        initial_state : numpy.ndarray, optional
            Initial state vector of shape (12, num_quadrotors). If None, zeros are used.
        leader_speed : float, optional
            Speed parameter for the leader trajectory
        position_gain : float, optional
            Gain for position error (k1)
        velocity_gain : float, optional
            Gain for velocity error (k2)

        Returns
        -------
        tuple
            (position_history, omega_history) containing the simulation results
        """
        # Initialize or use provided state
        if initial_state is None:
            s = self.initialize_state()
        else:
            s = initial_state

        # Generate leader trajectory
        leader_positions, leader_velocities, _ = self.generate_leader_trajectory(
            leader_speed
        )

        # Initialize history arrays
        position_history = np.zeros(
            (self.state_dim, self.num_quadrotors + 1, self.num_steps + 1)
        )
        omega_history = np.zeros((4, self.num_steps))

        # Store initial positions
        position_history[:, 0, 0] = leader_positions[:, 0]  # Leader
        position_history[:, 1:, 0] = s[0:3, :]  # Quadrotors

        # Run simulation
        for t in range(self.num_steps):
            # Current leader position and velocity
            xl = leader_positions[:, t]
            vl = leader_velocities[:, t]

            # For simplicity, we're handling a single quadrotor in this implementation
            # Get control inputs from the controller
            if self.num_quadrotors == 1:
                omega = quadrotor_controller(
                    s[:, 0], xl, vl, 0, self.parameters, position_gain, velocity_gain
                )

                # Store motor speeds
                omega_history[:, t] = omega

                # Update quadrotor state
                s[:, 0] = quadrotor_kinematics(s[:, 0], omega, self.parameters, self.dt)
            else:
                # For multiple quadrotors, apply control to each
                for i in range(self.num_quadrotors):
                    omega = quadrotor_controller(
                        s[:, i],
                        xl,
                        vl,
                        0,
                        self.parameters,
                        position_gain,
                        velocity_gain,
                    )

                    # Update quadrotor state
                    s[:, i] = quadrotor_kinematics(
                        s[:, i], omega, self.parameters, self.dt
                    )

                    # For demo purposes, only store the first quadrotor's motor speeds
                    if i == 0:
                        omega_history[:, t] = omega

            # Store positions
            position_history[:, 0, t + 1] = leader_positions[:, t + 1]  # Leader
            position_history[:, 1:, t + 1] = s[0:3, :]  # Quadrotors

        # Save history for later access
        self.position_history = position_history
        self.omega_history = omega_history

        return position_history, omega_history

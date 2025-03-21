#!/usr/bin/env python3
"""
Example script for quadrotor tracking simulation.

This is a Python/NumPy port of the MATLAB main.m script that demonstrates
the quadrotor tracking control simulation.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import sys
import os

# Add parent directory to path to import from quadrotor package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jqrtc.simulator import Simulator
from jqrtc.visualization import display_simulation

def main() -> None:
    """Run the quadrotor tracking simulation."""
    # Setup simulation parameters
    dt = 0.01
    total_time = 50
    
    # Create simulator
    simulator = Simulator(dt=dt, total_time=total_time)
    
    # Initialize quadrotor state
    # Format: [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
    init_state = np.zeros((12, 1))
    
    # Set control gains
    position_gain = 1.0
    velocity_gain = 10.0
    
    # Set leader speed
    leader_speed = 1.0
    
    print("Running quadrotor tracking simulation...")
    
    # Run simulation
    position_history, omega_history = simulator.run_simulation(
        initial_state=init_state,
        leader_speed=leader_speed,
        position_gain=position_gain,
        velocity_gain=velocity_gain
    )
    
    print("Simulation complete. Displaying results...")
    print(position_history.shape)
    print(omega_history.shape)

    # Display results
    display_simulation(position_history, omega_history, dt, tail_length=-1, speed=200, save_file='tracking_simulation')
    
if __name__ == "__main__":
    main()

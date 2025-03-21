"""
Example of JAX-based quadrotor tracking simulation with multiple quadrotors.

This example demonstrates the use of the JAX implementation with vmap
to efficiently simulate multiple quadrotors in parallel.
"""

import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random

from jqrtc.jax_simulator import JaxSimulator
from jqrtc.jax_visualization import plot_trajectory_3d, plot_tracking_error

# Set these parameters to control the simulation
SIMULATION_TIME = 20.0       # Total simulation time in seconds
DT = 0.01                    # Time step in seconds
NUM_QUADROTORS = 100         # Number of quadrotors to simulate in parallel
LEADER_SPEED = 1.0           # Speed parameter for leader trajectory
POSITION_GAIN = 1.0          # Gain for position error
VELOCITY_GAIN = 10.0         # Gain for velocity error
RANDOM_INIT = True           # Whether to randomize initial positions
INIT_RANGE = 0.5             # Range for random initialization

def run_jax_simulation():
    """Run the JAX-based simulation and measure performance."""
    print(f"Running JAX simulation with {NUM_QUADROTORS} quadrotors...")
    
    # Create a JAX simulator
    jax_simulator = JaxSimulator(
        dt=DT,
        total_time=SIMULATION_TIME,
        num_quadrotors=NUM_QUADROTORS
    )
    
    # Initialize PRNG key for random state initialization
    key = random.PRNGKey(0)
    
    # Warm-up JIT compilation (first run is slower due to compilation)
    _ = jax_simulator.run_simulation(
        key=key,
        leader_speed=LEADER_SPEED,
        position_gain=POSITION_GAIN,
        velocity_gain=VELOCITY_GAIN,
        random_init=RANDOM_INIT,
        init_range=INIT_RANGE
    )
    
    # Measure performance
    start_time = time.time()
    
    # Run the simulation
    leader_positions, quadrotor_positions, leader_velocities, omega_histories = (
        jax_simulator.run_simulation(
            key=key,
            leader_speed=LEADER_SPEED,
            position_gain=POSITION_GAIN,
            velocity_gain=VELOCITY_GAIN,
            random_init=RANDOM_INIT,
            init_range=INIT_RANGE
        )
    )
    
    elapsed_time = time.time() - start_time
    steps_per_second = jax_simulator.num_steps / elapsed_time
    quads_per_second = jax_simulator.num_steps * NUM_QUADROTORS / elapsed_time
    
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Performance: {steps_per_second:.2f} steps/second")
    print(f"Parallelized performance: {quads_per_second:.2f} quadrotor steps/second")
    
    # Plot the results
    # For visualization, we'll just show a subset of quadrotors if there are many
    num_to_display = min(5, NUM_QUADROTORS)
    
    # Use only a subset of quadrotors for plotting
    display_indices = jnp.linspace(0, NUM_QUADROTORS-1, num_to_display, dtype=int)
    displayed_positions = quadrotor_positions[:, display_indices, :]
    
    # Create trajectory plot
    fig_traj = plot_trajectory_3d(
        leader_positions, 
        displayed_positions,
        title=f"Quadrotor Tracking (showing {num_to_display} of {NUM_QUADROTORS})"
    )
    
    # Create error plot
    fig_error = plot_tracking_error(
        leader_positions, 
        displayed_positions, 
        DT,
        title=f"Tracking Error (showing {num_to_display} of {NUM_QUADROTORS})"
    )
    
    plt.show()
    
    return leader_positions, quadrotor_positions, leader_velocities, omega_histories


def compare_with_numpy():
    """
    Compare the JAX implementation with the NumPy implementation.
    
    This function runs both implementations with a single quadrotor
    and compares their performance.
    """
    try:
        # Import the NumPy-based simulator
        from jqrtc.simulator import Simulator
        
        # Parameters for comparison
        single_quad = 1
        
        # Create simulators
        jax_simulator = JaxSimulator(
            dt=DT,
            total_time=SIMULATION_TIME,
            num_quadrotors=single_quad
        )
        numpy_simulator = Simulator(
            dt=DT,
            total_time=SIMULATION_TIME,
            num_quadrotors=single_quad
        )
        
        # Run NumPy simulation
        print("Running NumPy simulation...")
        numpy_start = time.time()
        numpy_positions, _ = numpy_simulator.run_simulation(
            leader_speed=LEADER_SPEED,
            position_gain=POSITION_GAIN,
            velocity_gain=VELOCITY_GAIN
        )
        numpy_time = time.time() - numpy_start
        numpy_steps_per_second = numpy_simulator.num_steps / numpy_time
        
        # Run JAX simulation
        print("Running JAX simulation...")
        key = random.PRNGKey(0)
        
        # Warm-up JIT compilation
        _ = jax_simulator.run_simulation(
            key=key,
            leader_speed=LEADER_SPEED,
            position_gain=POSITION_GAIN,
            velocity_gain=VELOCITY_GAIN
        )
        
        jax_start = time.time()
        leader_positions, jax_positions, _, _ = jax_simulator.run_simulation(
            key=key,
            leader_speed=LEADER_SPEED,
            position_gain=POSITION_GAIN,
            velocity_gain=VELOCITY_GAIN
        )
        jax_time = time.time() - jax_start
        jax_steps_per_second = jax_simulator.num_steps / jax_time
        
        # Print performance comparison
        print("\nPerformance Comparison (Single Quadrotor):")
        print(f"NumPy: {numpy_time:.4f} seconds ({numpy_steps_per_second:.2f} steps/s)")
        print(f"JAX:   {jax_time:.4f} seconds ({jax_steps_per_second:.2f} steps/s)")
        print(f"Speedup: {numpy_time / jax_time:.2f}x")
        
        # Now run JAX with multiple quadrotors to demonstrate scaling
        num_test_quads = [1, 10, 100, 1000]
        jax_times = []
        
        print("\nJAX Scaling Test:")
        for num_quads in num_test_quads:
            # Create simulator with the specified number of quadrotors
            jax_multi_simulator = JaxSimulator(
                dt=DT,
                total_time=SIMULATION_TIME,
                num_quadrotors=num_quads
            )
            
            # Warm-up
            _ = jax_multi_simulator.run_simulation(
                key=key,
                leader_speed=LEADER_SPEED,
                position_gain=POSITION_GAIN,
                velocity_gain=VELOCITY_GAIN,
                random_init=True,
                init_range=0.1
            )
            
            # Measure
            start = time.time()
            _ = jax_multi_simulator.run_simulation(
                key=key,
                leader_speed=LEADER_SPEED,
                position_gain=POSITION_GAIN,
                velocity_gain=VELOCITY_GAIN,
                random_init=True,
                init_range=0.1
            )
            end = time.time()
            
            jax_times.append(end - start)
            steps_per_second = jax_multi_simulator.num_steps / (end - start)
            quads_per_second = jax_multi_simulator.num_steps * num_quads / (end - start)
            
            print(f"{num_quads} quadrotors: {end - start:.4f} s ({steps_per_second:.2f} steps/s, {quads_per_second:.2f} quadrotor steps/s)")
        
        # Plot scaling results
        plt.figure(figsize=(10, 6))
        plt.plot(num_test_quads, jax_times, 'o-', linewidth=2)
        plt.xlabel("Number of Quadrotors")
        plt.ylabel("Simulation Time (s)")
        plt.title("JAX Simulation Scaling")
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("NumPy simulator not available for comparison.")


if __name__ == "__main__":
    # Run the JAX simulation
    leader_positions, quadrotor_positions, leader_velocities, omega_histories = run_jax_simulation()
    
    # Optionally run comparison with NumPy implementation
    compare = input("Compare with NumPy implementation? (y/n): ")
    if compare.lower() == 'y':
        compare_with_numpy()

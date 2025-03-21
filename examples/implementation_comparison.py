"""
Comparison between NumPy and JAX implementations.

This example demonstrates how to use the factory module to easily switch
between NumPy and JAX implementations of the quadrotor simulation.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import random

from jqrtc.factory import create_simulator
from jqrtc.jax_visualization import plot_trajectory_3d


def compare_implementations(
    dt: float = 0.01,
    total_time: float = 20.0,
    leader_speed: float = 1.0,
    position_gain: float = 1.0,
    velocity_gain: float = 10.0,
    num_quadrotors: int = 10,
    random_init: bool = True,
    init_range: float = 0.5
):
    """
    Compare NumPy and JAX implementations with the same parameters.

    Parameters
    ----------
    dt : float
        Time step in seconds
    total_time : float
        Total simulation time in seconds
    leader_speed : float
        Speed parameter for leader trajectory
    position_gain : float
        Gain for position error
    velocity_gain : float
        Gain for velocity error
    num_quadrotors : int
        Number of quadrotors to simulate
    random_init : bool
        Whether to use random initialization
    init_range : float
        Range for random initialization
    """
    print(f"Comparing NumPy and JAX implementations with {num_quadrotors} quadrotors...")

    # Create simulators using the factory
    numpy_simulator = create_simulator(
        implementation="numpy",
        dt=dt,
        total_time=total_time,
        num_quadrotors=num_quadrotors
    )

    jax_simulator = create_simulator(
        implementation="jax",
        dt=dt,
        total_time=total_time,
        num_quadrotors=num_quadrotors
    )

    # Set up random key for JAX
    jax_key = random.PRNGKey(0)

    # Initialize states (using same random seed for fair comparison)
    np.random.seed(0)
    if random_init:
        numpy_initial_state = numpy_simulator.initialize_state(random_init=True, init_range=init_range)
        # Convert numpy initial state to JAX array for the JAX simulator
        jax_initial_state = jax.numpy.array(numpy_initial_state.T)  # JAX expects (num_quads, 12)
    else:
        numpy_initial_state = None
        jax_initial_state = None

    # Run NumPy simulation
    print("\nRunning NumPy simulation...")
    numpy_start = time.time()
    numpy_positions, _ = numpy_simulator.run_simulation(
        initial_state=numpy_initial_state,
        leader_speed=leader_speed,
        position_gain=position_gain,
        velocity_gain=velocity_gain
    )
    numpy_time = time.time() - numpy_start
    numpy_steps_per_second = numpy_simulator.num_steps / numpy_time
    numpy_quads_per_second = numpy_simulator.num_steps * num_quadrotors / numpy_time

    print(f"NumPy simulation took {numpy_time:.4f} seconds")
    print(f"NumPy performance: {numpy_steps_per_second:.2f} steps/second")
    print(f"NumPy throughput: {numpy_quads_per_second:.2f} quadrotor steps/second")

    # Warm up JAX JIT compilation
    _ = jax_simulator.run_simulation(
        initial_state=jax_initial_state,
        key=jax_key,
        leader_speed=leader_speed,
        position_gain=position_gain,
        velocity_gain=velocity_gain,
        random_init=jax_initial_state is None and random_init,
        init_range=init_range
    )

    # Run JAX simulation with timing
    print("\nRunning JAX simulation...")
    jax_start = time.time()
    jax_leader_positions, jax_quadrotor_positions, _, _ = jax_simulator.run_simulation(
        initial_state=jax_initial_state,
        key=jax_key,
        leader_speed=leader_speed,
        position_gain=position_gain,
        velocity_gain=velocity_gain,
        random_init=jax_initial_state is None and random_init,
        init_range=init_range
    )
    jax_time = time.time() - jax_start
    jax_steps_per_second = jax_simulator.num_steps / jax_time
    jax_quads_per_second = jax_simulator.num_steps * num_quadrotors / jax_time

    print(f"JAX simulation took {jax_time:.4f} seconds")
    print(f"JAX performance: {jax_steps_per_second:.2f} steps/second")
    print(f"JAX throughput: {jax_quads_per_second:.2f} quadrotor steps/second")

    # Calculate speedup
    speedup = numpy_time / jax_time
    throughput_speedup = jax_quads_per_second / numpy_quads_per_second
    print(f"\nJAX speedup: {speedup:.2f}x")
    print(f"JAX throughput speedup: {throughput_speedup:.2f}x")

    # Plot trajectory for JAX implementation
    # For visualization, limit to a reasonable number of quadrotors
    num_to_display = min(5, num_quadrotors)
    display_indices = jax.numpy.linspace(0, num_quadrotors-1, num_to_display, dtype=int)
    displayed_positions = jax_quadrotor_positions[:, display_indices, :]

    # Plot trajectory
    fig = plot_trajectory_3d(
        jax_leader_positions, 
        displayed_positions,
        title=f"JAX Quadrotor Tracking (showing {num_to_display} of {num_quadrotors})"
    )
    plt.show()


def scalability_test(max_quadrotors: int = 1000):
    """
    Test the scalability of the JAX implementation with increasing numbers of quadrotors.

    Parameters
    ----------
    max_quadrotors : int
        Maximum number of quadrotors to test
    """
    print(f"Testing JAX scalability up to {max_quadrotors} quadrotors...")
    
    # Parameters for testing
    dt = 0.01
    total_time = 10.0  # Shorter time for quicker benchmarking
    
    # Quadrotor counts to test (use logarithmic scale)
    num_quads_list = [1, 10, 100, 500, 1000, max_quadrotors] if max_quadrotors > 1000 else [1, 10, 100, 500, 1000]
    num_quads_list = [q for q in num_quads_list if q <= max_quadrotors]
    
    # Store results
    times = []
    steps_per_second = []
    quads_per_second = []
    
    # Set random key
    jax_key = random.PRNGKey(0)
    
    for num_quads in num_quads_list:
        print(f"\nTesting with {num_quads} quadrotors...")
        
        # Create simulator
        simulator = create_simulator(
            implementation="jax",
            dt=dt,
            total_time=total_time,
            num_quadrotors=num_quads
        )
        
        # Warm up JIT compilation
        _ = simulator.run_simulation(
            key=jax_key,
            leader_speed=1.0,
            position_gain=1.0,
            velocity_gain=10.0,
            random_init=True,
            init_range=0.5
        )
        
        # Run with timing
        start_time = time.time()
        _ = simulator.run_simulation(
            key=jax_key,
            leader_speed=1.0,
            position_gain=1.0,
            velocity_gain=10.0,
            random_init=True,
            init_range=0.5
        )
        elapsed = time.time() - start_time
        
        # Calculate metrics
        times.append(elapsed)
        steps_per_second.append(simulator.num_steps / elapsed)
        quads_per_second.append(simulator.num_steps * num_quads / elapsed)
        
        print(f"Time: {elapsed:.4f} seconds")
        print(f"Performance: {steps_per_second[-1]:.2f} steps/second")
        print(f"Throughput: {quads_per_second[-1]:.2f} quadrotor steps/second")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot simulation time
    ax1.plot(num_quads_list, times, 'o-')
    ax1.set_xlabel('Number of Quadrotors')
    ax1.set_ylabel('Simulation Time (s)')
    ax1.set_title('Simulation Time vs. Quadrotor Count')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot steps per second
    ax2.plot(num_quads_list, steps_per_second, 'o-')
    ax2.set_xlabel('Number of Quadrotors')
    ax2.set_ylabel('Steps per Second')
    ax2.set_title('Simulation Speed vs. Quadrotor Count')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot quadrotor steps per second (throughput)
    ax3.plot(num_quads_list, quads_per_second, 'o-')
    ax3.set_xlabel('Number of Quadrotors')
    ax3.set_ylabel('Quadrotor Steps per Second')
    ax3.set_title('Throughput vs. Quadrotor Count')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default number of quadrotors for comparison
    default_num_quadrotors = 10
    
    print("JAX and NumPy Quadrotor Simulation Comparison")
    print("=============================================")
    print("1. Compare NumPy and JAX implementations")
    print("2. Test JAX scalability")
    choice = input("Enter your choice (1/2): ")
    
    if choice == "1":
        num_quads = input(f"Enter number of quadrotors to simulate [{default_num_quadrotors}]: ")
        num_quads = int(num_quads) if num_quads.strip() else default_num_quadrotors
        compare_implementations(num_quadrotors=num_quads)
    elif choice == "2":
        max_quads = input("Enter maximum number of quadrotors to test [1000]: ")
        max_quads = int(max_quads) if max_quads.strip() else 1000
        scalability_test(max_quadrotors=max_quads)
    else:
        print("Invalid choice. Exiting.")

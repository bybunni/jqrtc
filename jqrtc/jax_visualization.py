"""
JAX-compatible visualization tools for quadrotor simulation.

This module provides visualization functions that work with JAX arrays
from the quadrotor tracking simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax.typing import ArrayLike
import jax.numpy as jnp


def plot_trajectory_3d(
    leader_positions: ArrayLike,
    quadrotor_positions: ArrayLike,
    title: str = "Quadrotor Tracking Simulation"
) -> plt.Figure:
    """
    Create a 3D plot of the leader and quadrotor trajectories.

    Parameters
    ----------
    leader_positions : jax.Array
        Leader positions of shape (num_steps, 3)
    quadrotor_positions : jax.Array
        Quadrotor positions of shape (num_steps, num_quadrotors, 3)
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Convert JAX arrays to NumPy for Matplotlib compatibility
    leader_positions_np = np.asarray(leader_positions)
    quadrotor_positions_np = np.asarray(quadrotor_positions)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot leader trajectory
    ax.plot(
        leader_positions_np[:, 0],
        leader_positions_np[:, 1],
        leader_positions_np[:, 2],
        'r-', linewidth=2, label='Leader'
    )
    
    # Plot quadrotor trajectories
    num_quadrotors = quadrotor_positions_np.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_quadrotors))
    
    for i in range(num_quadrotors):
        ax.plot(
            quadrotor_positions_np[:, i, 0],
            quadrotor_positions_np[:, i, 1],
            quadrotor_positions_np[:, i, 2],
            '-', color=colors[i], linewidth=1,
            label=f'Quadrotor {i+1}'
        )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Make the plot more visually balanced
    max_range = np.array([
        quadrotor_positions_np[:, :, 0].max() - quadrotor_positions_np[:, :, 0].min(),
        quadrotor_positions_np[:, :, 1].max() - quadrotor_positions_np[:, :, 1].min(),
        quadrotor_positions_np[:, :, 2].max() - quadrotor_positions_np[:, :, 2].min()
    ]).max() / 2.0
    
    mid_x = (quadrotor_positions_np[:, :, 0].max() + quadrotor_positions_np[:, :, 0].min()) / 2
    mid_y = (quadrotor_positions_np[:, :, 1].max() + quadrotor_positions_np[:, :, 1].min()) / 2
    mid_z = (quadrotor_positions_np[:, :, 2].max() + quadrotor_positions_np[:, :, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def plot_tracking_error(
    leader_positions: ArrayLike,
    quadrotor_positions: ArrayLike,
    dt: float,
    title: str = "Tracking Error"
) -> plt.Figure:
    """
    Plot the tracking error between the leader and quadrotors.

    Parameters
    ----------
    leader_positions : jax.Array
        Leader positions of shape (num_steps, 3)
    quadrotor_positions : jax.Array
        Quadrotor positions of shape (num_steps, num_quadrotors, 3)
    dt : float
        Time step in seconds
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Convert JAX arrays to NumPy for Matplotlib compatibility
    leader_positions_np = np.asarray(leader_positions)
    quadrotor_positions_np = np.asarray(quadrotor_positions)
    
    # Calculate errors
    num_steps = leader_positions_np.shape[0]
    num_quadrotors = quadrotor_positions_np.shape[1]
    
    time = np.arange(0, num_steps) * dt
    errors = np.zeros((num_steps, num_quadrotors))
    
    for t in range(num_steps):
        for i in range(num_quadrotors):
            errors[t, i] = np.linalg.norm(
                leader_positions_np[t] - quadrotor_positions_np[t, i]
            )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot error for each quadrotor
    colors = plt.cm.viridis(np.linspace(0, 1, num_quadrotors))
    
    for i in range(num_quadrotors):
        ax.plot(
            time, errors[:, i],
            color=colors[i], linewidth=1.5,
            label=f'Quadrotor {i+1}'
        )
    
    # Calculate and plot average error
    avg_error = np.mean(errors, axis=1)
    ax.plot(time, avg_error, 'k--', linewidth=2, label='Average Error')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (distance)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def animate_trajectory(
    leader_positions: ArrayLike,
    quadrotor_positions: ArrayLike,
    dt: float,
    save_path: str = None,
    show_animation: bool = True
) -> None:
    """
    Create an animation of the quadrotor tracking simulation.
    
    This is a simplified version that uses Matplotlib's animation
    capabilities to show the trajectory over time.

    Parameters
    ----------
    leader_positions : jax.Array
        Leader positions of shape (num_steps, 3)
    quadrotor_positions : jax.Array
        Quadrotor positions of shape (num_steps, num_quadrotors, 3)
    dt : float
        Time step in seconds
    save_path : str, optional
        Path to save the animation (if None, animation is not saved)
    show_animation : bool, optional
        Whether to display the animation
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("Matplotlib animation module not available.")
        return
    
    # Convert JAX arrays to NumPy for Matplotlib compatibility
    leader_positions_np = np.asarray(leader_positions)
    quadrotor_positions_np = np.asarray(quadrotor_positions)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine appropriate bounds for the animation
    x_min, x_max = leader_positions_np[:, 0].min(), leader_positions_np[:, 0].max()
    y_min, y_max = leader_positions_np[:, 1].min(), leader_positions_np[:, 1].max()
    z_min, z_max = leader_positions_np[:, 2].min(), leader_positions_np[:, 2].max()
    
    # Add a margin to the bounds
    margin = 0.5
    x_min, x_max = x_min - margin, x_max + margin
    y_min, y_max = y_min - margin, y_max + margin
    z_min, z_max = z_min - margin, z_max + margin
    
    # Set up animation properties
    num_steps = leader_positions_np.shape[0]
    num_quadrotors = quadrotor_positions_np.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_quadrotors))
    
    # Initialize empty plots
    leader_trajectory, = ax.plot([], [], [], 'r-', linewidth=2, label='Leader')
    leader_point, = ax.plot([], [], [], 'ro', markersize=8)
    
    quadrotor_trajectories = []
    quadrotor_points = []
    
    for i in range(num_quadrotors):
        traj, = ax.plot([], [], [], '-', color=colors[i], linewidth=1, label=f'Quadrotor {i+1}')
        point, = ax.plot([], [], [], 'o', color=colors[i], markersize=6)
        quadrotor_trajectories.append(traj)
        quadrotor_points.append(point)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Quadrotor Tracking Simulation')
    ax.legend(loc='upper right')
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Time text
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    def init():
        leader_trajectory.set_data([], [])
        leader_trajectory.set_3d_properties([])
        leader_point.set_data([], [])
        leader_point.set_3d_properties([])
        
        for traj, point in zip(quadrotor_trajectories, quadrotor_points):
            traj.set_data([], [])
            traj.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        
        time_text.set_text('')
        
        return [leader_trajectory, leader_point, *quadrotor_trajectories, *quadrotor_points, time_text]
    
    def update(frame):
        # Skip frames for smoother animation on slower machines
        frame = min(frame * 5, num_steps - 1)
        
        # Update leader
        leader_trajectory.set_data(leader_positions_np[:frame+1, 0], leader_positions_np[:frame+1, 1])
        leader_trajectory.set_3d_properties(leader_positions_np[:frame+1, 2])
        
        leader_point.set_data([leader_positions_np[frame, 0]], [leader_positions_np[frame, 1]])
        leader_point.set_3d_properties([leader_positions_np[frame, 2]])
        
        # Update quadrotors
        for i, (traj, point) in enumerate(zip(quadrotor_trajectories, quadrotor_points)):
            traj.set_data(quadrotor_positions_np[:frame+1, i, 0], quadrotor_positions_np[:frame+1, i, 1])
            traj.set_3d_properties(quadrotor_positions_np[:frame+1, i, 2])
            
            point.set_data([quadrotor_positions_np[frame, i, 0]], [quadrotor_positions_np[frame, i, 1]])
            point.set_3d_properties([quadrotor_positions_np[frame, i, 2]])
        
        # Update time text
        time_text.set_text(f'Time: {frame * dt:.2f} s')
        
        return [leader_trajectory, leader_point, *quadrotor_trajectories, *quadrotor_points, time_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=num_steps//5,
        init_func=init, blit=False, interval=20
    )
    
    # Save animation if path is provided
    if save_path is not None:
        try:
            anim.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
    
    # Show animation
    if show_animation:
        plt.tight_layout()
        plt.show()
    
    return anim

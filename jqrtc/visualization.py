"""
Visualization tools for quadrotor simulations.

This module provides functions for visualizing quadrotor trajectories and motor speeds.
"""

from typing import Any, List, Optional, Tuple, Union, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.artist as artist
from numpy.typing import NDArray

def plot_trajectory(position_history: NDArray, dt: float, tail_length: int = -1, 
                  speed: int = 200, title: Optional[str] = None) -> Tuple[Figure, FuncAnimation]:
    """
    Plot the trajectory of the quadrotor(s) in 3D.
    
    This is a Python/NumPy port of the MATLAB plotHis3 function.
    
    Parameters
    ----------
    position_history : numpy.ndarray
        Array of shape (3, n, loop) containing the position history, where:
        - 3 is the dimension (x, y, z)
        - n is the number of objects (leader + quadrotors)
        - loop is the number of time steps
    dt : float
        Time step in seconds
    tail_length : int, optional
        Length of the trajectory tail to display. If -1, show the full trajectory.
    speed : int, optional
        Animation speed (number of frames to display)
    title : str, optional
        Title for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    # Get dimensions
    _, n, loop = position_history.shape
    
    # Calculate total simulation time
    ts = dt * (loop - 1)
    
    # Adjust tail length if needed
    if tail_length > loop or tail_length < 0:
        tail_length = loop
    
    # Adjust speed if needed
    if speed > loop or speed < 0:
        speed = loop
    
    # Downsample for speed
    xy_his: List[NDArray[np.float64]] = []
    skip = round(loop / speed)
    if skip > 1:
        # Take every skip-th frame
        for k in range(1, loop + 1, skip):
            if len(xy_his) == 0:
                xy_his.append(position_history[:, :, 0])
            else:
                xy_his.append(position_history[:, :, k - 1])
        
        # Ensure the last frame is included
        if k != loop:
            xy_his.append(position_history[:, :, -1])
    else:
        # Use all frames
        xy_his = [position_history[:, :, i] for i in range(loop)]
    
    # Convert to numpy array
    xy_his_array: NDArray[np.float64] = np.array(xy_his)
    new_loop = len(xy_his_array)
    
    # Get axis limits
    x_min: float = float(np.min(position_history[0, :, :]))
    x_max: float = float(np.max(position_history[0, :, :]))
    y_min: float = float(np.min(position_history[1, :, :]))
    y_max: float = float(np.max(position_history[1, :, :]))
    z_min: float = float(np.min(position_history[2, :, :]))
    z_max: float = float(np.max(position_history[2, :, :]))
    
    # Add margins
    x_width: float = x_max - x_min
    x_width = x_width if x_width > 0 else 1.0
    y_width: float = y_max - y_min
    y_width = y_width if y_width > 0 else 1.0
    z_width: float = z_max - z_min
    z_width = z_width if z_width > 0 else 1.0
    
    x_min = x_min - 0.05 * x_width
    x_max = x_max + 0.05 * x_width
    y_min = y_min - 0.05 * y_width
    y_max = y_max + 0.05 * y_width
    z_min = z_min - 0.05 * z_width
    z_max = z_max + 0.05 * z_width
    
    # Reshape for easier plotting of trajectories
    xyz = np.transpose(position_history, (1, 0, 2))
    xyz = xyz.reshape(n * 3, loop)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Function to update the plot for each frame
    def update(frame: int) -> List[artist.Artist]:
        ax.clear()
        
        # Plot initial positions
        ax.plot(xy_his_array[0][0, :], xy_his_array[0][1, :], xy_his_array[0][2, :], 'kx')
        
        # Plot leader
        ax.plot(xy_his_array[frame][0, 0], xy_his_array[frame][1, 0], xy_his_array[frame][2, 0], 'rp')
        
        # Plot quadrotors
        ax.plot(xy_his_array[frame][0, 1:], xy_his_array[frame][1, 1:], xy_his_array[frame][2, 1:], 'bo', 
                markerfacecolor='b', markersize=5)
        
        # Plot leader trajectory
        if frame <= tail_length or tail_length < 0:
            indices = slice(0, frame+1)
        else:
            indices = slice(frame - tail_length, frame+1)
            
        ax.plot(xyz[0, indices], xyz[1, indices], xyz[2, indices], 'm--')
        
        # Plot quadrotor trajectories
        for i in range(1, n):
            idx = 3*i
            if frame <= tail_length or tail_length < 0:
                indices = slice(0, frame+1)
            else:
                indices = slice(frame - tail_length, frame+1)
                
            ax.plot(xyz[idx, indices], xyz[idx+1, indices], xyz[idx+2, indices], 'c-')
        
        # Set title with current time
        curr_time = frame / new_loop * ts
        ax.set_title(f'Time: {curr_time:.2f}s' if title is None else f'{title} - Time: {curr_time:.2f}s')
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)  # type: ignore[attr-defined] # Axes3D has this method
        
        # Set labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')  # type: ignore[attr-defined] # Axes3D has this method
        
        # Set grid
        ax.grid(True)
        
        # Return the artists that were added
        return ax.get_children()
        
    # Create animation
    ani = FuncAnimation(fig, update, frames=new_loop, interval=50, repeat=True)
    
    # Return the figure
    return fig, ani

def plot_motor_speeds(omega_history: NDArray, dt: float) -> Figure:
    """
    Plot the motor speeds over time.
    
    Parameters
    ----------
    omega_history : numpy.ndarray
        Array of shape (4, loop) containing the motor speeds
    dt : float
        Time step in seconds
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    # Get dimensions
    _, loop = omega_history.shape
    
    # Create time array
    time = np.arange(0, loop * dt, dt)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot motor speeds
    for i in range(4):
        ax.plot(time, omega_history[i, :], label=f'Motor {i+1}')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Speed (rad/s)')
    ax.set_title('Motor Speeds')
    
    # Add grid and legend
    ax.grid(True)
    ax.legend()
    
    return fig

def display_simulation(position_history: NDArray, omega_history: NDArray, dt: float, 
                     tail_length: int = -1, speed: int = 200, save_file: Optional[str] = None) -> None:
    """
    Display the complete simulation results.
    
    Parameters
    ----------
    position_history : numpy.ndarray
        Array of shape (3, n, loop) containing the position history
    omega_history : numpy.ndarray
        Array of shape (4, loop) containing the motor speeds
    dt : float
        Time step in seconds
    tail_length : int, optional
        Length of the trajectory tail to display. If -1, show the full trajectory.
    speed : int, optional
        Animation speed (number of frames to display)
    save_file : str, optional
        File to save the figures to. If None, show the plots.
    """
    # Plot trajectory
    fig1, ani = plot_trajectory(position_history, dt, tail_length, speed)
    plt.figure(fig1.number)
    plt.title('Quadrotor Trajectory')
    
    # Plot motor speeds
    fig2 = plot_motor_speeds(omega_history, dt)
    plt.figure(fig2.number)
    plt.title('Motor Speeds')
    
    # Save or show the plots
    if save_file is not None:
        ani.save(f'{save_file}_trajectory.mp4')
        fig2.savefig(f'{save_file}_motor_speeds.png')
    else:
        plt.show()

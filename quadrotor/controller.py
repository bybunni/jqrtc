"""
Quadrotor tracking controller implementation.

This module implements controllers for tracking a reference trajectory
with a quadrotor.
"""

import numpy as np
from .utils import angle_delta, saturate

def quadrotor_controller(s, xl, vl, psil, para, k1, k2):
    """
    Compute control inputs to track a reference trajectory.
    
    This is a Python/NumPy port of the MATLAB quadrotor_controller function.
    
    Parameters
    ----------
    s : numpy.ndarray
        Current state vector of shape (12,) containing:
        [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
    xl : numpy.ndarray
        Leader/reference position vector of shape (3,)
    vl : numpy.ndarray
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
    numpy.ndarray
        Motor speeds vector of shape (4,)
    """
    # Ensure state is real (in case of numerical errors)
    s = np.real(s)
    
    # Extract state variables
    x, y, z = s[0], s[1], s[2]
    vx, vy, vz = s[3], s[4], s[5]
    phi, theta, psi = s[6], s[7], s[8]
    vphi, vtheta, vpsi = s[9], s[10], s[11]
    
    # Initialize control vector
    u = np.zeros(4)
    
    # Thrust and torque transformation matrix
    tr = np.array([
        [para['b'], para['b'], para['b'], para['b']],
        [0, -para['b'], 0, para['b']],
        [-para['b'], 0, para['b'], 0],
        [-para['d'], para['d'], -para['d'], para['d']]
    ])
    
    # Compute control input based on position and velocity errors
    pos_error = np.array([x, y, z])
    vel_error = np.array([vx, vy, vz])
    input_delta = k1 * (xl - pos_error) + k2 * (vl - vel_error)
    
    # Apply saturation to control input
    attr = saturate(input_delta) * 2 + np.array([0, 0, para['g']])
    
    # Rotation matrix for yaw
    rot = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    attr = rot @ attr
    
    # Total thrust
    u[0] = para['m'] * np.linalg.norm(attr)
    
    # Desired roll and pitch angles
    phi_p = np.arcsin(-attr[1] / np.linalg.norm(attr))
    theta_p = np.arctan(attr[0] / attr[2])
    
    if attr[2] < 0:
        theta_p = theta_p - np.pi
    
    # Desired yaw angle
    psi_p = psil
    
    # Compute torque control inputs
    u[3] = para['Iz'] * saturate(angle_delta(psi_p, psi) + 0 - vpsi)
    u[1] = 20 * para['Ix'] * (u[0] + u[3]) / 2 * saturate(angle_delta(phi_p, phi) + 0 - 1 * vphi) / para['l']
    u[2] = 20 * para['Iy'] * (u[0] - u[3]) / 2 * saturate(angle_delta(theta_p, theta) + 0 - 1 * vtheta) / para['l']
    
    # Compute motor speeds from control inputs
    omega2 = np.linalg.solve(tr, u)
    
    # Ensure non-negative values
    omega2 = np.maximum(omega2, 0)
    
    # Convert to motor speeds
    omega = np.sqrt(omega2)
    
    return omega


class TrackingController:
    """
    A class for implementing quadrotor tracking control.
    
    This provides an object-oriented interface to the tracking controller.
    """
    
    def __init__(self, parameters=None, position_gain=1, velocity_gain=10):
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
            self.parameters = {
                'g': 9.8,        # Gravity acceleration (m/s^2)
                'm': 1.2,        # Mass (kg)
                'Ix': 0.05,      # Moment of inertia around x-axis (kg·m^2)
                'Iy': 0.05,      # Moment of inertia around y-axis (kg·m^2)
                'Iz': 0.1,       # Moment of inertia around z-axis (kg·m^2)
                'b': 1e-4,       # Thrust coefficient
                'l': 0.5,        # Arm length (m)
                'd': 1e-6,       # Drag coefficient
                'Jr': 0.01,      # Rotor inertia (kg·m^2)
                'k1': 0.02,      # Damping coefficient for x
                'k2': 0.02,      # Damping coefficient for y
                'k3': 0.02,      # Damping coefficient for z
                'k4': 0.1,       # Damping coefficient for phi
                'k5': 0.1,       # Damping coefficient for theta
                'k6': 0.1,       # Damping coefficient for psi
                'omegaMax': 330  # Maximum motor speed (rad/s)
            }
        else:
            self.parameters = parameters
            
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        
    def compute_control(self, state, leader_pos, leader_vel, desired_yaw=0):
        """
        Compute control inputs to track a reference trajectory.
        
        Parameters
        ----------
        state : numpy.ndarray
            Current state vector of shape (12,)
        leader_pos : numpy.ndarray
            Leader/reference position vector of shape (3,)
        leader_vel : numpy.ndarray
            Leader/reference velocity vector of shape (3,)
        desired_yaw : float, optional
            Desired yaw angle
            
        Returns
        -------
        numpy.ndarray
            Motor speeds vector of shape (4,)
        """
        return quadrotor_controller(
            state, leader_pos, leader_vel, desired_yaw,
            self.parameters, self.position_gain, self.velocity_gain
        )

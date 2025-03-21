"""
Quadrotor kinematics and dynamics simulation.

This module implements the equations of motion for a quadrotor and updates
the state based on the physical model.
"""

import numpy as np

def quadrotor_kinematics(s, omega, para, dt):
    """
    Update the state of the quadrotor based on its current state and motor speeds.
    
    This is a Python/NumPy port of the MATLAB quadrotor_kinematics function.
    
    Parameters
    ----------
    s : numpy.ndarray
        Current state vector of shape (12,) containing:
        [x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi]
        where:
        - (x, y, z) is the position
        - (vx, vy, vz) is the velocity
        - (phi, theta, psi) are the Euler angles (roll, pitch, yaw)
        - (vphi, vtheta, vpsi) are the angular velocities
        
    omega : numpy.ndarray
        Motor speeds vector of shape (4,)
        
    para : dict
        Dictionary of quadrotor parameters
        
    dt : float
        Time step in seconds
        
    Returns
    -------
    numpy.ndarray
        Updated state vector
    """
    # Extract state variables
    x, y, z = s[0], s[1], s[2]
    vx, vy, vz = s[3], s[4], s[5]
    phi, theta, psi = s[6], s[7], s[8]
    vphi, vtheta, vpsi = s[9], s[10], s[11]
    
    # Thrust and torque transformation matrix
    tr = np.array([
        [para['b'], para['b'], para['b'], para['b']],
        [0, -para['b'], 0, para['b']],
        [-para['b'], 0, para['b'], 0],
        [-para['d'], para['d'], -para['d'], para['d']]
    ])
    
    # Square of motor speeds
    omega2 = omega**2
    
    # Overall propeller speed
    Omega = -omega[0] + omega[1] - omega[2] + omega[3]
    
    # Control inputs (thrust and torques)
    u = tr @ omega2
    
    # Update linear velocities and positions
    vx_new = vx + dt * (u[0] * (np.cos(psi) * np.sin(theta) * np.cos(phi) + 
                                np.sin(psi) * np.sin(phi)) / para['m'] - 
                        para['k1'] * vx / para['m'])
    x_new = x + dt * vx
    
    vy_new = vy + dt * (u[0] * (np.sin(psi) * np.sin(theta) * np.cos(phi) - 
                                np.cos(psi) * np.sin(phi)) / para['m'] - 
                        para['k2'] * vy / para['m'])
    y_new = y + dt * vy
    
    vz_new = vz + dt * (u[0] * (np.cos(theta) * np.cos(phi)) / para['m'] - 
                        para['g'] - 
                        para['k3'] * vz / para['m'])
    z_new = z + dt * vz
    
    # Update angular velocities and angles
    vphi_new = vphi + dt * ((para['Iy'] - para['Iz']) / para['Ix'] * vtheta * vpsi - 
                           para['Jr'] * vtheta * Omega + 
                           u[1] * para['l'] / para['Ix'] - 
                           para['k4'] * vphi * para['l'] / para['Ix'])
    phi_new = phi + dt * vphi
    
    vtheta_new = vtheta + dt * ((para['Iz'] - para['Ix']) / para['Iy'] * vphi * vpsi + 
                               para['Jr'] * vphi * Omega + 
                               u[2] * para['l'] / para['Iy'] - 
                               para['k5'] * vtheta * para['l'] / para['Iy'])
    theta_new = theta + dt * vtheta
    
    vpsi_new = vpsi + dt * ((para['Ix'] - para['Iy']) / para['Iz'] * vphi * vtheta + 
                           u[3] * 1 / para['Iz'] - 
                           para['k6'] * vpsi / para['Iz'])
    psi_new = psi + dt * vpsi
    
    # Assemble and return the new state
    return np.array([
        x_new, y_new, z_new,
        vx_new, vy_new, vz_new,
        phi_new, theta_new, psi_new,
        vphi_new, vtheta_new, vpsi_new
    ])


class QuadrotorDynamics:
    """
    A class for simulating quadrotor dynamics.
    
    This provides an object-oriented interface to the quadrotor kinematics.
    """
    
    def __init__(self, parameters=None):
        """
        Initialize the quadrotor dynamics with physical parameters.
        
        Parameters
        ----------
        parameters : dict, optional
            Dictionary of quadrotor parameters. If None, default parameters are used.
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
    
    def step(self, state, omega, dt):
        """
        Update the quadrotor state for one time step.
        
        Parameters
        ----------
        state : numpy.ndarray
            Current state vector of shape (12,)
        omega : numpy.ndarray
            Motor speeds vector of shape (4,)
        dt : float
            Time step in seconds
            
        Returns
        -------
        numpy.ndarray
            Updated state vector
        """
        return quadrotor_kinematics(state, omega, self.parameters, dt)

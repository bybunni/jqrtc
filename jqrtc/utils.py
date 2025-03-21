"""
Utility functions for quadrotor simulation.

This module contains helper functions used across the quadrotor simulation.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

def angle_delta(p2: float, p1: float) -> float:
    """
    Calculate the angle between two angles on a plane.
    
    This is a Python/NumPy port of the MATLAB angleDelta function.
    
    Parameters
    ----------
    p2 : float
        Target angle in radians
    p1 : float
        Reference angle in radians
        
    Returns
    -------
    float
        The signed angle difference in radians
    """
    # Convert angles to unit vectors
    v1 = np.array([np.cos(p1), np.sin(p1)])
    v2 = np.array([np.cos(p2), np.sin(p2)])
    
    # Calculate the determinant to get the sign
    det_sign = np.sign(np.cross(np.append(v1, 0), np.append(v2, 0))[2])
    
    # Calculate the dot product for the angle
    dot_prod = np.dot(v1, v2)
    
    # Ensure dot product is in valid range for arccos
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    
    # Return the signed angle
    return float(det_sign * np.arccos(dot_prod))

def saturate(input_value: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Saturate an input value or array using hyperbolic tangent.
    
    Parameters
    ----------
    input_value : float or numpy.ndarray
        Input value or array to saturate
        
    Returns
    -------
    float or numpy.ndarray
        Saturated value or array
    """
    return np.tanh(input_value)

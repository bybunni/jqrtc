"""
Quadrotor Tracking Control
==========================

A Python/NumPy port of the MATLAB quadrotor tracking control simulation.

This package provides tools for simulating quadrotor dynamics and implementing
tracking control algorithms.

Author: Ported from MATLAB code by wjxjmj
Original MATLAB License: GPL
"""

__version__ = '0.1.0'

# Import key components for easier access
from .kinematics import quadrotor_kinematics
from .controller import quadrotor_controller
from .utils import angle_delta

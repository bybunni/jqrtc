# JQRTC: Quadrotor Tracking Control

A Python implementation of quadrotor tracking control simulation, ported from MATLAB.

This package provides tools for simulating quadrotor dynamics and implementing tracking control algorithms. The original MATLAB code has been ported to Python using NumPy to provide a more accessible and modern implementation.

## Features

- Quadrotor dynamics and kinematics simulation
- Tracking controller for following reference trajectories
- 3D visualization of flight trajectories
- Interactive animation of the flight process
- Modular architecture with OOP design

## Repository Structure

```
jqrtc/                      # Repository root
├── jqrtc/                  # Python package
│   ├── kinematics.py       # Quadrotor dynamics
│   ├── controller.py       # Tracking controller
│   ├── simulator.py        # Simulation engine
│   ├── utils.py            # Utility functions
│   └── visualization.py    # Plotting and animation
├── matlab/                 # Original MATLAB code
├── examples/               # Example scripts
├── docs/                   # Documentation
└── tests/                  # Unit tests (future implementation)
```

## Installation

Clone the repository and install in development mode:

```bash
# Clone the repository
git clone https://github.com/bybunni/jqrtc.git
cd jqrtc

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Usage

Run the example simulation:

```bash
# Run the tracking simulation example
python examples/tracking_simulation.py
```

Basic code example:

```python
from jqrtc.simulator import Simulator
from jqrtc.visualization import display_simulation

# Create simulator with default parameters
simulator = Simulator(dt=0.01, total_time=50)

# Run simulation
position_history, omega_history = simulator.run_simulation(
    leader_speed=1.0,
    position_gain=1.0,
    velocity_gain=10.0
)

# Visualize the results
display_simulation(position_history, omega_history, dt=0.01)
```

## Original MATLAB Implementation

The original MATLAB implementation is preserved in the `matlab/` directory. It includes:

- `main.m`: Main simulation script
- `quadrotor_kinematics.m`: Quadrotor dynamics implementation
- `quadrotor_controller.m`: Tracking controller implementation
- `angleDelta.m`: Utility function for angle calculations
- `plotHis3.m`: Visualization functions

## Documentation

For more detailed information about the port from MATLAB to Python, see [docs/numpy-port.md](docs/numpy-port.md).

## License

This project is licensed under the GNU General Public License (GPL) - see the LICENSE file for details.

## Acknowledgments

Original MATLAB code by wjxjmj (wjxjmj@126.com).

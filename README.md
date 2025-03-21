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

JQRTC uses modern Python packaging with pyproject.toml and hatchling as the build backend.

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/bybunni/jqrtc.git
cd jqrtc

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Option 2: Install using pip

```bash
# Install directly from GitHub (not yet available on PyPI)
pip install git+https://github.com/bybunni/jqrtc.git
```

### Using uv (modern Python package manager)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e .
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

## JAX Implementation

The project also includes a JAX-based implementation that enables efficient parallel simulation of multiple quadrotors. The JAX implementation includes:

- `jax_utils.py`: JAX-based utility functions
- `jax_kinematics.py`: Vectorized quadrotor dynamics
- `jax_controller.py`: Tracking controller with JAX support
- `jax_simulator.py`: Simulator with parallel computation capabilities
- `jax_visualization.py`: Visualization tools for JAX arrays
- `factory.py`: Factory pattern for easily switching between NumPy and JAX implementations

### Performance Comparison

#### Single Quadrotor Performance

| Implementation | Time (seconds) | Steps per Second | Speedup |
|----------------|-----------------|------------------|----------|
| NumPy | 0.4177 | 4,787.61 | 1.00x |
| JAX | 0.0095 | 210,182.86 | 43.90x |

#### JAX Scaling with Multiple Quadrotors

| Quadrotors | Time (seconds) | Steps per Second | Quadrotor Steps per Second |
|------------|----------------|------------------|----------------------------|
| 1 | 0.0110 | 181,673.84 | 181,673.84 |
| 10 | 0.0042 | 471,376.04 | 4,713,760.40 |
| 100 | 0.0027 | 745,521.51 | 74,552,150.73 |
| 1,000 | 0.0028 | 709,936.36 | 709,936,357.48 |

## Original MATLAB Implementation

The original MATLAB implementation is preserved in the `matlab/` directory. It includes:

- `main.m`: Main simulation script
- `quadrotor_kinematics.m`: Quadrotor dynamics implementation
- `quadrotor_controller.m`: Tracking controller implementation
- `angleDelta.m`: Utility function for angle calculations
- `plotHis3.m`: Visualization functions

## Documentation

For more detailed information about the ports, see:
- [docs/numpy-port.md](docs/numpy-port.md): Information on the NumPy implementation
- [docs/jax-port.md](docs/jax-port.md): Information on the JAX implementation

## License

This project is licensed under the GNU General Public License (GPL) - see the LICENSE file for details.

## Acknowledgments

Original MATLAB code by wjxjmj (wjxjmj@126.com).

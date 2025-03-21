# Quadrotor Tracking Control: MATLAB to NumPy Port

## Current Project Structure

The original MATLAB implementation (now in the `matlab/` directory) consists of the following files:

### 1. `matlab/main.m`
The main execution script that:
- Sets up simulation parameters (timestep, total simulation time)
- Initializes quadrotor state (position, velocity, orientation)
- Defines parameters for the quadrotor's physical model
- Defines the virtual leader trajectory
- Runs the simulation loop
- Records history of positions and motor speeds
- Visualizes results (trajectory and motor speeds)

### 2. `matlab/quadrotor_kinematics.m`
Implements the quadrotor dynamics and kinematics:
- Takes current state, motor speeds, parameters, and timestep as inputs
- Computes forces and torques from motor speeds
- Updates the quadrotor state using the equations of motion
- Returns the updated state vector

### 3. `matlab/quadrotor_controller.m`
Implements the tracking controller:
- Takes current state, leader position/velocity, and parameters as inputs
- Computes the control inputs to track the leader
- Uses a saturation function to limit control inputs
- Computes the required motor speeds to achieve the desired control inputs
- Returns the motor speeds

### 4. `matlab/angleDelta.m`
A utility function to calculate the angular difference between two angles.

### 5. `matlab/plotHis3.m`
Visualization function for the quadrotor trajectory:
- Plots the 3D trajectory of the quadrotor and leader
- Animates the flight process
- Supports options for visualization speed and tail length

## Python Port Implementation

### Module Structure

The Python port has been implemented with the following structure:

```
jqrtc/                       # Repository root
├── README.md                # Project overview
├── setup.py                 # Package installation
├── docs/                    # Documentation
│   └── numpy-port.md        # Porting details
├── matlab/                  # Original MATLAB code
│   ├── README.md
│   ├── main.m
│   ├── quadrotor_kinematics.m
│   ├── quadrotor_controller.m
│   ├── angleDelta.m
│   ├── plotHis3.m
│   ├── p1.png
│   └── p2.png
├── jqrtc/                   # Python package
│   ├── __init__.py          # Package initialization
│   ├── kinematics.py        # Quadrotor dynamics
│   ├── controller.py        # Tracking controller
│   ├── simulator.py         # Simulation engine
│   ├── utils.py             # Utility functions
│   └── visualization.py     # Plotting and animation
├── examples/                # Example scripts
│   └── tracking_simulation.py  # Main example script
└── tests/                   # Unit tests (future implementation)
```

### Implementation Details

#### 1. `jqrtc/__init__.py`
- Package initialization
- Version information
- Imports for user convenience

#### 2. `jqrtc/kinematics.py`
- `quadrotor_kinematics(state, omega, parameters, dt)`: Port of the MATLAB function
- Class-based implementation: `QuadrotorDynamics` class with methods for state update

#### 3. `jqrtc/controller.py`
- `quadrotor_controller(state, leader_pos, leader_vel, psi_desired, parameters, k1, k2)`: Port of the MATLAB function
- Class-based implementation: `TrackingController` class with methods for control computation

#### 4. `jqrtc/utils.py`
- `angle_delta(p2, p1)`: Port of the MATLAB function
- Other utility functions as needed

#### 5. `jqrtc/simulator.py`
- `Simulator` class for running simulations
- Methods for initialization, step execution, and result storage

#### 6. `jqrtc/visualization.py`
- `plot_trajectory(history, dt, tail_length, speed)`: Port of the MATLAB plotting function
- Interactive 3D visualization using Matplotlib
- Animation capabilities

#### 7. `examples/tracking_simulation.py`
- Main script demonstrating usage of the package
- Similar functionality to the original `main.m`
- Command-line arguments for simulation parameters

### NumPy Implementation Approach

1. **State Representation**:
   - Use NumPy arrays for state vectors and matrices
   - State vector: shape (12,) or (12, n) for multiple quadrotors

2. **Kinematics**:
   - Use NumPy's vectorized operations for efficient computation
   - Implement equations of motion using matrix operations

3. **Controller**:
   - Use NumPy's linear algebra functions for rotation matrices
   - Implement saturation functions using NumPy's element-wise operations

4. **Visualization**:
   - Use Matplotlib for 2D plots of motor speeds
   - Use Matplotlib's 3D plotting or Plotly for trajectory visualization
   - Implement animation using FuncAnimation or interactive Plotly plots

### Dependencies

- NumPy: Core numerical computation
- Matplotlib: Basic visualization and 2D plots
- Plotly (optional): Interactive 3D visualization
- pytest: Unit testing

### Testing Strategy

1. **Unit Tests**:
   - Test kinematics against known analytical solutions
   - Test controller outputs for simple inputs
   - Test utility functions against MATLAB equivalents

2. **Integration Tests**:
   - Compare trajectory results with MATLAB implementation
   - Verify stability and tracking performance

3. **Validation**:
   - Visual comparison of trajectories
   - Comparison of motor speeds and state evolution

### Implementation Timeline

1. **Phase 1: Core Functionality**
   - Port kinematics.py
   - Port controller.py
   - Port utils.py
   - Basic simulation functionality

2. **Phase 2: Simulation and Integration**
   - Implement simulator.py
   - Create main example script
   - Basic visualization

3. **Phase 3: Testing and Refinement**
   - Unit tests
   - Validation against MATLAB
   - Documentation
   - Performance optimization

4. **Phase 4: Advanced Features**
   - Enhanced visualization
   - Additional controller types
   - Multiple quadrotor support
   - Parameter estimation tools

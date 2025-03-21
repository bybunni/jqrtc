# JAX Port Development Plan

## Overview

This document outlines the plan for converting the `jqrtc` (Quadrotor Tracking Control Simulation) codebase from NumPy to JAX, with a focus on leveraging JAX's vectorized mapping (`vmap`) capabilities to parallelize computations.

## Current Architecture

The codebase currently consists of the following main components:

1. **kinematics.py** - Core quadrotor dynamics and kinematics equations
2. **controller.py** - Tracking controller implementation
3. **simulator.py** - Simulation engine that ties everything together
4. **utils.py** - Helper functions for angle calculations and saturation
5. **visualization.py** - Visualization tools for simulation results

The implementation uses NumPy arrays and operations, with proper type annotations throughout the codebase.

## JAX Migration Goals

1. Replace NumPy operations with JAX equivalents
2. Use JAX's functional programming paradigm
3. Leverage JAX's `vmap` for parallel computation of multiple quadrotors
4. Maintain existing API where possible for backward compatibility
5. Retain type annotations (possibly using `jax.typing`)
6. Keep CPU-only compatibility (no GPU/TPU required)

## Implementation Strategy

### 1. Dependencies Update

Update `pyproject.toml` to include JAX dependencies:
- Add `jax` and `jaxlib` as dependencies
- Keep `numpy` as a dependency for compatibility
- Keep `matplotlib` for visualization

### 2. Core Function Conversion

Convert the core computational functions to JAX:
- Convert `quadrotor_kinematics` to use JAX operations
- Convert `quadrotor_controller` to use JAX operations
- Convert `angle_delta` and `saturate` to use JAX operations
- Ensure these functions are pure with no side effects

### 3. Vectorization Implementation

Use JAX's `vmap` to vectorize operations:
- Vectorize kinematics calculations across multiple quadrotors
- Vectorize controller calculations across multiple quadrotors
- Modify simulator to handle batched operations

### 4. Class-Based API Compatibility

JAX works best with pure functions, but our codebase uses classes. To maintain compatibility:
- Refactor classes to use JAX's functional style internally
- Keep the class-based interface for backward compatibility
- Use JAX's state management patterns where appropriate

### 5. Random Number Generation

Replace NumPy's random number generation with JAX's PRNG system:
- Use JAX's key-based random number generation
- Ensure proper key splitting for stochastic operations

### 6. Type Annotations

Update type annotations to:
- Use `jax.Array` instead of `numpy.ndarray`
- Use `jaxlib.xla_extension.PrngKey` for random keys
- Maintain compatibility with existing type system

## Implementation Order

1. Create new JAX versions of utility functions in `utils.py`
2. Convert `quadrotor_kinematics` to JAX
3. Convert `quadrotor_controller` to JAX 
4. Implement JAX-based simulator with vmap
5. Update visualization to handle JAX arrays
6. Create examples demonstrating performance improvements

## Performance Benchmarking

We'll measure performance improvements by:
- Comparing simulation speed for single quadrotor (NumPy vs JAX)
- Testing simulation with increasing numbers of quadrotors
- Measuring memory usage for large simulations

## Testing Strategy

For each converted module:
1. Create equivalent JAX and NumPy functions
2. Verify outputs match within numerical precision
3. Test with increasing quadrotor counts to ensure scalability

## Potential Challenges

1. **Handling state mutation**: JAX prefers immutable state, while our simulation keeps updating the state.
2. **Numerical precision**: JAX may have different numerical precision than NumPy.
3. **Visualization compatibility**: Ensuring JAX arrays work properly with matplotlib.
4. **Compilation overhead**: JAX JIT compilation has overhead that may affect small simulations.

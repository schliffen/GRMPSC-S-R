# GRAMPC-S Rust Reimplementation Plan

Target upstream snapshot: `grampc/grampc-s` `main` at
`ce9d1ebdc69e6a2e96e303e262b47adad60d87ef`.

The goal is a pure Rust implementation of GRAMPC-S functionality. The target is
numerical equivalence with upstream examples and APIs within documented
tolerances. Bit-for-bit equality is not a realistic requirement across C++ and
Rust floating-point implementations, compiler settings, random generators, and
linear algebra backends.

## Equivalence Definition

For each upstream module and example, the Rust port should provide:

- The same mathematical model and public behavior.
- The same dimensions, parameter conventions, and error conditions.
- Golden tests against upstream outputs for representative inputs.
- Example-level tests that compare trajectories, controls, costs, constraints,
  and stochastic moments within explicit tolerances.

For solving optimal control problems exactly like upstream GRAMPC-S, the Rust
project must also reimplement the GRAMPC solver behavior used through
`grampc_interface`, or provide a Rust-native solver that is validated to the same
solutions within tolerance. The current `ShootingSolver` is useful for examples,
but it is not equivalent to GRAMPC.

## Current Rust Coverage

Already present in this crate:

- Chance-constraint approximations:
  - Gaussian
  - Chebyshev
  - Symmetric
- Distributions:
  - Gaussian
  - Uniform
  - Piecewise constant
  - Exponential
  - Gamma
  - Chi-squared
  - Log-normal
  - Weibull
  - Beta
  - Student-t
  - Fisher-F
  - Extreme value
  - Multivariate uncorrelated composition
- Polynomials:
  - Univariate polynomial arithmetic
  - Multivariate polynomial product basis
  - Hermite generator
  - Legendre generator
- Point transformations:
  - Monte Carlo
  - Unscented
  - Stirling first order
  - Stirling second order
  - Composed Gaussian quadrature
  - Polynomial chaos expansion
- Gaussian processes:
  - Squared exponential kernel
  - Matern 3/2 kernel
  - Matern 5/2 kernel
  - Periodic kernel
  - Locally periodic kernel
  - Kernel sum
  - Kernel product
  - GP mean and variance evaluation
- Stochastic dynamics approximations:
  - Sigma point
  - Resampling
  - Resampling with GP residuals
  - Monte Carlo
  - Taylor finite-difference approximation
- Examples:
  - Double integrator
  - Mass-spring-damper
  - Reactor
  - Inverted pendulum
  - Stochastic double-integrator MPC
- Local tests:
  - `cargo test` currently passes 34 tests.
  - `cargo test --examples` currently compiles all Rust examples.

## Missing Or Incomplete Areas

### 1. Upstream-compatible module API

Mirror upstream concepts more directly in Rust:

- `ChanceConstraintApproximation`
- `Distribution`
- `PointTransformation`
- `ProblemDescription`
- `Simulator`
- `StationaryKernel`
- `GaussianProcess`
- `GrampcInterface` or a Rust equivalent

The local API can stay idiomatic Rust, but each upstream concept should have a
clear Rust home and test mapping.

### 2. GRAMPC solver behavior

The largest functional gap is the optimizer. Upstream GRAMPC-S relies on the
external GRAMPC nonlinear MPC solver. To solve problems the same way, implement:

- GRAMPC parameter and option model.
- Time grid, scaling, and normalization behavior.
- Augmented Lagrangian and penalty update behavior.
- Gradient and adjoint handling.
- Control update, line search, and convergence criteria.
- Constraint handling and multiplier updates.
- Solution buffers equivalent to upstream outputs used by examples.

This should be developed as its own module, not mixed into the lightweight
`ShootingSolver`.

### 3. Problem descriptions

Port every upstream example problem description and derivative function:

- Double integrator
- Double integrator with GP
- Reactor
- Inverted pendulum
- Mass-spring-damper
- Vehicle
- PMSM
- 2D crane
- Nonlinear chain variants: 2, 3, 4, 6, 8, 10, 12, 14
- Python mobile robot examples if they are part of the desired Rust surface

Each problem should include tests for:

- Dynamics
- Vector-Jacobian products
- Stage and terminal costs
- Cost derivatives
- Path and terminal constraints
- Constraint derivatives

### 4. Simulator parity

The current Rust simulator has RK4 helpers. Upstream also exposes a simulator
configured by integration method strings such as `"heun"` in examples. Port:

- Heun integration
- Any additional integration modes used upstream
- Simulation output structure
- Closed-loop simulation flow used in upstream examples

### 5. File I/O compatibility

Port input/output behavior used by upstream examples:

- Trajectory output files
- Dimension and constraint files
- Gaussian-process data loading
- Any helper utilities from `grampc_s_util`

### 6. Python and MATLAB surfaces

If "all functions" includes bindings and scripts, define Rust replacements for:

- Python binding classes as Rust API tests or optional `pyo3` bindings.
- MATLAB helper behavior as Rust CLI/tools or documentation examples.

These are not needed for core Rust numerical equivalence unless the project
requires language-binding compatibility.

## Recommended Implementation Order

1. Freeze upstream commit and generate golden data.
2. Add an upstream fixture directory with compact numeric expected outputs.
3. Port and test all solver-independent modules to API parity.
4. Port all example problem descriptions and derivative functions.
5. Add simulator parity, especially Heun.
6. Add a GRAMPC-compatible Rust solver module.
7. Run example-level equivalence tests against golden outputs.
8. Port optional Python/MATLAB-facing behavior if required.

## Immediate Next Tasks

1. Add a script or test helper that stores upstream golden outputs.
2. Port the missing upstream examples one by one, starting with `vehicle` or
   `double_integrator_GP`.
3. Split the current `problem.rs` examples into separate modules so additional
   upstream examples can be added without making one large file harder to
   maintain.
4. Add explicit tolerances for equivalence tests:
   - Dynamics and derivative functions: `1e-10` to `1e-8`
   - Moment transformations: `1e-10` to `1e-7`
   - Closed-loop trajectories: problem-dependent, initially `1e-6` to `1e-4`
   - Optimizer results: problem-dependent until GRAMPC-compatible solver exists


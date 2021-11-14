### ODE
Header-only ordinary differential equation solvers in C++20 based on Eigen.

### Butcher Tableau
- The `[ |extended_]butcher_tableau` encapsulate the coefficients of a Butcher tableau. 
- The `type` of a tableau must be an arithmetic type (i.e. satisfy `std::is_arithmetic<type>`).
- The following tableau are currently provided:
  - Explicit:
    - Forward Euler
    - Runge Kutta 4
    - Dormand Prince 5
  - Implicit:
    - Backward Euler
    - Crank Nicolson

### Problems
- The `[initial_value|boundary_value]_problem` encapsulate the (initial) state of an ordinary differential equation problem.
- The `higher_order_initial_value_problem`s may be decomposed into order many coupled `initial_value_problem`s.
- The `boundary_value_problem`s may be decomposed into second order `higher_order_initial_value_problem`s using the shooting method.

### Methods
- The `[implicit|semi_implicit|explicit]_method`s encapsulate the operations to perform a single iteration of the solution to a `[initial_value|boundary_value]_problem`.
- The methods are constructed from `[ |extended_]butcher_tableau`.

### Iterators
- The `[fixed_size|adaptive_size]_iterator`s iterate a `[implicit|semi_implicit|explicit]_method` over a `[initial_value|boundary_value]_problem`.
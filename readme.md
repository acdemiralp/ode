### ODE
Header-only, dependency-free ordinary differential equation solvers in C++20.

### Design Notes
- The `[implicit|explicit]_method`s are constructed from `[extended_]butcher_tableau`s at compile-time.
- A `[fixed_size|adaptive_size]_iterator` takes an `[implicit|explicit]_method` and an `[initial_value|boundary_value]_problem`, iterating the method over a problem whose initial state is given. 
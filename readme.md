### ODE
Header-only, dependency-free ordinary differential equation solvers in C++20.

### Design Notes
- The `[implicit|explicit]_method`s are constructed from `[extended_]butcher_tableau`s at compile-time.
- A `[fixed_size|adaptive_size]_iterator` iterates an `[implicit|explicit]_method` over an `[initial_value|boundary_value]_problem`.
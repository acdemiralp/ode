### ODE
Header-only, dependency-free ordinary differential equation solvers in C++20.

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
- The `time_type`  of a problem must be an arithmetic type    (i.e. satisfy `std::is_arithmetic<type>`).
- The `value_type` of a problem must be default constructible (i.e. satisfy `std::is_default_constructible<type>`) and furthermore provide the following operators:
  - `value_type operator+(const value_type& lhs, const value_type& rhs)`.
  - `value_type operator-(const value_type& lhs, const value_type& rhs)`.
  - `value_type operator/(const value_type& lhs, const value_type& rhs)`.
  - `value_type operator*(const value_type& lhs, const time_type&  rhs)`.
  - Note that any decent linear algebra library such as Eigen supports this functionality out-of-the-box. If yours does not, you can implement them outside of the class.

### Methods
- The `[implicit|semi_implicit|explicit]_method`s encapsulate the operations to perform a single iteration of the solution to a `[initial_value|boundary_value]_problem`.
- The methods are constructed from `[ |extended_]butcher_tableau`.

### Iterators
- The `[fixed_size|adaptive_size]_iterator`s iterate a `[implicit|semi_implicit|explicit]_method` over a `[initial_value|boundary_value]_problem`.
### ODE
Header-only ordinary differential equation solvers in C++20.

### Example: Lorenz system
```cpp
#include <ode/ode.hpp>

using method_type  = ode::explicit_method<ode::runge_kutta_4_tableau<float>>;
using problem_type = ode::initial_value_problem<float, vector3f>;

std::int32_t main(std::int32_t argc, char** argv)
{
  const auto sigma   = 10.0f;
  const auto rho     = 28.0f;
  const auto beta    = 8.0f / 3.0f;

  const auto problem = problem_type
  {
    0.0f,                                  /* t0 */
    vector3f(16.0f, 16.0f, 16.0f),         /* y0 */         
    [&] (const float t, const vector3f& y) /* dy/dt = f(t, y) */
    {
      return vector3f(sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]);
    }
  };
  
  auto iterator = ode::fixed_step_iterator<method_type, problem_type>(problem, 1.0f /* h */);
  while (true)
    ++iterator;
}
```

### Butcher Tableau
- The `[ |extended_]butcher_tableau` encapsulate the coefficients of a Butcher tableau. 
- The `type` of a tableau must be an arithmetic type (i.e. satisfy `std::is_arithmetic<type>`).
- The following tableau are currently provided:
  - Explicit:
    - Forward Euler
    - Runge Kutta 4
    - Dormand Prince 5

### Problems
- The `initial_value_problem` encapsulate the (initial) state of an ordinary differential equation problem.
- The `higher_order_initial_value_problem`s may be decomposed into order many coupled `initial_value_problem`s.

### Methods
- The `[implicit|semi_implicit|explicit]_method`s encapsulate the operations to perform a single iteration of the solution to a `initial_value_problem`.
- The methods are constructed from `[ |extended_]butcher_tableau`.

### Iterators
- The `[fixed_size|adaptive_size]_iterator`s iterate a `explicit_method` over a `initial_value_problem`.
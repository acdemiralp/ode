#include <doctest/doctest.h>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Core>
#include <ode/ode.hpp>

TEST_CASE("Adaptive Test")
{
  using method_type             = ode::explicit_method<ode::dormand_prince_5_tableau<float>>;
  using problem_type            = ode::initial_value_problem<float, Eigen::Vector3f>;
  using i_controller_iterator   = ode::adaptive_step_iterator<method_type, problem_type, ode::integral_controller                        <method_type, problem_type>>;
  using pi_controller_iterator  = ode::adaptive_step_iterator<method_type, problem_type, ode::proportional_integral_controller           <method_type, problem_type>>;
  using pid_controller_iterator = ode::adaptive_step_iterator<method_type, problem_type, ode::proportional_integral_derivative_controller<method_type, problem_type>>;

  constexpr auto sigma   = 10.0f;
  constexpr auto rho     = 28.0f;
  constexpr auto beta    = 8.0f / 3.0f;
  const     auto problem = problem_type
  {
    0.0f,                                         /* t0 */
    Eigen::Vector3f(16.0f, 16.0f, 16.0f),         /* y0 */
    [&] (const float t, const Eigen::Vector3f& y) /* y' = f(t, y) */
    {
      return Eigen::Vector3f(sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]); /* Lorenz system */
    }
  };

  auto iterator_1 = i_controller_iterator  (problem, 1.0f /* h */);
  for (auto i = 0; i < 10000; ++i)
    ++iterator_1;

  auto iterator_2 = pi_controller_iterator (problem, 1.0f /* h */);
  for (auto i = 0; i < 10000; ++i)
    ++iterator_2;

  auto iterator_3 = pid_controller_iterator(problem, 1.0f /* h */);
  for (auto i = 0; i < 10000; ++i)
    ++iterator_3;
}
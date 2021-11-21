#include <doctest/doctest.h>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Core>
#include <ode/ode.hpp>

TEST_CASE("Coupled ODE Test")
{
  using method_type               = ode::explicit_method<ode::forward_euler_tableau<float>>;
  using second_order_problem_type = ode::higher_order_initial_value_problem<float, Eigen::Vector3f, 2>;
  using coupled_problem_type      = second_order_problem_type::coupled_type;

  constexpr auto alpha = 1.0f;

  const auto second_order_problem = second_order_problem_type
  {
    0.0f,                                                                              /* t0 */
    std::array {Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 0.0f)}, /* y0, y'0 */
    [&] (const float h, const Eigen::Vector3f& y, const Eigen::Vector3f& dydt)         /* y'' = f(t, y, y') */
    {
      return 2.0f * h * dydt + 2.0f * alpha * y; /* Hermite differential equation */
    }
  };

  auto iterator = ode::coupled_fixed_step_iterator<method_type, coupled_problem_type>(second_order_problem.to_coupled_first_order(), 1.0f /* h */);
  for (auto i = 0; i < 1000; ++i)
    ++iterator;
}
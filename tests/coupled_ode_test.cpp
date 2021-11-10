#include <doctest/doctest.h>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Core>
#include <ode/ode.hpp>

TEST_CASE("Coupled ODE Test")
{
  using method_type               = ode::explicit_method<ode::forward_euler_tableau<float>>;
  using second_order_problem_type = ode::higher_order_initial_value_problem<float, Eigen::Vector3f, 2>;
  using coupled_problem_type      = second_order_problem_type::coupled_type;

  const auto second_order_problem = second_order_problem_type
  {
    0.0f,
    std::array {Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 0.0f)},
    [ ] (const float h, const Eigen::Vector3f& y1, const Eigen::Vector3f& y2)
    {
      return y1;
    }
  };

  auto coupled  = second_order_problem.to_coupled_first_order();
  auto result_1 = method_type::apply(coupled[0], 1.0f);
  auto result_2 = method_type::apply(coupled[1], 1.0f);

  auto coupled_iterator = ode::coupled_fixed_step_iterator<method_type, coupled_problem_type>(coupled, 1.0f);
  coupled_iterator++;
}
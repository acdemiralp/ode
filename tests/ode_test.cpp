#include "doctest/doctest.h"

#include <Eigen/Core>

#include <ode/problem/initial_value_problem.hpp>

template <typename type>
struct is_problem : std::bool_constant<false> {};

template <typename type>
inline constexpr bool is_problem_v = is_problem<type>::value;

template <typename type>
concept problem = is_problem_v<type>;

template <typename time_type, typename value_type, typename function_type>
struct initial_value_problem
{
  
};

template <typename time_type, typename value_type, typename function_type>
struct is_problem<initial_value_problem<time_type, value_type, function_type>> : std::bool_constant<true> {};


TEST_CASE("Initial value problem test.")
{
  using scalar  = float;
  using vector3 = Eigen::Vector3f;

  // First order example: Lorenz system.
  ode::initial_value_problem<scalar, vector3> first_order_problem
  {
    0.0f,
    vector3(1.0f, 2.0f, 3.0f),
    [ ] (const scalar t, const vector3& y)
    {
      constexpr scalar sigma = 10.0f;
      constexpr scalar rho   = 28.0f;
      constexpr scalar beta  = 8.0f / 3.0f;
      return vector3(sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]);
    }
  };

  // Second order example: Van der Pol equation (y'' - mu (1 - y^2) y' + y = 0).
  ode::initial_value_problem<scalar, scalar, 2> second_order_problem
  {
    0.0f,
    {0.0f, 1.0f},
    [ ] (const scalar t, const scalar& y, const scalar& dydt)
    {
      constexpr scalar mu = 0.2f;
      return mu * (static_cast<scalar>(1) - static_cast<scalar>(std::pow(y, 2))) * dydt - y;
    }
  };
  auto converted_second_order_problem = second_order_problem.to_system_of_first_order();
}
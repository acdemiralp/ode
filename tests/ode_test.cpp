#include <doctest/doctest.h>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Core>

#include <ode/method/explicit_method.hpp>
#include <ode/tableau/explicit/forward_euler.hpp>
#include <ode/tableau/explicit/runge_kutta_4.hpp>

TEST_CASE("ODE Test")
{
  using method = ode::explicit_method<ode::runge_kutta_4_tableau<float>>;

  const Eigen::Vector3f y0 (0.0f, 0.0f, 0.0f);
  const float           t0 (0.0f);
  const float           h  (1.0f);
  const std::function   f  = [ ] (const Eigen::Vector3f& y, const float h)
  {
    return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
  };

  auto result = method::evaluate(y0, t0, f, h);

  REQUIRE(result[0] == 1.0f);
  REQUIRE(result[1] == 0.0f);
  REQUIRE(result[2] == 0.0f);
}
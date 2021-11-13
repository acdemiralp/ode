#include <doctest/doctest.h>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Core>
#include <ode/ode.hpp>

TEST_CASE("BVP Test")
{
  ode::boundary_value_problem<float, Eigen::Vector3f> bvp
  {
    {
      0.0f,
      1.0f
    },
    {
      Eigen::Vector3f(-0.5f, 0.0f, 0.0f),
      Eigen::Vector3f( 0.5f, 0.0f, 0.0f)
    },
    [ ] (const float t, const Eigen::Vector3f& y, const Eigen::Vector3f& dydt)
    {
      return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    }
  };

  auto ivp  = bvp.to_initial_value_problem(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
  auto ivps = ivp.to_coupled_first_order  ();

  auto iterator = ode::coupled_fixed_step_iterator<ode::explicit_method<ode::runge_kutta_4_tableau<float>>, decltype(ivp)::coupled_type>(ivps, (bvp.times[1] - bvp.times[0]) / 100);
  for (auto i = 0; i < 100; ++i)
    ++iterator;

  // TODO: Newton-Raphson is applied y(std::get<1>(times), s) - std::get<1>(values) = 0.
}
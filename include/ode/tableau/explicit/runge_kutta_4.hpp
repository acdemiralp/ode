#pragma once

#include <array>

#include <ode/parallel/cuda.hpp>
#include <ode/tableau/tableau.hpp>

namespace ode::tableau
{
template <typename type = double>
struct runge_kutta_4 {};

template <typename type>
__constant__ constexpr auto        a_v    <runge_kutta_4<type>> = std::array
{
  type(0.5),
  type(0.0), type(0.5),
  type(0.0), type(0.0), type(1.0)
};
template <typename type>
__constant__ constexpr auto        b_v    <runge_kutta_4<type>> = std::array
{
  type(1.0 / 6.0), type(1.0 / 3.0), type(1.0 / 3.0), type(1.0 / 6.0)
};
template <typename type>
__constant__ constexpr auto        c_v    <runge_kutta_4<type>> = std::array
{
  type(0.0), type(0.5), type(0.5), type(1.0)
};

template <typename type>
__constant__ constexpr std::size_t order_v<runge_kutta_4<type>> = 4;
}
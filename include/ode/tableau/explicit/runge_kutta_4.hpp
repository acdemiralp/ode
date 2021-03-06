#pragma once

#include <ode/tableau/butcher_tableau.hpp>
#include <ode/tableau/order.hpp>
#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename type = double>
using runge_kutta_4_tableau = butcher_tableau<
  sequence<type, 0.5,
                 0.0, 0.5,
                 0.0, 0.0, 1.0>,
  sequence<type, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0>,
  sequence<type, 0.0, 0.5, 0.5, 1.0>>;

template <typename type>
constexpr std::size_t order<runge_kutta_4_tableau<type>> = 4;
}
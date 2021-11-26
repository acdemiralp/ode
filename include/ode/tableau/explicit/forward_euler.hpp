#pragma once

#include <ode/tableau/butcher_tableau.hpp>
#include <ode/tableau/order.hpp>
#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename type = double>
using forward_euler_tableau = butcher_tableau<
  sequence<type, 0.0>,
  sequence<type, 1.0>,
  sequence<type, 0.0>>;

template <typename type>
constexpr std::size_t order<forward_euler_tableau<type>> = 1;
}
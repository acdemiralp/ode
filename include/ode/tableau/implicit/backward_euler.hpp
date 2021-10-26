#pragma once

#include <ode/tableau/butcher_tableau.hpp>

namespace ode
{
template <typename type = double>
using forward_euler_tableau = butcher_tableau<
  sequence<type, 1.0>,
  sequence<type, 1.0>,
  sequence<type, 1.0>>;
}
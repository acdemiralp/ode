#pragma once

#include <ode/tableau/butcher_tableau.hpp>

namespace ode
{
template <typename type = double>
using runge_kutta_4_tableau = butcher_tableau<
  sequence<type, 0.5,
                 0.0, 0.5,
                 0.0, 0.0, 1.0>,
  sequence<type, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0>,
  sequence<type, 0.0, 0.5, 0.5, 1.0>>;
}
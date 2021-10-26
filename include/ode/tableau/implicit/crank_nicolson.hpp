#pragma once

#include <ode/tableau/butcher_tableau.hpp>

namespace ode
{
template <typename type>
using crank_nicolson_tableau = butcher_tableau<
  sequence<type, 0.0, 0.0,
                 0.5, 0.5>,
  sequence<type, 0.5, 0.5>,
  sequence<type, 0.0, 1.0>>;
}
#pragma once

#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename type = double>
struct van_der_houwen_wray_3_tableau {};

template <typename type>
__constant__ constexpr auto tableau_a<van_der_houwen_wray_3_tableau<type>> = std::array
{
  type(8.0 / 15.0),
  type(1.0 / 4.0 ), type(5.0 / 12.0)
};
template <typename type>
__constant__ constexpr auto tableau_b<van_der_houwen_wray_3_tableau<type>> = std::array {type(1.0 / 4.0), type(0.0), type(3.0 / 4.0)};
template <typename type>
__constant__ constexpr auto tableau_c<van_der_houwen_wray_3_tableau<type>> = std::array {type(0.0), type(8.0 / 15.0), type(2.0 / 3.0)};
}
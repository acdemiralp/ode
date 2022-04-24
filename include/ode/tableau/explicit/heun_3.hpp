#pragma once

#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename type = double>
struct heun_3_tableau {};

template <typename type>
__constant__ constexpr auto tableau_a<heun_3_tableau<type>> = std::array
{
  type(1.0 / 3.0),
  type(0.0      ), type(2.0 / 3.0)
};
template <typename type>
__constant__ constexpr auto tableau_b<heun_3_tableau<type>> = std::array {type(0.25), type(0.0)      , type(0.75)};
template <typename type>
__constant__ constexpr auto tableau_c<heun_3_tableau<type>> = std::array {type(0.0) , type(1.0 / 3.0), type(2.0 / 3.0)};
}
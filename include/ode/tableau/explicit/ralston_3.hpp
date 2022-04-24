#pragma once

#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename type = double>
struct ralston_3_tableau {};

template <typename type>
__constant__ constexpr auto tableau_a<ralston_3_tableau<type>> = std::array
{
  type(0.5),
  type(0.0), type(0.75)
};
template <typename type>
__constant__ constexpr auto tableau_b<ralston_3_tableau<type>> = std::array {type(2.0 / 9.0), type(1.0 / 3.0), type(4.0 / 9.0)};
template <typename type>
__constant__ constexpr auto tableau_c<ralston_3_tableau<type>> = std::array {type(0.0), type(0.5), type(0.75)};
}
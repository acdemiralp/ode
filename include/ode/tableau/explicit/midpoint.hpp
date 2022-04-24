#pragma once

#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename type = double>
struct midpoint_tableau {};

template <typename type>
__constant__ constexpr auto tableau_a<midpoint_tableau<type>> = std::array {type(0.5)};
template <typename type>
__constant__ constexpr auto tableau_b<midpoint_tableau<type>> = std::array {type(0.0), type(1.0)};
template <typename type>
__constant__ constexpr auto tableau_c<midpoint_tableau<type>> = std::array {type(0.0), type(0.5)};
}
#pragma once

#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename type = double>
struct heun_2_tableau {};

template <typename type>
__constant__ constexpr auto tableau_a<heun_2_tableau<type>> = std::array {type(1.0)};
template <typename type>
__constant__ constexpr auto tableau_b<heun_2_tableau<type>> = std::array {type(0.5), type(0.5)};
template <typename type>
__constant__ constexpr auto tableau_c<heun_2_tableau<type>> = std::array {type(0.0), type(1.0)};
}
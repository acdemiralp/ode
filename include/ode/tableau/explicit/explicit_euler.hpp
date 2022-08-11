#pragma once

#include <array>

#include <ode/parallel/cuda.hpp>
#include <ode/tableau/tableau.hpp>

namespace ode::tableau
{
template <typename type = double>
struct explicit_euler {};

template <typename type>
__constant__ constexpr auto a_v<explicit_euler<type>> = std::array {type(0.0)};
template <typename type>
__constant__ constexpr auto b_v<explicit_euler<type>> = std::array {type(1.0)};
template <typename type>
__constant__ constexpr auto c_v<explicit_euler<type>> = std::array {type(0.0)};
}
#pragma once

#include <array>

#include <ode/parallel/cuda.hpp>
#include <ode/tableau/tableau.hpp>

namespace ode::tableau
{
template <typename type = double>
struct implicit_euler {};

template <typename type>
__constant__ constexpr auto a_v<implicit_euler<type>> = std::array {type(1.0)};
template <typename type>
__constant__ constexpr auto b_v<implicit_euler<type>> = std::array {type(1.0)};
template <typename type>
__constant__ constexpr auto c_v<implicit_euler<type>> = std::array {type(1.0)};
}
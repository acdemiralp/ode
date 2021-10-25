#pragma once

namespace ode
{
template <auto n>
constexpr auto triangular_number = n * (n + 1) / 2;
}
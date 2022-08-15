#pragma once

#include <type_traits>

#include <ode/parallel/cuda.hpp>

namespace ode
{
template <typename type, type index, typename = std::make_integer_sequence<type, index>>
__constant__        constexpr type factorial_v;
template <typename type, type index, type... index_sequence>
__constant__        constexpr type factorial_v<type, index, std::integer_sequence<type, index_sequence...>> = (static_cast<type>(1) * ... * (index_sequence + 1));

template <typename type>
__host__ __device__ constexpr type factorial(const type index)
{
  type result {1};
  for (auto i = type{2}; i <= index; ++i)
    result *= i;
  return result;
}
}
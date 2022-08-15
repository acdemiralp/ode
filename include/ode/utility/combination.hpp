#pragma once

#include <ode/parallel/cuda.hpp>
#include <ode/utility/factorial.hpp>

namespace ode
{
template <typename type, type n, type k>
__constant__        constexpr type combination_v = factorial_v<type, n> / (factorial_v<type, k> * factorial_v<type, n - k>);

template <typename type>
__host__ __device__ constexpr type combination(const type n, const type k)
{
  return factorial(n) / (factorial(k) * factorial(n - k));
}
}
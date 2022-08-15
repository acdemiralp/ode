#pragma once

#include <ode/parallel/cuda.hpp>
#include <ode/utility/factorial.hpp>

namespace ode
{
template <typename type, type n, type k>
__constant__        constexpr type permutation_v = factorial_v<type, n> / factorial_v<type, n - k>;

template <typename type>
__host__ __device__ constexpr type permutation(const type n, const type k)
{
  return factorial(n) / factorial(n - k);
}
}
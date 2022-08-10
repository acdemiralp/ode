#pragma once

#include <array>
#include <cstddef>

#include <ode/parallel/cuda.hpp>

namespace ode::tableau
{
template <typename type>
__constant__ constexpr type        a_v              ;
template <typename type>
__constant__ constexpr type        b_v              ;
template <typename type>
__constant__ constexpr type        bs_v             ;
template <typename type>
__constant__ constexpr type        c_v              ;
template <typename type>
__constant__ constexpr bool        is_extended_v    = false;
template <typename type>
__constant__ constexpr std::size_t order_v          = 1;
template <typename type>
__constant__ constexpr std::size_t extended_order_v = 1;
}
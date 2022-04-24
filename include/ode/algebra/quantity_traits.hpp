#pragma once

#include <complex>
#include <type_traits>

namespace ode
{
template <typename type>
inline constexpr bool is_complex_v                     = false;
template <typename type>                               
inline constexpr bool is_complex_v<std::complex<type>> = true ;
                                                       
template <typename type>                               
inline constexpr bool is_scalar_v                      = std::is_arithmetic_v<type> || is_complex_v<type>;

template <typename type>
struct is_complex : std::bool_constant<is_complex_v<type>> { };
template <typename type>
struct is_scalar  : std::bool_constant<is_scalar_v <type>> { };
}
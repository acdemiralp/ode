#pragma once

#include <type_traits>

#include <ode/method/explicit_method.hpp>
#include <ode/method/implicit_method.hpp>
#include <ode/method/semi_implicit_method.hpp>

namespace ode
{
template <typename type>
struct is_explicit_method                                     : std::false_type { };
template <typename tableau>
struct is_explicit_method<explicit_method<tableau>>           : std::true_type  { };

template <typename type>
struct is_implicit_method                                     : std::false_type { };
template <typename tableau>
struct is_implicit_method<implicit_method<tableau>>           : std::true_type  { };

template <typename type>
struct is_semi_implicit_method                                : std::false_type { };
template <typename tableau>
struct is_semi_implicit_method<semi_implicit_method<tableau>> : std::true_type  { };

template <typename type>
inline constexpr bool is_explicit_method_v      = is_explicit_method     <type>::value;
template <typename type>
inline constexpr bool is_implicit_method_v      = is_implicit_method     <type>::value;
template <typename type>
inline constexpr bool is_semi_implicit_method_v = is_semi_implicit_method<type>::value;
}
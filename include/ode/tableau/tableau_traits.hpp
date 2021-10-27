#pragma once

#include <type_traits>

#include <ode/tableau/butcher_tableau.hpp>
#include <ode/tableau/extended_butcher_tableau.hpp>

namespace ode
{
template <typename type>
struct is_butcher_tableau                                                                                                                             : std::false_type { };
template <typename type, type... a, type... b, type... c>
struct is_butcher_tableau<butcher_tableau<sequence<type, a...>, sequence<type, b...>, sequence<type, c...>>>                                          : std::true_type  { };
template <typename type, type... a, type... b, type... bs, type... c>
struct is_butcher_tableau<extended_butcher_tableau<sequence<type, a...>, sequence<type, b...>, sequence<type, bs...>, sequence<type, c...>>>          : std::true_type  { };

template <typename type>
struct is_extended_butcher_tableau                                                                                                                    : std::false_type { };
template <typename type, type... a, type... b, type... bs, type... c>
struct is_extended_butcher_tableau<extended_butcher_tableau<sequence<type, a...>, sequence<type, b...>, sequence<type, bs...>, sequence<type, c...>>> : std::true_type  { };

template <typename type>
inline constexpr bool is_butcher_tableau_v          = is_butcher_tableau         <type>::value;
template <typename type>
inline constexpr bool is_extended_butcher_tableau_v = is_extended_butcher_tableau<type>::value;
}
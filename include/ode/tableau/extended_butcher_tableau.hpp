#pragma once

#include <array>

#include <ode/tableau/butcher_tableau.hpp>
#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename a, typename b, typename bs, typename c>
struct extended_butcher_tableau { };
template <typename type_, type_... a_, type_... b_, type_... bs_, type_... c_>
struct extended_butcher_tableau<sequence<type_, a_...>, sequence<type_, b_...>, sequence<type_, bs_...>, sequence<type_, c_...>>
:      butcher_tableau         <sequence<type_, a_...>, sequence<type_, b_...>,                          sequence<type_, c_...>>
{
  static constexpr auto bs = std::array { bs_... };
};
}
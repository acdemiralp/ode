#pragma once

#include <array>

#include <ode/tableau/butcher_tableau.hpp>
#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename a, typename b, typename bs, typename c>
struct extended_butcher_tableau { };
template <typename type, type... a_, type... b_, type... bs_, type... c_>
struct extended_butcher_tableau<sequence<type, a_...>, sequence<type, b_...>, sequence<type, bs_...>, sequence<type, c_...>>
:      butcher_tableau         <sequence<type, a_...>, sequence<type, b_...>,                         sequence<type, c_...>>
{
  static constexpr auto bs = std::array { bs_... };
};
}
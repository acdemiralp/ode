#pragma once

#include <array>

#include <ode/tableau/butcher_tableau.hpp>
#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename a, typename b, typename bs, typename c>
struct extended_butcher_tableau { };
template <typename type, type... a, type... b, type... bs_, type... c>
struct extended_butcher_tableau<sequence<type, a...>, sequence<type, b...>, sequence<type, bs_...>, sequence<type, c...>>
:      butcher_tableau         <sequence<type, a...>, sequence<type, b...>,                         sequence<type, c...>>
{
  static constexpr auto bs = std::array { bs_... };
};
}
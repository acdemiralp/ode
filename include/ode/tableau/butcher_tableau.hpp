#pragma once

#include <array>

#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename a, typename b, typename c>
struct butcher_tableau { };
template <typename type_, type_... a_, type_... b_, type_... c_>
struct butcher_tableau<sequence<type_, a_...>, sequence<type_, b_...>, sequence<type_, c_...>>
{
  using type = type_;

  static constexpr auto stages = sizeof...(c_);
  static constexpr auto a      = std::array { a_...  };
  static constexpr auto b      = std::array { b_...  };
  static constexpr auto c      = std::array { c_...  };
};
}
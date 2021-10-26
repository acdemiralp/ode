#pragma once

#include <array>

#include <ode/utility/sequence.hpp>

namespace ode
{
template <typename a, typename b, typename c>
struct butcher_tableau { };
template <typename type, type... a_, type... b_, type... c_>
struct butcher_tableau<sequence<type, a_...>, sequence<type, b_...>, sequence<type, c_...>>
{
  static constexpr auto stages = sizeof...(c_);
  static constexpr auto a      = std::array { a_...  };
  static constexpr auto b      = std::array { b_...  };
  static constexpr auto c      = std::array { c_...  };
};
}
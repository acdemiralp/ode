#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>

#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/sequence.hpp>
#include <ode/utility/triangular_number.hpp>

namespace ode
{
template <typename a, typename b, typename c>
class explicit_butcher_tableau { };
template <typename type, type... a_, type... b_, type... c_>
class explicit_butcher_tableau<sequence<type, a_...>, sequence<type, b_...>, sequence<type, c_...>>
{
public:
  using t_type    = type;
  template <typename y_type>
  using dydt_type = std::function<y_type(const y_type&, t_type)>;

  static constexpr std::tuple  a      { a_... };
  static constexpr std::tuple  b      { b_... };
  static constexpr std::tuple  c      { c_... };
  static constexpr std::size_t stages { std::tuple_size_v<decltype(c)> };

  template <typename y_type>
  static constexpr y_type evaluate(const y_type& y, t_type t, const dydt_type<y_type>& f, t_type h)
  {
    std::array<y_type, stages> k;
    constexpr_for<0, stages, 1>([&y, &t, &f, &h, &k] (auto i)
    {
      y_type sum;
      constexpr_for<0, i.value, 1>([&k, &sum, &i] (auto j)
      {
        sum += k[j] * std::get<triangular_number<i.value - 1> + j.value>(a);
      });
      k[i] = f(y + sum * h, t + std::get<i>(c) * h);
    });

    y_type sum;
    constexpr_for<0, stages, 1>([&k, &sum] (auto i)
    {
      sum += k[i] * std::get<i>(b);
    });
    return y + sum * h;
  }

  template <class y_type>
  static constexpr std::function<y_type(const y_type&, t_type, const dydt_type<y_type>&, t_type)> function()
  {
    return std::bind(
      static_cast<y_type(*)(const y_type&, t_type, const dydt_type<y_type>&, t_type)>(evaluate),
      std::placeholders::_1,
      std::placeholders::_2,
      std::placeholders::_3,
      std::placeholders::_4);
  }
};
}
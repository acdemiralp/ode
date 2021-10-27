#pragma once

#include <array>
#include <functional>
#include <utility>

#include <ode/tableau/tableau_traits.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/triangular_number.hpp>

namespace ode
{
template <typename tableau>
class explicit_method
{
public:
  using t_type    = typename tableau::type;
  template <typename y_type>
  using dydt_type = std::function<y_type(const y_type&, t_type)>;

  template <typename y_type>
  static constexpr auto evaluate(const y_type& y, t_type t, const dydt_type<y_type>& f, t_type h)
  {
    std::array<y_type, tableau::stages> k;
    constexpr_for<0, tableau::stages, 1>([&y, &t, &f, &h, &k] (auto i)
    {
      y_type sum;
      constexpr_for<0, i.value, 1>([&k, &sum, &i] (auto j)
      {
        sum += k[j] * std::get<triangular_number<i.value - 1> + j.value>(tableau::a);
      });
      k[i] = f(y + sum * h, t + std::get<i>(tableau::c) * h);
    });

    if constexpr (ode::is_extended_butcher_tableau_v<tableau>)
    {
      y_type higher, lower;
      constexpr_for<0, tableau::stages, 1>([&k, &higher, &lower] (auto i)
      {
        higher += k[i] * std::get<i>(tableau::b );
        lower  += k[i] * std::get<i>(tableau::bs);
      });
      return std::array<y_type, 2>{y + higher * h, (higher - lower) * h};
    }
    else
    {
      y_type sum;
      constexpr_for<0, tableau::stages, 1>([&k, &sum] (auto i)
      {
        sum += k[i] * std::get<i>(tableau::b);
      });
      return y_type(y + sum * h);
    }
  }

  template <typename y_type>
  static constexpr auto function()
  {
    return std::bind(evaluate<y_type>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  }
};
}
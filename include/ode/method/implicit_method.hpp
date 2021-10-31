#pragma once

#include <array>
#include <functional>
#include <utility>

#include <ode/tableau/tableau_traits.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/extended_result.hpp>

namespace ode
{
template <typename type>
class root_finding_algorithm
{
  
};

template <typename tableau_type_>
class implicit_method
{
public:
  using tableau_type = tableau_type_;

  template <typename problem_type>
  static constexpr auto apply(const problem_type& problem, const typename problem_type::time_type step_size)
  {
    using value_type = typename problem_type::value_type;

    std::array<value_type, tableau_type::stages> stages;
    constexpr_for<0, tableau_type::stages, 1>([&problem, &step_size, &stages] (auto i)
    {
      // TODO: Compute each k_i using a root finding algorithm e.g. Newton-Raphson.
      // auto y_next    = y;
      // auto t_next    = t + h;
      // 
      // auto dydt      = f(y, t_next);
      // auto b_value   = dydt * h;
      // auto jacobian  = j(y, t_next) * h - j_type::Identity();
      // y_next        -= jacobian.partialPivLu().solve(b_value);
      // 
      // while (b_value.l2() > std::numeric_limits<t_type>::epsilon())
      // {
      //   dydt    = f(y, t_next);
      //   b_value = y - y_next + dydt * h;
      //   // jacobian = j(y_next, t_next) * h - j_type::Identity(); // Modified Newton-Raphson computes the Jacobian only once.
      //   y_next -= jacobian.partialPivLu().solve(b_value);
      // }
      std::get<i>(stages) = value_type(42);
    });

    if constexpr (is_extended_butcher_tableau_v<tableau_type>)
    {
      value_type higher, lower;
      constexpr_for<0, tableau_type::stages, 1>([&stages, &higher, &lower] (auto i)
      {
        higher = higher + std::get<i>(stages) * std::get<i>(tableau_type::b );
        lower  = lower  + std::get<i>(stages) * std::get<i>(tableau_type::bs);
      });
      return extended_result<value_type> {problem.value + higher * step_size, (higher - lower) * step_size};
    }
    else
    {
      value_type sum;
      constexpr_for<0, tableau_type::stages, 1>([&stages, &sum] (auto i)
      {
        sum = sum + std::get<i>(stages) * std::get<i>(tableau_type::b);
      });
      return value_type(problem.value + sum * step_size);
    }
  }

  template <typename problem_type>
  static constexpr auto function()
  {
    return std::bind(apply<problem_type>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
  }
};
}
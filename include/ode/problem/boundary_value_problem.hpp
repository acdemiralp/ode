#pragma once

#include <array>
#include <functional>
#include <utility>

#include <ode/problem/higher_order_initial_value_problem.hpp>

namespace ode
{
template <typename time_type_, typename value_type_>
struct boundary_value_problem
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const value_type&, const value_type&)>;
  
  [[nodiscard]]
  constexpr auto to_initial_value_problem(const value_type& initial_guess = value_type()) const
  {
    return higher_order_initial_value_problem<time_type, value_type, 2>
    {
      std::get<0>(times),
      {
        std::get<0>(values),
        initial_guess
      },
      function
    };
  }

  std::array<time_type , 2> times   ;
  std::array<value_type, 2> values  ;
  function_type             function;
};
}
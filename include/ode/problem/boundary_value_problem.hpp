#pragma once

#include <functional>

#include <ode/problem/initial_value_problem.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/parameter_pack_expander.hpp>

namespace ode
{
template <typename value_type_, typename time_type_>
struct boundary_value_problem
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const value_type&, const value_type&)>;
  using coupled_type  = std::array<initial_value_problem<time_type, value_type>, 2>;

  [[nodiscard]]
  constexpr coupled_type to_coupled_first_order_initial_value_problems() const
  {
    coupled_type result;
    constexpr_for<0, 2, 1>([&](auto i)
    {

    });
    return result;
  }

  std::pair<time_type, value_type> value_0 ;
  std::pair<time_type, value_type> value_1 ;
  function_type                    function;
};
}
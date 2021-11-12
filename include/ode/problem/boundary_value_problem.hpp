#pragma once

#include <array>
#include <functional>
#include <utility>

#include <ode/problem/initial_value_problem.hpp>

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

    std::get<0>(result).time     = std::get<0>(times );                   // a
    std::get<0>(result).value    = std::get<0>(values);                   // y(a) = alpha
    std::get<0>(result).function = [ ] (time_type x, const value_type& y) // TODO: y' = z. Choose randomly?
    {
      
    };

    std::get<1>(result).time     = std::get<0>(times );                                                   // a
    std::get<1>(result).value    = std::get<0>(result).function(std::get<0>(times), std::get<0>(values)); // TODO z(a) = y'(a) = s
    std::get<1>(result).function = [&] (time_type x, const value_type& z)                                 // z' = f(x,y,z)
    {
      return function(x, std::get<0>(result).value, z);
    };

    // TODO: Newton-Raphson is applied after integration: y(std::get<1>(times), s) - std::get<1>(values) = 0.

    return result;
  }

  std::array<time_type , 2> times   ; // a    , b
  std::array<value_type, 2> values  ; // alpha, beta
  function_type             function;
};
}
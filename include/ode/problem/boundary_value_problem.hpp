#pragma once

#include <functional>

namespace ode
{
template <typename value_type_, typename time_type_>
struct boundary_value_problem
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const value_type&, const value_type&)>; // TODO: Depends on the order.

  time_type     time_0   ; // TODO: Depends on the order.
  value_type    value_0  ; // TODO: Depends on the order.
  time_type     time_1   ; // TODO: Depends on the order.
  value_type    value_1  ; // TODO: Depends on the order.
  function_type function ;
};
}
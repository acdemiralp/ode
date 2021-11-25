#pragma once

#include <functional>

namespace ode
{
template <typename time_type_, typename value_type_>
struct initial_value_problem
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const value_type&)>;

  time_type     time    = time_type(0);
  value_type    value   = {};
  function_type function;
};
}
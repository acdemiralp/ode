#pragma once

#include <functional>

namespace ode
{
template <typename value_type_, typename time_type_>
struct initial_value_problem
{
  using value_type      = value_type_;
  using time_type       = time_type_ ;
  using derivative_type = std::function<value_type(const value_type&, time_type)>;

  value_type      value     ;
  time_type       time      ;
  derivative_type derivative;
};
}
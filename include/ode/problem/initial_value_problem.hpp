#pragma once

#include <functional>

namespace ode
{
template <typename value_type, typename time_type>
struct initial_value_problem
{
  using function_type = std::function<value_type(const value_type&, time_type)>;

  value_type    y;
  time_type     t;
  function_type f;
};
}
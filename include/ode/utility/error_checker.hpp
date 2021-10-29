#pragma once

namespace ode
{
struct error_checker
{
  template <typename value_type>
  static bool      accept    (const value_type& error)
  {
    return true;
  }
  template <typename value_type, typename time_type>
  static time_type adapt_step(const value_type& error, const time_type step_size)
  {
    return step_size;
  }
};
}
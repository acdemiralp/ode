#pragma once

namespace ode
{
struct error_checker
{
  template <typename value_type>
  static bool      accept    (const value_type& error)
  {
    constexpr static auto absolute_error = 1e-6;
    constexpr static auto relative_error = 1e-6;

    auto tolerance = || |err_i| / (absolute_error + relative_error * (a_x * |x_i| + a_dxdt_lower * |dxdt_i| )) ||

    return true;
  }
  template <typename value_type, typename time_type>
  static time_type adapt_step(const value_type& error, const time_type step_size)
  {
    return step_size;
  }
};
}
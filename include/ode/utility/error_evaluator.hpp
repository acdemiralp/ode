#pragma once

#include <ode/utility/error_evaluation.hpp>

namespace ode
{
struct error_evaluator
{
  constexpr static auto absolute_error = 1e-6;
  constexpr static auto relative_error = 1e-6;

  template <typename problem_type, typename time_type, typename result_type>
  static error_evaluation<time_type> evaluate(const problem_type& problem, const time_type step_size, const result_type& result)
  {
    // TODO: Compute tolerance.  auto tolerance = || |err_i| / (absolute_error + relative_error * (a_x * |x_i| + a_dxdt_lower * |dxdt_i| )) ||
    // TODO: Compute next step size according to tolerance.
    
    return {true, 42};
  }
};
}
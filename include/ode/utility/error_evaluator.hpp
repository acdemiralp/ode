#pragma once

#include <ode/utility/error_evaluation.hpp>

namespace ode
{
struct error_evaluator
{
  constexpr static auto absolute_tolerance = 1e-6;
  constexpr static auto relative_tolerance = 1e-3;

  template <typename problem_type, typename time_type, typename result_type>
  static error_evaluation<time_type> evaluate(
    const problem_type& problem  , 
    const time_type     step_size, 
    const result_type&  result   )
  {
    //time_type error = NORM(result.error / (absolute_tolerance + relative_tolerance * MAX(problem.value, result.value)));
    time_type error = NORM(result.error / (absolute_tolerance + relative_tolerance * ABS(result.value)));
    return {error < time_type(1), step_size / error};
  }
};
}
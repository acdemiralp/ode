#pragma once

#include <cmath>

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
    time_type sum;
    for (auto i = 0; i < std::size(problem.value); ++i)
      sum += std::pow(std::abs(result.error[i] / (absolute_tolerance + relative_tolerance * std::max(problem.value[i], result.value[i]))), 2);

    time_type error = std::sqrt(sum / std::size(problem.value));

    // TODO: INTEGRAL CONTROLLER.

    return {error < time_type(1), step_size / error};
  }
};
}
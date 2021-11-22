#pragma once

#include <cmath>

#include <ode/utility/error_evaluation.hpp>

namespace ode
{
template <typename type, type absolute_tolerance = type(1e-6), type relative_tolerance = type(1e-3)>
struct error_evaluator
{
  template <typename problem_type, typename time_type, typename result_type>
  static error_evaluation<time_type> evaluate(
    const problem_type& problem  , 
    const time_type     step_size, 
    const result_type&  result   )
  {
    time_type squared_sum;
    for (auto i = 0; i < problem.value.size(); ++i)                                                                                          // TODO: PROBLEMATIC FOR SCALAR, MATRIX, TENSOR VALUE_TYPE.
      squared_sum += std::pow(result.error[i] / (absolute_tolerance + relative_tolerance * std::max(problem.value[i], result.value[i])), 2); // TODO: PROBLEMATIC FOR SCALAR, MATRIX, TENSOR VALUE_TYPE.

    time_type hairer_norm = std::sqrt(squared_sum / problem.value.size());
    // TODO: INTEGRAL CONTROLLER.

    return {hairer_norm < time_type(1), step_size / hairer_norm};
  }
};
}
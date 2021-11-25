#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

#include <ode/algebra/quantity_operations.hpp>
#include <ode/error/error_evaluation.hpp>
#include <ode/error/extended_result.hpp>
#include <ode/tableau/order.hpp>

namespace ode
{
template <typename method_type_, typename problem_type_>
struct proportional_integral_controller
{
  using method_type  = method_type_                       ;
  using problem_type = problem_type_                      ;
  using tableau_type = typename method_type ::tableau_type;
  using time_type    = typename problem_type::time_type   ;
  using value_type   = typename problem_type::value_type  ;
  using operations   = quantity_operations<value_type>    ;

  // Reference: https://doi.org/10.1007/978-3-642-05221-7 Chapter: IV.2, Section: A PI Step Size Control, Equation: 2.43c
  error_evaluation<time_type> evaluate(const problem_type& problem, const time_type step_size, const extended_result<value_type>& result)
  {
    time_type squared_sum;
    operations::for_each([&] (const auto& p, const auto& r, const auto& e)
    {
      squared_sum += std::pow(std::abs(e) / (absolute_tolerance + relative_tolerance * std::max(std::abs(p), std::abs(r))), 2);
    }, problem.value, result.value, result.error);

    time_type error   = std::sqrt(std::real(squared_sum) / operations::size(problem.value));
    time_type optimal = factor * std::pow(time_type(1) / std::abs(error), alpha) * std::pow(std::abs(previous_error), beta);
    time_type limited = std::min (factor_maximum, std::max(factor_minimum, optimal));

    const bool accept = error <= time_type(1);
    if (accept)
      previous_error = error;

    return {accept, step_size * limited};
  }
  
  const time_type absolute_tolerance = time_type(1e-6);
  const time_type relative_tolerance = time_type(1e-3);
  const time_type factor             = time_type(0.8 );
  const time_type factor_minimum     = time_type(1e-2);
  const time_type factor_maximum     = time_type(1e+2);
  const time_type alpha              = time_type(7.0 / (10.0 * order<tableau_type>));
  const time_type beta               = time_type(4.0 / (10.0 * order<tableau_type>));
  time_type       previous_error     = time_type(1e-3);
};
}
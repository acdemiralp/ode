#pragma once

#include <cmath>

#include <ode/algebra/quantity_operations.hpp>
#include <ode/error/error_evaluation.hpp>
#include <ode/error/extended_result.hpp>
#include <ode/tableau/order.hpp>

namespace ode
{
template <typename method_type_, typename problem_type_>
struct integral_controller
{
  using method_type  = method_type_                     ;
  using problem_type = problem_type_                    ;
  using time_type    = typename problem_type::time_type ;
  using value_type   = typename problem_type::value_type;
  using operations   = quantity_operations<value_type>  ;

  error_evaluation<time_type> evaluate(const problem_type& problem, const time_type step_size, const extended_result<value_type>& result)
  {
    time_type squared_sum;
    operations::for_each(problem.value, result.value, result.error, [&] (const auto& p, const auto& r, const auto& e)
    {
      squared_sum += std::pow(std::abs(e / (absolute_tolerance + relative_tolerance * std::max(p, r))), 2);
    });

    time_type hairer_norm  = std::sqrt(std::real(squared_sum) / operations::size(problem.value));
    time_type intermediate = std::pow (hairer_norm, time_type(1) / order<typename method_type::tableau_type>) / gamma;
    time_type scaling      = std::max (time_type(1) / maximum_scaling, std::min(time_type(1) / minimum_scaling, intermediate));

    return {hairer_norm < time_type(1), step_size / scaling};
  }
  
  const time_type absolute_tolerance = time_type(1e-6);
  const time_type relative_tolerance = time_type(1e-3);
  const time_type minimum_scaling    = time_type(1e-3);
  const time_type maximum_scaling    = time_type(1e+3);
  const time_type gamma              = time_type(0.9 );
};
}
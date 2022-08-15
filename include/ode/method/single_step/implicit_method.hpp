#pragma once

#include <array>
#include <utility>

#include <ode/error/extended_result.hpp>
#include <ode/parallel/cuda.hpp>
#include <ode/root_finding/newton_raphson.hpp>
#include <ode/tableau/tableau.hpp>
#include <ode/utility/constexpr_for.hpp>

namespace ode
{
template <typename tableau_type_, typename root_finder_ = newton_raphson>
class implicit_method
{
public:
  using tableau_type = tableau_type_;
  using root_finder  = root_finder_ ;

  template <typename problem_type>
  __device__ __host__ static constexpr auto apply(const problem_type& problem, const typename problem_type::time_type step_size)
  {
    using value_type = typename problem_type::function_type::result_type;

    std::array<value_type, tableau::stages_v<tableau_type>> stages;
    constexpr_for<0, tableau::stages_v<tableau_type>, 1>([&problem, &step_size, &stages] (auto i)
    {
      value_type sum = root_finder::apply(problem.time, problem.value, problem.function, problem.function_derivative, 1 /* iterations */);
      std::get<i>(stages) = problem.function(problem.time + std::get<i>(tableau::c_v<tableau_type>) * step_size, problem.value + sum * step_size); // TODO
    });

    if constexpr (tableau::is_extended_v<tableau_type>)
    {
      value_type higher, lower;
      constexpr_for<0, tableau::stages_v<tableau_type>, 1>([&stages, &higher, &lower] (auto i)
      {
        higher += std::get<i>(stages) * std::get<i>(tableau::b_v <tableau_type>);
        lower  += std::get<i>(stages) * std::get<i>(tableau::bs_v<tableau_type>);
      });
      return extended_result<value_type> {problem.value + higher * step_size, (higher - lower) * step_size};
    }
    else
    {
      value_type sum;
      constexpr_for<0, tableau::stages_v<tableau_type>, 1>([&stages, &sum] (auto i)
      {
        sum += std::get<i>(stages) * std::get<i>(tableau::b_v<tableau_type>);
      });
      return value_type(problem.value + sum * step_size);
    }
  }
};
}
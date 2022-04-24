#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>

#include <ode/parallel/cuda.hpp>
#include <ode/tableau/tableau_traits.hpp>
#include <ode/utility/constexpr_for.hpp>

namespace ode
{
template <typename method_type_, typename problems_type_>
class coupled_fixed_step_iterator
{
public:
  using method_type       = method_type_;
  using problems_type     = problems_type_;
  using time_type         = typename problems_type::value_type::time_type;

  using iterator_category = std::input_iterator_tag; // Single pass read forward.
  using difference_type   = std::ptrdiff_t;
  using value_type        = problems_type;
  using pointer           = value_type const*;
  using reference         = value_type const&;

  __device__ __host__ constexpr reference                    operator* () const noexcept
  {
    return problems;
  }
  __device__ __host__ constexpr pointer                      operator->() const noexcept
  {
    return &problems;
  }

  __device__ __host__ constexpr coupled_fixed_step_iterator& operator++()
  {
    constexpr_for<0, std::tuple_size_v<problems_type>, 1> ([&] (auto i)
    {
      if constexpr (is_extended_butcher_tableau_v<typename method_type::tableau_type>)
        std::get<i>(problems).value = method_type::apply(std::get<i>(problems), step_size).value;
      else
        std::get<i>(problems).value = method_type::apply(std::get<i>(problems), step_size);

      std::get<i>(problems).time += step_size;
    });

    return *this;
  }
  __device__ __host__ constexpr coupled_fixed_step_iterator  operator++(std::int32_t)
  {
    coupled_fixed_step_iterator temp = *this;
    ++(*this);
    return temp;
  }

  __device__ __host__ friend constexpr bool                  operator==(const coupled_fixed_step_iterator& lhs, const coupled_fixed_step_iterator& rhs) noexcept
  {
    return lhs.problems == rhs.problems && lhs.step_size == rhs.step_size;
  }
  
  problems_type problems     ;
  time_type     step_size {1};
};
}
#pragma once

#include <cstddef>
#include <iterator>

#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename method_type_, typename problem_type_>
class fixed_step_iterator
{
public:
  using method_type       = method_type_;
  using problem_type      = problem_type_;
  using time_type         = typename problem_type::time_type;

  using iterator_category = std::input_iterator_tag; // Single pass read forward.
  using difference_type   = std::ptrdiff_t;
  using value_type        = problem_type_;
  using pointer           = value_type const*;
  using reference         = value_type const&;

  explicit fixed_step_iterator             (const problem_type& problem, const time_type step_size)
  : problem_(problem), step_size_(step_size)
  {
    
  }
  fixed_step_iterator                      (const fixed_step_iterator&  that) = default;
  fixed_step_iterator                      (      fixed_step_iterator&& temp) = default;
  virtual ~fixed_step_iterator             ()                                 = default;
  fixed_step_iterator&           operator= (const fixed_step_iterator&  that) = default;
  fixed_step_iterator&           operator= (      fixed_step_iterator&& temp) = default;

  constexpr reference            operator* () const noexcept
  {
    return problem_;
  }
  constexpr pointer              operator->() const noexcept
  {
    return &problem_;
  }

  constexpr fixed_step_iterator& operator++()
  {
    if constexpr (is_extended_butcher_tableau_v<typename method_type::tableau_type>)
      problem_.value = method_type::evaluate(problem_, step_size_).value;
    else
      problem_.value = method_type::evaluate(problem_, step_size_);

    problem_.time += step_size_;

    return *this;
  }
  constexpr fixed_step_iterator& operator++(std::int32_t)
  {
    fixed_step_iterator temp = *this;
    ++(*this);
    return temp;
  }

  friend constexpr bool          operator==(const fixed_step_iterator& lhs, const fixed_step_iterator& rhs) noexcept
  {
    return lhs.problem_ == rhs.problem_ && lhs.step_size_ == rhs.step_size_;
  }
  
protected:
  problem_type problem_  ;
  time_type    step_size_;
};
}
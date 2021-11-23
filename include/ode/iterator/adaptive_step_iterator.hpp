#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include <ode/error/controller/integral_controller.hpp>
#include <ode/tableau/tableau_traits.hpp>

namespace ode
{
template <typename method_type_, typename problem_type_, typename error_evaluator_type_ = integral_controller<method_type_, problem_type_>>
class adaptive_step_iterator
{
public:
  using method_type          = method_type_;
  using problem_type         = problem_type_;
  using error_evaluator_type = error_evaluator_type_;
  using time_type            = typename problem_type::time_type;

  using iterator_category    = std::input_iterator_tag; // Single pass read forward.
  using difference_type      = std::ptrdiff_t;
  using value_type           = problem_type_;
  using pointer              = value_type const*;
  using reference            = value_type const&;

  explicit adaptive_step_iterator             (
    const problem_type&         problem           , 
    const time_type             initial_step_size = time_type(1), 
    const error_evaluator_type& error_evaluator   = error_evaluator_type())
  : problem_(problem), step_size_(initial_step_size), error_evaluator_(error_evaluator)
  {
    
  }
  adaptive_step_iterator                      (const adaptive_step_iterator&  that) = default;
  adaptive_step_iterator                      (      adaptive_step_iterator&& temp) = default;
  virtual ~adaptive_step_iterator             ()                                    = default;
  adaptive_step_iterator&           operator= (const adaptive_step_iterator&  that) = default;
  adaptive_step_iterator&           operator= (      adaptive_step_iterator&& temp) = default;

  constexpr reference               operator* () const noexcept
  {
    return  problem_;
  }
  constexpr pointer                 operator->() const noexcept
  {
    return &problem_;
  }

  constexpr adaptive_step_iterator& operator++()
  {
    if constexpr (is_extended_butcher_tableau_v<typename method_type::tableau_type>)
    {
      const auto result     = method_type::apply       (problem_, step_size_);
      const auto evaluation = error_evaluator_.evaluate(problem_, step_size_, result);

      if (evaluation.accept)
      {
        problem_.value = result.value;
        problem_.time += step_size_;
        step_size_     = evaluation.next_step_size;
      }
      else
      {
        step_size_     = evaluation.next_step_size;
        return operator++(); // Retry with the adapted step size.
      }
    }
    else
    {
      // Non-extended Butcher tableau may be used with adaptive step iterators, but their step size will not be adapted.
      problem_.value = method_type::apply(problem_, step_size_);
      problem_.time += step_size_;
    }
    return *this;
  }
  constexpr adaptive_step_iterator& operator++(std::int32_t)
  {
    adaptive_step_iterator temp = *this;
    ++(*this);
    return temp;
  }

  friend constexpr bool             operator==(const adaptive_step_iterator& lhs, const adaptive_step_iterator& rhs) noexcept
  {
    return lhs.problem_ == rhs.problem_ && lhs.step_size_ == rhs.step_size_;
  }

protected:
  problem_type         problem_        ;
  time_type            step_size_      ;
  error_evaluator_type error_evaluator_;
};
}
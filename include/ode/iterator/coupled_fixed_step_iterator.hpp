#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>

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

  explicit coupled_fixed_step_iterator             (const problems_type& problems, const time_type step_size)
  : problems_(problems), step_size_(step_size)
  {
    
  }
  coupled_fixed_step_iterator                      (const coupled_fixed_step_iterator&  that) = default;
  coupled_fixed_step_iterator                      (      coupled_fixed_step_iterator&& temp) = default;
  virtual ~coupled_fixed_step_iterator             ()                                         = default;
  coupled_fixed_step_iterator&           operator= (const coupled_fixed_step_iterator&  that) = default;
  coupled_fixed_step_iterator&           operator= (      coupled_fixed_step_iterator&& temp) = default;

  constexpr reference                    operator* () const noexcept
  {
    return problems_;
  }
  constexpr pointer                      operator->() const noexcept
  {
    return &problems_;
  }

  constexpr coupled_fixed_step_iterator& operator++()
  {
    constexpr_for<0, std::tuple_size_v<problems_type>, 1> ([&] (auto i)
    {
      if constexpr (is_extended_butcher_tableau_v<typename method_type::tableau_type>)
        std::get<i>(problems_).value = method_type::apply(std::get<i>(problems_), step_size_).value;
      else
        std::get<i>(problems_).value = method_type::apply(std::get<i>(problems_), step_size_);

      std::get<i>(problems_).time += step_size_;
    });

    return *this;
  }
  constexpr coupled_fixed_step_iterator& operator++(std::int32_t)
  {
    coupled_fixed_step_iterator temp = *this;
    ++(*this);
    return temp;
  }

  friend constexpr bool                  operator==(const coupled_fixed_step_iterator& lhs, const coupled_fixed_step_iterator& rhs) noexcept
  {
    return lhs.problems_ == rhs.problems_ && lhs.step_size_ == rhs.step_size_;
  }
  
protected:
  problems_type problems_ ;
  time_type     step_size_;
};
}
#pragma once

#include <iterator>

namespace ode
{
template <typename type>
class fixed_step_size_iterator
{
public:
  using iterator_category = std::input_iterator_tag; // Single pass read forward.
  using difference_type   = std::ptrdiff_t;
  using value_type        = type;
  using pointer           = value_type*;
  using reference         = value_type&;

  fixed_step_size_iterator           ()                                      = default;
  fixed_step_size_iterator           (const fixed_step_size_iterator&  that) = default;
  fixed_step_size_iterator           (      fixed_step_size_iterator&& temp) = default;
  virtual ~fixed_step_size_iterator  ()                                      = default;
  fixed_step_size_iterator& operator=(const fixed_step_size_iterator&  that) = default;
  fixed_step_size_iterator& operator=(      fixed_step_size_iterator&& temp) = default;

protected:

};
}
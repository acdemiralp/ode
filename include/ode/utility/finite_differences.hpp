#pragma once

#include <ode/utility/combination.hpp>
#include <ode/utility/constexpr_for.hpp>

namespace ode
{
// First-order finite differences.
// Requires:
//   std::is_arithmetic_v<value_type> == true
//   value_type                 operator+(value_type                , value_type                ) Note: In-built for all arithmetic types.
//   value_type                 operator-(value_type                , value_type                ) Note: In-built for all arithmetic types.
//   value_type                 operator*(value_type                , value_type                ) Note: In-built for all arithmetic types.
//   function_type::result_type operator-(function_type::result_type, function_type::result_type)
//   function_type::result_type operator/(function_type::result_type, value_type                )
template <typename function_type, typename value_type>
typename function_type::result_type forward_difference (const function_type& function, const value_type& value, const value_type spacing = value_type(0.001))
{
  return (function(value + spacing) - function(value          )) / spacing;
}
template <typename function_type, typename value_type>
typename function_type::result_type backward_difference(const function_type& function, const value_type& value, const value_type spacing = value_type(0.001))
{
  return (function(value          ) - function(value - spacing)) / spacing;
}
template <typename function_type, typename value_type>
typename function_type::result_type central_difference (const function_type& function, const value_type& value, const value_type spacing = value_type(0.001))
{
  return (function(value + spacing) - function(value - spacing)) / (value_type(2) * spacing);
}

// Higher-order finite differences.
// Requires:
//   std::is_arithmetic_v<value_type> == true
//   value_type                  operator+ (value_type                 , value_type                ) Note: In-built for all arithmetic types.
//   value_type                  operator- (value_type                 , value_type                ) Note: In-built for all arithmetic types.
//   value_type                  operator* (value_type                 , value_type                ) Note: In-built for all arithmetic types.
//   value_type                  operator/ (value_type                 , value_type                ) Note: In-built for all arithmetic types.
//   function_type::result_type& operator+=(function_type::result_type&, function_type::result_type)
//   function_type::result_type  operator* (function_type::result_type , value_type                )
//   function_type::result_type  operator/ (function_type::result_type , value_type                )
template <std::size_t order, typename function_type, typename value_type>
typename function_type::result_type forward_difference (const function_type& function, const value_type& value, const value_type spacing = value_type(0.001))
{
  typename function_type::result_type sum {};
  constexpr_for<0, order + 1, 1>([&function, &value, &spacing, &sum] (auto i)
  {
    sum += 
      static_cast<value_type>((order - i.value) % 2 == 0 ? 1 : -1)
      * combination_v<value_type, order, i.value> 
      * function(value + static_cast<value_type>(i.value) * spacing);
  });
  return sum / static_cast<value_type>(std::pow(spacing, order));
}
template <std::size_t order, typename function_type, typename value_type>
typename function_type::result_type backward_difference(const function_type& function, const value_type& value, const value_type spacing = value_type(0.001))
{
  typename function_type::result_type sum {};
  constexpr_for<0, order + 1, 1>([&function, &value, &spacing, &sum] (auto i)
  {
    sum += 
      static_cast<value_type>(i.value % 2 == 0 ? 1 : -1)
      * combination_v<value_type, order, i.value> 
      * function(value - static_cast<value_type>(i.value) * spacing);
  });
  return sum / static_cast<value_type>(std::pow(spacing, order));
}
template <std::size_t order, typename function_type, typename value_type>
typename function_type::result_type central_difference (const function_type& function, const value_type& value, const value_type spacing = value_type(0.001))
{
  typename function_type::result_type sum {};
  constexpr_for<0, order + 1, 1>([&function, &value, &spacing, &sum] (auto i)
  {
    sum += 
      static_cast<value_type>(i.value % 2 == 0 ? 1 : -1)
      * combination_v<value_type, order, i.value> 
      * function(value + (static_cast<value_type>(order) / static_cast<value_type>(2) - static_cast<value_type>(i.value)) * spacing);
  });
  return sum / static_cast<value_type>(std::pow(spacing, order));
}

// TODO: Higher-order higher-stencil finite differences.
}
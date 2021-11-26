#pragma once

#include <cstddef>

#include <ode/algebra/quantity_traits.hpp>

namespace ode
{
template <typename value_type>
struct quantity_operations { };

template <scalar value_type>
struct quantity_operations<value_type>
{
  template <typename function_type, scalar... value_types>
  static void        for_each(const function_type& function, const value_types&... values)
  {
    function(values...);
  }
  static std::size_t size    (const value_type& value)
  {
    return 1;
  }
};

template <vector value_type>
struct quantity_operations<value_type>
{
  template <typename function_type, vector... value_types>
  static void        for_each(const function_type& function, const value_types&... values)
  {
    const auto element_count = std::get<0>(std::tie(values...)).size();
    for (auto i = 0; i < element_count; ++i)
      function(values.data()[i]...);
  }
  static std::size_t size    (const value_type& value)
  {
    return value.size();
  }
};

template <matrix value_type>
struct quantity_operations<value_type>
{
  template <typename function_type, matrix... value_types>
  static void        for_each(const function_type& function, const value_types&... values)
  {
    const auto element_count = std::get<0>(std::tie(values...)).size();
    for (auto i = 0; i < element_count; ++i)
      function(values.data()[i]...);
  }
  static std::size_t size    (const value_type& value)
  {
    return value.size();
  }
};

template <tensor value_type>
struct quantity_operations<value_type>
{
  template <typename function_type, tensor... value_types>
  static void        for_each(const function_type& function, const value_types&... values)
  {
    const auto element_count = std::get<0>(std::tie(values...)).size();
    for (auto i = 0; i < element_count; ++i)
      function(values.data()[i]...);
  }
  static std::size_t size    (const value_type& value)
  {
    return value.size();
  }
};
}
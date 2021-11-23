#pragma once

#include <cstddef>

#include <ode/algebra/quantity_traits.hpp>

namespace ode
{
template <typename value>
struct quantity_operations { };

template <scalar value>
struct quantity_operations<value>
{
  static std::size_t size    (const value& v)
  {
    return 1;
  }
  // TODO for_each
};

template <vector value>
struct quantity_operations<value>
{
  static std::size_t size    (const value& v)
  {
    return v.size();
  }
  static void        for_each(value&... v, const std::function<void(value::value_type&...)>& function)
  {
    for (auto i = 0; i < v.size(); ++i)
      function(v.data()[i]...);
  }
};

template <matrix value>
struct quantity_operations<value>
{
  static std::size_t size(const value& v)
  {
    return v.size();
  }
  // TODO for_each
};

template <tensor value>
struct quantity_operations<value>
{
  static std::size_t size(const value& v)
  {
    return v.size();
  }
  // TODO for_each
};
}
#pragma once

#include <functional>

namespace ode
{
class newton_raphson_method
{
public:
  template <typename type>
  static type apply(
    const type&                             value     ,
    const std::function<type(const type&)>& function  ,
    const std::function<type(const type&)>& derivative)
  {
    return value - function(value) / derivative(value);
  }
};
}
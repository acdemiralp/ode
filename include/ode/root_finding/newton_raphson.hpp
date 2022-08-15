#pragma once

namespace ode
{
class newton_raphson
{
public:
  template <
    typename time_type               , 
    typename value_type              , 
    typename function_type           , 
    typename function_derivative_type>
  static value_type apply(
    const time_type                 time                ,
    const value_type&               value               ,
    const function_type&            function            ,
    const function_derivative_type& function_derivative ,
    const std::size_t               iterations          = 1)
  {
    return {};
  }
};
}
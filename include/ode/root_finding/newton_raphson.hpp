#pragma once

namespace ode
{
template <
  typename time_type               ,
  typename value_type              , 
  typename function_type           , 
  typename derivative_function_type>
constexpr value_type newton_raphson(
  const time_type                 time               ,
  const value_type&               value              , 
  const function_type&            function           , 
  const derivative_function_type& derivative_function)
{
  return value - function(time, value) / derivative_function(time, value);
}
}
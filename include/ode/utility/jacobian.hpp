#pragma once

#include <ode/utility/finite_differences.hpp>

namespace ode
{
template <typename function_type>
std::size_t argument_count_v                                                 = 0;
template <typename return_value, typename... argument_types>
std::size_t argument_count_v<std::function<return_value(argument_types...)>> = sizeof...(argument_types);

template <typename function_type>
auto jacobian(const function_type& function)
{
  using result_type          = typename function_type::result_type;
  using result_gradient_type = ode::gradient<result_type>;

  // Example function_type:
  // - Scalar             valued function with scalar parameter x                            (Jacobian is   scalar):
  //   a                = x * x                                                              
  // - Vector             valued function with scalar parameter x                            (Jacobian is a vector of same dimensions as value):
  //   [a, b]           = [x, x * x]                                                         
  // - Second rank tensor valued function with scalar parameter x                            (Jacobian is a tensor of same dimensions as value):
  //   [[a, b], [c, d]] = [[x, x * x], [x * x, x]]                                           
  //                                                                                         
  // - Scalar             valued function with vector parameter [x, y]                       (Jacobian is a vector             of same dimensions as the parameter, classic scalar field):
  //   a                = x * y                                                              
  // - Vector             valued function with vector parameter [x, y]                       (Jacobian is a second rank tensor of value_size x parameter_size     , classic vector field):
  //   [a, b]           = [x * x, x * y]                                                     
  // - Second rank tensor valued function with vector parameter [x, y]                       (Jacobian is a third  rank tensor of value_size x parameter_size     , classic tensor field):
  //   [[a, b], [c, d]] = [[x, x * y], [x * y, y]]
  //
  // - Scalar             valued function with second rank tensor parameter [[x, y], [z, w]] (Jacobian is a second rank tensor of same dimensions as the parameter):
  //   a                = x * y * z * w
  // - Vector             valued function with second rank tensor parameter [[x, y], [z, w]] (Jacobian is a third  rank tensor of ...):
  //   [a, b]           = [x * y, z * w]
  // - Second rank tensor valued function with second rank tensor parameter [[x, y], [z, w]] (Jacobian is a fourth rank tensor of ...):
  //   [[a, b], [c, d]] = [[x * x, y * y], [z * z, w * w]]
}
}
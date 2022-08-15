#pragma once

#include <ode/utility/finite_differences.hpp>

namespace ode
{
template <typename function_type, typename parameter_type>
struct jacobian_type { };

template <typename function_type, typename parameter_type>
auto jacobian(const function_type& function, const parameter_type& parameter)
{
  using jacobian_type = typename jacobian_type<function_type, parameter_type>::value;

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
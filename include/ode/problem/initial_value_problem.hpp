#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ode/parallel/cuda.hpp>
#include <ode/utility/parameter_pack_expander.hpp>

namespace ode
{
template <typename time_type , typename value_type , std::size_t order, typename = std::make_index_sequence<order>>
struct initial_value_problem;

template <typename time_type_, typename value_type_, std::size_t order, std::size_t... sequence>
struct initial_value_problem<time_type_, value_type_, order, std::index_sequence<sequence...>>
{
  using time_type               = time_type_ ;
  using value_type              = value_type_;
  using function_type           = function<value_type(time_type, const parameter_pack_expander<value_type, sequence>&...)>;
  using first_order_system_type = std::array<initial_value_problem<time_type, value_type, 1>, order>;


  time_type                     time    = time_type(0);
  std::array<value_type, order> values  = {};
  function_type                 function;
};
}
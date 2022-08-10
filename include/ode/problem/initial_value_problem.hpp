#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ode/parallel/cuda.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/parameter_pack_expander.hpp>

namespace ode
{
template <typename time_type , typename value_type , std::size_t order = 1, typename = std::make_index_sequence<order>>
struct initial_value_problem;
template <typename time_type_, typename value_type_, std::size_t order, std::size_t... sequence>
struct initial_value_problem<time_type_, value_type_, order, std::index_sequence<sequence...>>
{
  using time_type               = time_type_ ;
  using value_type              = value_type_;
  using value_container_type    = std::conditional_t<order == 1, value_type, std::array<value_type, order>>;
  using function_type           = function<value_type(time_type, const parameter_pack_expander<value_type, sequence>&...)>;

  using first_order_type        = initial_value_problem<time_type, value_type, 1>;
  using first_order_system_type = std::array<first_order_type, order>;

  template <std::size_t index, std::size_t sequence_index> [[nodiscard]]
  constexpr const value_type&       select_argument         (const value_type& value, const first_order_system_type& result) const
  {
    if constexpr (index == sequence_index)
      return value;
    else
      return std::get<sequence_index>(result).value;
  }
  [[nodiscard]]
  constexpr first_order_system_type to_system_of_first_order() const
  {
    first_order_system_type result {};
    constexpr_for<0, order, 1>([&] (auto i)
    {
      // TODO: Check whether capturing the result by reference leads to correct behavior.
      std::get<i>(result) = {time, std::get<i>(values), [=, &result] __device__ (time_type t, const value_type& v)
      {
        return function(t, select_argument<i.value, sequence>(v, result)...);
      }};
    });
    return result;
  }

  time_type            time     {};
  value_container_type values   {};
  function_type        function {};
};
}
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

#include <ode/problem/boundary_value_problem.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/parameter_pack_expander.hpp>

namespace ode
{
template <typename time_type , typename value_type , std::size_t order, typename = std::make_index_sequence<order>>
struct higher_order_boundary_value_problem;
template <typename time_type_, typename value_type_, std::size_t order, std::size_t... sequence>
struct higher_order_boundary_value_problem<time_type_, value_type_, order, std::index_sequence<sequence...>>
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const parameter_pack_expander<value_type, sequence>&...)>;
  using coupled_type  = std::array<boundary_value_problem<time_type, value_type>, order>;

  [[nodiscard]]
  constexpr coupled_type to_coupled_first_order() const
  {
    coupled_type result;
    constexpr_for<0, order, 1>([&] (auto i)
    {

    });
    return result;
  }

  std::array<std::pair<time_type, value_type>, order> values  ;
  function_type                                       function;
};
}
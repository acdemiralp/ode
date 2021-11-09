#pragma once

#include <functional>

#include <ode/problem/initial_value_problem.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/parameter_pack_expander.hpp>

namespace ode
{
template <typename value_type_, typename time_type_>
struct two_point_boundary_value_problem
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const value_type&, const value_type&)>;
  using coupled_type  = std::array<initial_value_problem<time_type, value_type>, 2>;

  [[nodiscard]]
  constexpr coupled_type to_coupled_first_order_initial_value_problems() const
  {
    coupled_type result;
    constexpr_for<0, 2, 1>([&](auto i)
    {
      std::get<i>(result) = 
      {
        time,
        std::get<i>(values),
        [=, this, &result] (time_type t, const value_type& v)
        {
          return function(t, select_argument<i.value, sequence>(v, result)...);
        }
      };
    });
    return result;
  }

  std::pair<time_type, value_type> value_0 ;
  std::pair<time_type, value_type> value_1 ;
  function_type                    function;
};

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
      std::get<i>(result) = 
      {
        time,
        std::get<i>(values),
        [=, this, &result] (time_type t, const value_type& v)
        {
          // Example resolution for order 3:
          // return function(t, v              , result[1].value, result[2].value);
          // return function(t, result[0].value, v              , result[2].value);
          // return function(t, result[0].value, result[1].value, v              );
          return function(t, select_argument<i.value, sequence>(v, result)...);
        }
      };
    });
    return result;
  }

  std::array<std::pair<time_type, value_type>, order> values  ;
  function_type                                       function;
};
}
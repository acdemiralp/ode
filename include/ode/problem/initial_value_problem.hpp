#pragma once

#include <functional>

#include <ode/utility/constexpr_for.hpp>

namespace ode
{
template <typename time_type_, typename value_type_>
struct initial_value_problem
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const value_type&)>;

  time_type     time      ;
  value_type    value     ;
  function_type function  ;
};

template<typename type, std::size_t order, typename = std::make_index_sequence<order>>
struct expander;
template<typename type, std::size_t order, std::size_t... sequence>
struct expander<type, order, std::index_sequence<sequence...>>
{
  template<std::size_t>
  using type = type;

  std::function<std::size_t(decltype(sequence)...)> foo;
};

// Higher order initial value problems are decomposed into coupled first order initial value problems.
template <typename time_type_, typename value_type_, std::size_t order, typename = std::make_index_sequence<order>>
struct initial_value_problem_n;

template <typename time_type_, typename value_type_, std::size_t order, std::size_t... sequence>
struct initial_value_problem_n<time_type_, value_type_, order, std::index_sequence<sequence...>>
{
  using time_type     = time_type_ ;
  using value_type    = value_type_;
  using function_type = std::function<value_type(time_type, const std::array<value_type, order>&)>;

  std::unique_ptr<std::array<initial_value_problem<time_type, value_type>, order>> to_coupled_initial_value_problems()
  {
    auto result = std::make_unique<std::array<initial_value_problem<time_type, value_type>, order>>();
    constexpr_for<0, order, 1>([&] (auto& i)
    {
      std::get<i>(*result).time     = time;
      std::get<i>(*result).value    = std::get<i>(values);
      std::get<i>(*result).function = [=, &result] (time_type time, const value_type& value)
      {
        return function(time, {value          , result[1].value, result[2].value});
        return function(time, {result[0].value, value          , result[2].value}); // TODO
        return function(time, {result[0].value, result[1].value, value          }); // TODO
      };
    });
    return std::move(result);
  }

  time_type                     time    ;
  std::array<value_type, order> values  ;
  function_type                 function;
};
}
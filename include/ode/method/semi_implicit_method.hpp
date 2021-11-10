#pragma once

#include <array>
#include <functional>
#include <utility>

#include <ode/tableau/tableau_traits.hpp>
#include <ode/utility/constexpr_for.hpp>
#include <ode/utility/extended_result.hpp>

namespace ode
{
template <typename tableau_type_>
class semi_implicit_method
{
  using tableau_type = tableau_type_;
};
}
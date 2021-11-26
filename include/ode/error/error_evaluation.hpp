#pragma once

namespace ode
{
template <typename type>
struct error_evaluation
{
  bool accept;
  type next_step_size;
};
}
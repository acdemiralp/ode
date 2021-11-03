#pragma once

#include <cstddef>

namespace ode
{
template <typename type, std::size_t unused>
using parameter_pack_expander = type;
}
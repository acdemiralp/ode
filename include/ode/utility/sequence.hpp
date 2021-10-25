#pragma once

namespace ode
{
// A compile-time sequence of objects of type.
template <typename type, type...>
struct sequence { };
}
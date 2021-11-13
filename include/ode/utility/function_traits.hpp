#pragma once

#include <cstdint>
#include <type_traits>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ode
{
template <typename argument_type, typename = void>
struct scalar_function_traits { };
template <typename argument_type>
struct scalar_function_traits<argument_type, std::enable_if_t<std::is_arithmetic_v<argument_type>>>
{
  using gradient_type  = argument_type;
  using potential_type = argument_type;
};
template <typename value_type, std::int32_t size                      , std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct scalar_function_traits<Eigen::Matrix         <value_type, size, 1      , options, max_rows, max_columns>>
{
  using gradient_type  = Eigen::Matrix<value_type, 1, size, options, max_rows, max_columns>; // Row vector for coherence with the Jacobian.
  using potential_type = void;
};
template <typename value_type, std::int32_t rows, std::int32_t columns, std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct scalar_function_traits<Eigen::Matrix         <value_type, rows, columns, options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, typename dimensions                    , std::int32_t options, typename index_type>
struct scalar_function_traits<Eigen::TensorFixedSize<value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t dimensions                , std::int32_t options, typename index_type>
struct scalar_function_traits<Eigen::Tensor         <value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};

template <typename argument_type, typename = void>
struct vector_function_traits { };
template <typename argument_type>
struct vector_function_traits<argument_type, std::enable_if_t<std::is_arithmetic_v<argument_type>>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t size                      , std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct vector_function_traits<Eigen::Matrix         <value_type, size, 1      , options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t rows, std::int32_t columns, std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct vector_function_traits<Eigen::Matrix         <value_type, rows, columns, options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, typename dimensions                    , std::int32_t options, typename index_type>
struct vector_function_traits<Eigen::TensorFixedSize<value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t dimensions                , std::int32_t options, typename index_type>
struct vector_function_traits<Eigen::Tensor         <value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};

template <typename argument_type, typename = void>
struct matrix_function_traits { };
template <typename argument_type>
struct matrix_function_traits<argument_type, std::enable_if_t<std::is_arithmetic_v<argument_type>>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t size                      , std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct matrix_function_traits<Eigen::Matrix         <value_type, size, 1      , options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t rows, std::int32_t columns, std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct matrix_function_traits<Eigen::Matrix         <value_type, rows, columns, options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, typename dimensions                    , std::int32_t options, typename index_type>
struct matrix_function_traits<Eigen::TensorFixedSize<value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t dimensions                , std::int32_t options, typename index_type>
struct matrix_function_traits<Eigen::Tensor         <value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};

template <typename argument_type, typename = void>
struct tensor_function_traits { };
template <typename argument_type>
struct tensor_function_traits<argument_type, std::enable_if_t<std::is_arithmetic_v<argument_type>>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t size                      , std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct tensor_function_traits<Eigen::Matrix         <value_type, size, 1      , options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t rows, std::int32_t columns, std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct tensor_function_traits<Eigen::Matrix         <value_type, rows, columns, options, max_rows, max_columns>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, typename dimensions                    , std::int32_t options, typename index_type>
struct tensor_function_traits<Eigen::TensorFixedSize<value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};
template <typename value_type, std::int32_t dimensions                , std::int32_t options, typename index_type>
struct tensor_function_traits<Eigen::Tensor         <value_type, dimensions   , options, index_type>>
{
  using gradient_type  = void;
  using potential_type = void;
};
}
#pragma once

#include <cstdint>
#include <type_traits>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ode
{
template <typename type>
struct is_scalar : std::is_arithmetic<type> { };

template <typename type>
struct is_vector : std::false_type {};
template <typename type, std::int32_t size, std::int32_t options, std::int32_t max_rows>
struct is_vector    <Eigen::Matrix<type, size, 1, options, max_rows, 1>> : std::true_type {};

template <typename type>
struct is_row_vector : std::false_type {};
template <typename type, std::int32_t size, std::int32_t options, std::int32_t max_columns>
struct is_row_vector<Eigen::Matrix<type, 1, size, options, 1, max_columns>> : std::true_type {};

template <typename type, typename = void>
struct is_matrix : std::false_type {};
template <typename type, std::int32_t rows, std::int32_t columns, std::int32_t options, std::int32_t max_rows, std::int32_t max_columns>
struct is_matrix<Eigen::Matrix<type, rows, columns, options, max_rows, max_columns>, std::enable_if_t<rows != 1 && columns != 1>> : std::true_type {};

template <typename type>
struct is_tensor : std::false_type {};
template <typename type, typename     dimensions, std::int32_t options, typename index_type>
struct is_tensor<Eigen::TensorFixedSize<type, dimensions, options, index_type>> : std::true_type {};
template <typename type, std::int32_t dimensions, std::int32_t options, typename index_type>
struct is_tensor<Eigen::Tensor         <type, dimensions, options, index_type>> : std::true_type {};

template <typename type>
inline constexpr bool is_scalar_v     = is_scalar    <type>::value;
template <typename type>
inline constexpr bool is_vector_v     = is_vector    <type>::value;
template <typename type>
inline constexpr bool is_row_vector_v = is_row_vector<type>::value;
template <typename type>
inline constexpr bool is_matrix_v     = is_matrix    <type>::value;
template <typename type>
inline constexpr bool is_tensor_v     = is_tensor    <type>::value;

template <typename type>
concept scalar     = is_scalar_v    <type>;
template <typename type>
concept vector     = is_vector_v    <type>;
template <typename type>
concept row_vector = is_row_vector_v<type>;
template <typename type>
concept matrix     = is_matrix_v    <type>;
template <typename type>
concept tensor     = is_tensor_v    <type>;

template <scalar value_type, scalar argument_type>
struct function_traits
{
  // Scalar function with scalar argument.
  using gradient_type  = value_type;
  using potential_type = value_type;
};
template <scalar value_type, vector argument_type>
struct function_traits
{
  // Scalar function with vector argument.
  using gradient_type  = void; // TODO: A row vector.
  using potential_type = void; // TODO: A scalar?
};
template <scalar value_type, matrix argument_type>
struct function_traits
{
  // Scalar function with matrix argument.
  using gradient_type  = void;
  using potential_type = void;
};
template <scalar value_type, tensor argument_type>
struct function_traits
{
  // Scalar function with tensor argument.
  using gradient_type  = void;
  using potential_type = void;
};
}
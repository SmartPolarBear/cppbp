//
// Created by cleve on 5/25/2022.
//

#pragma once

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

namespace cppbp::base
{
using MatrixType = Eigen::MatrixXd;
using VectorType = Eigen::VectorXd;

template<typename Scalar, int NumIndices, int Options, typename IndexType>
using TensorType = Eigen::Tensor<Scalar, NumIndices, Options, IndexType>;

static inline constexpr double EPS = 1e-5;
}// namespace cppbp::base

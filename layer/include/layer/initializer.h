//
// Created by cleve on 5/25/2022.
//

#pragma once

#include <base/base.h>

#include <memory>

namespace cppbp::layer
{
class IWeightInitializer
{
 public:
	virtual base::MatrixType initialize_weights(size_t rows, size_t cols, size_t ni, size_t no) = 0;
};
}// namespace cppbp::layer

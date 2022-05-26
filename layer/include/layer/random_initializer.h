//
// Created by cleve on 5/25/2022.
//

#pragma once

#include <base/base.h>

#include <layer/initializer.h>

#include <memory>

namespace cppbp::layer
{
class RandomInitializer
	: public IWeightInitializer
{
public:
    base::MatrixType initialize_weights(size_t rows, size_t cols, size_t ni, size_t no) override;
};
}// namespace cppbp::layer
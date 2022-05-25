//
// Created by cleve on 5/25/2022.
//

#pragma once

#include <base/base.h>

#include <layer/initializer.h>

#include <memory>

namespace cppbp::layer
{
class XavierInitializer
	: public IWeightInitializer
{
public:
    base::MatrixType initialize_weights(size_t w, size_t h, size_t ni, size_t no) override;
};
}// namespace cppbp::layer

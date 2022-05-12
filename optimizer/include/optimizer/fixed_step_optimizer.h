//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <optimizer/optimizer.h>

#include <cstdint>

namespace cppbp::optimizer
{
class FixedStepOptimizer
	: public IOptimizer
{
 public:
	explicit FixedStepOptimizer(double lr);

	void step() override;
	double optimize(double prev, double grad) override;
 private:
	uint64_t step_{};
	double lr_{};
};
}
//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <optimizer/optimizer.h>

#include <Eigen/Eigen>

#include <cstdint>
#include "Eigen/Core"

namespace cppbp::optimizer
{
class FixedStepOptimizer
	: public IOptimizer
{
 public:
	explicit FixedStepOptimizer(double lr);

	void step() override;
	Eigen::MatrixXd optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads) override;
 private:
	uint64_t step_{};
	double lr_{};
};
}
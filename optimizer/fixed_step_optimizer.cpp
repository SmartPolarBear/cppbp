//
// Created by cleve on 5/12/2022.
//

#include <optimizer/fixed_step_optimizer.h>

#include <algorithm>
#include "Eigen/Core"

using namespace std;

cppbp::optimizer::FixedStepOptimizer::FixedStepOptimizer(double lr)
	: lr_(lr)
{
}

void cppbp::optimizer::FixedStepOptimizer::step()
{
	step_++;
}

Eigen::MatrixXd cppbp::optimizer::FixedStepOptimizer::optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads)
{
	return prev - lr_ * grads;
}


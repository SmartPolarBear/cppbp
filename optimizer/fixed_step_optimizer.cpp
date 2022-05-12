//
// Created by cleve on 5/12/2022.
//

#include <optimizer/fixed_step_optimizer.h>

#include <algorithm>

using namespace std;

void cppbp::optimizer::FixedStepOptimizer::step()
{
	step_++;
}

double cppbp::optimizer::FixedStepOptimizer::optimize(double prev, double grads)
{
	return prev - lr_ * grads;
}

cppbp::optimizer::FixedStepOptimizer::FixedStepOptimizer(double lr)
	: lr_(lr)
{
}

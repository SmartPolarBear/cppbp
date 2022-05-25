//
// Created by cleve on 5/11/2022.

#include <base/base.h>
#include <optimizer/sgd_optimizer.h>

#include <Eigen/Core>

using namespace std;

using namespace cppbp::base;

cppbp::optimizer::SGDOptimizer::SGDOptimizer(double lr, double momentum, double weight_decay, double dampening, bool nesterov, bool maximize)
	: lr_(lr),
	  momentum_(momentum),
	  weight_decay_(weight_decay),
	  dampening_(dampening),
	  nesterov_(nesterov),
	  maximize_(maximize),
	  step_(0)
{
}

void cppbp::optimizer::SGDOptimizer::step()
{
	step_++;
}

Eigen::MatrixXd cppbp::optimizer::SGDOptimizer::optimize(Eigen::MatrixXd params, Eigen::MatrixXd grads)
{
	if (std::fabs(weight_decay_) > EPS)
	{
		if (step_ > 0)
		{
			grads += weight_decay_ * params;
		}
	}

	auto b = grads;
	if (std::fabs(momentum_) > EPS)
	{
		if (step_ > 0)
		{
			b = momentum_ * prev_b_ + (1 - dampening_) * grads;
		}

		if (nesterov_ && step_ > 0)
		{
			grads = prev_grads_ + momentum_ * b;
		}
		else
		{
			grads = b;
		}
	}

	if (maximize_)
	{
		params += lr_ * grads;
	}
	else
	{
		params -= lr_ * grads;
	}

	prev_b_ = b;
	prev_grads_ = grads;

	return params;
}
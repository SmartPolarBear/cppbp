//
// Created by cleve on 5/11/2022.

#include <optimizer/sgd_optimizer.h>
#include "Eigen/Core"

using namespace std;

cppbp::optimizer::SGDOptimizer::SGDOptimizer(double lr,double mometum,double weight_decay,double dampening,bool nesterov,bool maximize)
:
	  lr_(lr),
	  momentum_(mometum),
	  weight_decay_(weight_decay),
	  dampening_(dampening),
	  nesterov_(nesterov),
	  maximize_(maximize)
{
}

void cppbp::optimizer::SGDOptimizer::step()
{
	step_++;
}
/*
 * Eigen::MatrixXd
 * element size : dynamic
 * element type : double
 * */
Eigen::MatrixXd cppbp::optimizer::SGDOptimizer::optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads)
{
	for (int i = 0; i < prev.rows(); ++i)
	{
		auto b = grads(i);
		if(weight_decay_)
			grads(i) += weight_decay_*prev(i);
		if(momentum_){
			if(i>0)
				b = momentum_ * b + (1 - dampening_)*grads(i);
			else
				b = grads(i);

			if(nesterov_)
				grads(i) = grads(i) + momentum_*b;
			else
				grads(i) = b;
		}

		if(maximize_)
		{
			prev(i) += weight_decay_*grads(i);
		}
		else
			prev(i) -= weight_decay_*grads(i);
	}

	return prev ;
}
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
class SGDOptimizer
	:public  IOptimizer
{
 public:
	explicit SGDOptimizer(double lr,double mometum,double weight_decay,double dampening,bool nesterov,bool maximize);

	void step() override;
	Eigen::MatrixXd optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads) override;
 private:
	uint64_t step_{}; //8 bytes int : The number of iterations
	double lr_{}; // learning rate
	//momentum factor
	double momentum_{};
	//weight_decay (L2 penalty)
	double dampening_{};
	double weight_decay_{};
	//enables Nesterov momentum
	bool nesterov_{false};
	//maximize the params based on the objective, instead of minimizing
	bool maximize_{};

};
}

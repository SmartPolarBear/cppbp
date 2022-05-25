//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <optimizer/optimizer.h>

#include <Eigen/Eigen>

#include "Eigen/Core"
#include <cstdint>

namespace cppbp::optimizer
{
class SGDOptimizer
	: public IOptimizer
{
 public:
	explicit SGDOptimizer(double lr, double mometum = 0, double weight_decay = 0, double dampening = 0, bool nesterov = 0, bool maximize = false);

	void step() override;
	Eigen::MatrixXd optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads) override;

 private:
	uint64_t step_{};//8 bytes int : The number of iterations

	double lr_{};    // learning rate
	//momentum factor
	double momentum_{};
	//weight_decay (L2 penalty)
	double dampening_{};
	double weight_decay_{};
	//enables Nesterov momentum
	bool nesterov_{false};
	//maximize the params based on the objective, instead of minimizing
	bool maximize_{};

	Eigen::MatrixXd prev_b_{};
	Eigen::MatrixXd prev_grads_{};

};
}// namespace cppbp::optimizer

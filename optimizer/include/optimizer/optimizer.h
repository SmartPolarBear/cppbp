//
// Created by cleve on 5/12/2022.
//

#pragma once


#include <vector>

#include <Eigen/Eigen>

namespace cppbp::optimizer
{
class IOptimizer
{
 public:
	virtual void step() = 0;
	virtual Eigen::MatrixXd optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads) = 0;
};

class IOptimizable
{
 public:
	virtual void optimize(IOptimizer&) = 0;
};
}
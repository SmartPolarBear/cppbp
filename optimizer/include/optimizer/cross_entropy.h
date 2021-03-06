//
// Created by δΉι»εζ on 2022/5/23.
//

#pragma once

#include <optimizer/loss.h>

namespace cppbp::optimizer
{
class CrossEntropyLoss
	: public ILossFunction
{
 public:
	double operator()(Eigen::VectorXd value, Eigen::VectorXd label) override;
	double eval(Eigen::VectorXd value, Eigen::VectorXd label) override;
	Eigen::VectorXd derive(Eigen::VectorXd value, Eigen::VectorXd label) override;
	uint32_t type_id() override;
};
}
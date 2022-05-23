//
// Created by 九黎千明 on 2022/5/23.
//

#pragma once

#include <optimizer/loss.h>

namespace cppbp::optimizer
{
class CELoss
	: public ILossFunction
{
 public:
	double operator()(Eigen::VectorXd value, Eigen::VectorXd label) override;
	double eval(Eigen::VectorXd value, Eigen::VectorXd label) override;
	Eigen::VectorXd derive(Eigen::VectorXd value, Eigen::VectorXd label) override;

};
}
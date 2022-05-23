//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <optimizer/loss.h>


namespace cppbp::optimizer
{
class MSELoss
	: public ILossFunction
{
 public:
	double operator()(Eigen::VectorXd value, Eigen::VectorXd label) override;
	double eval(Eigen::VectorXd value, Eigen::VectorXd label) override;
	Eigen::VectorXd derive(Eigen::VectorXd value, Eigen::VectorXd label) override;
	uint32_t type_id() override;

};


}
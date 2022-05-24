//
// Created by cleve on 5/11/2022.
//

#pragma once
#include <layer/activation_function.h>

namespace cppbp::layer
{

class Sigmoid final
	: public IActivationFunction
{
 public:
	uint32_t type_id() override;
	double operator()(double x) override;
	double eval(double x) override;
	double derive(double y) override;
	Eigen::VectorXd eval(Eigen::VectorXd x) override;
	Eigen::VectorXd derive(Eigen::VectorXd y) override;
};

}

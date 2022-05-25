//
// Created on 5/21/2022.
//

#pragma once
#include <layer/activation_function.h>

namespace cppbp::layer
{
class Tanh final
	: public IActivationFunction
{
 public:
	double operator()(double x) override;
	double eval(double x) override;
	double derive(double y) override;
	Eigen::VectorXd eval(Eigen::VectorXd x) override;
	Eigen::VectorXd derive(Eigen::VectorXd y) override;
};
}

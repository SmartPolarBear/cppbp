//
// Created by 九黎千明 on 2022/5/24.
//
#pragma once
#include <layer/activation_function.h>

namespace cppbp::layer
{
class softmax
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

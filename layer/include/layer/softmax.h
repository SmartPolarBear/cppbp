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
	softmax()
	{   MAX=0.0;
		total=0.0;}
	uint32_t type_id() override;
	double operator()(double x) override;
	double eval(double x) override;
	double derive(double y) override;
	Eigen::VectorXd eval(Eigen::VectorXd x) override;
	Eigen::MatrixXd derive(Eigen::VectorXd y) override;
 private:
	double MAX;
	double total;
};
}

//
// Created by 九黎千明 on 2022/5/23.
//
#include <optimizer/cross_entropy.h>

#include <iostream>

double cppbp::optimizer::CELoss::eval(Eigen::VectorXd value, Eigen::VectorXd label)
{
/*	auto diff = value - label;

	double ret = 0.5 * (diff.dot(diff)) / value.size();
	return ret;*/
	auto loss = 0.;
	for (auto i = 0; i < 3; ++i)
		loss += -label[i] * std::log(value[i]);
	return loss;
}

double cppbp::optimizer::CELoss::operator()(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return eval(value, label);
}

Eigen::VectorXd cppbp::optimizer::CELoss::derive(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return value - label;
}
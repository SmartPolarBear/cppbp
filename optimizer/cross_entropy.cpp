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
	auto loss = 0.0;
	for (auto i = 0; i < value.size(); ++i)
		loss += label[i] * std::log(value[i])+(1-label[i])*std::log(1-value[i]);
	return -loss/value.size();
}

double cppbp::optimizer::CELoss::operator()(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return eval(value, label);
}

Eigen::VectorXd cppbp::optimizer::CELoss::derive(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return value - label;
}
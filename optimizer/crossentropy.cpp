//
// Created by 九黎千明 on 2022/5/23.
//
#include <optimizer/crossentropy.h>

#include <iostream>

double cppbp::optimizer::CrossEntropyELoss::eval(Eigen::VectorXd value, Eigen::VectorXd label)
{
/*	auto diff = value - label;

	double ret = 0.5 * (diff.dot(diff)) / value.size();
	return ret;*/
//logistic回归交叉熵
/*	auto loss = 0.0;
	for (auto i = 0; i < value.size(); ++i)
		loss += label[i] * std::log(value[i])+(1-label[i])*std::log(1-value[i]);
	return -loss/value.size();*/
//softmax回归交叉熵
    auto loss=0.0;
	for (auto i = 0; i < value.size(); ++i)
			loss += -label[i] * std::log(value[i]);
	return loss;
}

double cppbp::optimizer::CrossEntropyELoss::operator()(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return eval(value, label);
}

Eigen::VectorXd cppbp::optimizer::CrossEntropyELoss::derive(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return value-label;
}
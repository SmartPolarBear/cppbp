//
// Created by 九黎千明 on 2022/5/23.
//
#include <optimizer/cross_entropy.h>

#include <iostream>

double cppbp::optimizer::CrossEntropyLoss::eval(Eigen::VectorXd value, Eigen::VectorXd label)
{
	auto loss_items = label.array().cwiseProduct(value.array().log());
	return -loss_items.sum();
}

double cppbp::optimizer::CrossEntropyLoss::operator()(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return eval(value, label);
}

Eigen::VectorXd cppbp::optimizer::CrossEntropyLoss::derive(Eigen::VectorXd value, Eigen::VectorXd label)
{
	auto ret = -label.cwiseQuotient(value);
	return ret;
}

uint32_t cppbp::optimizer::CrossEntropyLoss::type_id()
{
	return 1;
}
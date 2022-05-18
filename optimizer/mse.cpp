//
// Created by cleve on 5/12/2022.
//

#include <optimizer/mse.h>

#include <iostream>

double cppbp::optimizer::MSELoss::eval(Eigen::VectorXd value, Eigen::VectorXd label)
{
	auto diff = value - label;

	double ret = 0.5 * (diff.dot(diff)) / value.size();
	return ret;
}

double cppbp::optimizer::MSELoss::operator()(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return eval(value, label);
}

Eigen::VectorXd cppbp::optimizer::MSELoss::derive(Eigen::VectorXd value, Eigen::VectorXd label)
{
	return label - value;
}




//
// Created by cleve on 5/19/2022.
//

#include <layer/relu.h>

#include <algorithm>

double cppbp::layer::Relu::operator()(double x)
{
	return eval(x);
}

double cppbp::layer::Relu::eval(double x)
{
	return std::max(0.0, x);
}

double cppbp::layer::Relu::derive(double y)
{
	if (y > 0)
	{
		return 1;
	}
	return 0;
}

Eigen::VectorXd cppbp::layer::Relu::eval(Eigen::VectorXd x)
{
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = eval(x[i]);
	}
	return x;
}

Eigen::VectorXd cppbp::layer::Relu::derive(Eigen::VectorXd y)
{
	for (int i = 0; i < y.size(); i++)
	{
		y[i] = derive(y[i]);
	}
	return y;
}

//
// Created by 九黎千明 on 2022/5/24.
//

#include <layer/softmax.h>

#include <algorithm>
double cppbp::layer::softmax::operator()(double x)
{
	return eval(x);
}

double cppbp::layer::softmax::eval(double x)
{
	return std::exp(x);
}

double cppbp::layer::softmax::derive(double y)
{
	return y*(1-y);
}

Eigen::VectorXd cppbp::layer::softmax::eval(Eigen::VectorXd x)
{
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = eval(x[i]);
		total+=x[i];
	}
	x=x/total;
	return x;
}

Eigen::VectorXd cppbp::layer::softmax::derive(Eigen::VectorXd y)
{
	for (int i = 0; i < y.size(); i++)
	{
		y[i] = derive(y[i]);
	}
	return y;
}
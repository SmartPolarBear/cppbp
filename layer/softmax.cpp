//
// Created by 九黎千明 on 2022/5/24.
//

#include <layer/softmax.h>

#include <algorithm>
double total=0.0;
double MAX=0.0;
double cppbp::layer::softmax::operator()(double x)
{
	return eval(x);
}

double cppbp::layer::softmax::eval(double x)
{
	return std::exp(x-MAX);
}

double cppbp::layer::softmax::derive(double y)
{
	return y/total;
}

Eigen::VectorXd cppbp::layer::softmax::eval(Eigen::VectorXd x)
{
	MAX=x[0];
	for (int i = 0; i < x.size(); i++)
	{
		MAX=std::max(MAX,x[i]);
	}
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = eval(x[i]);
		total+=x[i];
	}
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
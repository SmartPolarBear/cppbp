/* #include <layer/sigmoid.h>

#include <cmath>

double cppbp::layer::Sigmoid::operator()(double x)
{
	return eval(x);
}

double cppbp::layer::Sigmoid::eval(double x)
{
	return 1 / (1 + std::exp(-x));
}

double cppbp::layer::Sigmoid::derive(double y)
{
	return y * (1 - y);
}

Eigen::VectorXd cppbp::layer::Sigmoid::eval(Eigen::VectorXd x)
{
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = eval(x[i]);
	}
	return x;
}

Eigen::VectorXd cppbp::layer::Sigmoid::derive(Eigen::VectorXd y)
{
	for (int i = 0; i < y.size(); i++)
	{
		y[i] = derive(y[i]);
	}
	return y;
}
 */
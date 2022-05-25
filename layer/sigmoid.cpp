#include <layer/sigmoid.h>

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

Eigen::MatrixXd cppbp::layer::Sigmoid::derive(Eigen::VectorXd y)
{
	Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(y.size(), y.size());
	for (int i = 0; i < y.size(); i++)
	{
		ret(i, i) = derive(y[i]);
	}
	return ret;
}

uint32_t cppbp::layer::Sigmoid::type_id()
{
	return 1;
}

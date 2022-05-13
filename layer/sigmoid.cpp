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

double cppbp::layer::Sigmoid::derive(double x)
{
	return eval(x) * (1 - eval(x));
}
#include <layer/tanh.h>

#include <cmath>

using namespace cppbp::base;

double cppbp::layer::Tanh::operator()(double x)
{
    return eval(x);
}

double cppbp::layer::Tanh::eval(double x)
{
    return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}

double cppbp::layer::Tanh::derive(double y)
{
    return 1 - y * y;
}

Eigen::VectorXd cppbp::layer::Tanh::eval(Eigen::VectorXd x)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = eval(x[i]);
    }
    return x;
}

MatrixType cppbp::layer::Tanh::derive(Eigen::VectorXd y)
{
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(y.size(), y.size());
    for (int i = 0; i < y.size(); i++)
    {
        ret(i, i) = derive(y[i]);
    }
    return y;
}

uint32_t cppbp::layer::Tanh::type_id()
{
    return 3;
}

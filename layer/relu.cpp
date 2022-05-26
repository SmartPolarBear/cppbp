//
// Created by cleve on 5/19/2022.
//

#include <layer/relu.h>
#include <layer/he_initializer.h>

#include <algorithm>

using namespace cppbp::layer;

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

Eigen::MatrixXd cppbp::layer::Relu::derive(Eigen::VectorXd y)
{
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(y.size(), y.size());
    for (int i = 0; i < y.size(); i++)
    {
        ret(i, i) = derive(y[i]);
    }
    return ret;
}

uint32_t cppbp::layer::Relu::type_id()
{
    return 2;
}

std::shared_ptr<IWeightInitializer> cppbp::layer::Relu::default_initializer()
{
    return IWeightInitializer::make<HeInitializer>();
}

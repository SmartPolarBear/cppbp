//
// Created by 九黎千明 on 2022/5/24.
//

#include <layer/Softmax.h>
#include <layer/xavier_initializer.h>
#include <layer/random_initializer.h>

#include <base/exceptions.h>

#include <algorithm>
#include <iostream>

#include <gsl/assert>

using namespace std;
using namespace gsl;

using namespace cppbp::layer;

using namespace Eigen;

double cppbp::layer::Softmax::operator()(double x)
{
    return eval(x);
}

double cppbp::layer::Softmax::eval(double x)
{
    throw base::not_implemented{};
}

double cppbp::layer::Softmax::derive(double y)
{
    throw base::not_implemented{};
}

Eigen::VectorXd cppbp::layer::Softmax::eval(Eigen::VectorXd x)
{
    Expects(!x.hasNaN());

    auto exps = x.array().exp();
    Eigen::VectorXd ret(exps.cwiseQuotient(exps.sum() * Eigen::ArrayXd::Ones(exps.size())));

    Ensures(!ret.hasNaN());

    return ret;
}

Eigen::MatrixXd cppbp::layer::Softmax::derive(Eigen::VectorXd y)
{
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(y.size(), y.size());
    for (int i = 0; i < y.size(); i++)
    {
        for (int j = 0; j < y.size(); j++)
        {
            if (i == j)
            {
                ret(i, j) = y[i] * (1 - y[i]);
            }
            else
            {
                ret(i, j) = -y[i] * y[j];
            }
        }
    }
    return ret;
}

uint32_t cppbp::layer::Softmax::type_id()
{
    return 1;
}

shared_ptr<IWeightInitializer> cppbp::layer::Softmax::default_initializer()
{
    return IWeightInitializer::make<RandomInitializer>();
}

//
// Created by 九黎千明 on 2022/5/24.
//

#include <layer/softmax.h>

#include <algorithm>
#include <iostream>

#include <gsl/assert>

using namespace std;
using namespace gsl;

using namespace Eigen;

double cppbp::layer::softmax::operator()(double x)
{
    return eval(x);
}

double cppbp::layer::softmax::eval(double x)
{
    return 0.0;//TODO:throw
}

double cppbp::layer::softmax::derive(double y)
{
    return 0.0;//TODO:throw
}

Eigen::VectorXd cppbp::layer::softmax::eval(Eigen::VectorXd x)
{
    Expects(!x.hasNaN());

    auto exps = x.array().exp();
    Eigen::VectorXd ret(exps.cwiseQuotient(exps.sum() * Eigen::ArrayXd::Ones(exps.size())));

    Ensures(!ret.hasNaN());

    return ret;
}

Eigen::MatrixXd cppbp::layer::softmax::derive(Eigen::VectorXd y)
{
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(y.size(), y.size());
    for (int i = 0; i < y.size(); i++)
    {
        for (int j = 0; j < y.size(); j++)
        {
            if (i == j)
            {
                ret(i, j) = y[i] * (1 - y[i]);
            } else
            {
                ret(i, j) = -y[i] * y[j];
            }
        }
    }
    return ret;
}

uint32_t cppbp::layer::softmax::type_id()
{
    return 1;
}
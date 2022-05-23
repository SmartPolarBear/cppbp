//
// Created by cleve on 5/11/2022.

#include <optimizer/sgd_optimizer.h>
#include "Eigen/Core"

using namespace std;

cppbp::optimizer::Sgd_optimizer::Sgd_optimizer(double lr)
:lr_(lr)
{
}

void cppbp::optimizer::Sgd_optimizer::step()
{
	step_++;
}
Eigen::MatrixXd cppbp::optimizer::Sgd_optimizer::optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads)
{

}
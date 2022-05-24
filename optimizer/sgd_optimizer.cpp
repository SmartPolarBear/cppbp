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
/*
 * Eigen::MatrixXd
 * element size : dynamic
 * element type : double
 * */
Eigen::MatrixXd cppbp::optimizer::Sgd_optimizer::optimize(Eigen::MatrixXd prev, Eigen::MatrixXd grads)
{	//
	Eigen::VectorXd g = Eigen::VectorXd::Ones(prev.rows());

	for (int i = 0; i < prev.rows()/10; ++i)
	{
		int k = rand()%prev.rows()+0;
		g[k] = 1;
	}

	for (int i = 0; i < g.size(); ++i)
	{
		if(g(i) == 0)
		{
			for(int j = 0;j < prev.cols();j++)
				grads(i,j) = 0;
		}
	}
	return prev + lr_ * grads;
}
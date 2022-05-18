//
// Created by cleve on 5/11/2022.
//
#pragma once

#include <Eigen/Eigen>

namespace cppbp::layer
{
class IActivationFunction
{
 public:
	virtual double operator()(double x) = 0;

	virtual double eval(double x) = 0;
	virtual double derive(double y) = 0;

	virtual Eigen::VectorXd eval(Eigen::VectorXd x) = 0;
	virtual Eigen::VectorXd derive(Eigen::VectorXd y) = 0;
};
}
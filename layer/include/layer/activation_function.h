//
// Created by cleve on 5/11/2022.
//
#pragma once

#include <base/type_id.h>

#include <Eigen/Eigen>

namespace cppbp::layer
{
class IActivationFunction
	: public base::ITypeId
{
 public:
	virtual double operator()(double x) = 0;

	virtual double eval(double x) = 0;
	virtual double derive(double y) = 0;

	virtual Eigen::VectorXd eval(Eigen::VectorXd x) = 0;
	virtual Eigen::VectorXd derive(Eigen::VectorXd y) = 0;

};
}
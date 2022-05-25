//
// Created by cleve on 5/13/2022.
//

#pragma once

#include <layer/activation_function.h>

#include <base/type_id.h>

#include <Eigen/Eigen>

#include <vector>

namespace cppbp::optimizer
{
class ILossFunction
	: public base::ITypeId
{
 public:
	virtual double operator()(Eigen::VectorXd value, Eigen::VectorXd label) = 0;
	virtual double eval(Eigen::VectorXd value, Eigen::VectorXd label) = 0;
	virtual Eigen::VectorXd derive(Eigen::VectorXd value, Eigen::VectorXd label) = 0;
};
}
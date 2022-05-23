//
// Created by cleve on 5/19/2022.
//

#pragma once

#include <layer/activation_function.h>

#include <model/persist.h>

namespace cppbp::layer
{
class Relu
	: public IActivationFunction
{
 public:
	double operator()(double x) override;
	double eval(double x) override;
	double derive(double y) override;
	Eigen::VectorXd eval(Eigen::VectorXd x) override;
	Eigen::VectorXd derive(Eigen::VectorXd y) override;
};

template<>
struct cppbp::model::persist::ActivationFunctionTypeId<Relu>
{
	static inline constexpr uint32_t value = 2;
};

}
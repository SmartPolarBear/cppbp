//
// Created by cleve on 5/11/2022.
//

#pragma once
#include <layer/activation_function.h>

namespace cppbp::layer
{
class Sigmoid final
	: public IActivationFunction
{
 public:
	double operator()(double x) override;
	double eval(double x) override;
	double derive(double x) override;
};
}

//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <layer/activation_function.h>

namespace cppbp::layer
{
class Neuron
{
 public:
	Neuron(IActivationFunction& af, double val);

	double operator()();
	double operator()(double x);

	double derive();

	[[nodiscard]] double value() const
	{
		return val_;
	}
 private:
	IActivationFunction* act_func_;
	double val_;
};
}

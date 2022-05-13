//
// Created by cleve on 5/13/2022.
//

#pragma once

#include <layer/activation_function.h>

#include <vector>

namespace cppbp::optimizer
{
class ILossFunction
{
 public:
	virtual double operator()(std::vector<double> value, std::vector<double> label) = 0;
	virtual double eval(std::vector<double> value, std::vector<double> label) = 0;
	virtual double derive(std::vector<double> value, std::vector<double> label, uint64_t idx) = 0;

	virtual std::vector<double> error(std::vector<double> value,
		std::vector<double> label,
		layer::IActivationFunction& f) = 0;
};
}
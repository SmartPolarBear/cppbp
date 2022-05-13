//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <optimizer/loss.h>

namespace cppbp::optimizer
{
class MSELoss
	: public ILossFunction
{
 public:
	double operator()(std::vector<double> value, std::vector<double> label) override;
	double eval(std::vector<double> value, std::vector<double> label) override;
	double derive(std::vector<double> value, std::vector<double> label, uint64_t idx) override;

	std::vector<double> error(std::vector<double> value,
		std::vector<double> label,
		layer::IActivationFunction& f) override;

}
}
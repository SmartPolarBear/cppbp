//
// Created by cleve on 5/12/2022.
//

#include <optimizer/mse.h>

double cppbp::optimizer::MSELoss::eval(std::vector<double> value, std::vector<double> label)
{
	double ssum = 0.0;
	for (int i = 0; i < value.size(); i++)
	{
		ssum += (value[i] - label[i]) * (value[i] - label[i]);
	}

	return 0.5 * ssum / value.size();
}

double cppbp::optimizer::MSELoss::operator()(std::vector<double> value, std::vector<double> label)
{
	return eval(value, label);
}

double cppbp::optimizer::MSELoss::derive(std::vector<double> value, std::vector<double> label, uint64_t idx)
{
	return label[idx] - value[idx];
}

std::vector<double> cppbp::optimizer::MSELoss::error(
	std::vector<double> value,
	std::vector<double> label,
	layer::IActivationFunction& f)
{
	std::vector<double> errors{};
	for (int i = 0; i < value.size(); i++)
	{
		errors.emplace_back(-derive(value, label, i) * value[i] * f.derive(value[i]));
	}
	return errors;
}



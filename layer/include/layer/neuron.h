//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <layer/activation_function.h>

#include <base/forward.h>
#include <base/backprop.h>

#include <unordered_map>
#include <memory>

namespace cppbp::layer
{
class Neuron
	: public std::enable_shared_from_this<Neuron>,
	  public base::IForward,
	  public base::IBackProp
{
 public:
	Neuron(IActivationFunction& af, double bias);

	void set(double val);

	void set_derivative(double d);

	void operator()(const std::shared_ptr<Neuron>& from, double x);

	void connect(const std::shared_ptr<Neuron>& next);

	void forward() override;

	void backprop() override;

 private:
	void update_derivative(const std::shared_ptr<Neuron>& from,double x);

	IActivationFunction* act_func_;

	double bias_;

	double value_;

	double derivative_;

	std::unordered_map<std::shared_ptr<Neuron>, double> in_;

	std::unordered_map<std::shared_ptr<Neuron>, double> act_values_;

	std::unordered_map<std::shared_ptr<Neuron>, double> derivative_values_;

	std::vector<std::weak_ptr<Neuron>> out_;
};
}

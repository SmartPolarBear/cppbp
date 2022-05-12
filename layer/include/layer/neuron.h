//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <layer/activation_function.h>

#include <base/forward.h>
#include <base/backprop.h>
#include <base/summary.h>

#include <optimizer/optimizer.h>

#include <unordered_map>
#include <memory>

namespace cppbp::layer
{
class Neuron
	: public std::enable_shared_from_this<Neuron>,
	  public base::IForward,
	  public base::IBackProp,
	  public base::ISummary,
	  public optimizer::IOptimizable
{
 public:
	explicit Neuron(IActivationFunction& af);

	void set(double val);

	void set_derivative(double d);

	void operator()(const std::shared_ptr<Neuron>& from, double x);

	void connect(const std::shared_ptr<Neuron>& next);

	void forward() override;

	void backprop() override;

	void optimize(optimizer::IOptimizer& opt) override;
	std::string summary() const override;

 private:
	void update_error(const std::shared_ptr<Neuron>& from, double x);

	IActivationFunction* act_func_{ nullptr };

	double value_{};

	double error_{};

	std::unordered_map<std::shared_ptr<Neuron>, std::pair<double, double>> in_{};

	std::unordered_map<std::shared_ptr<Neuron>, double> act_values_{};

	std::unordered_map<std::shared_ptr<Neuron>, double> error_values_{};

	std::vector<std::weak_ptr<Neuron>> out_{};
};
}

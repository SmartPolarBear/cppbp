//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <layer/layer.h>
#include <layer/neuron.h>

#include <vector>

namespace cppbp::layer
{
class FullyConnected
	: public ILayer,
	  public base::IForward,
	  public base::IBackProp,
	  public optimizer::IOptimizable
{
 public:
	explicit FullyConnected(size_t len, IActivationFunction& af);

	void connect(const std::shared_ptr<FullyConnected>& next);

	void set(std::vector<double> values);

	void set_derivatives(std::vector<double> d);

	void backprop() override;

	void forward() override;

	void optimize(optimizer::IOptimizer& optimizable) override;
 private:
	IActivationFunction* act_func_{ nullptr };

	std::vector<std::shared_ptr<Neuron>> neurons_{};
	std::shared_ptr<FullyConnected> next_{};
};

}

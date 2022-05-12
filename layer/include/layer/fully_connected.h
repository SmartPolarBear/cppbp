//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <layer/layer.h>
#include <layer/neuron.h>

#include <utils/counter.h>

#include <cstdint>
#include <vector>

namespace cppbp::layer
{
class FullyConnected
	: public ILayer,
	  public optimizer::IOptimizable,
	  public utils::Counter<FullyConnected>
{
 public:
	explicit FullyConnected(size_t len, IActivationFunction& af);

	FullyConnected& connect(FullyConnected& next);

	void set(std::vector<double> values);

	std::vector<double> get() const;

	void set_derivatives(std::vector<double> d);

	void backprop() override;

	void forward() override;

	void optimize(optimizer::IOptimizer& optimizable) override;

	std::string name() const override;

	std::string summary() const override;
 private:
	uint64_t id_{};

	IActivationFunction* act_func_{ nullptr };

	std::vector<std::shared_ptr<Neuron>> neurons_{};

	FullyConnected* next_{};
	FullyConnected* prev_{};
};

}

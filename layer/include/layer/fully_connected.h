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
	ILayer* next() override;
	ILayer* prev() override;
 public:
	explicit FullyConnected(size_t len, IActivationFunction& af);

	ILayer& connect(ILayer& next) override;

	ILayer& operator|(ILayer& next) override;

	void set(std::vector<double> values) override;

	[[nodiscard]] std::vector<double> get() const override;

	void set_derivatives(std::vector<double> d);

	void backprop() override;

	void forward() override;

	void optimize(optimizer::IOptimizer& opt) override;

	[[nodiscard]] std::string name() const override;

	[[nodiscard]] std::string summary() const override;
 private:
	uint64_t id_{};

	IActivationFunction* act_func_{ nullptr };

	std::vector<std::shared_ptr<Neuron>> neurons_{};

	FullyConnected* next_{};
	FullyConnected* prev_{};
};

}

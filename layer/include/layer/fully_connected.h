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
	  public utils::Counter<FullyConnected>
{
 public:
	ILayer* next() override;
	ILayer* prev() override;
 public:
	explicit FullyConnected(size_t len, IActivationFunction& af);

	void reshape(size_t input) override;

	ILayer& connect(ILayer& next) override;

	ILayer& operator|(ILayer& next) override;

	void set(Eigen::VectorXd vec) override ;

	[[nodiscard]] std::vector<double> get() const override;

	void set_deltas(Eigen::VectorXd dlts) override;

	void backprop() override;

	void forward() override;

	void set_errors(Eigen::VectorXd errors) override;

	void optimize(optimizer::IOptimizer& opt) override;

	[[nodiscard]] std::string name() const override;

	[[nodiscard]] std::string summary() const override;

	IActivationFunction& activation_function() override;

 private:
	uint64_t id_{};

	IActivationFunction* act_func_{ nullptr };

	Eigen::MatrixXd weights_{};

	Eigen::VectorXd input_;
	Eigen::VectorXd activations_;

	Eigen::VectorXd deltas_;
	Eigen::VectorXd errors_;

	size_t len_{};

	FullyConnected* next_{};
	FullyConnected* prev_{};
};

}

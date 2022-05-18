//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <layer/activation_function.h>

#include <base/forward.h>
#include <base/backprop.h>
#include <base/summary.h>

#include <optimizer/optimizer.h>

#include <Eigen/Eigen>

#include <unordered_map>
#include <memory>

namespace cppbp::layer
{
class Neuron
	: public std::enable_shared_from_this<Neuron>,
	  public base::ISummary,
	  public base::INamable,
	  public optimizer::IOptimizable
{
 public:
	explicit Neuron(IActivationFunction& af);

	Neuron(IActivationFunction& af, uint64_t parent_id, uint64_t id);

	double operator()(Eigen::VectorXd input);


	void reshape(size_t input);

	void optimize(optimizer::IOptimizer& opt) override;

	std::string summary() const override;

	std::string name() const override;

 private:
	std::string name_{};

	uint64_t parent_id_;

	uint64_t id_;

	IActivationFunction* act_func_{ nullptr };

	double bias_{};

	double error_{};

	double activation_{};

	Eigen::VectorXd weights_;

	Eigen::VectorXd input_;

};
}

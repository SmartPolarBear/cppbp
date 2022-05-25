//
// Created by cleve on 5/24/2022.
//

#pragma once

#include <base/serializable.h>

#include <layer/layer.h>

#include <model/persist.h>

#include <utils/counter.h>

#include <Eigen/Eigen>

#include <layer/sigmoid.h>

#include <cstdint>
#include <vector>

namespace cppbp::layer
{
class DropOut
	: public ILayer,
	  public utils::Counter<DropOut>
{
 public:
	explicit DropOut(double drop_prob = 0.1);

	void backprop() override;
	void forward() override;
	ILayer* next() override;
	ILayer* prev() override;
	void set_prev(ILayer* prev) override;
	void set_next(ILayer* next) override;
	std::tuple<std::shared_ptr<char[]>, size_t> serialize() override;
	char* deserialize(char* data) override;
	std::string name() const override;
	std::string summary() const override;
	ILayer& connect(ILayer& next) override;
	void set(Eigen::VectorXd vec) override;
	void set_deltas(Eigen::VectorXd deltas) override;
	void set_errors(Eigen::VectorXd errors) override;
	Eigen::VectorXd get() const override;
	IActivationFunction& activation_function() override;
	ILayer& operator|(ILayer& next) override;
	void reshape(size_t input) override;
	void optimize(optimizer::IOptimizer& iOptimizer) override;

 private:
	uint64_t id_{};
	size_t input_{};

	ILayer* next_{};
	ILayer* prev_{};

	double drop_prob_{};

	Eigen::VectorXd errors_;
	Eigen::VectorXd values_;

	Sigmoid placeholder{};
};
}// namespace cppbp::layer
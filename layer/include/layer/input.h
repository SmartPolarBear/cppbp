//
// Created by cleve on 5/18/2022.
//

#pragma once

#include <base/serializable.h>

#include <layer/layer.h>
#include <layer/sigmoid.h>

#include <model/persist.h>

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace cppbp::layer
{
class Input
	: public ILayer
{
 public:
	explicit Input(size_t size);

	void backprop() override;
	void forward() override;
	std::tuple<std::shared_ptr<char[]>, size_t> serialize() override;
	char* deserialize(char* data) override;
	ILayer* next() override;
	ILayer* prev() override;
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
	void set_prev(ILayer* prev) override;
	void set_next(ILayer* next) override;

 private:
	size_t len_;
	Eigen::VectorXd values_;

	ILayer* next_{};

	Sigmoid placeholder{};

};

}

template<>
struct cppbp::model::persist::LayerTypeId<cppbp::layer::Input>
{
	static inline constexpr uint32_t value = 1;
};

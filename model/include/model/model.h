//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <layer/layer.h>

#include <optimizer/optimizer.h>
#include <optimizer/loss.h>

#include <dataloader/dataloader.h>

#include <vector>

namespace cppbp::model
{
class Model
	: public base::IForward,
	  public base::IBackProp,
	  public base::ISummary,
	  public base::INamable,
	  public optimizer::IOptimizable
{
 public:
	explicit Model(layer::ILayer& layer, optimizer::ILossFunction& loss);

	explicit Model(std::vector<layer::ILayer>& layers, optimizer::ILossFunction& loss);

	std::vector<double> operator()(std::vector<double> input);

	void set(std::vector<double> values);

	void forward() override;

	void backprop() override;

	void fit(cppbp::dataloader::DataLoader& dl,
		size_t epoches,
		cppbp::optimizer::IOptimizer& opt,
		bool varbose);

	std::string name() const override;

	std::string summary() const override;

	void optimize(optimizer::IOptimizer& iOptimizer) override;

 private:
	layer::ILayer* input_{}, * output_{};
	std::string name_{ "Model" };
	optimizer::ILossFunction* loss_{};
};
}
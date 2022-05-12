//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <layer/layer.h>
#include <optimizer/optimizer.h>

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
	explicit Model(layer::ILayer& input);

	explicit Model(std::vector<layer::ILayer> layers);

	// TODO: in train mode, automatically do a backprop
	std::vector<double> operator()(std::vector<double> input);

	void forward() override;

	void backprop() override;

	void train();

	void eval();

 private:

};
}
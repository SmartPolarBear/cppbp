//
// Created by cleve on 5/11/2022.
//

#include <model/model.h>

#include <utility>

cppbp::model::Model::Model(cppbp::layer::ILayer& layer)
	: input_(&layer), output_(&layer)
{
	while (input_->prev())
	{
		input_ = input_->prev();
	}

	while (output_->next())
	{
		output_ = output_->next();
	}
}

cppbp::model::Model::Model(std::vector<layer::ILayer>& layers)
{
	for (int i = 1; i < layers.size(); ++i)
	{
		layers[i - 1].connect(layers[i]);
	}

	input_ = &layers.front();
	output_ = &layers.back();
}

void cppbp::model::Model::set(std::vector<double> values)
{
	input_->set(std::move(values));
}

std::vector<double> cppbp::model::Model::operator()(std::vector<double> input)
{
	input_->set(std::move(input));
	input_->forward();
	return output_->get();
}

void cppbp::model::Model::forward()
{
	input_->forward();
}

void cppbp::model::Model::backprop()
{
	output_->backprop();
}

void cppbp::model::Model::train()
{

}

void cppbp::model::Model::eval()
{

}

std::string cppbp::model::Model::name() const
{
	return name_;
}

std::string cppbp::model::Model::summary() const
{
	return input_->summary();
}

void cppbp::model::Model::optimize(cppbp::optimizer::IOptimizer& iOptimizer)
{

}

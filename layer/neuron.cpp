//
// Created by cleve on 5/11/2022.
//

#include <layer/neuron.h>
#include <utils/random.h>

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af, double bias)
	: act_func_(&af), bias_(bias)
{
}

void cppbp::layer::Neuron::set(double val)
{
	value_ = val;
}

void cppbp::layer::Neuron::set_derivative(double d)
{
	derivative_ = d;
}

void cppbp::layer::Neuron::operator()(const std::shared_ptr<Neuron>& from, double x)
{
	auto weight = in_.at(from);
	this->act_values_[from] = weight * x;
	value_ += weight * x;
}

void cppbp::layer::Neuron::update_derivative(const std::shared_ptr<Neuron>& from, double x)
{
	derivative_ += (*act_func_).derive(value_) * x;
	this->derivative_values_[from] = (*act_func_).derive(value_) * x
}

void cppbp::layer::Neuron::connect(const std::shared_ptr<Neuron>& next)
{
	next->in_[this->shared_from_this()] = utils::random::uniform(0.1, 1.0);
	this->out_.emplace_back(next);
}

void cppbp::layer::Neuron::forward()
{
	auto act_val = (*act_func_)(this->value_ + bias_);
	for (auto& next : out_)
	{
		(*next.lock())(this->shared_from_this(), act_val);
	}
}

void cppbp::layer::Neuron::backprop()
{
	for (auto& [prev, w] : in_)
	{
		prev->update_derivative(this->shared_from_this(), w * this->derivative_);
	}
}


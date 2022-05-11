//
// Created by cleve on 5/11/2022.
//

#include <layer/neuron.h>

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af, double val)
	: act_func_(&af), val_(val)
{
}

double cppbp::layer::Neuron::operator()(double x)
{
	val_ = x;
	return (*this)();
}

double cppbp::layer::Neuron::operator()()
{
	return (*act_func_)(val_);
}

double cppbp::layer::Neuron::derive()
{
	return act_func_->derive(val_);
}

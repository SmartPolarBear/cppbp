//
// Created by cleve on 5/11/2022.
//

#include <layer/neuron.h>
#include <utils/random.h>

#include <sstream>
#include <utility>

using namespace std;

using namespace Eigen;

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af)
	: act_func_(&af), bias_(utils::random::uniform(0, 0.5))
{
}

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af, uint64_t parent_id, uint64_t id)
	: Neuron(af)
{
	parent_id_ = parent_id;
	id_ = id;
}

double cppbp::layer::Neuron::operator()(Eigen::VectorXd input)
{
	input_ = std::move(input);
	activation_ = act_func_->eval(input_.dot(weights_.transpose()) + bias_);
	return activation_;
}

void cppbp::layer::Neuron::optimize(cppbp::optimizer::IOptimizer& opt)
{
}

string cppbp::layer::Neuron::summary() const
{
	stringstream ss{};

	ss << "Neuron " << name() << "[" << weights_.transpose() << "], bias = " << bias_;

	return ss.str();
}

string cppbp::layer::Neuron::name() const
{
	return "(" + to_string(parent_id_) + "," + to_string(id_) + ")";
}

void cppbp::layer::Neuron::reshape(size_t input)
{
	weights_ = VectorXd(input);
	for (int i = 0; i < input; i++)
	{
		weights_[i] = utils::random::uniform(-1, 1);
	}
}

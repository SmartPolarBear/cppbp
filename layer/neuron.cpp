//
// Created by cleve on 5/11/2022.
//

#include <layer/neuron.h>
#include <utils/random.h>

#include <sstream>

using namespace std;

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af)
	: act_func_(&af)
{
}

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af, uint64_t parent_id, uint64_t id)
	: Neuron(af)
{
	parent_id_ = parent_id;
	id_ = id;
}

void cppbp::layer::Neuron::set(double val)
{
	value_ = val;
}

void cppbp::layer::Neuron::set_derivative(double d)
{
	error_ = d;
}

double cppbp::layer::Neuron::get() const
{
	return (*act_func_)(value_);
}

void cppbp::layer::Neuron::operator()(const std::shared_ptr<Neuron>& from, double x)
{
	auto p = in_.at(from);
	this->act_values_[from] = p.first * x + p.second;
	value_ += this->act_values_[from];
}

void cppbp::layer::Neuron::update_error(const std::shared_ptr<Neuron>& from, double x)
{
	error_ += (*act_func_).derive(value_) * x;
	this->error_values_[from] = x;
}

void cppbp::layer::Neuron::connect(const std::shared_ptr<Neuron>& next)
{
	next->in_[this->shared_from_this()] =
		make_pair(utils::random::uniform(0.1, 1.0), utils::random::uniform(0.1, 1.0));
	this->out_.emplace_back(next);
}

void cppbp::layer::Neuron::forward()
{
	auto act_val = (*act_func_)(this->value_);
	for (auto& next : out_)
	{
		(*next.lock())(this->shared_from_this(), act_val);
	}
}

void cppbp::layer::Neuron::backprop()
{
	for (auto& [prev, w] : in_)
	{
		prev->update_error(this->shared_from_this(), w.first * this->error_);
	}
}

void cppbp::layer::Neuron::optimize(cppbp::optimizer::IOptimizer& opt)
{
	for (auto& [f, v] : error_values_)
	{
		in_[f].first -= opt.optimize(in_[f].first, v * act_values_[f]);
		in_[f].second -= opt.optimize(in_[f].second, v);
	}
}
string cppbp::layer::Neuron::summary() const
{
	stringstream ss{};

	ss << "Neuron " << name() << "[" << in_.size() << "]:{\n";
	for (auto& [f, w] : in_)
	{
		ss << "[" << w.first << "," << w.second << "] from " << f->name() << "\n";
	}
	if (in_.empty())
	{
		ss << "<No incoming edge connected>\n";
	}
	ss << "}";
	return ss.str();
}

string cppbp::layer::Neuron::name() const
{
	return "(" + to_string(parent_id_) + "," + to_string(id_) + ")";
}



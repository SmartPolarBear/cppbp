//
// Created by cleve on 5/11/2022.
//

#include <layer/neuron.h>
#include <utils/random.h>

#include <sstream>

using namespace std;

cppbp::layer::Neuron::Neuron(cppbp::layer::IActivationFunction& af)
	: act_func_(&af), bias_(0)
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

void cppbp::layer::Neuron::set_error(double d)
{
	error_ = d;
}

double cppbp::layer::Neuron::get() const
{
	return (*act_func_)(value_ + bias_);
}

void cppbp::layer::Neuron::operator()(const std::shared_ptr<Neuron>& from, double x)
{
	auto p = in_.at(from);
	this->act_values_[from] = p * x;
	value_ += this->act_values_[from];
}

void cppbp::layer::Neuron::update_error(const std::shared_ptr<Neuron>& from, double x)
{
	error_ += act_func_->derive(get()) * x;
	this->error_values_[from] = x;
}

void cppbp::layer::Neuron::connect(const std::shared_ptr<Neuron>& next)
{
	next->in_[this->shared_from_this()] = utils::random::uniform(-1, 1);
	this->out_.emplace_back(next);
}

void cppbp::layer::Neuron::forward()
{
	for (auto& next : out_)
	{
		(*next.lock())(this->shared_from_this(), get());
	}
}

void cppbp::layer::Neuron::backprop()
{
	for (auto& [prev, w] : in_)
	{
		prev->update_error(this->shared_from_this(), w * this->error_);
	}
}

void cppbp::layer::Neuron::optimize(cppbp::optimizer::IOptimizer& opt)
{
	double db = 0;
	for (auto& [f, v] : act_values_)
	{
		db += error_;
	}
//	db /= act_values_.size();
//	bias_ -= db;

	for (auto& [f, v] : act_values_)
	{
		in_[f] = opt.optimize(in_[f], error_ * v);
	}
}
string cppbp::layer::Neuron::summary() const
{
	stringstream ss{};

	ss << "Neuron " << name() << "[" << in_.size() << "], bias = " << bias_ << ":{\n";
	for (auto& [f, w] : in_)
	{
		ss << "[" << w << "] from " << f->name() << "\n";
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



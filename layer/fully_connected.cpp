//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>

#include <sstream>

using namespace std;

cppbp::layer::FullyConnected::FullyConnected(size_t len, cppbp::layer::IActivationFunction& af)
	: act_func_(&af), next_(nullptr)
{
	id_ = cppbp::layer::FullyConnected::objects_alive;

	for (int i = 0; i < len; i++)
	{
		neurons_.emplace_back(std::make_shared<Neuron>(af, id_, i));
	}
}

cppbp::layer::ILayer& cppbp::layer::FullyConnected::connect(ILayer& n)
{
	auto& next = dynamic_cast<FullyConnected&>(n); //FIXME

	for (const auto& t : neurons_)
	{
		for (const auto& n : next.neurons_)
		{
			t->connect(n);
		}
	}
	next_ = &next;
	next.prev_ = this;

	return next;
}

void cppbp::layer::FullyConnected::backprop()
{
	for (auto& n : neurons_)
	{
		n->backprop();
	}

	if (prev_)
	{
		prev_->backprop();
	}
}

void cppbp::layer::FullyConnected::forward()
{
	for (auto& n : neurons_)
	{
		n->forward();
	}

	if (next_)
	{
		next_->forward();
	}
}

void cppbp::layer::FullyConnected::optimize(cppbp::optimizer::IOptimizer& opt)
{
	for (auto& n : neurons_)
	{
		n->optimize(opt);
	}

	if (next_)
	{
		next_->optimize(opt);
	}
}

void cppbp::layer::FullyConnected::set(std::vector<double> values)
{
	for (int i = 0; i < values.size(); i++)
	{
		neurons_[i]->set(values[i]);
	}
}

void cppbp::layer::FullyConnected::set_derivatives(std::vector<double> d)
{
	for (int i = 0; i < d.size(); i++)
	{
		neurons_[i]->set_derivative(d[i]);
	}
}
std::string cppbp::layer::FullyConnected::summary() const
{
	stringstream ss{};
	ss << "Fully Connected [" << neurons_.size() << "]:{\n";
	for (const auto& n : neurons_)
	{
		ss << "" << n->summary() << "\n";
	}
	ss << "}";
	if (next_)
	{
		ss << "\n";
		ss << next_->summary();
	}

	return ss.str();
}
std::vector<double> cppbp::layer::FullyConnected::get() const
{
	std::vector<double> vals{};
	for (const auto& n : neurons_)
	{
		vals.emplace_back(n->get());
	}
	return vals;
}

string cppbp::layer::FullyConnected::name() const
{
	return "fc" + to_string(id_);
}

cppbp::layer::ILayer& cppbp::layer::FullyConnected::operator|(cppbp::layer::ILayer& next)
{
	return connect(next);
}

cppbp::layer::ILayer* cppbp::layer::FullyConnected::next()
{
	return next_;
}

cppbp::layer::ILayer* cppbp::layer::FullyConnected::prev()
{
	return prev_;
}


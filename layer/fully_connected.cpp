//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>

#include <sstream>

using namespace std;

using namespace Eigen;

cppbp::layer::FullyConnected::FullyConnected(size_t len, cppbp::layer::IActivationFunction& af)
	: act_func_(&af), len_(len), next_(nullptr)
{
	id_ = cppbp::layer::FullyConnected::objects_alive;

	for (int i = 0; i < len; i++)
	{
		neurons_.emplace_back(std::make_shared<Neuron>(af, id_, i));
		neurons_.back()->reshape(len);
	}
}

cppbp::layer::ILayer& cppbp::layer::FullyConnected::connect(ILayer& n)
{
	auto& next = dynamic_cast<FullyConnected&>(n); //FIXME

	next.reshape(len_);

	next_ = &next;
	next.prev_ = this;

	return next;
}

void cppbp::layer::FullyConnected::backprop()
{
//	for (auto& n : neurons_)
//	{
//		n->backprop();
//	}
//
//	if (prev_)
//	{
//		prev_->backprop();
//	}
}

void cppbp::layer::FullyConnected::forward()
{
	if (next_)
	{
		next_->set(activations_);
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

void cppbp::layer::FullyConnected::set(VectorXd vec)
{
	activations_ = VectorXd(len_);
	for (int i = 0; i < len_; i++)
	{
		activations_[i] = (*neurons_[i])(vec);
	}
}

void cppbp::layer::FullyConnected::set_errors(Eigen::VectorXd error)
{
//	for (int i = 0; i < error.size(); i++)
//	{
//		neurons_[i]->set_error(d[i]);
//	}
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
	for (int i = 0; i < len_; i++)
	{
		vals.emplace_back(activations_[i]);
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

cppbp::layer::IActivationFunction& cppbp::layer::FullyConnected::activation_function()
{
	return *act_func_;
}

void cppbp::layer::FullyConnected::reshape(size_t input)
{
	for (auto& n : neurons_)
	{
		n->reshape(input);
	}
}


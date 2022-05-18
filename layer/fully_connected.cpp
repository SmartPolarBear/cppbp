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
}

cppbp::layer::ILayer& cppbp::layer::FullyConnected::connect(ILayer& next)
{
	next.reshape(len_);

	this->set_next(&next);
	next.set_prev(this);

	return next;
}

void cppbp::layer::FullyConnected::backprop()
{
	VectorXd prev_activation;

	prev_activation << 1, prev_->get(); // TODO: add a separate input layer class

	prev_->set_deltas(errors_.cwiseProduct(act_func_->derive(activations_)));
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
	VectorXd aug;
	aug << 1, activations_;
	weights_ += 0.1 * deltas_.transpose() * aug;// TODO: use optimizer

	if (next_)
	{
		next_->optimize(opt);
	}
}

void cppbp::layer::FullyConnected::set(VectorXd vec)
{
	input_ = vec;
	VectorXd aug;
	aug << 1, vec;
	activations_ = act_func_->eval(weights_ * aug);
}

void cppbp::layer::FullyConnected::set_deltas(Eigen::VectorXd dlts)
{
	deltas_ = dlts;
	errors_ = deltas_ * weights_.block(0, 1, weights_.rows(), weights_.cols() - 1);
}

std::string cppbp::layer::FullyConnected::summary() const
{
	stringstream ss{};
	ss << "Fully Connected [" << len_ << "]:{\n";
	for (const auto& row : weights_.rowwise())
	{
		ss << "" << row << "\n";
	}
	ss << "}";

	if (next_)
	{
		ss << "\n";
		ss << next_->summary();
	}

	return ss.str();
}

Eigen::VectorXd cppbp::layer::FullyConnected::get() const
{
	return activations_;
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
	weights_ = MatrixXd::Random(len_, input + 1);
}

void cppbp::layer::FullyConnected::set_errors(Eigen::VectorXd errors)
{
	errors_ = errors;
}

void cppbp::layer::FullyConnected::set_prev(cppbp::layer::ILayer* prev)
{
	prev_ = prev;
}

void cppbp::layer::FullyConnected::set_next(cppbp::layer::ILayer* next)
{
	next_ = next;
}

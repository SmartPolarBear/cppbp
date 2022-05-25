//
// Created by cleve on 5/24/2022.
//

#include <layer/dropout.h>

#include <fmt/format.h>

#include <iostream>
#include <sstream>

#include <gsl/assert>

using namespace std;
using namespace gsl;

using namespace Eigen;

cppbp::layer::DropOut::DropOut(double drop_prob)
	: next_(nullptr), drop_prob_(drop_prob)
{
	id_ = cppbp::layer::DropOut::objects_alive;
}

void cppbp::layer::DropOut::backprop()
{
	if (prev())
	{
		prev()->set_errors(errors_);
		prev()->backprop();
	}
}

void cppbp::layer::DropOut::forward()
{
	if (next())
	{
		next()->set(values_);
		next()->forward();
	}
}

cppbp::layer::ILayer* cppbp::layer::DropOut::next()
{
	return next_;
}

cppbp::layer::ILayer* cppbp::layer::DropOut::prev()
{
	return prev_;
}

void cppbp::layer::DropOut::set_prev(cppbp::layer::ILayer* prev)
{
	prev_ = prev;
}

void cppbp::layer::DropOut::set_next(cppbp::layer::ILayer* next)
{
	next_ = next;
}

std::tuple<std::shared_ptr<char[]>, size_t> cppbp::layer::DropOut::serialize()
{
	// TODO
	return std::tuple<std::shared_ptr<char[]>, size_t>();
}

char* cppbp::layer::DropOut::deserialize(char* data)
{
	// TODO
	return nullptr;
}

std::string cppbp::layer::DropOut::name() const
{
	return fmt::format("Dropout {}", id_);
}

std::string cppbp::layer::DropOut::summary() const
{
	stringstream ss{};
	ss << name() << fmt::format("[Probability: {}]", drop_prob_);

	if (next_)
	{
		ss << "\n";
		ss << next_->summary();
	}

	return ss.str();
}

cppbp::layer::ILayer& cppbp::layer::DropOut::connect(cppbp::layer::ILayer& next)
{
	next.reshape(input_);

	this->set_next(&next);
	next.set_prev(this);

	return next;
}

void cppbp::layer::DropOut::set(Eigen::VectorXd vec)
{
	Expects(!vec.hasNaN());

	auto p = 1 - drop_prob_;
	VectorXd mask = ((VectorXd::Random(vec.size()) + VectorXd::Ones(vec.size())) / 2);

	for (int i = 0; i < mask.size(); i++)
	{
		mask[i] = static_cast<double>(mask[i] < p);
	}

	mask /= p;

	values_ = vec.cwiseProduct(mask);

	Ensures(!values_.hasNaN());
}

void cppbp::layer::DropOut::set_deltas(Eigen::VectorXd deltas)
{
}

void cppbp::layer::DropOut::set_errors(Eigen::VectorXd errors)
{
	errors_ = errors;
}

Eigen::VectorXd cppbp::layer::DropOut::get() const
{
	return values_;
}

cppbp::layer::IActivationFunction& cppbp::layer::DropOut::activation_function()
{
	return placeholder;//FIXME: should return nothing
}

cppbp::layer::ILayer& cppbp::layer::DropOut::operator|(cppbp::layer::ILayer& next)
{
	return connect(next);
}

void cppbp::layer::DropOut::reshape(size_t input)
{
	//	if (input == weights_.cols()) return;
	//
	//	weights_ = MatrixXd::Random(len_, input + 1);
	input_ = input;
}

void cppbp::layer::DropOut::optimize(cppbp::optimizer::IOptimizer& opt)
{
	if (next())
	{
		next()->optimize(opt);
	}
}

//
// Created by cleve on 5/18/2022.
//

#include <layer/input.h>

#include <model/persist.h>

#include <fmt/format.h>

using namespace std;
using namespace Eigen;

using namespace cppbp::model::persist;

cppbp::layer::Input::Input(size_t size)
	: len_(size)
{
}

void cppbp::layer::Input::backprop()
{
	// Do nothing
}

void cppbp::layer::Input::forward()
{
	if (next_)
	{
		next_->set(values_);
		next_->forward();
	}
}

cppbp::layer::ILayer* cppbp::layer::Input::next()
{
	return next_;
}

cppbp::layer::ILayer* cppbp::layer::Input::prev()
{
	return nullptr; // It must be the leftmost layer
}

std::string cppbp::layer::Input::name() const
{
	return "Input";
}

std::string cppbp::layer::Input::summary() const
{
	stringstream ss{};
	ss << fmt::format("Input [{}]", len_);

	if (next_)
	{
		ss << "\n";
		ss << next_->summary();
	}

	return ss.str();
}

cppbp::layer::ILayer& cppbp::layer::Input::connect(cppbp::layer::ILayer& next)
{
	next.reshape(len_);

	this->set_next(&next);
	next.set_prev(this);

	return next;
}

void cppbp::layer::Input::set(Eigen::VectorXd vec)
{
	values_ = vec;
}

void cppbp::layer::Input::set_deltas(Eigen::VectorXd deltas)
{
	// Do nothing
}

void cppbp::layer::Input::set_errors(Eigen::VectorXd errors)
{
	// Do nothing
}

Eigen::VectorXd cppbp::layer::Input::get() const
{
	return values_;
}

cppbp::layer::IActivationFunction& cppbp::layer::Input::activation_function()
{
	return placeholder; //FIXME: should return nothing
}

cppbp::layer::ILayer& cppbp::layer::Input::operator|(cppbp::layer::ILayer& next)
{
	return connect(next);
}

void cppbp::layer::Input::reshape(size_t input)
{
	len_ = input;
}

void cppbp::layer::Input::optimize(cppbp::optimizer::IOptimizer& opt)
{
	if (next())
	{
		next()->optimize(opt);
	}
}

void cppbp::layer::Input::set_prev(cppbp::layer::ILayer* prev)
{
	// Do nothing
}

void cppbp::layer::Input::set_next(cppbp::layer::ILayer* next)
{
	next_ = next;
}

std::tuple<std::unique_ptr<char>, size_t> cppbp::layer::Input::serialize()
{
	auto size = sizeof(LayerDescriptor);

	auto ret = make_unique<char[]>(size);

	auto desc = reinterpret_cast<LayerDescriptor*>(ret.get());
	desc->type = LDT_INPUT;
	desc->act_func = ACF_NONE;

	return { ret, size };
}

std::unique_ptr<char> cppbp::layer::Input::deserialize(std::unique_ptr<char> data)
{

}

//
// Created by cleve on 5/26/2022.
//

#include <layer/layer_norm.h>

#include <sstream>

#include <fmt/format.h>
#include <gsl/assert>

using namespace std;

cppbp::layer::BatchNorm::BatchNorm()
        : id_(cppbp::layer::BatchNorm::objects_alive)
{
}

void cppbp::layer::BatchNorm::backprop()
{

}

void cppbp::layer::BatchNorm::forward()
{
    if (next())
    {
        next()->set(activations_);
        next()->forward();
    }
}

void cppbp::layer::BatchNorm::set(Eigen::VectorXd vec)
{
    Expects(!vec.hasNaN());

    input_ = vec;

    auto average = input_.sum() / input_.size();
    auto sq_average = input_.cwiseProduct(input_).sum() / input_.size();

    sigma_ = std::sqrt(sq_average - average * average + base::EPS);

    scaled_input_ = input_ - Eigen::VectorXd::Ones(input_.size()) * average;
    normalized_input_ = scaled_input_ / sigma_;

    activations_ = normalized_input_.cwiseProduct(gammas_) + betas_;

    Ensures(!activations_.hasNaN());
}

void cppbp::layer::BatchNorm::optimize(cppbp::optimizer::IOptimizer &optimizer_1)
{

}

void cppbp::layer::BatchNorm::set_deltas(Eigen::VectorXd dlts)
{
    deltas_ = dlts;
}

void cppbp::layer::BatchNorm::set_errors(Eigen::VectorXd errors)
{
    errors_ = errors;
}

Eigen::VectorXd cppbp::layer::BatchNorm::get() const
{
    return activations_;
}

cppbp::layer::ILayer *cppbp::layer::BatchNorm::next()
{
    return next_;
}

cppbp::layer::ILayer *cppbp::layer::BatchNorm::prev()
{
    return prev_;
}

void cppbp::layer::BatchNorm::set_prev(cppbp::layer::ILayer *prev)
{
    prev_ = prev;
}

void cppbp::layer::BatchNorm::set_next(cppbp::layer::ILayer *next)
{
    next_ = next;
}

std::tuple<std::shared_ptr<char[]>, size_t> cppbp::layer::BatchNorm::serialize()
{
    return {};
}

char *cppbp::layer::BatchNorm::deserialize(char *data)
{
    return data;
}

std::string cppbp::layer::BatchNorm::name() const
{
    return fmt::format("Layer Norm {}", id_);
}

std::string cppbp::layer::BatchNorm::summary() const
{
    stringstream ss{};
    ss << name();

    if (next_)
    {
        ss << "\n";
        ss << next_->summary();
    }

    return ss.str();
}

cppbp::layer::ILayer &cppbp::layer::BatchNorm::connect(cppbp::layer::ILayer &next)
{
    next.reshape(input_len_);

    this->set_next(&next);
    next.set_prev(this);

    return next;
}

cppbp::layer::IActivationFunction &cppbp::layer::BatchNorm::activation_function()
{
    return placeholder;//FIXME: should return nothing
}

cppbp::layer::ILayer &cppbp::layer::BatchNorm::operator|(cppbp::layer::ILayer &next)
{
    return connect(next);
}

void cppbp::layer::BatchNorm::reshape(size_t input)
{
    input_len_ = input;
}

//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>


#include <utils/utils.h>

#include <iostream>
#include <sstream>

#include <fmt/format.h>

#include <gsl/assert>

using namespace cppbp::base;
using namespace cppbp::utils;

using namespace std;
using namespace gsl;

using namespace Eigen;

cppbp::layer::FullyConnected::FullyConnected(size_t len, cppbp::layer::IActivationFunction &af)
        : act_func_(&af), len_(len), next_(nullptr)
{
    id_ = cppbp::layer::FullyConnected::objects_alive;
}

cppbp::layer::ILayer &cppbp::layer::FullyConnected::connect(ILayer &next)
{
    next.reshape(len_);

    this->set_next(&next);
    next.set_prev(this);

    return next;
}

void cppbp::layer::FullyConnected::backprop()
{
    VectorXd prev_activation(1 + prev()->get().size());
    prev_activation << 1, prev()->get();

    auto derives = act_func_->derive(activations_);
    deltas_ = act_func_->derive(activations_).transpose() * errors_;

    VectorXd errors = deltas_.transpose() * weights_.block(0, 1, weights_.rows(), weights_.cols() - 1);

    if (prev())
    {
        prev()->set_errors(errors);
        prev()->backprop();
    }
}

void cppbp::layer::FullyConnected::forward()
{
    if (next())
    {
        next()->set(activations_);
        next()->forward();
    }
}

void cppbp::layer::FullyConnected::optimize(cppbp::optimizer::IOptimizer &opt)
{
    Expects(!weights_.hasNaN());

    VectorXd aug{1 + prev()->get().size()};
    aug << 1, prev()->get();
    weights_ = opt.optimize(weights_, deltas_ * aug.transpose());

    if (next_)
    {
        next_->optimize(opt);
    }

    Ensures(!weights_.hasNaN());
}

void cppbp::layer::FullyConnected::set(VectorXd vec)
{
    Expects(!vec.hasNaN());

    input_ = vec;
    VectorXd aug(vec.size() + 1);
    aug << 1, vec;
    activations_ = act_func_->eval(weights_ * aug);

    Ensures(!activations_.hasNaN());
}

void cppbp::layer::FullyConnected::set_deltas(Eigen::VectorXd dlts)
{
    deltas_ = dlts;
}

std::string cppbp::layer::FullyConnected::summary() const
{
    stringstream ss{};
    ss << fmt::format("Fully Connected [{} neurons]:{{\n", len_);
    for (const auto &row: weights_.rowwise())
    {
        ss << fmt::format("[1 Bias, {} weights]=", len_) << row << "\n";// TODO: custom formatter
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
    return fmt::format("fc {}", id_);
}

cppbp::layer::ILayer &cppbp::layer::FullyConnected::operator|(cppbp::layer::ILayer &next)
{
    return connect(next);
}

cppbp::layer::ILayer *cppbp::layer::FullyConnected::next()
{
    return next_;
}

cppbp::layer::ILayer *cppbp::layer::FullyConnected::prev()
{
    return prev_;
}

cppbp::layer::IActivationFunction &cppbp::layer::FullyConnected::activation_function()
{
    return *act_func_;
}

void cppbp::layer::FullyConnected::reshape(size_t input)
{
    if (input == weights_.cols()) return;

    if (act_func_)
    {
        auto initializer = act_func_->default_initializer();
        weights_ = initializer->initialize_weights(len_, input + 1, input, len_);
    }
    else
    {
        weights_ = MatrixXd::Random(len_, input + 1);
    }
}

void cppbp::layer::FullyConnected::set_errors(Eigen::VectorXd errors)
{
    errors_ = errors;
}

void cppbp::layer::FullyConnected::set_prev(cppbp::layer::ILayer *prev)
{
    prev_ = prev;
}

void cppbp::layer::FullyConnected::set_next(cppbp::layer::ILayer *next)
{
    next_ = next;
}

ostream &cppbp::layer::FullyConnected::serialize(std::ostream &out)
{
    out << magic();
    out << weights_.size() << weights_.rows() << weights_.cols();

    for (int i = 0; i < weights_.size(); i++)
    {
        out << weights_.coeff(i);
    }

    return out;
}

istream &cppbp::layer::FullyConnected::deserialize(istream &input)
{
    if (!check_magic<uint16_t>(*this, input))
    {
        throw; //TODO
    }

    int32_t size{0}, rows{0}, cols{0};
    input >> size >> rows >> cols;

    weights_ = MatrixType(rows, cols);

    for (int i = 0; i < size; i++)
    {
        input >> weights_.coeffRef(i);
    }

    return input;
}

uint16_t cppbp::layer::FullyConnected::magic() const
{
    return utils::magic_from_string<uint16_t>("FC");
}
